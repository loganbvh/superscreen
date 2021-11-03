import logging
from typing import Any, Optional, Union, List, Dict, Tuple

import numpy as np
from scipy import interpolate
from scipy import optimize
from scipy import spatial

from .solve import solve
from .device import Device
from .geometry import close_curve
from .solution import Solution


logger = logging.getLogger(__name__)


def make_fluxoid_polygons(
    device: Device,
    holes: Optional[Union[List[str], str]] = None,
    interp_points: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generates polygons enclosing the given holes to calculate the fluxoid.

    Note that this function may fail for non-convex holes.

    Args:
        device: The Device for which to generate polygons.
        holes: Name(s) of the hole(s) in the device for which to generate polygons.
            Defaults to all holes in the device.
        interp_points: If provided, the resulting polygons will be interpolated to
            ``inter_points`` vertices.

    Returns:
        A dict of ``{hole_name: fluxoid_polygon}``.
    """
    device_polygons = device.polygons
    device_holes = device.holes
    if holes is None:
        holes = list(device_holes)
    if isinstance(holes, str):
        holes = [holes]
    polygons = {}
    for name in holes:
        hole = device_holes[name]
        points = hole.points
        center = points.mean(axis=0)
        other_polygons = [
            poly
            for poly in device_polygons.values()
            if poly.layer == hole.layer and poly.name != name
        ]
        other_points = np.concatenate([poly.points for poly in other_polygons])
        min_dist = spatial.distance.cdist(hole.points, other_points).min()
        margin = min_dist / max(hole.extents)
        new_points = (1 + margin) * (points - center) + center
        if interp_points:
            _, ix = np.unique(new_points, axis=0, return_index=True)
            new_points = new_points[np.sort(ix)]
            tck, _ = interpolate.splprep(new_points.T, k=1, s=0)
            x, y = interpolate.splev(np.linspace(0, 1, interp_points), tck)
            new_points = close_curve(np.stack([x, y], axis=1))
        polygons[name] = new_points
    return polygons


def find_single_fluxoid_solution(
    device: Device,
    hole_name: str,
    polygon: Optional[np.ndarray] = None,
    fluxoid_n: float = 0.0,
    minimize_kwargs: Optional[Dict[str, Any]] = None,
    **solve_kwargs,
) -> Tuple[Solution, optimize.RootResults]:
    """Calculates the current circulating around a given hole in the device required
    to realize the specified fluxoid state.

    Args:
        device: The Device for which to find the given fluxoid solution.
        hole_name: The name of the hole for which to optimize the current.
        polygon: Vertices defining a polygon for which to calculate the fluxoid.
            If None is given, the polygon is calculated using
            :func:`make_fluxoid_polygons`.
        fluxoid_n: The desired fluxoid number.
        minimize_kwargs: A dict of keyword arguments passed to
            :func:`scipy.optimize.root_scalar`.
        solve_kwargs: Additional keyword arguments are passed to
            :func:`superscreen.solve.solve`.

    Returns:
        The optimized :class:`superscreen.Solution` and the minimizer results object.
    """
    if minimize_kwargs is None:
        minimize_kwargs = dict(x0=-1, x1=1)
    solve_kwargs = solve_kwargs.copy()

    solution = None

    def residual_fluxoid(current: float) -> float:
        """Returns (actual_fluxoid - desired_fluxoid) in units of Phi_0."""
        nonlocal solution
        circulating_currents = solve_kwargs.get("circulating_currents", {})
        circulating_currents[hole_name] = current
        solve_kwargs["circulating_currents"] = circulating_currents
        solution = solve(device=device, **solve_kwargs)[-1]
        total_fluxoid = sum(solution.hole_fluxoid(hole_name, points=polygon))
        return total_fluxoid.to("Phi_0").magnitude - fluxoid_n

    # Replace optimize.root_scalar with optimize.minimize for multiple holes.
    result = optimize.root_scalar(residual_fluxoid, **minimize_kwargs)
    # Return the optimal Solution
    return solution, result


def find_fluxoid_solution(
    device: Device,
    hole_polygon_fluxoid_mapping: Optional[
        Dict[str, Tuple[Optional[np.ndarray], Optional[float]]]
    ] = None,
    x0: Optional[np.ndarray] = None,
    minimize_kwargs: Optional[Dict[str, Any]] = None,
    **solve_kwargs,
) -> Tuple[Solution, Union[optimize.RootResults, optimize.OptimizeResult]]:
    """Calculates the current(s) circulating around hole(s) in the device required
    to realize the specified fluxoid state.

    Args:
        device: The Device for which to find the given fluxoid solution.
        hole_polygon_fluxoid_mapping: A dict of
            ``{hole_name: (polygon_points, fluxoid_number)}`` specifying the desired
            fluxoid state. If ``polygon_points`` for any ``hole_name``, the polygon
            will be generated using :func:`make_fluxoid_polygons`. ``fluxoid_number``
            defaults to 0.
        x0: Initial guess for the circulating currents.
        minimize_kwargs: A dict of keyword arguments passed to
            :func:`scipy.optimize.minimize` (or to :func:`scipy.optimize.root_scalar`
            if there is only a single hole).
        solve_kwargs: Additional keyword arguments are passed to
            :func:`superscreen.solve.solve`.

    Returns:
        The optimized :class:`superscreen.Solution` and the minimizer results object.
    """
    if hole_polygon_fluxoid_mapping is None:
        hole_polygon_fluxoid_mapping = {}
        for name, poly in make_fluxoid_polygons(device).items():
            hole_polygon_fluxoid_mapping[name] = (poly, 0)
    else:
        for name, (polygon, fluxoid_n) in hole_polygon_fluxoid_mapping.items():
            if polygon is None:
                polygon = make_fluxoid_polygons(device, holes=name)[name]
            if fluxoid_n is None:
                fluxoid_n = 0
            hole_polygon_fluxoid_mapping[name] = (polygon, fluxoid_n)
    for name in device.holes:
        # Constrain all unspecified holes to have zero fluxoid.
        if name not in hole_polygon_fluxoid_mapping:
            polygon = make_fluxoid_polygons(device, holes=name)[name]
            fluxoid_n = 0
            hole_polygon_fluxoid_mapping[name] = (polygon, fluxoid_n)

    if len(hole_polygon_fluxoid_mapping) < 1:
        raise ValueError("find_fluxoid_solution requires one or more holes.")
    if len(hole_polygon_fluxoid_mapping) == 1:
        for hole_name, (polygon, fluxoid_n) in hole_polygon_fluxoid_mapping.items():
            logger.info("Finding fluxoid solution using root finding...")
            return find_single_fluxoid_solution(
                device,
                hole_name=hole_name,
                polygon=polygon,
                fluxoid_n=fluxoid_n,
                minimize_kwargs=minimize_kwargs,
                **solve_kwargs,
            )

    logger.info("Finding fluxoid solution using least-squares minimization...")
    solve_kwargs = solve_kwargs.copy()
    hole_names = list(hole_polygon_fluxoid_mapping)
    solution = None

    def fluxoid_cost(currents: np.ndarray) -> float:
        """Returns sum((actual_fluxoid - desired_fluxoid)**2) in units of Phi_0."""
        nonlocal solution
        circulating_currents = solve_kwargs.get("circulating_currents", {})
        for name, current in zip(hole_names, currents):
            circulating_currents[name] = current
        logger.info(f"Solving device with circulating_currents={circulating_currents}.")
        solve_kwargs["circulating_currents"] = circulating_currents
        solution = solve(device=device, **solve_kwargs)[-1]
        errors = []
        for name, (polygon, target_fluxoid) in hole_polygon_fluxoid_mapping.items():
            total_fluxoid = (
                sum(solution.hole_fluxoid(name, points=polygon)).to("Phi_0").magnitude
            )
            logger.info(
                f"Hole {name}: target = {target_fluxoid:.3e} Phi_0, "
                f"actual = {total_fluxoid:.3e} Phi_0."
            )
            errors.append(total_fluxoid - target_fluxoid)
        total_cost = np.sum(np.square(errors))
        logger.info(f"Total cost = {total_cost:.3e} Phi_0 ** 2.")
        return total_cost

    if x0 is None:
        x0 = np.zeros(len(hole_names))
    minimize_kwargs = minimize_kwargs or {}

    result = optimize.minimize(fluxoid_cost, x0, **minimize_kwargs)
    return solution, result
