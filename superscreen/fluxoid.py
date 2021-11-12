import logging
from typing import Any, Optional, Union, List, Dict, Tuple

import numpy as np
from scipy import optimize

from .solve import solve
from .device import Device
from .solution import Solution


logger = logging.getLogger(__name__)


def make_fluxoid_polygons(
    device: Device,
    holes: Optional[Union[List[str], str]] = None,
    join_style: str = "mitre",
    interp_points: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generates polygons enclosing the given holes to calculate the fluxoid.

    Args:
        device: The Device for which to generate polygons.
        holes: Name(s) of the hole(s) in the device for which to generate polygons.
            Defaults to all holes in the device.
        join_style: See :meth:`superscreen.components.Polygon.buffer`.
        interp_points: If provided, the resulting polygons will be interpolated to
            ``interp_points`` vertices.

    Returns:
        A dict of ``{hole_name: fluxoid_polygon}``.
    """
    device_polygons = {**device.films, **device.holes}
    device_holes = device.holes
    if holes is None:
        holes = list(device_holes)
    if isinstance(holes, str):
        holes = [holes]
    polygons = {}
    for name in holes:
        hole = device_holes[name]
        hole_poly = hole.polygon
        min_dist = min(
            hole_poly.exterior.distance(other.polygon.exterior)
            for other in device_polygons.values()
            if other.layer == hole.layer and other.name != name
        )
        delta = min_dist / 2
        new_poly = hole.buffer(delta, join_style=join_style)
        if interp_points:
            new_poly = new_poly.resample(interp_points)
        polygons[name] = new_poly.points
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
        The optimized :class:`superscreen.solution.Solution` and the minimizer
        results object.
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
    *,
    fluxoids: Dict[str, float],
    x0: Optional[np.ndarray] = None,
    minimize_kwargs: Optional[Dict[str, Any]] = None,
    **solve_kwargs,
) -> Tuple[Solution, Union[optimize.RootResults, optimize.OptimizeResult]]:
    """Calculates the current(s) circulating around hole(s) in the device required
    to realize the specified fluxoid state.

    Args:
        device: The Device for which to find the given fluxoid solution.
        fluxoids: A dict of ``{hole_name: fluxoid_value}``, where ``fluxoid_value`` is
            in units of :math:`\\Phi_0`. The fluxoid for any holes not in this dict
            will not be constrained.
        x0: Initial guess for the circulating currents.
        minimize_kwargs: A dict of keyword arguments passed to
            :func:`scipy.optimize.minimize` (or to :func:`scipy.optimize.root_scalar`
            if there is only a single hole).
        solve_kwargs: Additional keyword arguments are passed to
            :func:`superscreen.solve.solve`.

    Returns:
        The optimized :class:`superscreen.solution.Solution` and the minimizer
        results object.
    """
    if len(fluxoids) < 1:
        raise ValueError("find_fluxoid_solution requires one or more holes.")

    hole_polygon_fluxoid_mapping = {}
    for hole, fluxoid_n in fluxoids.items():
        polygon = polygon = make_fluxoid_polygons(device, holes=hole)[hole]
        hole_polygon_fluxoid_mapping[hole] = (polygon, fluxoid_n)

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
