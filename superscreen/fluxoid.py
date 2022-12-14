import logging
from typing import Dict, List, Optional, Union

import numpy as np

from .device import Device
from .solution import Solution
from .solve import solve

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


def find_fluxoid_solution(
    device: Device,
    *,
    fluxoids: Optional[Dict[str, float]] = None,
    **solve_kwargs,
) -> Solution:
    """Calculates the current(s) circulating around hole(s) in the device required
    to realize the specified fluxoid state.

    Args:
        device: The Device for which to find the given fluxoid solution.
        fluxoids: A dict of ``{hole_name: fluxoid_value}``, where ``fluxoid_value`` is
            in units of :math:`\\Phi_0`. The fluxoid for any holes not in this dict
            will default to 0.
        solve_kwargs: Additional keyword arguments are passed to
            :func:`superscreen.solve.solve`.

    Returns:
        The optimized :class:`superscreen.Solution`.
    """
    fluxoids = fluxoids or {}
    holes = list(device.holes)
    solve_kwargs = solve_kwargs.copy()
    current_units = solve_kwargs.get("current_units", "uA")
    terminal_currents = solve_kwargs.pop("terminal_currents", None)
    applied_field = solve_kwargs.pop("applied_field", None)
    M = device.mutual_inductance_matrix(
        units=f"Phi_0 / {current_units}", **solve_kwargs
    )
    target_fluxoids = np.array([fluxoids.get(name, 0) for name in holes])

    # Find the hole fluxoids assuming no circulating currents.
    solution_no_circ = solve(
        device,
        applied_field=applied_field,
        terminal_currents=terminal_currents,
        circulating_currents=None,
        **solve_kwargs,
    )[-1]
    fluxoids = [
        sum(solution_no_circ.hole_fluxoid(name)).to("Phi_0").magnitude for name in holes
    ]
    fluxoids = np.array(fluxoids)
    # Solve for the circulating currents needed to realize the target_fluxoids.
    I_circ = np.linalg.solve(M.magnitude, target_fluxoids - fluxoids)
    circulating_currents = dict(zip(holes, I_circ))
    # Solve the model with the optimized circulating currents.
    solution = solve(
        device,
        applied_field=applied_field,
        terminal_currents=terminal_currents,
        circulating_currents=circulating_currents,
        **solve_kwargs,
    )[-1]
    return solution
