import logging
from typing import Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la

from ..device import Device, TransportDevice
from ..device.transport import stream_from_terminal_current
from .utils import FilmInfo, FilmSolution

logger = logging.getLogger("solve")


class LinearSystem(NamedTuple):
    A: np.ndarray
    indices: np.ndarray
    lu_piv: Optional[Tuple[np.ndarray, np.ndarray]] = None
    grad_Lambda_term: Union[float, np.ndarray] = 0.0


def build_linear_systems(
    device: Device, film_info_dict: Dict[str, FilmInfo]
) -> Tuple[Dict[str, LinearSystem], Dict[str, Dict[str, LinearSystem]]]:
    film_systems = {}
    hole_systems = {}
    for film_name, film_info in film_info_dict.items():
        hole_systems[film_name] = {}
        Lambda_info = film_info.lambda_info
        inhomogeneous = Lambda_info.inhomogeneous
        Lambda = Lambda_info.Lambda
        if inhomogeneous:
            grad = film_info.gradient
            grad_Lambda_term = np.einsum("ijk, ijk -> jk", (grad @ Lambda), grad)
        else:
            grad_Lambda_term = 0

        hole_indices = film_info.hole_indices
        for hole_name, indices in hole_indices.items():
            # Effective field associated with the circulating currents:
            # current is in [current_units], Lambda is in [device.length_units],
            # and Del2 is in [device.length_units ** (-2)], so
            # Ha_eff has units of [current_unit / device.length_units]
            A = _build_system_1d(
                film_info.kernel,
                film_info.weights,
                Lambda,
                film_info.laplacian,
                grad_Lambda_term,
                indices,
                inhomogeneous=inhomogeneous,
            )
            hole_systems[film_name][hole_name] = LinearSystem(
                A=A, indices=indices, grad_Lambda_term=grad_Lambda_term
            )

        # We want all points that are in a film and not in a hole.
        film_indices = film_info.film_indices
        if hole_indices:
            film_indices = np.setdiff1d(
                film_indices, np.concatenate(list(hole_indices.values()))
            )
        if isinstance(device, TransportDevice):
            film_indices = np.setdiff1d(film_indices, film_info.boundary_indices)

        # Form the linear system for the film:
        # gf = -K @ h, where K = inv(Q * w - Lambda * Del2 - grad_Lambda_term) = inv(A)
        # Eqs. 15-17 in [Brandt], Eqs 12-14 in [Kirtley1], Eqs. 12-14 in [Kirtley2].
        A = _build_system_2d(
            film_info.kernel,
            film_info.weights,
            Lambda,
            film_info.laplacian,
            grad_Lambda_term,
            film_indices,
            inhomogeneous=inhomogeneous,
        )
        lu_piv = la.lu_factor(-A)
        film_systems[film_name] = LinearSystem(
            A=A, indices=film_indices, lu_piv=lu_piv, grad_Lambda_term=grad_Lambda_term
        )
    return film_systems, hole_systems


def _build_system_1d(
    Q, weights, Lambda, laplacian, grad_Lambda_term, ix, inhomogeneous=False
):
    """Builds the linear system for the 'effective applied fields'."""
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[:, ix]
    else:
        grad_Lambda = 0
    return Q[:, ix] * weights[ix, 0] - Lambda[ix, 0] * laplacian[:, ix] - grad_Lambda


def _build_system_2d(
    Q, weights, Lambda, laplacian, grad_Lambda_term, ix1d, inhomogeneous=False
):
    """Builds the linear system to solve for the stream function."""
    ix2d = np.ix_(ix1d, ix1d)
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[ix2d]
    else:
        grad_Lambda = 0
    return Q[ix2d] * weights[ix1d, 0] - Lambda[ix1d, 0] * laplacian[ix2d] - grad_Lambda


def solve_for_terminal_current_stream(
    device: TransportDevice,
    film_info: FilmInfo,
    film_system: LinearSystem,
    terminal_currents: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Solve for the stream function in the film assuming no applied field and
    ignoring the presence of any holes.
    2. Set the stream function in each hole to the weighted average of the stream
    function found in step 1.
    3. Re-solve for the stream function in the film with the new hole boundary conditions.
    """
    device._check_current_mapping(terminal_currents)
    terminal_currents = terminal_currents.copy()
    # The drain terminal must sink all current
    if device.drain_terminal is not None:
        terminal_currents[device.drain_terminal.name] = -sum(
            terminal_currents.get(term.name, 0) for term in device.source_terminals
        )
    mesh = device.meshes[film_info.name]
    points = mesh.sites
    inhomogeneous = film_info.lambda_info.inhomogeneous
    Lambda = film_info.lambda_info.Lambda
    Q = film_info.kernel
    laplacian = film_info.laplacian
    weights = film_info.weights
    hole_indices = film_info.hole_indices
    in_hole = film_info.in_hole
    grad_Lambda_term = film_system.grad_Lambda_term
    boundary_indices = device.boundary_vertices()
    on_boundary = np.zeros(len(points), dtype=bool)
    boundary_points = points[boundary_indices]
    on_boundary[boundary_indices] = True
    npoints = len(points)

    # if gpu:
    #     g = jnp.zeros(npoints)
    #     Ha_eff = jnp.zeros(npoints)
    # ix_points = np.arange(npoints, dtype=int)
    g = np.zeros(npoints)
    Ha_eff = np.zeros(npoints)

    if not any(terminal_currents.values()):
        return g, on_boundary

    Lambda = 1e10 * np.ones_like(Lambda)

    def min_index(terminal):
        return terminal.contains_points(boundary_points, index=True).max()

    terminals = sorted(device.source_terminals + [device.drain_terminal], key=min_index)
    # Rotate terminals so that the drain is the last element
    while terminals[-1] is not device.drain_terminal:
        terminals.append(terminals.pop(0))
    for terminal in terminals:
        current = terminal_currents[terminal.name]
        ix_boundary = sorted(
            terminal.contains_points(boundary_points, index=True).tolist()
        )
        remaining_boundary = boundary_indices[(ix_boundary[-1] + 1) :]
        ix_terminal = boundary_indices[ix_boundary]
        stream = stream_from_terminal_current(points[ix_terminal], -current)
        # if gpu:
        #     g = g.at[ix_terminal].add(stream)
        #     g = g.at[remaining_boundary].add(stream[-1])
        g[ix_terminal] += stream
        g[remaining_boundary] += stream[-1]
    # The stream function on the "reference boundary" (i.e., the boundary
    # immediately after the output terminal in a CCW direction) should be zero.
    g_ref = g[ix_terminal.max()]
    ix = boundary_indices
    # if gpu:
    #     g = g.at[ix].add(-g_ref)
    g[ix] += -g_ref
    A = _build_system_1d(
        Q, weights, Lambda, laplacian, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
    )
    # if gpu:
    #     Ha_eff = Ha_eff.at[ix_points].add(-(A @ g[ix]))
    Ha_eff += -(A @ g[ix])
    # First solve for the stream function inside the film, ignoring the
    # presence of holes completely.
    film = device.film
    ix1d = np.logical_and.reduce(
        [film.contains_points(points), np.logical_not(on_boundary)]
    )
    ix1d = np.where(ix1d)[0]
    A = _build_system_2d(
        Q,
        weights,
        Lambda,
        laplacian,
        grad_Lambda_term,
        ix1d,
        inhomogeneous=inhomogeneous,
    )
    h = -Ha_eff[ix1d]
    # if gpu:
    #     lu_piv = jla.lu_factor(-A)
    #     gf = jla.lu_solve(lu_piv, h)
    #     g = g.at[ix1d].set(gf)
    lu_piv = la.lu_factor(-A)
    gf = la.lu_solve(lu_piv, h)
    g[ix1d] = gf

    holes = {
        name: hole for name, hole in device.holes.items() if hole.layer == film.layer
    }
    if len(holes) == 0:
        return g, on_boundary

    # Set the stream function in each hole to the average value
    # obtained when ignoring holes.
    # if gpu:
    #     Ha_eff = jnp.zeros(points.shape[0])
    Ha_eff = np.zeros(points.shape[0])
    for name in holes:
        ix = hole_indices[name]
        # if gpu:
        #     g = g.at[ix].set(jnp.average(g[ix], weights=weights[ix, 0]))
        g[ix] = np.average(g[ix], weights=weights[ix, 0])
        A = _build_system_1d(
            Q,
            weights,
            Lambda,
            laplacian,
            grad_Lambda_term,
            ix,
            inhomogeneous=inhomogeneous,
        )
        # if gpu:
        #     Ha_eff = Ha_eff.at[ix_points].add(-(A @ g[ix]))
        Ha_eff += -(A @ g[ix])

    ix = boundary_indices
    A = _build_system_1d(
        Q, weights, Lambda, laplacian, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
    )
    # if gpu:
    #     Ha_eff = Ha_eff.at[ix_points].add(-(A @ g[ix]))
    Ha_eff += -(A @ g[ix])

    # Solve for the stream function inside the superconducting film again,
    # now with the new boundary conditions for the holes.
    ix1d = np.logical_and.reduce(
        [
            film.contains_points(points),
            np.logical_not(in_hole),
            np.logical_not(on_boundary),
        ]
    )
    ix1d = np.where(ix1d)[0]
    A = _build_system_2d(
        Q,
        weights,
        Lambda,
        laplacian,
        grad_Lambda_term,
        ix1d,
        inhomogeneous=inhomogeneous,
    )
    h = -Ha_eff[ix1d]
    # if gpu:
    #     lu_piv = jla.lu_factor(-A)
    #     gf = jla.lu_solve(lu_piv, h)
    #     g = g.at[ix1d].set(gf)
    lu_piv = la.lu_factor(-A)
    gf = la.lu_solve(lu_piv, h)
    g[ix1d] = gf
    return g, on_boundary


def solve_film(
    *,
    device: Device,
    applied_field: np.ndarray,
    film_info: FilmInfo,
    film_system: LinearSystem,
    hole_systems: Dict[str, LinearSystem],
    field_conversion: float,
    vortex_flux: float,
    field_from_other_films: Optional[np.ndarray] = None,
    check_inversion: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the stream function and magnetic field within a single layer of a ``Device``.

    Args:
        device: The Device to simulate.
        layer: Name of the layer to analyze.
        applied_field: The applied magnetic field evaluated at the mesh vertices.
        weights: The Device's weight vector.
        kernel: The Device's kernel matrix ``Q``.
        Del2: The Device's Laplacian operator.
        grad: The Device's vertex gradient matrix, shape (num_vertices, 2, num_vertices).
        Lambda_info: A LambdaInfo instance defining Lambda(x, y).
        terminal_currents: A dict of ``{source_name: source_current}`` for
            each source terminal. This argument is only allowed if ``device``
            as an instance of ``TransportDevice``.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        vortices: A list of Vortex objects located in films in this layer.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of ``{current_units} / {device.length_units}``.
        check_inversion: Whether to verify the accuracy of the matrix inversion.

    Returns:
        A FilmSolution containing the results
    """
    if isinstance(device, TransportDevice):
        transport_device = True
    else:
        # if terminal_currents:
        #     raise TypeError("Terminal currents are only allowed for TransportDevices.")
        transport_device = False

    circulating_currents = film_info.circulating_currents
    terminal_currents = film_info.terminal_currents
    mesh = device.meshes[film_info.name]
    points = mesh.sites
    # Dense arrays
    weights = film_info.weights
    Q = film_info.kernel
    grad_x = mesh.operators.gradient_x
    grad_y = mesh.operators.gradient_y

    # Units: current_units / device.length_units.
    Hz_applied = applied_field
    if field_from_other_films is not None:
        Hz_applied = Hz_applied + field_from_other_films

    g = np.zeros_like(Hz_applied)
    Ha_eff = np.zeros_like(Hz_applied)
    # if gpu:
    #     g = jax.device_put(g)
    #     ix_points = np.arange(len(Ha_eff), dtype=int)
    #     Ha_eff = jax.device_put(Ha_eff)

    # Set the boundary conditions for all holes:
    # 1. g[hole] = I_circ_hole
    # 2. Effective field associated with I_circ_hole
    # See Section II(a) in [Brandt], Eqs. 18-19 in [Kirtley1],
    # and Eqs 17-18 in [Kirtley2].
    for name, system in hole_systems.items():
        indices = system.indices
        A = system.A
        current = circulating_currents.get(name, 0)
        # if gpu:
        #     g = g.at[indices].add(current)
        #     Ha_eff = Ha_eff.at[ix_points].add(-(A @ g[indices]))
        g[indices] += current  # g[hole] = I_circ
        Ha_eff += -(A @ g[indices])

    if transport_device:
        g_transport, on_boundary = solve_for_terminal_current_stream(
            device,
            film_info,
            film_system,
            terminal_currents,
        )
        # if gpu:
        #     g = g.at[ix_points].add(g_transport)
        g += g_transport

    indices = film_system.indices
    A = film_system.A
    lu_piv = film_system.lu_piv
    h = Hz_applied[indices] - Ha_eff[indices]
    # if gpu:
    #     gf = jla.lu_solve(lu_piv, h)
    #     g = g.at[indices].add(gf)
    gf = la.lu_solve(lu_piv, h)
    g[indices] += gf

    if check_inversion:
        # Validate solution
        hsim = -(A @ gf)
        if not np.allclose(hsim, h):
            logger.warning(
                f"Unable to solve for stream function in {film_info.name}), "
                f"maximum error {np.abs(hsim - h).max():.3e}."
            )
    K = None  # Matrix inverse of A
    for vortex in film_info.vortices:
        if K is None:
            # Compute K only once if needed
            # if gpu:
            #     K = -jla.lu_solve(lu_piv, jnp.eye(A.shape[0]))
            K = -la.lu_solve(lu_piv, np.eye(A.shape[0]))
        # Index of the mesh vertex that is closest to the vortex position:
        # in the film-specific sub-mesh
        j_film = np.argmin(la.norm(points[indices] - [[vortex.x, vortex.y]], axis=1))
        # ... and in the full device mesh.
        j_device = np.argmin(la.norm(points - [[vortex.x, vortex.y]], axis=1))
        # Eq. 28 in [Brandt]
        g_vortex = vortex_flux * vortex.nPhi0 * K[:, j_film] / weights[j_device]
        # if gpu:
        #     g = g.at[indices].add(g_vortex)
        g[indices] += g_vortex
    # Current density J = curl(g \hat{z}) = [dg/dy, -dg/dx]
    J = np.array([grad_y @ g, -(grad_x @ g)]).T
    # Eq. 7 in [Kirtley1], Eq. 7 in [Kirtley2]
    screening_field = np.asarray(Q @ (weights[:, 0] * g))
    # Above is equivalent to the following, but faster
    # screening_field = np.einsum("ij, ji, j -> i", Q, weights, g)
    if field_from_other_films is not None:
        field_from_other_films = field_from_other_films / field_conversion
    return FilmSolution(
        stream=g,
        current_density=J,
        applied_field=applied_field / field_conversion,
        self_field=screening_field / field_conversion,
        field_from_other_films=field_from_other_films,
    )
