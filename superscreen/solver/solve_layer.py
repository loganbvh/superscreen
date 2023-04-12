import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pint
import scipy.linalg as la

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jla
except (ModuleNotFoundError, ImportError, RuntimeError):
    jax = None

from ..device import Device, TransportDevice
from ..device.transport import stream_from_terminal_current
from ..solution import Vortex
from .utils import LambdaInfo

logger = logging.getLogger("solve")


def _make_system(Q, weights, Lambda, Del2, grad_Lambda_term, ix, inhomogeneous=False):
    """Builds the linear system for the 'effective applied fields'."""
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[:, ix]
    else:
        grad_Lambda = 0
    return Q[:, ix] * weights[ix, 0] - Lambda[ix, 0] * Del2[:, ix] - grad_Lambda


def _make_system_2d(
    Q, weights, Lambda, Del2, grad_Lambda_term, ix1d, inhomogeneous=False
):
    """Builds the linear system to solve for the stream function."""
    ix2d = np.ix_(ix1d, ix1d)
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[ix2d]
    else:
        grad_Lambda = 0
    return Q[ix2d] * weights[ix1d, 0] - Lambda[ix1d, 0] * Del2[ix2d] - grad_Lambda


def _solve_for_terminal_current_stream(
    device: TransportDevice,
    points: np.ndarray,
    in_hole: np.ndarray,
    Q: np.ndarray,
    weights: np.ndarray,
    Del2: np.ndarray,
    Lambda: np.ndarray,
    grad_Lambda_term: np.ndarray,
    terminal_currents: Dict[str, float],
    hole_indices: np.ndarray,
    gpu: bool = False,
    inhomogeneous: bool = False,
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
    boundary_indices = device.boundary_vertices()
    on_boundary = np.zeros(points.shape[0], dtype=bool)
    boundary_points = points[boundary_indices]
    on_boundary[boundary_indices] = True

    if gpu:
        g = jnp.zeros(points.shape[0])
        Ha_eff = jnp.zeros(points.shape[0])
    else:
        g = np.zeros(points.shape[0])
        Ha_eff = np.zeros(points.shape[0])

    if not any(terminal_currents.values()):
        return g, on_boundary

    ix_points = np.arange(points.shape[0], dtype=int)
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
        if gpu:
            g = g.at[ix_terminal].add(stream)
            g = g.at[remaining_boundary].add(stream[-1])
        else:
            g[ix_terminal] += stream
            g[remaining_boundary] += stream[-1]
    # The stream function on the "reference boundary" (i.e., the boundary
    # immediately after the output terminal in a CCW direction) should be zero.
    g_ref = g[ix_terminal.max()]
    ix = boundary_indices
    if gpu:
        g = g.at[ix].add(-g_ref)
    else:
        g[ix] += -g_ref
    A = _make_system(
        Q, weights, Lambda, Del2, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
    )
    if gpu:
        Ha_eff = Ha_eff.at[ix_points].add(-A @ g[ix])
    else:
        Ha_eff += -A @ g[ix]
    # First solve for the stream function inside the film, ignoring the
    # presence of holes completely.
    film = device.film
    ix1d = np.logical_and.reduce(
        [film.contains_points(points), np.logical_not(on_boundary)]
    )
    ix1d = np.where(ix1d)[0]
    A = _make_system_2d(
        Q, weights, Lambda, Del2, grad_Lambda_term, ix1d, inhomogeneous=inhomogeneous
    )
    h = -Ha_eff[ix1d]
    if gpu:
        lu_piv = jla.lu_factor(-A)
        gf = jla.lu_solve(lu_piv, h)
        g = g.at[ix1d].set(gf)
    else:
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
    if gpu:
        Ha_eff = jnp.zeros(points.shape[0])
    else:
        Ha_eff = np.zeros(points.shape[0])
    for name in holes:
        ix = hole_indices[name]
        if gpu:
            g = g.at[ix].set(jnp.average(g[ix], weights=weights[ix, 0]))
        else:
            g[ix] = np.average(g[ix], weights=weights[ix, 0])
        A = _make_system(
            Q, weights, Lambda, Del2, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
        )
        if gpu:
            Ha_eff = Ha_eff.at[ix_points].add(-A @ g[ix])
        else:
            Ha_eff += -A @ g[ix]

    ix = boundary_indices
    A = _make_system(
        Q, weights, Lambda, Del2, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
    )
    if gpu:
        Ha_eff = Ha_eff.at[ix_points].add(-A @ g[ix])
    else:
        Ha_eff += -A @ g[ix]

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
    A = _make_system_2d(
        Q, weights, Lambda, Del2, grad_Lambda_term, ix1d, inhomogeneous=inhomogeneous
    )
    h = -Ha_eff[ix1d]
    if gpu:
        lu_piv = jla.lu_factor(-A)
        gf = jla.lu_solve(lu_piv, h)
        g = g.at[ix1d].set(gf)
    else:
        lu_piv = la.lu_factor(-A)
        gf = la.lu_solve(lu_piv, h)
        g[ix1d] = gf
    return g, on_boundary


def solve_layer(
    *,
    device: Device,
    layer: str,
    applied_field: np.ndarray,
    kernel: np.ndarray,
    weights: np.ndarray,
    Del2: np.ndarray,
    grad: np.ndarray,
    Lambda_info: LambdaInfo,
    terminal_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[List[Vortex]] = None,
    current_units: str = "uA",
    check_inversion: bool = False,
    gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        gpu: Solve on a GPU if available (requires JAX and CUDA).

    Returns:
        stream function, current density, total field, film screening field
    """

    circulating_currents = circulating_currents or {}
    terminal_currents = terminal_currents or {}
    for name in circulating_currents:
        if name not in device.holes:
            raise ValueError(f"Circulating current specified for unknown hole: {name}.")
    if isinstance(device, TransportDevice):
        transport_device = True
    else:
        if terminal_currents:
            raise TypeError("Terminal currents are only allowed for TransportDevices.")
        transport_device = False
    vortices = vortices or []
    # Vortex flux in magnetization-like units,
    # i.e. H * area as opposed to B * area = mu_0 * H * area.
    # ([current] / [length]) * [length]^2 = [current] * [length]
    vortex_flux = (
        device.ureg("Phi_0 / mu_0")
        .to(f"{current_units} * {device.length_units}")
        .magnitude
    )

    films = {name: film for name, film in device.films.items() if film.layer == layer}
    holes = {name: hole for name, hole in device.holes.items() if hole.layer == layer}
    Q = kernel
    points = device.points

    if gpu:
        if jax is None:
            raise ValueError("Running solve_layer(..., gpu=True) requires JAX.")
        einsum_ = jnp.einsum
    else:
        einsum_ = np.einsum

    # Units: current_units / device.length_units.
    Hz_applied = applied_field
    inhomogeneous = Lambda_info.inhomogeneous
    Lambda = Lambda_info.Lambda
    if inhomogeneous:
        grad_Lambda_term = einsum_("ijk, ijk -> jk", (grad @ Lambda), grad)
    else:
        grad_Lambda_term = None

    # Identify holes in the superconductor
    hole_indices = {}
    in_hole = np.zeros(points.shape[0], dtype=bool)
    for name, hole in holes.items():
        ix = hole.contains_points(points)
        hole_indices[name] = np.where(ix)[0]
        in_hole = np.logical_or(in_hole, ix)

    g = np.zeros_like(Hz_applied)
    Ha_eff = np.zeros_like(Hz_applied)
    if gpu:
        g = jax.device_put(g)
        ix_points = np.arange(Ha_eff.shape[0], dtype=int)
        Ha_eff = jax.device_put(Ha_eff)

    # Set the boundary conditions for all holes:
    # 1. g[hole] = I_circ_hole
    # 2. Effective field associated with I_circ_hole
    # See Section II(a) in [Brandt], Eqs. 18-19 in [Kirtley1],
    # and Eqs 17-18 in [Kirtley2].
    for name in holes:
        current = circulating_currents.get(name, 0)
        ix = hole_indices[name]
        if gpu:
            g = g.at[ix].add(current)
        else:
            g[ix] += current  # g[hole] = I_circ
        # Effective field associated with the circulating currents:
        # current is in [current_units], Lambda is in [device.length_units],
        # and Del2 is in [device.length_units ** (-2)], so
        # Ha_eff has units of [current_unit / device.length_units]
        A = _make_system(
            Q, weights, Lambda, Del2, grad_Lambda_term, ix, inhomogeneous=inhomogeneous
        )
        if gpu:
            Ha_eff = Ha_eff.at[ix_points].add(-A @ g[ix])
        else:
            Ha_eff += -A @ g[ix]

    if transport_device:
        g_transport, on_boundary = _solve_for_terminal_current_stream(
            device,
            points,
            in_hole,
            Q,
            weights,
            Del2,
            Lambda,
            grad_Lambda_term,
            terminal_currents,
            hole_indices,
            gpu=gpu,
            inhomogeneous=inhomogeneous,
        )
        if gpu:
            g = g.at[ix_points].add(g_transport)
        else:
            g += g_transport

    film_to_vortices = defaultdict(list)
    for vortex in vortices:
        for name, film in films.items():
            if film.contains_points([vortex.x, vortex.y]):
                film_to_vortices[name].append(vortex)
                # A given vortex can only lie in a single film.
                continue

    # Now solve for the stream function inside the superconducting films
    for name, film in films.items():
        # We want all points that are in a film and not in a hole.
        ix1d = np.logical_and(film.contains_points(points), np.logical_not(in_hole))
        if transport_device:
            ix1d = np.logical_and(ix1d, np.logical_not(on_boundary))
        ix1d = np.where(ix1d)[0]
        # # Form the linear system for the film:
        # # gf = -K @ h, where K = inv(Q * w - Lambda * Del2 - grad_Lambda_term) = inv(A)
        # # Eqs. 15-17 in [Brandt], Eqs 12-14 in [Kirtley1], Eqs. 12-14 in [Kirtley2].
        A = _make_system_2d(
            Q,
            weights,
            Lambda,
            Del2,
            grad_Lambda_term,
            ix1d,
            inhomogeneous=inhomogeneous,
        )
        h = Hz_applied[ix1d] - Ha_eff[ix1d]
        if gpu:
            lu_piv = jla.lu_factor(-A)
            gf = jla.lu_solve(lu_piv, h)
            g = g.at[ix1d].add(gf)
        else:
            lu_piv = la.lu_factor(-A)
            gf = la.lu_solve(lu_piv, h)
            g[ix1d] += gf

        if check_inversion:
            # Validate solution
            hsim = -A @ gf
            if not np.allclose(hsim, h):
                logger.warning(
                    f"Unable to solve for stream function in {layer} ({name}), "
                    f"maximum error {np.abs(hsim - h).max():.3e}."
                )
        K = None  # Matrix inverse of A
        for vortex in film_to_vortices[name]:
            if K is None:
                # Compute K only once if needed
                if gpu:
                    K = -jla.lu_solve(lu_piv, jnp.eye(A.shape[0]))
                else:
                    K = -la.lu_solve(lu_piv, np.eye(A.shape[0]))
            # Index of the mesh vertex that is closest to the vortex position:
            # in the film-specific sub-mesh
            j_film = np.argmin(la.norm(points[ix1d] - [[vortex.x, vortex.y]], axis=1))
            # ... and in the full device mesh.
            j_device = np.argmin(la.norm(points - [[vortex.x, vortex.y]], axis=1))
            # Eq. 28 in [Brandt]
            g_vortex = vortex_flux * vortex.nPhi0 * K[:, j_film] / weights[j_device]
            if gpu:
                g = g.at[ix1d].add(g_vortex)
            else:
                g[ix1d] += g_vortex
            del g_vortex
    # Current density J = curl(g \hat{z}) = [dg/dy, -dg/dx]
    Gx = grad[0]
    Gy = grad[1]
    Jx = np.asarray(Gy @ g)
    Jy = np.asarray(-Gx @ g)
    J = np.stack([Jx, Jy], axis=1)
    # Eq. 7 in [Kirtley1], Eq. 7 in [Kirtley2]
    screening_field = np.asarray(Q @ (weights[:, 0] * g))
    # Above is equivalent to the following, but much faster
    # screening_field = np.einsum("ij, ji, j -> i", Q, weights, g)
    total_field = np.asarray(Hz_applied) + screening_field
    return g, J, total_field, screening_field
