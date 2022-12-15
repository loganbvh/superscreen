import contextlib
import itertools
import logging
import os
import tempfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pint
import psutil
import scipy.linalg as la
import scipy.sparse as sp
from scipy.spatial import distance

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jla

    HAS_JAX = True
except (ModuleNotFoundError, ImportError, RuntimeError):
    HAS_JAX = False

from .device import Device
from .device.transport import TransportDevice, stream_from_terminal_current
from .parameter import Constant, Parameter
from .solution import Solution, Vortex
from .sources import ConstantField

lambda_str = "\u03bb"
Lambda_str = "\u039b"

logger = logging.getLogger(__name__)


class LambdaInfo:
    def __init__(
        self,
        *,
        layer: str,
        Lambda: np.ndarray,
        london_lambda: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
    ):
        self.layer = layer
        self.Lambda = Lambda
        self.london_lambda = london_lambda
        self.thickness = thickness
        self.inhomogeneous = (
            np.ptp(self.Lambda) / max(np.min(np.abs(self.Lambda)), np.finfo(float).eps)
            > 1e-6
        )
        if self.inhomogeneous:
            logger.warning(
                f"Inhomogeneous {Lambda_str} in layer '{self.layer}', "
                f"which violates the assumptions of the London model. "
                f"Results may not be reliable."
            )
        if self.london_lambda is not None:
            assert self.thickness is not None
            assert np.allclose(self.Lambda, self.london_lambda**2 / self.thickness)
        if np.any(self.Lambda < 0):
            raise ValueError(f"Negative Lambda in layer '{layer}'.")


def q_matrix(
    points: np.ndarray, dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """Computes the denominator matrix, q:

    .. math::

        q_{ij} = \\frac{1}{4\\pi|\\vec{r}_i-\\vec{r}_j|^3}

    See Eq. 7 in [Brandt-PRB-2005]_, Eq. 8 in [Kirtley-RSI-2016]_,
    and Eq. 8 in [Kirtley-SST-2016]_.

    Args:
        points: Shape (n, 2) array of x,y coordinates of vertices.
        dtype: Output dtype.

    Returns:
        Shape (n, n) array, qij
    """
    # Euclidean distance between points
    distances = distance.cdist(points, points, metric="euclidean")
    if dtype is not None:
        distances = distances.astype(dtype, copy=False)
    with np.errstate(divide="ignore"):
        q = 1 / (4 * np.pi * distances**3)
    np.fill_diagonal(q, np.inf)
    return q.astype(dtype, copy=False)


def C_vector(
    points: np.ndarray,
    dtype: Optional[Union[str, np.dtype]] = None,
) -> np.ndarray:
    """Computes the edge vector, C:

    .. math::
        C_i &= \\frac{1}{4\\pi}\\sum_{p,q=\\pm1}\\sqrt{(\\Delta x - px_i)^{-2}
            + (\\Delta y - qy_i)^{-2}}\\\\
        \\Delta x &= \\frac{1}{2}(\\mathrm{max}(x) - \\mathrm{min}(x))\\\\
        \\Delta y &= \\frac{1}{2}(\\mathrm{max}(y) - \\mathrm{min}(y))

    See Eq. 12 in [Brandt-PRB-2005]_, Eq. 16 in [Kirtley-RSI-2016]_,
    and Eq. 15 in [Kirtley-SST-2016]_.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        dtype: Output dtype.

    Returns:
        Shape (n, ) array, Ci
    """
    x = points[:, 0]
    y = points[:, 1]
    x = x - x.mean()
    y = y - y.mean()
    a = np.ptp(x) / 2
    b = np.ptp(y) / 2
    with np.errstate(divide="ignore"):
        C = sum(
            np.sqrt((a - p * x) ** (-2) + (b - q * y) ** (-2))
            for p, q in itertools.product((-1, 1), repeat=2)
        )
    C[np.isinf(C)] = 1e30
    C /= 4 * np.pi
    if dtype is not None:
        C = C.astype(dtype, copy=False)
    return C


def Q_matrix(
    q: np.ndarray,
    C: np.ndarray,
    weights: np.ndarray,
    dtype: Optional[Union[str, np.dtype]] = None,
) -> np.ndarray:
    """Computes the kernel matrix, Q:

    .. math::

        Q_{ij} = (\\delta_{ij}-1)q_{ij}
        + \\delta_{ij}\\frac{1}{w_{ij}}\\left(C_i
        + \\sum_{l\\neq i}q_{il}w_{il}\\right)

    See Eq. 10 in [Brandt-PRB-2005]_, Eq. 11 in [Kirtley-RSI-2016]_,
    and Eq. 11 in [Kirtley-SST-2016]_.

    Args:
        q: Shape (n, n) matrix qij.
        C: Shape (n, ) vector Ci.
        weights: Shape (n, ) weight vector.
        dtype: Output dtype.

    Returns:
        Shape (n, n) array, Qij
    """
    if sp.issparse(weights):
        weights = weights.diagonal()
    # q[i, i] are np.inf, but Q[i, i] involves a sum over only the
    # off-diagonal elements of q, so we can just set q[i, i] = 0 here.
    q = q.copy()
    np.fill_diagonal(q, 0)
    Q = -q
    np.fill_diagonal(Q, (C + np.einsum("ij, j -> i", q, weights)) / weights)
    if dtype is not None:
        Q = Q.astype(dtype, copy=False)
    return Q


def convert_field(
    value: Union[np.ndarray, float, str, pint.Quantity],
    new_units: Union[str, pint.Unit],
    old_units: Optional[Union[str, pint.Unit]] = None,
    ureg: Optional[pint.UnitRegistry] = None,
    with_units: bool = True,
) -> Union[pint.Quantity, np.ndarray, float]:
    """Converts a value between different field units, either magnetic field H
    [current] / [length] or flux density B = mu0 * H [mass] / ([curret] [time]^2)).

    Args:
        value: The value to convert. It can either be a numpy array (no units),
            a float (no units), a string like "1 uA/um", or a scalar or array
            ``pint.Quantity``. If value is not a string wiht units or a ``pint.Quantity``,
            then old_units must specify the units of the float or array.
        new_units: The units to convert to.
        old_units: The old units of ``value``. This argument is required if ``value``
            is not a string with units or a ``pint.Quantity``.
        ureg: The ``pint.UnitRegistry`` to use for conversion. If None is given,
            a new instance is created.
        with_units: Whether to return a ``pint.Quantity`` with units attached.

    Returns:
        The converted value, either a pint.Quantity (scalar or array with units),
        or an array or float without units, depending on the ``with_units`` argument.
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    if isinstance(value, str):
        value = ureg(value)
    if isinstance(value, pint.Quantity):
        old_units = value.units
    if old_units is None:
        raise ValueError(
            "Old units must be specified if value is not a string or pint.Quantity."
        )
    if isinstance(old_units, str):
        old_units = ureg(old_units)
    if isinstance(new_units, str):
        new_units = ureg(new_units)
    if not isinstance(value, pint.Quantity):
        value = value * old_units
    if new_units.dimensionality == old_units.dimensionality:
        value = value.to(new_units)
    elif "[length]" in old_units.dimensionality:
        # value is H in units with dimensionality [current] / [length]
        # and we want B = mu0 * H
        value = (value * ureg("mu0")).to(new_units)
    else:
        # value is B = mu0 * H in units with dimensionality
        # [mass] / ([current] [time]^2) and we want H = B / mu0
        value = (value / ureg("mu0")).to(new_units)
    if not with_units:
        value = value.magnitude
    return value


def field_conversion_factor(
    field_units: str,
    current_units: str,
    length_units: str = "m",
    ureg: Optional[pint.UnitRegistry] = None,
) -> pint.Quantity:
    """Returns a conversion factor from ``field_units`` to ``current_units / length_units``.

    Args:
        field_units: Magnetic field/flux unit to convert, having dimensionality
            either of magnetic field ``H`` (e.g. A / m or Oe) or of
            magnetic flux density ``B = mu0 * H`` (e.g. Tesla or Gauss).
        current_units: Current unit to use for the conversion.
        length_units: Lenght/distance unit to use for the conversion.
        ureg: pint UnitRegistry to use for the conversion. If None is provided,
            a new UnitRegistry is created.

    Returns:
        Conversion factor as a ``pint.Quantity``. ``conversion_factor.magnitude``
        gives you the numerical value of the conversion factor.
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    field = ureg(field_units)
    target_units = f"{current_units} / {length_units}"
    try:
        field = field.to(target_units)
    except pint.DimensionalityError:
        # field_units is flux density B = mu0 * H
        field = (field / ureg("mu0")).to(target_units)
    return field / ureg(field_units)


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
        if not HAS_JAX:
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


def solve(
    device: Device,
    *,
    applied_field: Optional[Callable] = None,
    terminal_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[List[Vortex]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    return_solutions: bool = True,
    directory: Optional[str] = None,
    cache_memory_cutoff: float = np.inf,
    log_level: int = logging.INFO,
    gpu: bool = False,
    _solver: str = "superscreen.solve",
) -> List[Solution]:
    """Computes the stream functions and magnetic fields for all layers in a ``Device``.

    The simulation strategy is:

    1. Compute the stream functions and fields for each layer given
    only the applied field.

    2. If iterations > 1 and there are multiple layers, then for each layer,
    calculate the screening field from all other layers and recompute the
    stream function and fields based on the sum of the applied field
    and the screening fields from all other layers.

    3. Repeat step 2 (iterations - 1) times.

    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        terminal_currents: A dict of ``{source_name: source_current}`` for
            each source terminal. This argument is only allowed if ``device``
            as an instance of ``TransportDevice``.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        vortices: A list of Vortex objects located in the Device.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        iterations: Number of times to compute the interactions between layers.
        return_solutions: Whether to return a list of Solution objects.
        directory: If not None, resulting Solutions will be saved in this directory.
        cache_memory_cutoff: If the memory needed for layer-to-layer kernel
            matrices exceeds ``cache_memory_cutoff`` times the current available
            system memory, then the kernel matrices will be cached to disk rather than
            in memory. Setting this value to ``inf`` disables caching to disk. In this
            case, the arrays will remain in memory unless they are swapped to disk
            by the operating system.
        log_level: Logging level to use, if any.
        gpu: Solve on a GPU if available (requires JAX and CUDA).
        _solver: Name of the solver method used.

    Returns:
        If ``return_solutions`` is True, returns a list of Solutions of
        length ``iterations + 1``.
    """

    if log_level is not None:
        logging.basicConfig(level=log_level)

    if directory is not None:
        os.makedirs(directory, exist_ok=True)

    if device.points is None:
        raise ValueError(
            "The device does not have a mesh. Call device.make_mesh() to generate it."
        )
    if device.weights is None or device.Del2 is None:
        raise ValueError(
            "The device does not have a Laplace operator. "
            "Call device.compute_matrices() to calculate it."
        )
    if gpu:
        if not HAS_JAX:
            raise ValueError("Running solve(..., gpu=True) requires JAX.")
        jax_device = jax.devices()[0]
        logger.info(f"Using JAX with device {jax_device}.")
        if "cpu" in jax_device.device_kind:
            logger.warning("No GPU found. Using JAX on the CPU.")
            _solver = _solver + ":jax:cpu"
        else:
            _solver = _solver + ":jax:gpu"
        dtype = np.float32
    else:
        dtype = device.solve_dtype

    # Convert all circulating and terminal currents to floats.
    def current_to_float(value):
        if isinstance(value, str):
            value = device.ureg(value)
        if isinstance(value, pint.Quantity):
            value = value.to(current_units).magnitude
        return value

    circulating_currents = circulating_currents or {}
    _circ_currents = circulating_currents.copy()
    circulating_currents = {}
    for name, current in _circ_currents.items():
        circulating_currents[name] = current_to_float(current)
    terminal_currents = terminal_currents or {}
    _term_currents = terminal_currents.copy()
    terminal_currents = {}
    for name, current in _term_currents.items():
        terminal_currents[name] = current_to_float(current)

    points = device.points.astype(dtype, copy=False)
    weights = device.weights.astype(dtype, copy=False)
    Q = device.Q.astype(dtype, copy=False)
    if weights.ndim == 1:
        # Shape (n, ) --> (n, 1)
        weights = weights[:, np.newaxis]
    Del2 = device.Del2
    gradx = device.gradx
    grady = device.grady
    if sp.issparse(Del2):
        Del2 = Del2.toarray().astype(dtype, copy=False)
    if sp.issparse(gradx):
        gradx = gradx.toarray().astype(dtype, copy=False)
    if sp.issparse(grady):
        grady = grady.toarray().astype(dtype, copy=False)
    grad = np.stack([gradx, grady], axis=0)

    if gpu:
        weights = jax.device_put(weights)
        Del2 = jax.device_put(Del2)
        grad = jax.device_put(grad)
        Q = jax.device_put(Q)
        grad = jax.device_put(grad)

    solutions = []
    streams = {}
    currents = {}
    fields = {}
    screening_fields = {}
    applied_field = applied_field or ConstantField(0)
    vortices = vortices or []

    field_conversion = field_conversion_factor(
        field_units,
        current_units,
        length_units=device.length_units,
        ureg=device.ureg,
    )
    logger.debug(
        f"Conversion factor from {device.ureg(field_units).units:~P} to "
        f"{device.ureg(current_units) / device.ureg(device.length_units):~P}: "
        f"{field_conversion:~P}."
    )
    field_conversion_magnitude = field_conversion.magnitude
    # Only compute the applied field and Lambda once.
    layer_fields = {}
    layer_Lambdas = {}
    for name, layer in device.layers.items():
        # Units: current_units / device.length_units
        layer_field = (
            applied_field(device.points[:, 0], device.points[:, 1], layer.z0)
            * field_conversion_magnitude
        ).astype(dtype, copy=False)
        # Check and cache penetration depth
        london_lambda = layer.london_lambda
        d = layer.thickness
        Lambda = layer.Lambda
        if isinstance(london_lambda, (int, float)) and london_lambda <= d:
            length_units = device.ureg(device.length_units).units
            logger.warning(
                f"Layer '{name}': The film thickness, d = {d:.4f} {length_units:~P}"
                f", is greater than or equal to the London penetration depth, resulting "
                f"in an effective penetration depth {Lambda_str} = {Lambda:.4f} "
                f"{length_units:~P} <= {lambda_str} = {london_lambda:.4f} {length_units:~P}. "
                f"The assumption that the current density is nearly constant over the "
                f"thickness of the film may not be valid. "
            )
        if isinstance(Lambda, (int, float)):
            Lambda = Constant(Lambda)
        Lambda = Lambda(device.points[:, 0], device.points[:, 1]).astype(
            dtype, copy=False
        )[:, np.newaxis]
        if london_lambda is not None:
            if isinstance(london_lambda, (int, float)):
                london_lambda = Constant(london_lambda)
            london_lambda = london_lambda(
                device.points[:, 0], device.points[:, 1]
            ).astype(dtype, copy=False)[:, np.newaxis]
        if gpu:
            layer_field = jax.device_put(layer_field)
            Lambda = jax.device_put(Lambda)
            if london_lambda is not None:
                london_lambda = jax.device_put(london_lambda)
        layer_fields[name] = layer_field
        layer_Lambdas[name] = LambdaInfo(
            layer=name,
            Lambda=Lambda,
            london_lambda=london_lambda,
            thickness=layer.thickness,
        )

    vortices_by_layer = defaultdict(list)
    for vortex in vortices:
        if not isinstance(vortex, Vortex):
            raise TypeError(f"Expected a Vortex, but got {type(vortex)}.")
        if vortex.layer not in device.layers:
            raise ValueError(f"Vortex located in unknown layer: {vortex}.")
        films_in_layer = [f for f in device.films.values() if f.layer == vortex.layer]
        holes_in_layer = [h for h in device.holes.values() if h.layer == vortex.layer]
        if not any(
            film.contains_points([vortex.x, vortex.y]) for film in films_in_layer
        ):
            raise ValueError(f"Vortex {vortex} is not located in a film.")
        if any(hole.contains_points([vortex.x, vortex.y]) for hole in holes_in_layer):
            raise ValueError(f"Vortex {vortex} is located in a hole.")
        vortices_by_layer[vortex.layer].append(vortex)

    # Compute the stream functions and fields for each layer
    # given only the applied field.
    for name, layer in device.layers.items():
        logger.info(f"Calculating {name} response to applied field.")
        g, J, total_field, screening_field = solve_layer(
            device=device,
            layer=name,
            applied_field=layer_fields[name],
            kernel=Q,
            terminal_currents=terminal_currents,
            circulating_currents=circulating_currents,
            vortices=vortices_by_layer[name],
            weights=weights,
            Del2=Del2,
            grad=grad,
            Lambda_info=layer_Lambdas[name],
            current_units=current_units,
            check_inversion=check_inversion,
            gpu=gpu,
        )
        # Units: current_units
        streams[name] = g.astype(dtype)
        # Units: currents_units / device.length_units
        currents[name] = J.astype(dtype, copy=False)
        # Units: current_units / device.length_units
        fields[name] = total_field.astype(dtype, copy=False)
        screening_fields[name] = screening_field.astype(dtype, copy=False)

    solution = Solution(
        device=device,
        streams={
            layer: np.asarray(stream, dtype=dtype) for layer, stream in streams.items()
        },
        current_densities=currents,
        fields={
            # Units: field_units
            layer: (field / field_conversion_magnitude).astype(dtype, copy=False)
            for layer, field in fields.items()
        },
        screening_fields={
            # Units: field_units
            layer: (field / field_conversion_magnitude).astype(dtype, copy=False)
            for layer, field in screening_fields.items()
        },
        applied_field=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
        vortices=vortices,
        solver=_solver,
    )
    if directory is not None:
        solution.to_file(os.path.join(directory, str(0)))
    if return_solutions:
        solutions.append(solution)
    else:
        del solution

    layer_names = list(device.layers)
    nlayers = len(layer_names)
    if nlayers < 2 or iterations < 1:
        if return_solutions:
            return solutions
        return
    rho2 = distance.cdist(points, points, metric="sqeuclidean").astype(
        dtype,
        copy=False,
    )
    # Cache kernel matrices.
    kernels = {}
    if cache_memory_cutoff is None:
        cache_memory_cutoff = np.inf
    if cache_memory_cutoff < 0:
        raise ValueError(
            f"Kernel cache memory cutoff must be greater than zero "
            f"(got {cache_memory_cutoff})."
        )
    nkernels = nlayers * (nlayers - 1) // 2
    total_bytes = nkernels * rho2.nbytes
    available_bytes = psutil.virtual_memory().available
    if total_bytes > cache_memory_cutoff * available_bytes:
        # Save kernel matrices to disk to avoid filling up memory.
        cache_kernels_to_disk = True
        context = tempfile.TemporaryDirectory()
    else:
        # Cache kernels in memory (much faster than saving to/loading from disk).
        cache_kernels_to_disk = False
        context = contextlib.nullcontext()
    if gpu:
        cache_kernels_to_disk = False
        rho2 = jax.device_put(rho2)
    with context as tempdir:
        for i in range(iterations):
            # Calculate the screening fields at each layer from every other layer
            zeros = jnp.zeros if gpu else np.zeros
            other_screening_fields = {
                name: zeros(points.shape[0], dtype=dtype) for name in layer_names
            }
            for layer, other_layer in itertools.product(device.layers_list, repeat=2):
                if layer.name == other_layer.name:
                    continue
                if (
                    i == 0
                    and layer.name == layer_names[0]
                    and other_layer.name == layer_names[1]
                ):
                    logger.info(
                        f"Caching {nkernels} layer-to-layer kernel(s) "
                        f"({total_bytes / 1024 ** 2:.0f} MB total) "
                        f"{'to disk' if cache_kernels_to_disk else 'in memory'}."
                    )
                logger.debug(
                    f"Calculating screening field at {layer.name} "
                    f"from {other_layer.name} ({i+1}/{iterations})."
                )
                g = streams[other_layer.name]
                dz = other_layer.z0 - layer.z0
                key = frozenset((layer.name, other_layer.name))
                # Get the cached kernel matrix,
                # or calculate it if it's not yet in the cache.
                q = kernels.get(key, None)
                if q is None:
                    q = (2 * dz**2 - rho2) / (4 * np.pi * (dz**2 + rho2) ** (5 / 2))
                    if cache_kernels_to_disk:
                        fname = os.path.join(tempdir, "_".join(key))
                        np.save(fname, q)
                        kernels[key] = f"{fname}.npy"
                    else:
                        kernels[key] = q
                elif isinstance(q, str):
                    # Kernel was cached to disk, so load it into memory.
                    q = np.load(q)
                # Calculate the dipole kernel and integrate
                # Eqs. 1-2 in [Brandt], Eqs. 5-6 in [Kirtley1], Eqs. 5-6 in [Kirtley2].
                screening_field = q @ (weights[:, 0] * g)
                other_screening_fields[layer.name] += screening_field
                del q, g
            # Solve again with the screening fields from all layers.
            # Calculate applied fields only once per iteration.
            new_layer_fields = {}
            for name, layer in device.layers.items():
                # Units: current_units / device.length_units
                new_layer_fields[name] = (
                    layer_fields[name] + other_screening_fields[name]
                )
            streams = {}
            fields = {}
            screening_fields = {}
            for name, layer in device.layers.items():
                logger.info(
                    f"Calculating {name} response to applied field and "
                    f"screening field from other layers ({i+1}/{iterations})."
                )
                g, J, total_field, screening_field = solve_layer(
                    device=device,
                    layer=name,
                    applied_field=new_layer_fields[name],
                    kernel=Q,
                    weights=weights,
                    Del2=Del2,
                    grad=grad,
                    Lambda_info=layer_Lambdas[name],
                    circulating_currents=circulating_currents,
                    vortices=vortices_by_layer[name],
                    current_units=current_units,
                    check_inversion=check_inversion,
                    gpu=gpu,
                )
                # Units: current_units
                streams[name] = g
                # Units: current_units / device.length_units
                currents[name] = J
                # Units: current_units / device.length_units
                fields[name] = total_field
                screening_fields[name] = screening_field

            solution = Solution(
                device=device,
                streams={
                    layer: np.asarray(g, dtype=dtype) for layer, g in streams.items()
                },
                current_densities=currents,
                fields={
                    # Units: field_units
                    layer: (np.asarray(field) / field_conversion_magnitude).astype(
                        dtype, copy=False
                    )
                    for layer, field in fields.items()
                },
                screening_fields={
                    # Units: field_units
                    layer: (np.asarray(field) / field_conversion_magnitude).astype(
                        dtype, copy=False
                    )
                    for layer, field in screening_fields.items()
                },
                applied_field=applied_field,
                field_units=field_units,
                current_units=current_units,
                circulating_currents=circulating_currents,
                terminal_currents=terminal_currents,
                vortices=vortices,
                solver=_solver,
            )
            if directory is not None:
                solution.to_file(os.path.join(directory, str(i + 1)))
            if return_solutions:
                solutions.append(solution)
            else:
                del solution
    if cache_kernels_to_disk:
        context.cleanup()
    if return_solutions:
        return solutions


def solve_many(
    device: Device,
    *,
    parallel_method: Optional[str] = None,
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
    vortices: Optional[Union[List[Vortex], List[List[Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_memory_cutoff: float = np.inf,
    log_level: int = logging.INFO,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
    gpu: bool = False,
) -> Tuple[Optional[Union[List[Solution], List[List[Solution]]]], Optional[List[str]]]:
    """Solves many models involving the same device, optionally in parallel using
    multiple processes.

    Args:
        device: The Device to simulate.
        parallel_method: The method to use for multiprocessing (None, "mp", or "ray").
        applied_fields: A callable or list of callables that compute(s) the applied
            magnetic field as a function of x, y, z coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}`` or list
            of such dicts. If circulating_current is a float, then it is assumed to
            be in units of current_units. If circulating_current is a string, then
            it is converted to a pint.Quantity.
        vortices: A list (list of lists) of ``Vortex`` objects.
        layer_updater: A callable with signature
            ``layer_updater(layer: Layer, **kwargs) -> Layer`` that updates
            parameter(s) of each layer in a device.
        layer_update_kwargs: A list of dicts of keyword arguments passed to
            ``layer_updater``.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        iterations: Number of times to compute the interactions between layers
        product: If True, then all combinations of applied_fields,
            circulating_currrents, and layer_update_kwargs are simulated (the
            behavior is given by itertools.product(), i.e. a nested for loop).
            Otherwise, the behavior is similar to zip().
            See superscreen.parallel.create_models for more details.
        directory: The directory in which to save the results. If None is given, then
            the results are not automatically saved to disk.
        return_solutions: Whether to return the Solution objects.
        keep_only_final_solution: Whether to keep/save only the Solution from
            the final iteration of superscreen.solve.solve for each setup.
        cache_memory_cutoff: If the memory needed for layer-to-layer kernel
            matrices exceeds ``cache_memory_cutoff`` times the current available
            system memory, then the kernel matrices will be cached to disk rather than
            in memory. Setting this value to ``inf`` disables caching to disk. In this
            case, the arrays will remain in memory unless they are swapped to disk
            by the operating system.
        log_level: Logging level to use, if any.
        use_shared_memory: Whether to use shared memory if parallel_method is not None.
        num_cpus: The number of processes to utilize.
        gpu: Solve on a GPU if available (requires JAX and CUDA). gpu = True is only allowed
            for serial execution, i.e., ``parallel_method in {None, False, "serial"}``.

    Returns:
        solutions, paths. If return_solutions is True, solutions is either a list of
        lists of Solutions (if keep_only_final_solution is False), or a list
        of Solutions (the final iteration for each setup). If directory is True,
        paths is a list of paths to the saved solutions, otherwise paths is None.
    """
    from . import parallel

    parallel_methods = {
        None: parallel.solve_many_serial,
        False: parallel.solve_many_serial,
        "serial": parallel.solve_many_serial,
        "multiprocessing": parallel.solve_many_mp,
        "mp": parallel.solve_many_mp,
        "ray": parallel.solve_many_ray,
    }

    if parallel_method not in parallel_methods:
        raise ValueError(
            f"Unknown parallel method, {parallel_method}. "
            f"Valid methods are {list(parallel_methods)}."
        )
    serial_methods = {None, False, "serial"}
    if num_cpus is not None and parallel_method in serial_methods:
        logger.warning(
            f"Ignoring num_cpus because parallel_method = {parallel_method!r}."
        )

    kwargs = dict(
        device=device,
        applied_fields=applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        field_units=field_units,
        current_units=current_units,
        check_inversion=check_inversion,
        iterations=iterations,
        product=product,
        directory=directory,
        return_solutions=return_solutions,
        keep_only_final_solution=keep_only_final_solution,
        cache_memory_cutoff=cache_memory_cutoff,
        log_level=log_level,
        use_shared_memory=use_shared_memory,
        num_cpus=num_cpus,
    )

    if parallel_method in serial_methods:
        kwargs["gpu"] = gpu
    elif gpu:
        raise ValueError(
            "Running solve_many() with gpu = True requires serial execution "
            "(i.e., parallel method in {None, False, 'serial'})."
        )

    solutions, paths = parallel_methods[parallel_method](**kwargs)
    return solutions, paths
