import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.linalg as la

from ..device import Device
from ..solution import FilmSolution
from .utils import FilmInfo, stream_from_terminal_current

logger = logging.getLogger("solve")


@dataclass
class LinearSystem:
    r"""The linear system representing a given film or hole.

    Args:
        A: The matrix quantity to be inverted,
            :math:`\mathbf{Q}.\mathbf{w}^T-\mathbf{\Lambda}^T.\mathbf{\nabla}^2-\vec{\nabla}\mathbf{\Lambda}\cdot\vec{\nabla}`
        indices: The indices into the corresponding mesh
        lu_piv: The LU factorization ``lu_piv = scipy.linalg.lu_factor(-A)``,
            see :func:`scipy.linalg.lu_factor`
        grad_Lambda_term: The term corresponding to the gradient of the
            effective penetration depth.
    """

    A: np.ndarray
    indices: np.ndarray
    lu_piv: Optional[Tuple[np.ndarray, np.ndarray]] = None
    grad_Lambda_term: Union[float, np.ndarray] = 0.0

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save a :class:`superscreen.solver.LinearSystem` to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` to which to save the ``LinearSystem``
        """
        h5group["A"] = self.A
        h5group["indices"] = self.indices
        if self.lu_piv is not None:
            h5group["lu"] = self.lu_piv[0]
            h5group["piv"] = self.lu_piv[1]
        if isinstance(self.grad_Lambda_term, np.ndarray):
            h5group["grad_Lambda_term"] = self.grad_Lambda_term
        else:
            h5group.attrs["grad_Lambda_term"] = self.grad_Lambda_term

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "LinearSystem":
        """Load a :class:`superscreen.solver.LinearSystem` to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` from which to load the ``LinearSystem``

        Returns:
            The loaded :class:`superscreen.solver.LinearSystem`
        """
        A = np.array(h5group["A"])
        indices = np.array(h5group["indices"])
        lu_piv = None
        if "lu" in h5group:
            lu = np.array(h5group["lu"])
            piv = np.array(h5group["piv"])
            lu_piv = (lu, piv)
        grad_Lambda_term = 0.0
        if "grad_Lambda_term" in h5group:
            grad_Lambda_term = np.array(h5group["grad_Lambda_term"])
        else:
            grad_Lambda_term = h5group.attrs["grad_Lambda_term"]
        return LinearSystem(
            A, indices, lu_piv=lu_piv, grad_Lambda_term=grad_Lambda_term
        )


@dataclass
class TerminalSystems:
    """A container for the :class:`superscreen.solver.LinearSystem` objects
    needed to calculate the stream function associated with transport currents.

    Args:
        film: The film name:
        boundary: The :class:`superscreen.solver.LinearSystem` for the film boundary
        holes: A dict of :class:`superscreen.solver.LinearSystem` objects for the
            holes in the film
        film_without_boundary: The :class:`superscreen.solver.LinearSystem` for the
            interior of the film, excluding the boundary
        film_without_boundary_or_holes: The :class:`superscreen.solver.LinearSystem` for
            the interior of the film, excluding the boundary and any holes in the film
    """

    film: str
    boundary: LinearSystem
    holes: Dict[str, LinearSystem]
    film_without_boundary: LinearSystem
    film_without_boundary_or_holes: Optional[LinearSystem] = None

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save a :class:`superscreen.solver.TerminalSystems` to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` to which to save the ``TerminalSystems``
        """
        h5group.attrs["film"] = self.film
        self.boundary.to_hdf5(h5group.create_group("boundary"))
        holes_grp = h5group.create_group("holes")
        for name, system in self.holes.items():
            system.to_hdf5(holes_grp.create_group(name))
        self.film_without_boundary.to_hdf5(
            h5group.create_group("film_without_boundary")
        )
        if self.film_without_boundary_or_holes is not None:
            self.film_without_boundary_or_holes.to_hdf5(
                h5group.create_group("film_without_boundary_or_holes")
            )

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "TerminalSystems":
        """Load a :class:`superscreen.solver.TerminalSystems` to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` from which to load the ``TerminalSystems``

        Returns:
            The loaded :class:`superscreen.solver.TerminalSystems`
        """
        film = h5group.attrs["film"]
        boundary = LinearSystem.from_hdf5(h5group["boundary"])
        holes = {}
        for name, grp in h5group["holes"].items():
            holes[name] = LinearSystem.from_hdf5(grp)
        film_without_boundary = LinearSystem.from_hdf5(h5group["film_without_boundary"])
        film_without_boundary_or_holes = None
        if "film_without_boundary_or_holes" in h5group:
            film_without_boundary_or_holes = LinearSystem.from_hdf5(
                h5group["film_without_boundary_or_holes"]
            )
        return TerminalSystems(
            film=film,
            boundary=boundary,
            holes=holes,
            film_without_boundary=film_without_boundary,
            film_without_boundary_or_holes=film_without_boundary_or_holes,
        )


def factorize_linear_systems(
    device: Device, film_info_dict: Dict[str, FilmInfo]
) -> Tuple[
    Dict[str, LinearSystem],
    Dict[str, Dict[str, LinearSystem]],
    Dict[str, TerminalSystems],
]:
    """Build and factorize the linear systems for all films, holes, and terminals.

    Args:
        device: The :class:`superscreen.Device` to solve
        film_info_dict: A dict of ``{film_name: film_info}``, where each ``film_info``
            is a :class:`superscreen.solver.FilmInfo` instance

    Returns:
        A dict of ``{film_name: film_system}``,
        a dict of ``{film_name: {hole_name: hole_system}}``,
        and a dict of ``{film_name: TerminalSystems}``
    """
    film_systems = {}
    hole_systems = {}
    terminal_systems = {}
    for film_name, film_info in film_info_dict.items():
        hole_systems[film_name] = {}
        interior_indices = film_info.interior_indices
        boundary_indices = film_info.boundary_indices
        hole_indices = film_info.hole_indices
        Lambda_info = film_info.lambda_info
        inhomogeneous = Lambda_info.inhomogeneous
        Lambda = Lambda_info.Lambda
        terminal_Lambda = 1e10 * np.ones_like(Lambda)
        if inhomogeneous:
            grad = film_info.gradient
            grad_Lambda_term = np.einsum("ijk, ijk -> jk", (grad @ Lambda), grad)
        else:
            grad_Lambda_term = 0

        def make_system_1d(indices, Lambda):
            return _build_system_1d(
                film_info.kernel,
                film_info.weights,
                Lambda,
                film_info.laplacian,
                grad_Lambda_term,
                indices,
                inhomogeneous=inhomogeneous,
            )

        def make_system_2d(indices, Lambda):
            return _build_system_2d(
                film_info.kernel,
                film_info.weights,
                Lambda,
                film_info.laplacian,
                grad_Lambda_term,
                indices,
                inhomogeneous=inhomogeneous,
            )

        for hole_name, indices in hole_indices.items():
            # Effective field associated with the circulating currents:
            # current is in [current_units], Lambda is in [device.length_units],
            # and Del2 is in [device.length_units ** (-2)], so
            # Ha_eff has units of [current_unit / device.length_units]
            hole_systems[film_name][hole_name] = LinearSystem(
                A=make_system_1d(indices, Lambda),
                indices=indices,
                grad_Lambda_term=grad_Lambda_term,
            )

        if film_name in device.terminals:
            # Make the boundary linear system
            boundary_system = LinearSystem(
                A=make_system_1d(boundary_indices, terminal_Lambda),
                indices=boundary_indices,
                grad_Lambda_term=grad_Lambda_term,
            )
            # Make the film interior linear system (including holes)
            A = make_system_2d(interior_indices, terminal_Lambda)
            film_without_boundary_system = LinearSystem(
                A=A,
                indices=interior_indices,
                lu_piv=la.lu_factor(-A),
                grad_Lambda_term=grad_Lambda_term,
            )
            # Make the hole linear systems
            terminal_hole_systems = {}
            for hole_name, indices in hole_indices.items():
                terminal_hole_systems[hole_name] = LinearSystem(
                    A=make_system_1d(indices, terminal_Lambda),
                    indices=indices,
                    grad_Lambda_term=grad_Lambda_term,
                )
            # Make the film interior linear system (excluding holes)
            film_without_boundary_or_holes_system = None
            if hole_indices:
                interior_indices = np.setdiff1d(
                    interior_indices, np.concatenate(list(hole_indices.values()))
                )
                A = make_system_2d(interior_indices, terminal_Lambda)
                film_without_boundary_or_holes_system = LinearSystem(
                    A=A,
                    indices=interior_indices,
                    lu_piv=la.lu_factor(-A),
                    grad_Lambda_term=grad_Lambda_term,
                )

            terminal_systems[film_name] = TerminalSystems(
                film=film_name,
                boundary=boundary_system,
                holes=terminal_hole_systems,
                film_without_boundary=film_without_boundary_system,
                film_without_boundary_or_holes=film_without_boundary_or_holes_system,
            )

        # Form the linear system for the film:
        # gf = -K @ h, where K = inv(Q * w - Lambda * Del2 - grad_Lambda_term) = inv(A)
        # Eqs. 15-17 in [Brandt], Eqs 12-14 in [Kirtley1], Eqs. 12-14 in [Kirtley2].
        # We want all points that are in a film and not in a hole.
        if hole_indices:
            interior_indices = np.setdiff1d(
                interior_indices, np.concatenate(list(hole_indices.values()))
            )
        if film_name in device.terminals:
            interior_indices = np.setdiff1d(interior_indices, boundary_indices)
        A = make_system_2d(interior_indices, Lambda)
        film_systems[film_name] = LinearSystem(
            A=A,
            indices=interior_indices,
            lu_piv=la.lu_factor(-A),
            grad_Lambda_term=grad_Lambda_term,
        )
    return film_systems, hole_systems, terminal_systems


def _build_system_1d(
    Q, weights, Lambda, laplacian, grad_Lambda_term, ix, inhomogeneous=False
):
    """Builds the linear system for the 'effective applied fields'."""
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[:, ix]
    else:
        grad_Lambda = 0
    return Q[:, ix] * weights[ix] - Lambda[ix, 0] * laplacian[:, ix] - grad_Lambda


def _build_system_2d(
    Q, weights, Lambda, laplacian, grad_Lambda_term, ix1d, inhomogeneous=False
):
    """Builds the linear system to solve for the stream function."""
    ix2d = np.ix_(ix1d, ix1d)
    if inhomogeneous:
        grad_Lambda = grad_Lambda_term[ix2d]
    else:
        grad_Lambda = 0
    return Q[ix2d] * weights[ix1d] - Lambda[ix1d, 0] * laplacian[ix2d] - grad_Lambda


def solve_for_terminal_current_stream(
    device: Device,
    film_info: FilmInfo,
    terminal_systems: TerminalSystems,
    terminal_currents: Dict[str, float],
) -> np.ndarray:
    """Solves for the stream function associated with transport currents in a single film.

    The algorithm is:

    1. Solve for the stream function in the film assuming no applied field and
    ignoring the presence of any holes.
    2. Set the stream function in each hole to the weighted average of the stream
    function found in step 1.
    3. Re-solve for the stream function in the film with the new hole boundary conditions.

    Args:
        device: The :class:`superscreen.Device` to solve
        film_info: The :class:`superscreen.solver.FilmInfo` instance for the film
        terminal_systems: The :class:`superscreen.solver.TerminalSystems` instance
            for the film
        terminal_currents: A dict of ``{terminal_name: terminal_current}``

    Returns:
        The stream function associated with the transport current
    """
    terminal_currents = terminal_currents.copy()
    mesh = device.meshes[film_info.name]
    points = mesh.sites
    weights = mesh.operators.weights
    npoints = len(points)
    if not any(terminal_currents.values()):
        return np.zeros(npoints)

    terminals = device.terminals[film_info.name].copy()
    boundary_indices = terminal_systems.boundary.indices
    boundary_points = points[boundary_indices]

    # 1. Set the stream function on the boundary
    # and solve for the effective applied field.
    g = np.zeros(npoints)
    Ha_eff = np.zeros(npoints)
    for terminal in terminals:
        current = terminal_currents[terminal.name]
        ix_boundary = np.sort(terminal.contains_points(boundary_points, index=True))
        remaining_boundary = boundary_indices[(ix_boundary[-1] + 1) :]
        ix_terminal = boundary_indices[ix_boundary]
        stream = stream_from_terminal_current(points[ix_terminal], -current)
        g[ix_terminal] += stream
        g[remaining_boundary] += stream[-1]
    # The stream function on the "reference boundary" (i.e., the boundary
    # immediately after the output terminal in a CCW direction) should be zero.
    g_ref = g[ix_terminal.max()]
    g[boundary_indices] += -g_ref
    A = terminal_systems.boundary.A
    Ha_eff += -(A @ g[boundary_indices])

    # 2. Solve for the stream function inside the film given the effective applied
    # field, ignoring the presence of holes completely.
    lu_piv = terminal_systems.film_without_boundary.lu_piv
    h = -Ha_eff[terminal_systems.film_without_boundary.indices]
    gf = la.lu_solve(lu_piv, h)
    g[terminal_systems.film_without_boundary.indices] = gf
    if len(terminal_systems.holes) == 0:
        return g

    # Set the stream function in each hole to the average value
    # obtained when ignoring holes.
    Ha_eff = np.zeros(points.shape[0])
    for system in terminal_systems.holes.values():
        ix = system.indices
        g[ix] = np.average(g[ix], weights=weights[ix])
        A = system.A
        Ha_eff += -(A @ g[ix])

    A = terminal_systems.boundary.A
    Ha_eff += -(A @ g[boundary_indices])

    # 3. Solve for the stream function inside the superconducting film again,
    # now with the new boundary conditions for the holes.
    ix = terminal_systems.film_without_boundary_or_holes.indices
    lu_piv = terminal_systems.film_without_boundary_or_holes.lu_piv
    gf = la.lu_solve(lu_piv, -Ha_eff[ix])
    g[ix] = gf
    return g


def solve_film(
    *,
    device: Device,
    applied_field: np.ndarray,
    film_info: FilmInfo,
    film_system: LinearSystem,
    hole_systems: Dict[str, LinearSystem],
    field_conversion: float,
    vortex_flux: float,
    terminal_systems: Optional[TerminalSystems] = None,
    field_from_other_films: Optional[np.ndarray] = None,
    check_inversion: bool = False,
) -> FilmSolution:
    """Computes the stream function and magnetic field within a single film
    in a :class:`superscreen.Device`.

    Args:
        device: The :class:`superscreen.Device` to simulate.
        applied_field: The applied magnetic field evaluated at the mesh vertices.
        film_info: The :class:`superscreen.solver.FilmInfo` instance for the film
        film_system: The :class:`superscreen.solver.LinearSystem` for the film
        hole_systems: A dict of ``{hole_name: hole_system}``, where each ``hole_system``
            is an instance of :class:`superscreen.solver.LinearSystem`
        field_conversion: The field conversion factor from user units to solver units
        vortex_flux: The flux associated with a single Phi_0 vortex in the solver units
        terminal_systems: An instance of :class:`superscreen.solver.TerminalSystems`
            containing the linear systems required to calculate the stream function
            associated with transport currents.
        field_from_other_films: The magnetic field from any other films in the ``Device``
        check_inversion: Whether to verify the accuracy of the matrix inversion.

    Returns:
        A :class:`superscreen.FilmSolution` containing the results
    """

    circulating_currents = film_info.circulating_currents
    terminal_currents = film_info.terminal_currents or {}
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

    # Set the boundary conditions for all holes:
    # 1. g[hole] = I_circ_hole
    # 2. Effective field associated with I_circ_hole
    # See Section II(a) in [Brandt], Eqs. 18-19 in [Kirtley1],
    # and Eqs 17-18 in [Kirtley2].
    for name, system in hole_systems.items():
        indices = system.indices
        A = system.A
        current = circulating_currents.get(name, 0)
        g[indices] += current  # g[hole] = I_circ
        Ha_eff += -(A @ g[indices])

    if film_info.name in device.terminals:
        g_transport = solve_for_terminal_current_stream(
            device,
            film_info,
            terminal_systems,
            terminal_currents,
        )
        g += g_transport

    indices = film_system.indices
    A = film_system.A
    lu_piv = film_system.lu_piv
    h = Hz_applied[indices] - Ha_eff[indices]
    gf = la.lu_solve(lu_piv, h)
    g[indices] += gf

    if check_inversion:
        # Validate solution
        hsim = -(A @ gf)
        if not np.allclose(hsim, h):
            logger.warning(
                f"Unable to solve for stream function in {film_info.name!r}), "
                f"maximum error {np.abs(hsim - h).max():.3e}."
            )
    K = None  # Matrix inverse of A
    for vortex in film_info.vortices:
        if K is None:
            # Compute K only once if needed
            K = -la.lu_solve(lu_piv, np.eye(A.shape[0]))
        # Index of the mesh vertex that is closest to the vortex position:
        # in the film-specific sub-mesh
        xy = (vortex.x, vortex.y)
        j_film = np.argmin(la.norm(points[indices] - xy, axis=1))
        # ... and in the full device mesh.
        j_device = np.argmin(la.norm(points - xy, axis=1))
        # Eq. 28 in [Brandt]
        g_vortex = vortex_flux * vortex.nPhi0 * K[:, j_film] / weights[j_device].T
        g[indices] += g_vortex
    # Current density J = curl(g \hat{z}) = [dg/dy, -dg/dx]
    J = np.array([grad_y @ g, -(grad_x @ g)]).T
    # Eq. 7 in [Kirtley1], Eq. 7 in [Kirtley2]
    screening_field = Q @ (weights * g)
    if field_from_other_films is not None:
        field_from_other_films = field_from_other_films / field_conversion
    return FilmSolution(
        stream=g,
        current_density=J,
        applied_field=applied_field / field_conversion,
        self_field=screening_field / field_conversion,
        field_from_other_films=field_from_other_films,
    )
