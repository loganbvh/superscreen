import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pint
from scipy import integrate

from ..device import Device, Polygon
from ..geometry import path_vectors
from ..parameter import Constant
from ..solution import Vortex

logger = logging.getLogger("solve")


class LambdaInfo:
    """An object containing information about the effective penetration depth for a given film.

    Args:
        film: The name of the film
        Lambda: The effective penetration depth at each mesh site
        london_lambda: The London penetration depth at each mesh site
        thickness: The thickness of the layer in which the film exists
    """

    lambda_str = "\u03bb"
    Lambda_str = "\u039b"

    def __init__(
        self,
        *,
        film: str,
        Lambda: np.ndarray,
        london_lambda: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
    ):
        self.film = film
        self.Lambda = Lambda
        self.london_lambda = london_lambda
        self.thickness = thickness
        self.inhomogeneous = (
            np.ptp(self.Lambda) / max(np.min(np.abs(self.Lambda)), np.finfo(float).eps)
            > 1e-6
        )
        if self.inhomogeneous:
            logger.info(
                f"Inhomogeneous {LambdaInfo.Lambda_str} in film {self.film!r}, "
                f"which violates the assumptions of the London model. "
                f"Results may not be reliable."
            )
        if self.london_lambda is not None:
            assert self.thickness is not None
            assert np.allclose(self.Lambda, self.london_lambda**2 / self.thickness)
        if np.any(self.Lambda < 0):
            raise ValueError(f"Negative Lambda in film {film!r}.")

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save the ``LambdaInfo`` instance to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` in which to save the
                ``LambdaInfo`` instance.
        """
        h5group.attrs["film"] = self.film
        if self.london_lambda is not None:
            h5group["london_lambda"] = self.london_lambda
        if self.thickness is not None:
            h5group.attrs["thickness"] = self.thickness
        h5group["Lambda"] = self.Lambda

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "LambdaInfo":
        """Load a LambdaInfo instance from an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` from which to load the
                ``LambdaInfo`` instance.

        Returns:
            The loaded ``LambdaInfo`` instance
        """
        london_lambda = None
        if "london_lambda" in h5group:
            london_lambda = np.array(h5group["london_lambda"])
        return LambdaInfo(
            film=h5group.attrs["film"],
            Lambda=np.array(h5group["Lambda"]),
            london_lambda=london_lambda,
            thickness=h5group.attrs.get("thickness", None),
        )


@dataclass
class FilmInfo:
    """A container for information about a single film required for the solver.

    Args:
        name: The film name
        layer: The layer in which the film exists
        lambda_info: A :class:`superscreen.solver.LambdaInfo` instance defining the
            effective penetration depth in the film
        vortices: Any :class:`superscreen.Vortex` instances located in the film
        interior_indices: The indices of the film in its mesh
        boundary_indices: Indices of the boundary vertices for the mesh,
            ordered counterclockwise
        hole_indices: A dict containing the indices of each hole in the film's mesh
        in_hole: A boolean array indicated which mesh sites lie inside a hole
        circulating_currents: A dict of ``{hole_name, circulating_current}``
        weights: The mesh weights
        kernel: The mesh self-field kernel :math:`\\mathbf{Q}`
        laplacian: The mesh Laplacian :math:`\\nabla^2`
        gradient: The mesh gradient operator :math:`\\vec{\\nabla}`
        terminal_currents: A dict of ``{terminal_name: terminal_current}``
    """

    name: str
    layer: str
    lambda_info: LambdaInfo
    vortices: Tuple[Vortex]
    interior_indices: np.ndarray
    boundary_indices: np.ndarray
    hole_indices: Dict[str, np.ndarray]
    in_hole: np.ndarray
    circulating_currents: Dict[str, float]
    weights: np.ndarray
    kernel: np.ndarray
    laplacian: np.ndarray
    gradient: Optional[np.ndarray] = None
    terminal_currents: Optional[Dict[str, float]] = None

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save the :class:`superscreen.solver.FilmInfo` instance to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` in which to save the ``FilmInfo``
        """
        h5group.attrs["name"] = self.name
        h5group.attrs["layer"] = self.layer
        self.lambda_info.to_hdf5(h5group.create_group("lambda_info"))
        vortices_grp = h5group.create_group("vortices")
        for i, vortex in enumerate(self.vortices):
            vortex.to_hdf5(vortices_grp.create_group(str(i)))
        h5group["interior_indices"] = self.interior_indices
        h5group["boundary_indices"] = self.boundary_indices
        hole_indices_grp = h5group.create_group("hole_indices")
        for hole, indices in self.hole_indices.items():
            hole_indices_grp[hole] = indices
        h5group["in_hole"] = self.in_hole
        circ_grp = h5group.create_group("circulating_currents")
        for hole, current in self.circulating_currents.items():
            circ_grp.attrs[hole] = current
        h5group["weights"] = self.weights
        h5group["kernel"] = self.kernel
        h5group["laplacian"] = self.laplacian
        if self.gradient is not None:
            h5group["gradient"] = self.gradient
        if self.terminal_currents is not None:
            term_grp = h5group.create_group("terminal_currents")
            for name, current in self.terminal_currents.items():
                term_grp.attrs[name] = current

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "FilmInfo":
        """Load a :class:`superscreen.solver.FilmInfo` instance from an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` from which to load the ``FilmInfo``

        Returns:
            The loaded :class:`superscreen.solver.FilmInfo` instance
        """
        name = h5group.attrs["name"]
        layer = h5group.attrs["layer"]
        lambda_info = LambdaInfo.from_hdf5(h5group["lambda_info"])
        vortices = []
        for i in sorted(h5group["vortices"], key=int):
            vortices.append(Vortex.from_hdf5(h5group[f"vortices/{i}"]))
        interior_indices = np.array(h5group["interior_indices"])
        boundary_indices = np.array(h5group["boundary_indices"])
        hole_indices = {}
        for hole, indices in h5group["hole_indices"].items():
            hole_indices[hole] = np.array(indices)
        in_hole = np.array(h5group["in_hole"])
        circulating_currents = dict(h5group["circulating_currents"].attrs)
        weights = np.array(h5group["weights"])
        kernel = np.array(h5group["kernel"])
        laplacian = np.array(h5group["laplacian"])
        gradient = terminal_currents = boundary_indices = None
        if "gradient" in h5group:
            gradient = np.array(h5group["gradient"])
        if "terminal_currents" in h5group:
            terminal_currents = dict(h5group["terminal_currents"].attrs)
        return FilmInfo(
            name=name,
            layer=layer,
            lambda_info=lambda_info,
            vortices=tuple(vortices),
            interior_indices=interior_indices,
            boundary_indices=boundary_indices,
            hole_indices=hole_indices,
            in_hole=in_hole,
            circulating_currents=circulating_currents,
            weights=weights,
            kernel=kernel,
            laplacian=laplacian,
            gradient=gradient,
            terminal_currents=terminal_currents,
        )


def get_holes_and_vortices_by_film(
    device: Device, vortices: List[Vortex]
) -> Tuple[Dict[str, List[Polygon]], Dict[str, List[Vortex]]]:
    vortices_by_film = {film_name: [] for film_name in device.films}
    holes_by_film = device.holes_by_film()
    for vortex in vortices:
        if not isinstance(vortex, Vortex):
            raise TypeError(f"Expected a Vortex, but got {type(vortex)}.")
        if not device.films[vortex.film].contains_points((vortex.x, vortex.y)).all():
            raise ValueError(
                f"Vortex {vortex!r} is not located in film {vortex.film!r}."
            )
        for hole in holes_by_film[vortex.film]:
            if hole.contains_points((vortex.x, vortex.y)).all():
                # Film contains hole and hole contains vortex.
                raise ValueError(f"Vortex {vortex} is located in hole {hole.name!r}.")
        vortices_by_film[vortex.film].append(vortex)
    return holes_by_film, vortices_by_film


def make_film_info(
    *,
    device: Device,
    vortices: List[Vortex],
    circulating_currents: Dict[str, float],
    terminal_currents: Dict[str, float],
) -> Dict[str, FilmInfo]:
    dtype = device.solve_dtype
    holes_by_film, vortices_by_film = get_holes_and_vortices_by_film(device, vortices)
    film_info = {}
    for name, film in device.films.items():
        mesh = device.meshes[name]
        layer = device.layers[film.layer]
        # Check and cache penetration depth
        london_lambda = layer.london_lambda
        d = layer.thickness
        Lambda = layer.Lambda
        if isinstance(london_lambda, (int, float)) and london_lambda <= d:
            length_units = device.ureg(device.length_units).units
            logger.info(
                f"Layer {name!r}: The film thickness, d = {d:.4f} {length_units:~P},"
                f" is greater than or equal to the London penetration depth, resulting"
                f" in an effective penetration depth {LambdaInfo.Lambda_str} = {Lambda:.4f}"
                f" {length_units:~P} <= {LambdaInfo.lambda_str} = {london_lambda:.4f}"
                f" {length_units:~P}. The assumption that the current density is nearly"
                f" constant over the thickness of the film may not be valid."
            )
        if isinstance(Lambda, (int, float)):
            Lambda = Constant(Lambda)
        Lambda = Lambda(mesh.sites[:, 0], mesh.sites[:, 1]).astype(dtype, copy=False)
        Lambda = Lambda[:, np.newaxis]
        if london_lambda is not None:
            if isinstance(london_lambda, (int, float)):
                london_lambda = Constant(london_lambda)
            london_lambda = london_lambda(mesh.sites[:, 0], mesh.sites[:, 1])
            london_lambda = london_lambda.astype(dtype, copy=False)[:, np.newaxis]

        hole_indices = {
            hole.name: hole.contains_points(mesh.sites, index=True)
            for hole in holes_by_film[name]
        }
        in_hole = np.zeros((len(mesh.sites)), dtype=bool)
        if hole_indices:
            in_hole_indices = np.concatenate(list(hole_indices.values()))
            in_hole[in_hole_indices] = True
        circ_currents = {
            hole_name: current
            for hole_name, current in circulating_currents.items()
            if hole_name in hole_indices
        }
        lambda_info = LambdaInfo(
            film=name,
            Lambda=Lambda,
            london_lambda=london_lambda,
            thickness=layer.thickness,
        )
        weights = mesh.operators.weights.astype(dtype, copy=False)
        Q = mesh.operators.Q.astype(dtype, copy=False)
        laplacian = mesh.operators.laplacian.toarray().astype(dtype, copy=False)
        grad = None
        if lambda_info.inhomogeneous:
            grad_x = mesh.operators.gradient_x.toarray().astype(dtype, copy=False)
            grad_y = mesh.operators.gradient_y.toarray().astype(dtype, copy=False)
            grad = np.array([grad_x, grad_y])
        if name in device.terminals:
            boundary_indices = device.boundary_vertices(name)
        else:
            boundary_indices = mesh.boundary_indices
        interior_indices = np.setdiff1d(
            film.contains_points(mesh.sites, index=True), boundary_indices
        )
        term_currents = None
        if name in terminal_currents:
            term_currents = terminal_currents[name]
        film_info[name] = FilmInfo(
            name=name,
            layer=layer.name,
            lambda_info=lambda_info,
            vortices=vortices_by_film[name],
            interior_indices=interior_indices,
            boundary_indices=boundary_indices,
            hole_indices=hole_indices,
            in_hole=in_hole,
            circulating_currents=circ_currents,
            terminal_currents=term_currents,
            weights=weights,
            kernel=Q,
            gradient=grad,
            laplacian=laplacian,
        )
    return film_info


def current_to_float(
    value: Union[float, str, pint.Quantity], ureg: pint.UnitRegistry, current_units: str
):
    """Convert a current to a float in the given units."""
    if isinstance(value, str):
        value = ureg(value)
    if isinstance(value, pint.Quantity):
        value = value.to(current_units).magnitude
    return value


def currents_to_floats(
    currents: Dict[str, Union[float, str, pint.Quantity]],
    ureg: pint.UnitRegistry,
    current_units: str,
) -> Dict[str, float]:
    """Converts a dict of currents to a dict of floats in the given units."""
    return {
        key: current_to_float(value, ureg, current_units)
        for key, value in currents.items()
    }


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
            ``pint.Quantity``. If value is not a string with units or a ``pint.Quantity``,
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


def stream_from_current_density(points: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Computes the scalar stream function corresonding to a
    given current density :math:`J`, according to:

    .. math::

        g(\\vec{r})=g(\\vec{r}_0)+\\int_{\\vec{r}_0}^\\vec{r}
        (\\hat{z}\\times\\vec{J})\\cdot\\mathrm{d}\\vec{\\ell}

    Args:
        points: Shape ``(n, 2)`` array of ``(x, y)`` positions at which to
            compute the stream function :math:`g`.
        J: Shape ``(n, 2)`` array of the current density ``(Jx, Jy)`` at the
            given ``points``.s

    Returns:
        A shape ``(n, )`` array of the stream function at the given ``points``.
    """
    # (0, 0, 1) X (Jx, Jy, 0) == (-Jy, Jx, 0)
    zhat_cross_J = J[:, [1, 0]]
    zhat_cross_J[:, 0] *= -1
    dl = np.diff(points, axis=0, prepend=points[:1])
    integrand = np.sum(zhat_cross_J * dl, axis=1)
    return integrate.cumulative_trapezoid(integrand, initial=0)


def stream_from_terminal_current(points: np.ndarray, current: float) -> np.ndarray:
    """Computes the terminal stream function corresponding to a given terminal current.

    We assume that the current :math:`I` is uniformly distributed along the terminal
    with a current density :math:`\\vec{J}` which is perpendicular to the terminal.
    Then for :math:`\\vec{r}` along the terminal, the stream function is given by

    .. math::

        g(\\vec{r})=g(\\vec{r}_0)+\\int_{\\vec{r}_0}^\\vec{r}
        (\\hat{z}\\times\\vec{J})\\cdot\\mathrm{d}\\vec{\\ell}

    Args:
        points: A shape ``(n, 2)`` array of terminal vertex positions.
        current: The total current sources by the terminal.

    Returns:
        A shape ``(n, )`` array of the stream function along the terminal.
    """
    edge_lengths, unit_normals = path_vectors(points)
    J = current * unit_normals / np.sum(edge_lengths)
    g = stream_from_current_density(points, J)
    return g * current / g[-1]
