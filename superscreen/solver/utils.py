import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pint

try:
    import jax
except (ModuleNotFoundError, ImportError, RuntimeError):
    jax = None

from ..device import Device
from ..device.polygon import Polygon
from ..parameter import Constant

logger = logging.getLogger("solve")


@dataclass
class Vortex:
    """A vortex located at position ``(x, y)`` in ``layer`` containing
    a total flux ``nPhi0`` in units of the flux quantum :math:`\\Phi_0`.

    Args:
        x: Vortex x-position.
        y: Vortex y-position.
        layer: The layer in which the vortex is pinned.
        nPhi0: The number of flux quanta contained in the vortex.
    """

    x: float
    y: float
    layer: str
    nPhi0: float = 1

    def to_hdf5(self, h5group: h5py.Group) -> None:
        h5group.attrs["x"] = self.x
        h5group.attrs["y"] = self.y
        h5group.attrs["layer"] = self.layer
        h5group.attrs["nPhi0"] = self.nPhi0

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "Vortex":
        return Vortex(
            x=h5group.attrs["x"],
            y=h5group.attrs["y"],
            layer=h5group.attrs["layer"],
            nPhi0=h5group.attrs["nPhi0"],
        )


class FilmSolution:
    def __init__(
        self,
        stream: np.ndarray,
        current_density: np.ndarray,
        applied_field: np.ndarray,
        self_field: np.ndarray,
        field_from_other_films: Optional[np.ndarray] = None,
    ):
        self.stream = np.asarray(stream)
        self.current_density = np.asarray(current_density)
        self.applied_field = np.asarray(applied_field)
        self.self_field = np.asarray(self_field)
        if field_from_other_films is not None:
            field_from_other_films = np.asarray(field_from_other_films)
        self.field_from_other_films = field_from_other_films
        self._total_field: Optional[np.ndarray] = None

    @property
    def total_field(self) -> np.ndarray:
        if self._total_field is None:
            self._total_field = self.applied_field + self.self_field
            if self.field_from_other_films is not None:
                self._total_field += self.field_from_other_films
        return self._total_field

    def to_hdf5(self, h5group: h5py.Group) -> None:
        h5group["stream"] = self.stream
        h5group["current_density"] = self.current_density
        h5group["applied_field"] = self.applied_field
        h5group["self_field"] = self.self_field
        if self.field_from_other_films is not None:
            h5group["field_from_other_films"] = self.field_from_other_films

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "FilmSolution":
        field_from_other_films = h5group.get("field_from_other_films", None)
        if field_from_other_films is not None:
            field_from_other_films = np.array(field_from_other_films)
        return FilmSolution(
            stream=np.array(h5group["stream"]),
            current_density=np.array(h5group["current_density"]),
            applied_field=np.array(h5group["applied_field"]),
            self_field=np.array(h5group["self_field"]),
            field_from_other_films=field_from_other_films,
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, FilmSolution):
            return False
        if self.field_from_other_films is None:
            if other.field_from_other_films is not None:
                return False
        if other.field_from_other_films is None:
            if self.field_from_other_films is not None:
                return False
        return (
            np.allclose(self.stream, other.stream)
            and np.allclose(self.current_density, other.current_density)
            and np.allclose(self.applied_field, other.applied_field)
            and np.allclose(self.self_field, other.self_field)
        )


class LambdaInfo:
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
            logger.warning(
                f"Inhomogeneous {LambdaInfo.Lambda_str} in layer '{self.layer}', "
                f"which violates the assumptions of the London model. "
                f"Results may not be reliable."
            )
        if self.london_lambda is not None:
            assert self.thickness is not None
            assert np.allclose(self.Lambda, self.london_lambda**2 / self.thickness)
        if np.any(self.Lambda < 0):
            raise ValueError(f"Negative Lambda in film {film!r}.")


@dataclass
class FilmInfo:
    name: str
    layer: str
    lambda_info: LambdaInfo
    vortices: Tuple[Vortex]
    film_indices: np.ndarray
    hole_indices: Dict[str, np.ndarray]
    in_hole: np.ndarray
    circulating_currents: Dict[str, float]
    weights: np.ndarray
    kernel: np.ndarray
    laplacian: np.ndarray
    gradient: Optional[np.ndarray] = None
    terminal_currents: Optional[Dict[str, float]] = None


def get_holes_vortices_by_film(
    device: Device, vortices: List[Vortex]
) -> Tuple[Dict[str, List[Polygon]], Dict[str, List[Vortex]]]:
    films_by_layer = device.polygons_by_layer("film")
    vortices_by_film = {film_name: [] for film_name in device.films}
    holes_by_film = device.holes_by_film()
    for vortex in vortices:
        if not isinstance(vortex, Vortex):
            raise TypeError(f"Expected a Vortex, but got {type(vortex)}.")
        if vortex.layer not in device.layers:
            raise ValueError(f"Vortex located in unknown layer: {vortex}.")
        vortex_film = None
        for film in films_by_layer[vortex.layer]:
            for hole in holes_by_film[film.name]:
                if hole.contains_points((vortex.x, vortex.y)).all():
                    # Film contains hole and hole contains vortex.
                    raise ValueError(
                        f"Vortex {vortex} is located in hole {hole.name!r}."
                    )
            if film.contains_points((vortex.x, vortex.y)).all():
                # Vortex is located in the film and not in the hole.
                vortex_film = film.name
        if vortex_film is None:
            raise ValueError(f"Vortex {vortex} is not located in a film.")
        vortices_by_film[vortex_film].append(vortex)
    return holes_by_film, vortices_by_film


def make_film_info(
    *,
    device: Device,
    vortices: List[Vortex],
    circulating_currents: Dict[str, float],
    terminal_currents: Dict[str, float],
    dtype: Union[np.dtype, str] = "float32",
    gpu: bool = False,
) -> Dict[str, FilmInfo]:
    holes_by_film, vortices_by_film = get_holes_vortices_by_film(device, vortices)
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
            logger.warning(
                f"Layer '{name}': The film thickness, d = {d:.4f} {length_units:~P},"
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
        weights = mesh.operators.weights.astype(dtype, copy=False)[:, np.newaxis]
        Q = mesh.operators.Q.astype(dtype, copy=False)
        laplacian = mesh.operators.laplacian.toarray().astype(dtype, copy=False)
        grad = None
        if lambda_info.inhomogeneous:
            grad_x = mesh.operators.gradient_x.toarray().astype(dtype, copy=False)
            grad_y = mesh.operators.gradient_y.toarray().astype(dtype, copy=False)
            grad = np.array([grad_x, grad_y])
        if gpu and jax is not None:
            weights = jax.device_put(weights)
            Q = jax.device_put(Q)
            laplacian = jax.device_put(laplacian)
            if grad is not None:
                grad = jax.device_put(grad)
        film_info[name] = FilmInfo(
            name=name,
            layer=layer.name,
            lambda_info=lambda_info,
            vortices=vortices_by_film[name],
            film_indices=film.contains_points(mesh.sites, index=True),
            hole_indices=hole_indices,
            in_hole=in_hole,
            circulating_currents=circ_currents,
            terminal_currents=terminal_currents,
            weights=weights,
            kernel=Q,
            gradient=grad,
            laplacian=laplacian,
        )
    return film_info


# Convert all circulating and terminal currents to floats.
def current_to_float(value, ureg, current_units: str):
    if isinstance(value, str):
        value = ureg(value)
    if isinstance(value, pint.Quantity):
        value = value.to(current_units).magnitude
    return value


def currents_to_floats(currents, ureg, current_units) -> Dict[str, float]:
    _currents = currents.copy()
    for name, val in _currents.items():
        currents[name] = current_to_float(val, ureg, current_units)
    return currents


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
