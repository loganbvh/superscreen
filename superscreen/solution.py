import logging
import os
from contextlib import nullcontext
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pint
from scipy import interpolate

from .about import version_dict
from .device import Device, Polygon
from .fem import cdist_batched, in_polygon
from .io import deserialize_obj, serialize_obj
from .parameter import Constant
from .solver.utils import FilmSolution, Vortex
from .sources.current import biot_savart_2d

logger = logging.getLogger(__name__)


class Fluxoid(NamedTuple):
    """The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
    is defined as:

    .. math::

        \\Phi^f_S = \\underbrace{
            \\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r
        }_{\\text{flux part}}
        + \\underbrace{
            \\oint_{\\partial S}
            \\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}
        }_{\\text{supercurrent part}}

    Args:
        flux_part: :math:`\\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r`.
        supercurrent_part: :math:`\\oint_{\\partial S}\\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}`.
    """

    flux_part: Union[float, pint.Quantity]
    supercurrent_part: Union[float, pint.Quantity]


class Solution:
    """A container for the calculated stream functions and fields,
    with some convenient data processing methods.

    Args:
        device: The ``Device`` that was solved
        streams: A dict of ``{film_name: stream_function}``
        current_densities: A dict of ``{film_name: current_density}``
        fields: A dict of ``{film_name: total_field}``
        screening_fields: A dict of ``{film_name: screening_field}``
        applied_fields: A dict of ``{film_name: applied_field}``
        applied_field_func: The function defining the applied field
        field_units: Units of the applied field
        current_units: Units used for current quantities.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
        terminal_currents: A dict of ``{terminal_name: terminal_current}``.
        vortices: A list of ``Vortex`` objects located in the ``Device``.
        solver: The solver method that generated the solution.
    """

    def __init__(
        self,
        *,
        device: Device,
        film_solutions: Dict[str, FilmSolution],
        applied_field_func: Callable,
        field_units: str,
        current_units: str,
        circulating_currents: Optional[Dict[str, float]] = None,
        terminal_currents: Optional[Dict[str, float]] = None,
        vortices: Optional[List[Vortex]] = None,
        solver: str = "superscreen.solve",
    ):
        self.device = device.copy(with_mesh=True, copy_mesh=False)
        self.film_solutions = film_solutions
        self.applied_field_func = applied_field_func
        self.circulating_currents = circulating_currents or {}
        self.terminal_currents = terminal_currents or {}
        self.vortices = vortices or []
        # Make field_units and current_units "read-only" attributes.
        # The should never be changed after instantiation.
        self._field_units = field_units
        self._current_units = current_units
        self._solver = solver
        self._time_created = datetime.now()
        self._version_info = version_dict()

    @property
    def field_units(self) -> str:
        """The units in which magnetic fields are specified."""
        return self._field_units

    @property
    def current_units(self) -> str:
        """The units in which currents are specified."""
        return self._current_units

    @property
    def solver(self) -> str:
        """The solver method that generated the solution."""
        return self._solver

    @property
    def time_created(self) -> datetime:
        """The time at which the solution was originally created."""
        return self._time_created

    @property
    def version_info(self) -> Dict[str, str]:
        """A dictionary of dependency versions."""
        return self._version_info

    @staticmethod
    def _select_interpolator(
        method: str,
    ) -> Union[
        interpolate.NearestNDInterpolator,
        interpolate.LinearNDInterpolator,
        interpolate.CloughTocher2DInterpolator,
    ]:
        valid_methods = ("nearest", "linear", "cubic")
        if method not in valid_methods:
            raise ValueError(
                f"Interpolation method must be one of {valid_methods} (got {method})."
            )
        if method == "nearest":
            return interpolate.NearestNDInterpolator
        if method == "linear":
            return interpolate.LinearNDInterpolator
        return interpolate.CloughTocher2DInterpolator

    def interp_current_density(
        self,
        positions: np.ndarray,
        *,
        film: str,
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Interpolates the current density ``J = [dg/dy, -dg/dx]`` within a film.

        Additional keyword arguments are passed to the relevant interpolator:
        :class``scipy.interpolate.NearestNDInterpolator`,
        :class:`scipy.interpolate.LinearNDInterpolator`, or
        :class:`scipy.interpolate.CloughTocher2DInterpolator`.

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the current density.
            film: The name of the film in which to interpolate current density.
            method: Interpolation method to use (see scipy.interpolate.griddata).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            The interpolated current density
        """
        default_units = f"{self.current_units} / {self.device.length_units}"
        if units is None:
            units = default_units
        positions = np.atleast_2d(positions)
        interpolator = self._select_interpolator(method)
        xy = self.device.meshes[film].sites
        J = self.film_solutions[film].current_density
        Jx_interp = interpolator(xy, J[:, 0], **kwargs)
        Jy_interp = interpolator(xy, J[:, 1], **kwargs)
        J = np.stack([Jx_interp(positions), Jy_interp(positions)], axis=1)
        # J[~np.isfinite(J)] = 0
        J = (J * self.device.ureg(default_units)).to(units)
        if with_units:
            return J
        return J.magnitude

    def interp_field(
        self,
        positions: np.ndarray,
        *,
        film: str,
        dataset: str = "field",
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ):
        """Interpolates the z-component of the field within a film.

        Additional keyword arguments are passed to the relevant interpolator:
        :class``scipy.interpolate.NearestNDInterpolator`,
        :class:`scipy.interpolate.LinearNDInterpolator`, or
        :class:`scipy.interpolate.CloughTocher2DInterpolator`.

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the fields.
            film: The name of the film in which to interpolate the field.
            dataset: The dataset to interpolate. One of 'field', 'screening_field',
                or 'applied_field'.
            method: Interpolation method to use: 'nearest', 'linear', or 'cubic'.
            units: The desired units for the current density. Defaults to
                ``self.field_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            The interpolated field
        """
        from .solver import convert_field

        interpolator = self._select_interpolator(method)
        device = self.device
        if units is None:
            units = self.field_units
        valid_datasets = (
            "field",
            "self_field",
            "applied_field",
            "field_from_other_films",
        )
        if dataset not in valid_datasets:
            raise ValueError(
                f"Invalid dataset: {dataset!r}. Expected one of {valid_datasets!r}"
            )
        if dataset == "field":
            field = self.film_solutions[film].total_field
        elif dataset == "self_field":
            field = self.film_solutions[film].self_field
        elif dataset == "applied_field":
            field = self.film_solutions[film].applied_field
        else:
            field = self.film_solutions[film].field_from_other_films
            if field is None:
                field = np.zeros(len(positions))
        positions = np.atleast_2d(positions)
        points = self.device.meshes[film].sites
        Hz_interp = interpolator(points, field, **kwargs)
        Hz = convert_field(
            Hz_interp(positions),
            units,
            old_units=self.field_units,
            ureg=device.ureg,
            with_units=with_units,
        )
        return Hz

    def polygon_flux(
        self,
        name: str,
        units: Optional[Union[str, pint.Unit]] = None,
        with_units: bool = True,
    ) -> Dict[str, Union[float, pint.Quantity]]:
        """Computes the flux through a given polygon.

        Args:
            name: The name of the polygon for which to compute the flux.
            units: The flux units to use.
            with_units: Whether to a return a pint.Quantity with units attached.

        Returns:
            The polygon flux.
        """
        from .solver import convert_field

        if name not in self.device.polygons:
            raise ValueError(f"Unknown polygon: {name!r}.")

        device = self.device
        ureg = device.ureg
        new_units = units or f"{self.field_units} * {device.length_units}**2"
        if isinstance(new_units, str):
            new_units = ureg(new_units)

        polygon = device.polygons[name]
        if name in device.films:
            mesh = device.meshes[name]
            polygon_name = name
        else:
            for film in device.films.values():
                if (
                    film.layer == polygon.layer
                    and film.contains_points(polygon.points).all()
                ):
                    break
            mesh = device.meshes[film.name]
            polygon_name = film.name
        points = mesh.sites
        areas = mesh.vertex_areas * ureg(f"{self.device.length_units}") ** 2
        total_field = self.film_solutions[polygon_name].total_field

        ix = polygon.contains_points(points, index=True)
        field = total_field[ix] * ureg(self.field_units)
        area = areas[ix]
        # Convert field to B = mu0 * H
        field = convert_field(field, "mT", ureg=ureg)
        flux = np.einsum("i, i -> ", field, area).to(new_units)
        if with_units:
            return flux
        return flux.magnitude

    def polygon_fluxoid(
        self,
        polygon_coords: Union[np.ndarray, Polygon],
        *,
        film: str,
        interp_method: str = "linear",
        units: Optional[str] = "Phi_0",
        with_units: bool = True,
    ) -> Dict[str, Fluxoid]:
        """Computes the :class:`Fluxoid` (flux + supercurrent) for
        a given polygonal region.

        The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
        is defined as:

        .. math::

            \\Phi^f_S = \\underbrace{
                \\int_S \\mu_0 H_z(\\vec{r})\\,\\mathrm{d}^2r
            }_{\\text{flux part}}
            + \\underbrace{
                \\oint_{\\partial S}
                \\mu_0\\Lambda(\\vec{r})\\vec{J}(\\vec{r})\\cdot\\mathrm{d}\\vec{r}
            }_{\\text{supercurrent part}}

        Args:
            polygon_coords: A shape ``(n, 2)`` array of ``(x, y)`` coordinates of
                polygon vertices defining the closed region :math:`S`.
            film: The name of the film in which to evaluate the field and current.
            interp_method: Interpolation method to use.
            units: The desired units for the current density.
                Defaults to :math:`\\Phi_0`.
            with_units: Whether to return values as pint.Quantities with units attached.

        Returns:
            The :class:`Fluxoid` for the given polygon.
        """
        device = self.device
        ureg = device.ureg
        if units is None:
            units = f"{self.field_units} * {self.device.length_units} ** 2"
        polygon = Polygon(points=polygon_coords)
        points = polygon.points

        if not device.films[film].contains_points(points).all():
            raise ValueError(
                f"The polygon is not contained within the film ({film!r})."
            )

        poly_mesh = device.meshes[film]
        ix = polygon.contains_points(poly_mesh.sites)
        fields = self.film_solutions[film].total_field * ureg(self.field_units)
        areas = poly_mesh.vertex_areas * ureg(f"{device.length_units} ** 2")
        flux_part = np.einsum("i, i ->", fields[ix], areas[ix]).to(units)

        # Evaluate the supercurrent density at the polygon coordinates.
        J_units = f"{self.current_units} / {device.length_units}"
        J_poly = self.interp_current_density(
            points,
            film=film,
            method=interp_method,
            units=J_units,
            with_units=False,
        )
        # Compute the supercurrent part of the fluxoid:
        # \oint_{\\partial poly} \mu_0\Lambda \vec{J}\cdot\mathrm{d}\vec{r}
        Lambda = device.layers[device.films[film].layer].Lambda
        if not callable(Lambda):
            Lambda = Constant(Lambda)
        Lambda_poly = Lambda(points[:, 0], points[:, 1])
        # \oint_{poly}\Lambda\vec{J}\cdot\mathrm{d}\vec{r}
        dl = np.diff(points, axis=0)
        int_J = np.trapz(Lambda_poly[:-1] * np.sum(J_poly[:-1] * dl, axis=1))
        int_J = int_J * ureg(J_units) * ureg(device.length_units) ** 2
        supercurrent_part = (ureg("mu_0") * int_J).to(units)
        if not with_units:
            flux_part = flux_part.magnitude
            supercurrent_part = supercurrent_part.magnitude
        return Fluxoid(flux_part, supercurrent_part)

    def hole_fluxoid(
        self,
        hole_name: str,
        points: Optional[np.ndarray] = None,
        interp_method: str = "linear",
        units: Optional[str] = "Phi_0",
        with_units: bool = True,
    ) -> Fluxoid:
        """Calculcates the fluxoid for a polygon enclosing the specified hole.

        Args:
            hole_name: The name of the hole for which to calculate the fluxoid.
            points: The vertices of the polygon enclosing the hole. If None is given,
                a polygon is generated using
                :func:`supercreen.fluxoid.make_fluxoid_polygons`.
            interp_method: Interpolation method to use.
            units: The desired units for the current density.
                Defaults to :math:`\\Phi_0`.
            with_units: Whether to return values as pint.Quantities with units attached.

        Returns:
            The hole's Fluxoid.
        """
        if points is None:
            from .fluxoid import make_fluxoid_polygons

            points = make_fluxoid_polygons(self.device, holes=hole_name)[hole_name]

        device = self.device
        hole = device.holes[hole_name]

        if not in_polygon(points, hole.points).all():
            raise ValueError(
                f"Hole {hole.name} is not completely enclosed by the given polygon."
            )
        for film_name, holes in self.device.holes_by_film().items():
            if hole.name in [h.name for h in holes]:
                break
        return self.polygon_fluxoid(
            points,
            film=film_name,
            interp_method=interp_method,
            units=units,
            with_units=with_units,
        )

    def screening_field_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Union[float, np.ndarray, None] = None,
        vector: bool = False,
        interp_method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the field due to currents in the device at any point(s) in space
        (excluding the applied field).

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the magnetic field.
                A single sequence like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            vector: Whether to return the full vector magnetic field
                or just the z component.
            interp_method: Interpolation method to use.
            units: Units to which to convert the fields (can be either magnetic field H
                or magnetic flux density B = mu0 * H). If not given, then the fields
                are returned in units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity``
                with units attached.
            return_sum: Whether to return the sum of the fields from all layers in
                the device, or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{film_name: field_from_film}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``field_from_film`` will have
            shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        from .solver import convert_field

        device = self.device
        dtype = device.solve_dtype
        ureg = device.ureg
        layers = device.layers
        meshes = device.meshes
        units = units or self.field_units
        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        else:
            zs = np.squeeze(zs)
            if zs.ndim == 0:
                zs = zs.item() * np.ones(positions.shape[0], dtype=dtype)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")

        fields = {}
        shape = (len(positions),)
        if vector:
            shape = shape + (3,)
            zeros = np.zeros(len(positions), dtype=dtype)
        # Compute the fields at the specified positions from the currents in each film
        for name, film in device.films.items():
            layer = layers[film.layer]
            field_from_film = np.zeros(len(positions), dtype=dtype)
            in_film = np.zeros(len(positions), dtype=bool)
            if np.all(zs == layer.z0):
                # Evaluate the screening field within a film.
                in_film[film.contains_points(positions)] = True
                field_in_film = self.interp_field(
                    positions[in_film],
                    film=film.name,
                    dataset="self_field",
                    method=interp_method,
                    units="tesla",
                    with_units=False,
                )
                if vector:
                    # Make shape (m, 3)
                    field_in_film = np.array([zeros, zeros, field_in_film]).T
                field_from_film[in_film] = field_in_film
            # Evaluate the screening field outside of any films.
            not_in_film = ~in_film
            field_from_film[not_in_film] = biot_savart_2d(
                positions[not_in_film, 0],
                positions[not_in_film, 1],
                zs[not_in_film],
                positions=meshes[name].sites,
                areas=meshes[name].vertex_areas,
                current_densities=self.film_solutions[name].current_density,
                z0=layer.z0,
                length_units=device.length_units,
                current_units=self.current_units,
                vector=vector,
            )
            fields[name] = convert_field(
                field_from_film,
                units,
                old_units="tesla",
                ureg=ureg,
                with_units=with_units,
            )
        if return_sum:
            return sum(fields.values())
        return fields

    def field_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Union[float, np.ndarray, None] = None,
        interp_method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the field due to currents in the device at any point(s) in space.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the magnetic field.
                A single sequence like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            interp_method: Interpolation method to use.
            units: Units to which to convert the fields (can be either magnetic field H
                or magnetic flux density B = mu0 * H). If not given, then the fields
                are returned in units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity``
                with units attached.
            return_sum: Whether to return the sum of the fields from all layers in
                the device, or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{film_name: field_from_film}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``field_from_film`` will have
            shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """

        device = self.device
        dtype = device.solve_dtype
        ureg = device.ureg
        units = units or self.field_units
        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        else:
            zs = np.squeeze(zs)
            if zs.ndim == 0:
                zs = zs.item() * np.ones(positions.shape[0], dtype=dtype)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        # Evaluate the screening fields
        fields = self.screening_field_at_position(
            positions,
            zs=zs,
            vector=False,
            interp_method=interp_method,
            units=units,
            with_units=with_units,
            return_sum=False,
        )
        # Evaluate the applied fields
        films_by_layer = device.polygons_by_layer("film")
        Hz_applied = np.zeros(len(positions), dtype=dtype)
        in_film = np.zeros(len(positions), dtype=bool)
        for name, layer in device.layers.items():
            if np.all(zs == layer.z0):
                for film in films_by_layer[name]:
                    ix = film.contains_points(positions)
                    in_film[ix] = True
                    Hz_applied[ix] = self.interp_field(
                        positions[ix],
                        film=film.name,
                        dataset="applied_field",
                        method=interp_method,
                        units=self.field_units,
                        with_units=False,
                    )
                    Hz_applied[ix] += self.interp_field(
                        positions[ix],
                        film=film.name,
                        dataset="field_from_other_films",
                        method=interp_method,
                        units=self.field_units,
                        with_units=False,
                    )
                break
        mask = ~in_film
        Hz_applied[mask] = self.applied_field_func(
            positions[mask, 0], positions[mask, 1], zs[mask, np.newaxis]
        )
        Hz_applied = (Hz_applied * ureg(self.field_units)).to(units)
        if not with_units:
            Hz_applied = Hz_applied.magnitude
        fields["applied_field"] = np.atleast_1d(Hz_applied).squeeze()
        if return_sum:
            return sum(fields.values())
        return fields

    def vector_potential_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Union[float, np.ndarray, None] = None,
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
        batch_size: int = 1000,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the vector potential due to currents in the device at any
        point(s) in space. Note that this only considers the vector potential
        due to currents in the device, so only represents the total vector potential
        in cases where the applied field is zero (e.g. models with only vortices and/or
        circulating currents).

        The vector potential :math:`\\vec{A}` at position :math:`\\vec{r}`
        due to sheet current density :math:`\\vec{J}(\\vec{r}')` flowing in a film
        with lateral geometry :math:`S` is:

        .. math::

            \\vec{A}(\\vec{r}) = \\frac{\\mu_0}{4\\pi}
            \\int_S\\frac{\\vec{J}(\\vec{r}')}{|\\vec{r}-\\vec{r}'|}\\mathrm{d}^2r'.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the vector potential.
                A single list like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the potential. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            units: Units to which to convert the vector potential.
            with_units: Whether to return the vector potential as a ``pint.Quantity``
                with units attached.
            return_sum: Whether to return the sum of the potential from all layers in
                the device, or a dict of ``{layer_name: potential_from_layer}``.
            batch_size: The maximum size of each batch of positions.
                See :func:`superscreen.fem.cdist_batched`.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{film_name: potential_from_film}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``potential_from_film`` will have
            shape ``(m, 3)``.
        """
        device = self.device
        layers = device.layers
        meshes = device.meshes
        dtype = device.solve_dtype
        ureg = device.ureg
        units = units or f"{self.field_units} * {device.length_units}"

        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        else:
            zs = np.squeeze(zs)
            if zs.ndim == 0:
                zs = zs.item() * np.ones(positions.shape[0], dtype=dtype)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]

        # Compute the vector potential at the specified positions
        # from the currents in each film
        vector_potentials = {}
        for name, film in device.films.items():
            dz = zs - layers[name].z0
            if np.all(dz == 0) and film.contains_points(positions).all():
                raise ValueError(
                    f"Cannot evaluate vector potential inside the film ({name!r})."
                )
            mesh = meshes[name]
            rho2 = cdist_batched(
                positions, mesh.sites, batch_size=batch_size, metric="sqeuclidean"
            ).astype(dtype, copy=False)
            areas = mesh.vertex_areas
            # J has units of [current / length], shape = (device.points.shape[0], 2)
            J = self.film_solutions[name].current_density
            # rho has units of [length] and
            # shape = (postitions.shape[0], device.points.shape[0], 1)
            rho = np.sqrt(rho2 + dz**2)[:, :, np.newaxis]
            Axy = np.einsum("ijk, j -> ik", J / rho, areas, dtype=dtype)
            # z-component is zero because currents are parallel to the x-y plane.
            A = np.concatenate([Axy, np.zeros_like(Axy[:, :1])], axis=1)
            A = A * ureg(self.current_units)
            A = (ureg("mu_0") / (4 * np.pi) * A).to(units)
            if not with_units:
                A = A.magnitude
            vector_potentials[name] = A
        if return_sum:
            return sum(vector_potentials.values())
        return vector_potentials

    def to_hdf5(
        self,
        path_or_group: Union[os.PathLike, h5py.Group],
        device_path: Optional[str] = None,
        compress: bool = True,
    ) -> None:
        """Save the Solution to an HDF5 file.

        path_or_group: An HDF5 file path or an open h5py.Group in which to save
            the Solution.
        device_path: Path within the HDF5 file in which the Solution's Device
            is saved. If None, the Device will be saved at ``"/device"``.
        compress: Save the mesh in a compressed format.
        """
        if isinstance(path_or_group, h5py.Group):
            save_context = nullcontext(path_or_group)
        else:
            save_context = h5py.File(path_or_group, "x")
        with save_context as h5group:
            h5group.attrs["time_created"] = self.time_created.isoformat()
            h5group.attrs["field_units"] = self.field_units
            h5group.attrs["current_units"] = self.current_units
            h5group.attrs["solver"] = self.solver
            version_grp = h5group.create_group("version_info")
            version_grp.attrs.update(self.version_info)
            if device_path is None:
                self.device.to_hdf5(
                    h5group.create_group("device"), save_mesh=True, compress=compress
                )
            else:
                h5group["device"] = h5py.SoftLink(device_path)
            grp = h5group.create_group("film_solutions")
            for name, film_solution in self.film_solutions.items():
                film_solution.to_hdf5(grp.create_group(name))
            vortices_grp = h5group.create_group("vortices")
            for i, vortex in enumerate(self.vortices):
                vortex.to_hdf5(vortices_grp.create_group(str(i)))
            serialize_obj(h5group, self.applied_field_func, "applied_field_func")
            circ_grp = h5group.create_group("circulating_currents")
            circ_grp.attrs.update(self.circulating_currents)
            term_grp = h5group.create_group("terminal_currents")
            term_grp.attrs.update(self.terminal_currents)

    @staticmethod
    def from_hdf5(
        path_or_group: Union[os.PathLike, h5py.Group],
    ) -> "Solution":
        """Load a Solution from and HDF5 file.

        Args:
            path_or_group: An HDF5 file path or an open h5py.Group from which to load
                the Solution.

        Returns:
            The loaded Solution
        """
        if isinstance(path_or_group, h5py.Group):
            read_context = nullcontext(path_or_group)
        else:
            read_context = h5py.File(path_or_group, "r")
        with read_context as h5group:
            device = Device.from_hdf5(h5group["device"])
            film_solutions = {}
            for name, grp in h5group["film_solutions"].items():
                film_solutions[name] = FilmSolution.from_hdf5(grp)
            applied_field_func = deserialize_obj(h5group, "applied_field_func")
            vortices = []
            for i in sorted(h5group["vortices"], key=int):
                vortices.append(Vortex.from_hdf5(h5group[f"vortices/{i}"]))
            time_created = datetime.fromisoformat(h5group.attrs["time_created"])
            version_info = dict(h5group["version_info"].attrs)

            solution = Solution(
                device=device,
                film_solutions=film_solutions,
                applied_field_func=applied_field_func,
                vortices=vortices,
                circulating_currents=dict(h5group["circulating_currents"].attrs),
                terminal_currents=dict(h5group["terminal_currents"].attrs),
                current_units=h5group.attrs["current_units"],
                field_units=h5group.attrs["field_units"],
                solver=h5group.attrs["solver"],
            )
            # Set "read-only" attributes
            solution._time_created = time_created
            solution._version_info = version_info

        return solution

    @staticmethod
    def save_solutions(
        solutions: Sequence["Solution"],
        path_or_group: Union[os.PathLike, h5py.Group],
        compress: bool = True,
    ) -> None:
        """Save a series of Solutions to an HDF5 file.

        Args:
            solutions: A series of Solutions to save.
            path_or_group: An HDF5 file path or an open h5py.Group in which to save
                the Solutions.
            compress: Save the meshes in a compressed format.
        """
        if isinstance(path_or_group, h5py.Group):
            save_context = nullcontext(path_or_group)
        else:
            save_context = h5py.File(path_or_group, "x")
        with save_context as h5group:
            for i, solution in enumerate(solutions):
                solution.to_hdf5(
                    h5group.create_group(str(i)),
                    device_path="/device" if i else None,
                    compress=compress,
                )

    @staticmethod
    def load_solutions(
        path_or_group: Union[os.PathLike, h5py.Group]
    ) -> List["Solution"]:
        """Load a series of Solutions from an HDF5 file.

        Args:
            path_or_group: An HDF5 file path or an open h5py.Group from which to load
                the Solutions.

        Returns:
            A list of loaded Solutions.
        """
        if isinstance(path_or_group, h5py.Group):
            read_context = nullcontext(path_or_group)
        else:
            read_context = h5py.File(path_or_group, "x")
        solutions = []
        with read_context as h5group:
            groups = sorted((key for key in h5group if key.isdigit()), key=int)
            for group in groups:
                solutions.append(Solution.from_hdf5(h5group[group]))
        return solutions

    def equals(
        self,
        other: Any,
        require_same_timestamp: bool = False,
    ) -> bool:
        """Checks whether two solutions are equal.

        Args:
            other: The Solution to compare for equality.
            require_same_timestamp: If True, two solutions are only considered
                equal if they have the exact same time_created.

        Returns:
            A boolean indicating whether the two solutions are equal
        """
        # First check things that are "easy" to check
        if other is self:
            return True
        if not isinstance(other, Solution):
            return False

        if not (
            (self.device == other.device)
            and (self.field_units == other.field_units)
            and (self.current_units == other.current_units)
            and (self.circulating_currents == other.circulating_currents)
            and (
                getattr(self, "terminal_currents", None)
                == getattr(other, "terminal_currents", None)
            )
            and (self.applied_field_func == other.applied_field_func)
            and (self.vortices == other.vortices)
        ):
            return False
        if require_same_timestamp and (self.time_created != other.time_created):
            return False
        # Then check the film_solutions, which will take longer
        return self.film_solutions == other.film_solutions

    def __eq__(self, other) -> bool:
        return self.equals(other, require_same_timestamp=True)

    def plot_streams(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`superscreen.visualization.plot_streams`."""
        from .visualization import plot_streams

        return plot_streams(self, **kwargs)

    def plot_currents(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`superscreen.visualization.plot_currents`."""
        from .visualization import plot_currents

        return plot_currents(self, **kwargs)

    def plot_fields(self, **kwargs) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`superscreen.visualization.plot_fields`."""
        from .visualization import plot_fields

        return plot_fields(self, **kwargs)

    def plot_field_at_positions(
        self, points: np.ndarray, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`superscreen.visualization.plot_field_at_positions`."""
        from .visualization import plot_field_at_positions

        return plot_field_at_positions(self, points, **kwargs)
