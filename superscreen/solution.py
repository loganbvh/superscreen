import os
import json
import logging
import zipfile
from datetime import datetime
from typing import Optional, Union, Callable, Dict, Tuple, List, Any, NamedTuple

import dill
import pint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy import interpolate
from scipy.spatial import distance

from .about import version_dict
from .device import Device, Polygon
from .fem import in_polygon
from .parameter import Constant

from backports.datetime_fromisoformat import MonkeyPatch

MonkeyPatch.patch_fromisoformat()


logger = logging.getLogger(__name__)


class Vortex(NamedTuple):
    """A vortex located at position ``(x, y)`` in ``layer`` containing
    a total flux ``nPhi0`` in units of the flux quantum :math:`\\Phi_0`.
    """

    x: float
    y: float
    layer: str
    nPhi0: float = 1


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
    """

    flux_part: Union[float, pint.Quantity]
    supercurrent_part: Union[float, pint.Quantity]


class Solution(object):
    """A container for the calculated stream functions and fields,
    with some convenient data processing methods.

    Args:
        device: The ``Device`` that was solved
        streams: A dict of ``{layer_name: stream_function}``
        fields: A dict of ``{layer_name: total_field}``
        screening_fields: A dict of ``{layer_name: screening_field}``
        applied_field: The function defining the applied field
        field_units: Units of the applied field
        current_units: Units used for current quantities.
        circulating_currents: A dict of ``{hole_name: circulating_current}``
        vortices: A list of ``Vortex`` objects located in the ``Device``.
        solver: The solver method that generated the solution.
    """

    def __init__(
        self,
        *,
        device: Device,
        streams: Dict[str, np.ndarray],
        fields: Dict[str, np.ndarray],
        screening_fields: Dict[str, np.ndarray],
        applied_field: Callable,
        field_units: str,
        current_units: str,
        circulating_currents: Optional[
            Dict[str, Union[float, str, pint.Quantity]]
        ] = None,
        vortices: Optional[List[Vortex]] = None,
        solver: str = "superscreen.solve",
    ):
        self.device = device.copy(with_arrays=True, copy_arrays=False)
        self.streams = streams
        self.fields = fields
        self.applied_field = applied_field
        self.screening_fields = screening_fields
        self.circulating_currents = circulating_currents or {}
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

    def grid_data(
        self,
        dataset: str,
        *,
        layers: Optional[Union[str, List[str]]] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        method: str = "linear",
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Interpolates results from the triangular mesh to a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            dataset: Name of the dataset to interpolate
                (one of "streams", "fields", or "screening_fields", "current_density").
            layers: Name(s) of the layer(s) for which to interpolate results.
            grid_shape: Shape of the desired rectangular grid. If a single integer
                N is given, then the grid will be square, shape = (N, N).
            method: Interpolation method to use (see scipy.interpolate.griddata).
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated data for each layer
        """
        valid_data = ("streams", "fields", "screening_fields", "current_density")
        if dataset not in valid_data:
            raise ValueError(f"Expected one of {', '.join(valid_data)}, not {dataset}.")
        if dataset == "current_density":
            return self.grid_current_density(
                layers=layers,
                grid_shape=grid_shape,
                method=method,
                with_units=with_units,
                **kwargs,
            )
        datasets = getattr(self, dataset)

        if isinstance(layers, str):
            layers = [layers]
        if layers is None:
            layers = list(self.device.layers)
        else:
            for layer in layers:
                if layer not in self.device.layers:
                    raise ValueError(f"Unknown layer, {layer}.")

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise TypeError(
                f"Expected a tuple of length 2, but got {grid_shape} "
                f"({type(grid_shape)})."
            )

        points = self.device.points
        x = points[:, 0]
        y = points[:, 1]
        xgrid, ygrid = np.meshgrid(
            np.linspace(x.min(), x.max(), grid_shape[1]),
            np.linspace(y.min(), y.max(), grid_shape[0]),
        )
        zgrids = {}
        for name, array in datasets.items():
            if name in layers:
                zgrid = interpolate.griddata(
                    points, array, (xgrid, ygrid), method=method, **kwargs
                )
                zgrids[name] = zgrid
        if with_units:
            xgrid = xgrid * self.device.ureg(self.device.length_units)
            ygrid = ygrid * self.device.ureg(self.device.length_units)
            if dataset in ("fields", "screening_fields"):
                units = self.field_units
            else:
                units = self.current_units
            zgrids = {
                layer: data * self.device.ureg(units) for layer, data in zgrids.items()
            }
        return xgrid, ygrid, zgrids

    def grid_current_density(
        self,
        *,
        layers: Optional[Union[str, List[str]]] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Computes the current density ``J = [dg/dy, -dg/dx]`` on a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            layers: Name(s) of the layer(s) for which to interpolate current density.
            grid_shape: Shape of the desired rectangular grid. If a single integer
                N is given, then the grid will be square, shape = (N, N).
            method: Interpolation method to use (see scipy.interpolate.griddata).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated current density for each layer
        """
        xgrid, ygrid, streams = self.grid_data(
            dataset="streams",
            layers=layers,
            grid_shape=grid_shape,
            method=method,
            with_units=True,
            **kwargs,
        )
        units = units or f"{self.current_units} / {self.device.length_units}"
        Js = {}
        for name, g in streams.items():
            # J = [dg/dy, -dg/dx]
            # y is axis 0 (rows), x is axis 1 (columns)
            dg_dy, dg_dx = np.gradient(
                g.magnitude, ygrid[:, 0].magnitude, xgrid[0, :].magnitude
            )
            J = (np.array([dg_dy, -dg_dx]) * g.units / xgrid.units).to(units)
            if not with_units:
                J = J.magnitude
            Js[name] = J
        if not with_units:
            xgrid = xgrid.magnitude
            ygrid = ygrid.magnitude
        return xgrid, ygrid, Js

    def interp_current_density(
        self,
        positions: np.ndarray,
        *,
        layers: Optional[Union[str, List[str]]] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        method: str = "linear",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ):
        """Computes the current density ``J = [dg/dy, -dg/dx]``
        at unstructured coordinates via interpolation.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            positions: Shape ``(m, 2)`` array of x, y coordinates at which to evaluate
                the current density.
            layers: Name(s) of the layer(s) for which to interpolate current density.
            grid_shape: Shape of the desired rectangular grid. If a single integer
                N is given, then the grid will be square, shape = (N, N).
            method: Interpolation method to use (see scipy.interpolate.griddata).
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            A dict of interpolated current density for each layer.
        """
        valid_methods = ("nearest", "linear", "cubic")
        if method not in valid_methods:
            raise ValueError(
                f"Interpolation method must be one of {valid_methods} (got {method})."
            )
        if method == "nearest":
            interpolator = interpolate.NearestNDInterpolator
        elif method == "linear":
            interpolator = interpolate.LinearNDInterpolator
        else:  # "cubic"
            interpolator = interpolate.CloughTocher2DInterpolator
        if units is None:
            units = f"{self.current_units} / {self.device.length_units}"
        positions = np.atleast_2d(positions)
        xgrid, ygrid, Jgrids = self.grid_current_density(
            layers=layers,
            grid_shape=grid_shape,
            method=method,
            units=units,
            with_units=False,
            **kwargs,
        )
        xy = np.stack([xgrid.ravel(), ygrid.ravel()], axis=1)
        interpolated_Js = {}
        for layer, (Jx, Jy) in Jgrids.items():
            Jx_interp = interpolator(xy, Jx.ravel())
            Jy_interp = interpolator(xy, Jy.ravel())
            J = np.stack([Jx_interp(positions), Jy_interp(positions)], axis=1)
            if with_units:
                J = J * self.device.ureg(units)
            interpolated_Js[layer] = J
        return interpolated_Js

    def polygon_flux(
        self,
        *,
        polygons: Optional[Union[str, List[str]]] = None,
        units: Optional[Union[str, pint.Unit]] = None,
        with_units: bool = True,
    ) -> Dict[str, Union[float, pint.Quantity]]:
        """Computes the flux through all polygons (films, holes, and flux regions)
        by integrating the calculated fields.

        Args:
            polygons: Name(s) of the polygon(s) for which to compute the flux.
                Default: All polygons.
            units: The flux units to use.
            with_units: Whether to a dict of pint.Quantities with units attached.

        Returns:
            A dict of ``{polygon_name: polygon_flux}``
        """
        from .solve import convert_field

        films = list(self.device.films)
        holes = list(self.device.holes)
        abstract_regions = list(self.device.abstract_regions)
        all_polygons = films + holes + abstract_regions

        if isinstance(polygons, str):
            polygons = [polygons]
        if polygons is None:
            polygons = all_polygons
        else:
            for poly in polygons:
                if poly not in all_polygons:
                    raise ValueError(f"Unknown polygon, {poly}.")

        ureg = self.device.ureg
        new_units = units or f"{self.field_units} * {self.device.length_units}**2"
        if isinstance(new_units, str):
            new_units = ureg(new_units)

        points = self.device.points
        areas = self.device.weights * ureg(f"{self.device.length_units}") ** 2
        fluxes = {}
        for name in polygons:
            if name in films:
                poly = self.device.films[name]
            elif name in holes:
                poly = self.device.holes[name]
            else:
                poly = self.device.abstract_regions[name]
            ix = poly.contains_points(points, index=True)
            field = self.fields[poly.layer][ix] * ureg(self.field_units)
            area = areas[ix]
            # Convert field to B = mu0 * H
            field = convert_field(field, "mT", ureg=ureg)
            flux = np.einsum("i,i -> ", field, area).to(new_units)
            if with_units:
                fluxes[name] = flux
            else:
                fluxes[name] = flux.magnitude
        return fluxes

    def polygon_fluxoid(
        self,
        polygon_points: np.ndarray,
        layers: Optional[Union[str, List[str]]] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
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
            polygon_points: A shape ``(n, 2)`` array of ``(x, y)`` coordinates of
                polygon vertices defining the closed region :math:`S`.
            layers: Name(s) of the layer(s) for which to compute the fluxoid.
            grid_shape: Shape of the desired rectangular grid to use for interpolation.
                If a single integer N is given, then the grid will be square,
                shape = (N, N).
            interp_method: Interpolation method to use.
            units: The desired units for the current density.
                Defaults to :math:`\\Phi_0`.
            with_units: Whether to return values as pint.Quantities with units attached.

        Returns:
            A dict of ``{layer_name: fluxoid}`` for each specified layer, where
            ``fluxoid`` is an instance of :class:`Fluxoid`.
        """
        device = self.device
        ureg = device.ureg
        if layers is None:
            layers = list(device.layers)
        if isinstance(layers, str):
            layers = [layers]
        if units is None:
            units = f"{self.field_units} * {self.device.length_units} ** 2"
        polygon = Polygon(
            name="__polygon",
            layer=layers[0],
            points=polygon_points,
        )
        points = polygon.points
        if not any(
            film.contains_points(points).all() for film in device.films.values()
        ):
            raise ValueError(
                "The polygon must lie completely within a superconducting film."
            )

        # Evaluate the supercurrent density at the polygon coordinates.
        J_units = f"{self.current_units} / {device.length_units}"
        J_polys = self.interp_current_density(
            points,
            grid_shape=grid_shape,
            layers=layers,
            method=interp_method,
            units=J_units,
            with_units=False,
        )

        old_regions = device.abstract_regions
        temp_regions = old_regions.copy()
        temp_regions[polygon.name] = polygon
        fluxoids = {}
        for layer in layers:
            # Compute the flux part of the fluxoid:
            # \int_{poly} \mu_0 H_z(x, y) dx dy
            try:
                polygon.layer = layer
                device.abstract_regions = temp_regions
                flux_part = self.polygon_flux(
                    polygons=polygon.name,
                    units=units,
                    with_units=True,
                )[polygon.name]
            finally:
                device.abstract_regions = old_regions

            # Compute the supercurrent part of the fluxoid:
            # \oint_{\\partial poly} \mu_0\Lambda \vec{J}\cdot\mathrm{d}\vec{r}
            J_poly = J_polys[layer][:-1]
            Lambda = device.layers[layer].Lambda
            if not callable(Lambda):
                Lambda = Constant(Lambda)
            Lambda_poly = Lambda(points[:, 0], points[:, 1])[1:]
            # \oint_{poly}\Lambda\vec{J}\cdot\mathrm{d}\vec{r}
            int_J = np.trapz(
                Lambda_poly * np.sum(J_poly * np.diff(points, axis=0), axis=1)
            )
            int_J = int_J * ureg(J_units) * ureg(device.length_units) ** 2
            supercurrent_part = (ureg("mu_0") * int_J).to(units)
            if not with_units:
                flux_part = flux_part.magnitude
                supercurrent_part = supercurrent_part.magnitude
            fluxoids[layer] = Fluxoid(flux_part, supercurrent_part)
        return fluxoids

    def hole_fluxoid(
        self,
        hole_name: str,
        points: Optional[np.ndarray] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
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
            grid_shape: Shape of the desired rectangular grid to use for interpolation.
                If a single integer N is given, then the grid will be square,
                shape = (N, N).
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
        hole = self.device.holes[hole_name]
        if not in_polygon(points, hole.points).all():
            raise ValueError(
                f"Hole {hole_name} is not completely enclosed by the given polygon."
            )
        fluxoids = self.polygon_fluxoid(
            points,
            hole.layer,
            grid_shape=grid_shape,
            interp_method=interp_method,
            units=units,
            with_units=with_units,
        )
        return fluxoids[hole.layer]

    def field_at_position(
        self,
        positions: np.ndarray,
        *,
        zs: Optional[Union[float, np.ndarray]] = None,
        vector: bool = False,
        units: Optional[str] = None,
        with_units: bool = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculates the field due to currents in the device at any point(s) in space.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array
                of (x, y, z) coordinates at which to calculate the magnetic field.
                A single list like [x, y] or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape
                (m, 3), then this argument is not allowed. If zs is a scalar, then
                the fields are calculated in a plane parallel to the x-y plane.
                If zs is any array, then it must be same length as positions.
            vector: Whether to return the full vector magnetic field
                or just the z component.
            units: Units to which to convert the fields (can be either magnetic field H
                or magnetic flux density B = mu0 * H). If not given, then the fields
                are returned in units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity``
                with units attached.
            return_sum: Whether to return the sum of the fields from all layers in
                the device, or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of
            ``{layer_name: field_from_layer}``. If with_units is True, then the
            array(s) will contain pint.Quantities. ``field_from_layer`` will have
            shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        from .solve import convert_field

        device = self.device
        dtype = device.solve_dtype
        ureg = device.ureg
        points = device.points.astype(dtype, copy=False)
        triangles = device.triangles
        areas = device.weights

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
        elif isinstance(zs, (int, float, np.generic)):
            # constant zs
            zs = zs * np.ones(positions.shape[0], dtype=dtype)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]

        rho2 = distance.cdist(positions, points, metric="sqeuclidean").astype(
            dtype, copy=False
        )
        Hz_applied = self.applied_field(positions[:, 0], positions[:, 1], zs)
        if vector:
            H_applied = np.stack(
                [np.zeros_like(Hz_applied), np.zeros_like(Hz_applied), Hz_applied],
                axis=1,
            )
        else:
            H_applied = Hz_applied
        H_applied = convert_field(
            H_applied,
            units,
            old_units=self.field_units,
            ureg=ureg,
            with_units=with_units,
        )

        fields = {}
        # Compute the fields at the specified positions from the currents in each layer
        for name, layer in device.layers.items():
            dz = zs - layer.z0
            if np.all(dz == 0):
                # Interpolate field in the plane of an existing layer
                tri = Triangulation(points[:, 0], points[:, 1], triangles=triangles)
                Hz_interp = LinearTriInterpolator(tri, self.screening_fields[name])
                Hz = np.asarray(Hz_interp(positions[:, 0], positions[:, 1]))
                # Convert to {self.current_units} / {device.length_units}
                Hz = convert_field(
                    Hz,
                    f"{self.current_units} / {device.length_units}",
                    old_units=self.field_units,
                    ureg=ureg,
                    with_units=False,
                )
            else:
                if np.any(dz == 0):
                    raise ValueError(
                        f"Cannot calculate fields in the same plane as layer {name}."
                    )
                # g has units of [current]
                g = self.streams[name]
                # Q is the dipole kernel for the z component, Hz
                # Q has units of [length]^(2*(1-5/2)) = [length]^(-3)
                Q = (
                    (2 * dz ** 2 - rho2) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                ).astype(dtype, copy=False)
                # tri_areas has units of [length]^2
                # So here Hz is in units of [current] * [length]^(-1)
                Hz = np.einsum("ij,j -> i", Q, areas * g, dtype=dtype)
            if vector:
                if np.any(dz == 0):
                    raise ValueError(
                        f"Cannot calculate fields in the same plane as layer {name}."
                    )
                # See Eq. 15 of Kirtley RSI 2016, arXiv:1605.09483
                # Pairwise difference between all x positions
                d = np.subtract.outer(positions[:, 0], points[:, 0], dtype=dtype)
                # Kernel for x component, Hx
                Q = ((3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))).astype(
                    dtype, copy=False
                )
                Hx = np.einsum("ij,j -> i", Q, areas * g)

                # Pairwise difference between all y positions
                d = np.subtract.outer(positions[:, 1], points[:, 1], dtype=dtype)
                # Kernel for y component, Hy
                Q = ((3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))).astype(
                    dtype, copy=False
                )
                Hy = np.einsum("ij,j -> i", Q, areas * g, dtype=dtype)

                H = np.stack([Hx, Hy, Hz], axis=1)
            else:
                H = Hz
            fields[name] = convert_field(
                H,
                units,
                old_units=f"{self.current_units} / {device.length_units}",
                ureg=ureg,
                with_units=with_units,
            )
        if return_sum:
            fields = sum(fields.values()) + H_applied.squeeze()
        else:
            fields["applied_field"] = H_applied.squeeze()
        return fields

    def to_file(
        self,
        directory: str,
        save_mesh: bool = True,
        compressed: bool = True,
        to_zip: bool = False,
    ) -> None:
        """Saves a Solution to disk.

        Args:
            directory: The name of the directory in which to save the solution
                (must either be empty or not yet exist).
            save_mesh: Whether to save the device mesh.
            compressed: Whether to use numpy.savez_compressed rather than numpy.savez.
            to_zip: Whether to save the Solution to a zip file.
        """
        if to_zip:
            from .io import zip_solution

            zip_solution(self, directory)
            return

        if os.path.isdir(directory) and len(os.listdir(directory)):
            raise IOError(f"Directory '{directory}' already exists and is not empty.")
        os.makedirs(directory, exist_ok=True)

        # Save device
        device_path = "device"
        self.device.to_file(os.path.join(directory, device_path), save_mesh=save_mesh)

        # Save arrays
        array_paths = []
        save_npz = np.savez_compressed if compressed else np.savez
        for layer in self.device.layers:
            path = f"{layer}_arrays.npz"
            save_npz(
                os.path.join(directory, path),
                streams=self.streams[layer],
                fields=self.fields[layer],
                screening_fields=self.screening_fields[layer],
            )
            array_paths.append(path)

        # Save applied field function
        applied_field_path = "applied_field.dill"
        with open(os.path.join(directory, applied_field_path), "wb") as f:
            dill.dump(self.applied_field, f)

        # Handle circulating current formatting
        circ_currents = {}
        for name, val in self.circulating_currents.items():
            if isinstance(val, pint.Quantity):
                val = str(val)
            circ_currents[name] = val

        metadata = {
            "device": device_path,
            "arrays": array_paths,
            "applied_field": applied_field_path,
            "circulating_currents": circ_currents,
            "vortices": self.vortices,
            "field_units": self.field_units,
            "current_units": self.current_units,
            "solver": self.solver,
            "time_created": self.time_created.isoformat(),
            "version_info": self.version_info,
        }

        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def from_file(cls, directory: str, compute_matrices: bool = False) -> "Solution":
        """Loads a Solution from file.

        Args:
            directory: The directory from which to load the solution.
            compute_matrices: Whether to compute the field-independent
                matrices for the device if the mesh already exists.

        Returns:
            The loaded Solution instance
        """
        if directory.endswith(".zip") or zipfile.is_zipfile(directory):
            from .io import unzip_solution

            solution = unzip_solution(directory)
            if compute_matrices:
                solution.device.compute_matrices()
            return solution

        with open(os.path.join(directory, "metadata.json"), "r") as f:
            info = json.load(f)

        # Load device
        device_path = os.path.join(directory, info.pop("device"))
        device = Device.from_file(device_path, compute_matrices=compute_matrices)

        # Load arrays
        streams = {}
        fields = {}
        screening_fields = {}
        array_paths = info.pop("arrays")
        for path in array_paths:
            layer = path.replace("_arrays.npz", "")
            with np.load(os.path.join(directory, path)) as arrays:
                streams[layer] = arrays["streams"]
                fields[layer] = arrays["fields"]
                screening_fields[layer] = arrays["screening_fields"]

        # Load applied field function
        with open(os.path.join(directory, info.pop("applied_field")), "rb") as f:
            applied_field = dill.load(f)

        time_created = datetime.fromisoformat(info.pop("time_created"))
        version_info = info.pop("version_info", None)
        vortices = info.pop("vortices", [])
        vortices = [Vortex(*v) for v in vortices]

        solution = cls(
            device=device,
            streams=streams,
            fields=fields,
            screening_fields=screening_fields,
            applied_field=applied_field,
            vortices=vortices,
            **info,
        )
        # Set "read-only" attributes
        solution._time_created = time_created
        solution._version_info = version_info

        return solution

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
            self.device == other.device
            and self.field_units == other.field_units
            and self.current_units == other.current_units
            and self.circulating_currents == other.circulating_currents
            and self.applied_field == other.applied_field
            and self.vortices == other.vortices
        ):
            return False
        if require_same_timestamp and (self.time_created != other.time_created):
            return False

        # Then check the arrays, which will take longer
        for name, array in self.streams.items():
            if not np.allclose(array, other.streams[name]):
                return False
        for name, array in self.fields.items():
            if not np.allclose(array, other.fields[name]):
                return False

        return True

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
        self, positions: np.ndarray, **kwargs
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for :func:`superscreen.visualization.plot_field_at_positions`."""
        from .visualization import plot_field_at_positions

        return plot_field_at_positions(self, positions, **kwargs)
