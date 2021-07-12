import os
import json
from datetime import datetime
from typing import Optional, Union, Callable, Dict, Tuple, List, Any

import dill
import pint
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from .device import Device
from .fem import areas, centroids


class BrandtSolution(object):
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
    ):
        self.device = device
        self.streams = streams
        self.fields = fields
        self.screening_fields = screening_fields
        self.applied_field = applied_field
        self.circulating_currents = circulating_currents or {}
        # Make field_units and current_units "read-only" attributes.
        # The should never be changed after instantiation.
        self._field_units = field_units
        self._current_units = current_units
        self._time_created = datetime.now()

    @property
    def field_units(self) -> str:
        """The units in which magnetic fields are specified."""
        return self._field_units

    @property
    def current_units(self) -> str:
        """The units in whic currents are specified."""
        return self._current_units

    @property
    def time_created(self) -> datetime:
        """The time at which the solution was originally created."""
        return self._time_created

    def grid_data(
        self,
        dataset: str,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        layers: Optional[Union[str, List[str]]] = None,
        method: Optional[str] = "cubic",
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Interpolates results from the triangular mesh to a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            dataset: Name of the dataset to interpolate
                (one of "streams", "fields", or "screening_fields").
            grid_shape: Shape of the desired rectangular grid. If a single integer N is given,
                then the grid will be square, shape = (N, N).
            layers: Name(s) of the layer(s) for which to interpolate results.
            method: Interpolation method to use (see scipy.interpolate.griddata).
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated data for each layer
        """
        valid_data = ("streams", "fields", "screening_fields")
        if dataset not in valid_data:
            raise ValueError(f"Expected one of {', '.join(valid_data)}, not {dataset}.")
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
                f"Expected a tuple of length 2, but got {grid_shape} ({type(grid_shape)})."
            )

        points = self.device.points
        x, y = points.T
        xgrid, ygrid = np.meshgrid(
            np.linspace(x.min(), x.max(), grid_shape[1]),
            np.linspace(y.min(), y.max(), grid_shape[0]),
        )
        zgrids = {}
        for name, array in datasets.items():
            if name in layers:
                zgrid = griddata(points, array, (xgrid, ygrid), method=method, **kwargs)
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

    def current_density(
        self,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        layers: Optional[Union[str, List[str]]] = None,
        method: Optional[str] = "cubic",
        units: Optional[str] = None,
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Computes the current density ``J = [dg/dy, -dg/dx]`` on a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            grid_shape: Shape of the desired rectangular grid. If a single integer N is given,
                then the grid will be square, shape = (N, N).
            layers: Name(s) of the layer(s) for which to interpolate current density.
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

    def polygon_flux(
        self,
        polygons: Optional[Union[str, List[str]]] = None,
        units: Optional[Union[str, pint.Unit]] = None,
        with_units: bool = True,
    ) -> Dict[str, Union[float, pint.Quantity]]:
        """Compute the flux through all polygons (films, holes, and flux regions)
        by integrating the calculated fields.

        Args:
            polygons: Name(s) of the polygon(s) for which to compute the flux.
                Default: All polygons.
            units: The flux units to use.
            with_units: Whether to a dict of pint.Quantities with units attached.

        Returns:
            dict of ``{polygon_name: polygon_flux}``
        """
        from .brandt import convert_field

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
        triangles = self.device.triangles

        tri_areas = areas(points, triangles) * ureg(f"{self.device.length_units}") ** 2
        xt, yt = centroids(points, triangles).T
        fluxes = {}
        for name in polygons:
            if name in films:
                poly = self.device.films[name]
            elif name in holes:
                poly = self.device.holes[name]
            else:
                poly = self.device.abstract_regions[name]
            ix = poly.contains_points(xt, yt, index=True)
            field = self.fields[poly.layer][triangles].mean(axis=1)
            field = field[ix] * ureg(self.field_units)
            area = tri_areas[ix]
            # Convert field to B = mu0 * H
            field = convert_field(field, "mT", ureg=ureg)
            flux = np.sum(field * area).to(new_units)
            if with_units:
                fluxes[name] = flux
            else:
                fluxes[name] = flux.magnitude
        return fluxes

    def field_at_position(
        self,
        positions: np.ndarray,
        zs: Optional[Union[float, np.ndarray]] = None,
        vector: bool = False,
        units: Optional[str] = None,
        with_units: Optional[bool] = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculate the field due to currents in the device at any point(s) in space.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array of (x, y, z)
                coordinates at which to calculate the magnetic field. A single list like [x, y]
                or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape (m, 3), then
                this argument is not allowed. If zs is a scalar, then the fields are calculated in
                a plane parallel to the x-y plane. If zs is any array, then it must be same length
                as positions.
            vector: Whether to return the full vector magnetic field or just the z component.
            units: Units to which to convert the fields (can be either magnetic field H or
                magnetic flux density B = mu0 * H). If not give, then the fields are returned in
                units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity`` with units attached.
            return_sum: Whether to return the sum of the fields from all layers in the device,
                or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of ``{layer_name: field_from_layer}``.
            If with_units is True, then the array(s) will contain pint.Quantities. ``field_from_layer``
            will have shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        from .brandt import convert_field

        device = self.device
        ureg = device.ureg
        points = device.points
        triangles = device.triangles

        units = units or self.field_units
        old_units = f"{self.current_units} / {device.length_units}"

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
            zs = zs * np.ones(positions.shape[0], dtype=float)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]

        tri_points = centroids(points, triangles)
        tri_areas = areas(points, triangles)
        # Compute ||(x, y) - (xt, yt)||^2
        rho2 = cdist(positions, tri_points, metric="sqeuclidean")

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
            old_units=old_units,
            ureg=ureg,
            magnitude=(not with_units),
        )

        fields = {}
        # Compute the fields at the specified positions from the currents in each layer
        for name, layer in device.layers.items():
            dz = zs - layer.z0
            if np.any(dz == 0):
                raise ValueError(
                    f"Cannot calculate fields in the same plane as layer {name}."
                )
            # g has units of [current]
            g = self.streams[name][triangles].mean(axis=1)
            # Q is the dipole kernel for the z component, Hz
            # Q has units of [length]^(2*(1-5/2)) = [length]^(-3)
            Q = (2 * dz ** 2 - rho2) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
            # tri_areas has units of [length]^2
            # So here Hz is in units of [current] * [length]^(-1)
            Hz = np.asarray(np.sum(tri_areas * Q * g, axis=1))
            if vector:
                # See Eq. 15 of Kirtley RSI 2016, arXiv:1605.09483
                # Pairwise difference between all x positions
                d = np.subtract.outer(positions[:, 0], tri_points[:, 0])
                # Kernel for x component, Hx
                Q = (3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                Hx = np.asarray(np.sum(tri_areas * Q * g, axis=1))

                # Pairwise difference between all y positions
                d = np.subtract.outer(positions[:, 1], tri_points[:, 1])
                # Kernel for y component, Hy
                Q = (3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                Hy = np.asarray(np.sum(tri_areas * Q * g, axis=1))

                H = np.stack([Hx, Hy, Hz], axis=1)
            else:
                H = Hz
            fields[name] = convert_field(
                H,
                units,
                old_units=old_units,
                ureg=ureg,
                magnitude=(not with_units),
            )
        if return_sum:
            fields = sum(fields.values()) + H_applied
        else:
            fields["applied_field"] = H_applied
        return fields

    def to_file(self, directory: str, save_mesh: bool = True) -> None:
        """Saves a BrandtSolution to disk.

        Args:
            directory: The name of the directory in which to save the solution
                (must either be empty or not yet exist).
            save_mesh: Whether to save the device mesh.
        """
        if os.path.isdir(directory) and len(os.listdir(directory)):
            raise IOError(f"Directory '{directory}' already exists and is not empty.")
        os.makedirs(directory, exist_ok=True)

        # Save device
        device_path = "device"
        self.device.to_file(os.path.join(directory, device_path), save_mesh=save_mesh)

        # Save arrays
        array_paths = []
        for layer in self.device.layers:
            path = f"{layer}_arrays.npz"
            np.savez(
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
            "field_units": self.field_units,
            "current_units": self.current_units,
            "time_created": self.time_created.isoformat(),
        }

        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def from_file(
        cls, directory: str, compute_matrices: bool = False
    ) -> "BrandtSolution":
        """Loads a BrandtSolution from file.

        Args:
            directory: The directory from which to load the solution.
            compute_matrices: Whether to compute the field-independent
                matrices for the device if the mesh already exists.

        Returns:
            The loaded BrandtSolution instance
        """
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
            layer = path.split("_")[0]
            with np.load(os.path.join(directory, path)) as arrays:
                streams[layer] = arrays["streams"]
                fields[layer] = arrays["fields"]
                screening_fields[layer] = arrays["screening_fields"]

        # Load applied field function
        with open(os.path.join(directory, info.pop("applied_field")), "rb") as f:
            applied_field = dill.load(f)

        # Requires Python >= 3.7
        time_created = datetime.fromisoformat(info.pop("time_created"))

        solution = cls(
            device=device,
            streams=streams,
            fields=fields,
            screening_fields=screening_fields,
            applied_field=applied_field,
            **info,
        )
        # Set "read-only" attribute
        solution._time_created = time_created

        return solution

    def equals(self, other: Any, require_same_timestamp: bool = False) -> bool:
        """Check whether two solutions are equal.

        Args:
            other: The BrandtSolution to compare for equality.
            require_same_timestamp: If True, two solutions are only considered
                equal if they have the exact same time_created.

        Returns:
            bool indicating whether the two solutions are equal
        """

        if other is self:
            return True

        if not isinstance(other, BrandtSolution):
            return False

        # First check things that are "easy" to check
        if not (
            self.device == other.device
            and self.field_units == other.field_units
            and self.current_units == other.current_units
            and self.circulating_currents == other.circulating_currents
            and self.applied_field == other.applied_field
        ):
            return False

        if require_same_timestamp and (self.time_created != other.time_created):
            return False

        # Then check the arrays, which may take longer
        for name, array in self.streams.items():
            if not np.allclose(array, other.streams[name]):
                return False
        for name, array in self.fields.items():
            if not np.allclose(array, other.fields[name]):
                return False
        for name, array in self.screening_fields.items():
            if not np.allclose(array, other.screening_fields[name]):
                return False

        return True

    def __eq__(self, other) -> bool:
        return self.equals(other, require_same_timestamp=True)

    def plot_streams(
        self,
        layers: Optional[Union[List[str], str]] = None,
        units: Optional[str] = None,
        max_cols: int = 3,
        cmap: str = "magma",
        levels: int = 101,
        colorbar: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for superscreen.visualization.plot_streams."""
        from .visualization import plot_streams

        return plot_streams(self, **kwargs)

    def plot_currents(
        self,
        layers: Optional[Union[List[str], str]] = None,
        units: Optional[str] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        grid_method: str = "cubic",
        max_cols: int = 3,
        cmap: str = "inferno",
        colorbar: bool = True,
        auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
        share_color_scale: bool = False,
        symmetric_color_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        streamplot: bool = True,
        min_stream_amp: float = 0.025,
        cross_section_xs: Optional[Union[float, List[float]]] = None,
        cross_section_ys: Optional[Union[float, List[float]]] = None,
        cross_section_angle: Optional[float] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for superscreen.visualization.plot_currents."""
        from .visualization import plot_currents

        return plot_currents(self, **kwargs)

    def plot_fields(
        self,
        layers: Optional[Union[List[str], str]] = None,
        dataset: str = "fields",
        normalize: bool = False,
        units: Optional[str] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        grid_method: str = "cubic",
        max_cols: int = 3,
        cmap: str = "cividis",
        colorbar: bool = True,
        auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
        share_color_scale: bool = False,
        symmetric_color_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cross_section_xs: Optional[Union[float, List[float]]] = None,
        cross_section_ys: Optional[Union[float, List[float]]] = None,
        cross_section_angle: Optional[float] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for superscreen.visualization.plot_fields."""
        from .visualization import plot_fields

        return plot_fields(self, **kwargs)

    def plot_field_at_positions(
        self,
        positions: np.ndarray,
        zs: Optional[Union[float, np.ndarray]] = None,
        vector: bool = False,
        units: Optional[str] = None,
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        grid_method: str = "cubic",
        cmap: str = "cividis",
        colorbar: bool = True,
        auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
        share_color_scale: bool = False,
        symmetric_color_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cross_section_xs: Optional[Union[float, List[float]]] = None,
        cross_section_ys: Optional[Union[float, List[float]]] = None,
        cross_section_angle: Optional[float] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Alias for superscreen.visualization.plot_field_at_positions."""
        from .visualization import plot_field_at_positions

        return plot_field_at_positions(self, **kwargs)
