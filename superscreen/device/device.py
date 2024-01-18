import logging
import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import dill
import h5py
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely import geometry as geo
from tqdm import tqdm

from .. import fem
from ..geometry import ensure_unique
from ..units import ureg
from . import utils
from .layer import Layer
from .mesh import Mesh
from .polygon import Polygon

logger = logging.getLogger("device")


class Device:
    """An object representing a device composed of one or more layers of
    thin film superconductor.

    Args:
        name: Name of the device.
        layers: ``Layers`` making up the device.
        films: ``Polygons`` representing regions of superconductor.
        holes: ``Holes`` representing holes in superconducting films.
        terminals: A dict of ``{film_name: terminals}`` representing transport
            terminals in the device.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
        length_units: Distance units for the coordinate system.
        solve_dtype: The float data type to use when solving the device.
    """

    ureg = ureg

    def __init__(
        self,
        name: str,
        *,
        layers: Union[Sequence[Layer], Dict[str, Layer]],
        films: Union[Sequence[Polygon], Dict[str, Polygon]],
        holes: Optional[Union[Sequence[Polygon], Dict[str, Polygon]]] = None,
        terminals: Optional[Dict[str, List[Polygon]]] = None,
        abstract_regions: Optional[Union[Sequence[Polygon], Dict[str, Polygon]]] = None,
        length_units: str = "um",
        solve_dtype: Union[str, np.dtype] = "float32",
    ):
        self.name = name

        if isinstance(layers, dict):
            layers = list(layers.values())
        self.layers = {layer.name: layer for layer in layers}

        if isinstance(films, dict):
            films = list(films.values())
        self.films = {film.name: film for film in films}

        if holes is None:
            holes = []
        if isinstance(holes, dict):
            holes = list(holes.values())
        self.holes = {hole.name: hole for hole in holes}

        if terminals is None:
            terminals = {}
        self.terminals = terminals
        if not set(self.terminals).issubset(self.films):
            raise ValueError(
                f"terminals.keys() must be a subset of films.keys() ({list(self.films)!r})."
            )
        for film, terminals in self.terminals.items():
            for terminal in terminals:
                terminal.layer = self.films[film].layer

        if abstract_regions is None:
            abstract_regions = []
        if isinstance(abstract_regions, dict):
            abstract_regions = list(abstract_regions.values())
        self.abstract_regions = {region.name: region for region in abstract_regions}

        for polygons, label in [
            (self.films.values(), "film"),
            (self.holes.values(), "hole"),
        ]:
            for polygon in polygons:
                if not polygon.is_valid:
                    raise ValueError(f"The following {label} is not valid: {polygon}.")
                if polygon.layer not in self.layers:
                    raise ValueError(
                        f"The following {label} is assigned to a layer that doesn not "
                        f"exist in the device: {polygon}."
                    )

        # Make units a "read-only" attribute.
        # It should never be changed after instantiation.
        self._length_units = length_units
        self.solve_dtype = solve_dtype
        self.meshes: Union[Dict[str, Mesh], None] = None

    @property
    def length_units(self) -> str:
        """Length units used for the device geometry."""
        return self._length_units

    @property
    def solve_dtype(self) -> np.dtype:
        """Numpy dtype to use for floating point numbers."""
        return self._solve_dtype

    @solve_dtype.setter
    def solve_dtype(self, dtype) -> None:
        try:
            _ = np.finfo(dtype)
        except ValueError as e:
            raise ValueError(f"Invalid float dtype: {dtype}") from e
        self._solve_dtype = np.dtype(dtype)

    def get_polygons(self, include_terminals: bool = True) -> List[Polygon]:
        """Returns list of all Polygons in the device.

        Args:
            include_terminals: Include transport terminals in the list.

        Returns:
            A list of all Polygons in the device.
        """
        polygons = []
        for attr_name in ("films", "holes", "abstract_regions"):
            polygons.extend(list(getattr(self, attr_name).values()))
        if include_terminals:
            for terminals in self.terminals.values():
                polygons.extend(terminals)
        return polygons

    @property
    def poly_points(self) -> np.ndarray:
        """Shape (n, 2) array of (x, y) coordinates of all Polygons in the Device."""
        points = np.concatenate(
            [poly.points for poly in self.get_polygons(include_terminals=False)]
        )
        # Remove duplicate points to avoid meshing issues.
        # If you don't do this and there are duplicate points,
        # meshpy.triangle will segfault.
        return ensure_unique(points)

    def polygons_by_layer(
        self,
        polygon_type: Optional[
            Literal["film", "hole", "abstract", "terminal", "all"]
        ] = None,
    ) -> Dict[str, List[Polygon]]:
        """Returns a dict of ``{layer_name: list of polygons in layer}``.

        Args:
            polygon_type: One of 'film', 'hole', 'abstract', 'terminal', or 'all', specifying
                which types of polygons to include.

        Returns:
           A dict of ``{layer_name: list of polygons in layer of the given type}``.
        """
        valid_types = ("film", "hole", "abstract", "terminal", "all")
        if polygon_type is None:
            polygon_type = "all"
        polygon_type = polygon_type.lower()
        if polygon_type not in valid_types:
            raise ValueError(
                f"Invalid polygon type ({polygon_type}). Expected one of {valid_types!r}."
            )
        if polygon_type == "film":
            all_polygons = self.films.values()
        elif polygon_type == "hole":
            all_polygons = self.holes.values()
        elif polygon_type == "abstract":
            all_polygons = self.abstract_regions.values()
        elif polygon_type == "terminal":
            all_polygons = []
            for terminals in self.terminals.values():
                all_polygons.extend(terminals)
        else:
            # Films + holes + terminals + abstract regions
            all_polygons = self.get_polygons()
        polygons = {}
        for layer in self.layers:
            polygons[layer] = [p for p in all_polygons if p.layer == layer]
        return polygons

    def holes_by_film(self) -> Dict[str, List[Polygon]]:
        """Generates a mapping of films to holes contained in the film.

        Returns:
           A dict of ``{film_name: list of holes in the film}``.
        """
        holes_by_layer = self.polygons_by_layer("hole")
        holes_by_film = {}
        for film in self.films.values():
            holes_by_film[film.name] = []
            for hole in holes_by_layer[film.layer]:
                if film.contains_points(hole.points).all():
                    holes_by_film[film.name].append(hole)
        return holes_by_film

    def copy(self, with_mesh: bool = True, copy_mesh: bool = False) -> "Device":
        """Copy this Device to create a new one.

        Args:
            with_mesh: Whether to shallow copy the ``meshes`` dictionary.
            copy_mesh: Whether to deepcopy the arrays defining the mesh.

        Returns:
            A new Device instance, copied from self
        """
        layers = [layer.copy() for layer in self.layers.values()]
        films = [film.copy() for film in self.films.values()]
        holes = [hole.copy() for hole in self.holes.values()]
        terminals = {
            film: [term.copy() for term in film_terms]
            for film, film_terms in self.terminals.items()
        }
        abstract_regions = [region.copy() for region in self.abstract_regions.values()]

        device = Device(
            self.name,
            layers=layers,
            films=films,
            holes=holes,
            terminals=terminals,
            abstract_regions=abstract_regions,
            length_units=self.length_units,
        )
        if with_mesh and self.meshes is not None:
            meshes = self.meshes
            if copy_mesh:
                meshes = {name: mesh.copy() for name, mesh in meshes.items()}
            device.meshes = meshes
        return device

    def __copy__(self) -> "Device":
        # Shallow copy (create new references to existing arrays).
        return self.copy(with_mesh=True, copy_mesh=False)

    def __deepcopy__(self, memo) -> "Device":
        # Deep copy (copy all arrays and return references to copies)
        return self.copy(with_mesh=True, copy_mesh=True)

    def _warn_if_mesh_exist(self, method: str) -> None:
        if not self.meshes:
            return
        message = (
            f"Calling device.{method} on a device whose mesh already exists returns "
            f"a new device with no mesh. Call new_device.make_mesh() to generate the mesh "
            f"for the new device."
        )
        logger.warning(message)

    def scale(
        self, xfact: float = 1, yfact: float = 1, origin: Tuple[float, float] = (0, 0)
    ) -> "Device":
        """Returns a new device with polygons scaled horizontally and/or vertically.

        Negative ``xfact`` (``yfact``) can be used to reflect the device horizontally
        (vertically) about the ``origin``.

        Args:
            xfact: Factor by which to scale the device horizontally.
            yfact: Factor by which to scale the device vertically.
            origin: (x, y) coorindates of the origin.

        Returns:
            The scaled :class:`superscreen.Device`.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("scale()")
        device = self.copy(with_mesh=False)
        for polygon in device.get_polygons():
            polygon.scale(xfact=xfact, yfact=yfact, origin=origin, inplace=True)
        return device

    def rotate(self, degrees: float, origin: Tuple[float, float] = (0, 0)) -> "Device":
        """Returns a new device with polygons rotated a given amount
        counterclockwise about specified origin.

        Args:
            degrees: The amount by which to rotate the polygons.
            origin: (x, y) coorindates of the origin.

        Returns:
            The rotated :class:`superscreen.Device`.
        """
        if not (
            isinstance(origin, tuple)
            and len(origin) == 2
            and all(isinstance(val, (int, float)) for val in origin)
        ):
            raise TypeError("Origin must be a tuple of floats (x, y).")
        self._warn_if_mesh_exist("rotate()")
        device = self.copy(with_mesh=False)
        for polygon in device.get_polygons():
            polygon.rotate(degrees, origin=origin, inplace=True)
        return device

    def mirror_layers(self, about_z: float = 0.0) -> "Device":
        """Returns a new device with its layers mirrored about the plane
        ``z = about_z``.

        Args:
            about_z: The z-position of the plane (parallel to the x-y plane)
                about which to mirror the layers.

        Returns:
            The mirrored :class:`superscreen.Device`.
        """
        self._warn_if_mesh_exist("mirror_layers()")
        device = self.copy(with_mesh=False)
        for layer in device.layers.values():
            layer.z0 = about_z - layer.z0
        return device

    def translate(
        self,
        dx: float = 0,
        dy: float = 0,
        dz: float = 0,
        inplace: bool = False,
    ) -> "Device":
        """Translates the device polygons, layers, and meshes in space by a given amount.

        Args:
            dx: Distance by which to translate along the x-axis.
            dy: Distance by which to translate along the y-axis.
            dz: Distance by which to translate layers along the z-axis.
            inplace: If True, modifies the device (``self``) in-place and returns None,
                otherwise, creates a new device, translates it, and returns it.

        Returns:
            The translated device.
        """
        if inplace:
            device = self
        else:
            device = self.copy(with_mesh=True, copy_mesh=True)
        for polygon in device.get_polygons():
            polygon.translate(dx, dy, inplace=True)
        if device.meshes:
            for mesh in device.meshes.values():
                mesh.sites += np.array([[dx, dy]])
        if dz:
            for layer in device.layers.values():
                layer.z0 += dz
        return device

    @contextmanager
    def translation(self, dx: float, dy: float, dz: float = 0) -> None:
        """A context manager that temporarily translates a device in-place,
        then returns it to its original position.

        Args:
            dx: Distance by which to translate polygons along the x-axis.
            dy: Distance by which to translate polygons along the y-axis.
            dz: Distance by which to translate layers along the z-axis.
        """
        try:
            self.translate(dx, dy, dz=dz, inplace=True)
            yield
        finally:
            self.translate(-dx, -dy, dz=-dz, inplace=True)

    def make_mesh(
        self,
        buffer_factor: Union[float, Dict[str, float], None] = 0.05,
        buffer: Union[float, Dict[str, float], None] = None,
        join_style: str = "round",
        min_points: Union[int, Dict[str, int], None] = None,
        max_edge_length: Union[float, Dict[str, float], None] = None,
        preserve_boundary: bool = False,
        smooth: Union[int, Dict[str, int]] = 0,
        **meshpy_kwargs,
    ) -> None:
        """Generates the triangular mesh for each film and stores them in the
        ``self.meshes`` dictionary.

        The arguments ``buffer_factor``, ``buffer``, ``min_points``,
        ``max_edge_length``, and ``smooth`` can be specified either as a
        single value for all films or as a dict of ``{film_name: argument_value}``.

        Args:
            buffer_factor: Buffer for the film bounding box(es), in units of the maximum
                film dimension. This argument is ignored if ``buffer`` is not None.
            buffer: Buffer for the film bounding box(es), in ``length_units``.
            join_style: The join style for the buffered region (see :meth:`superscreen.Polygon.buffer`).
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs and the
                number of vertices in the underlying polygons.
            max_edge_length: The maximum distance between vertices in the resulting mesh.
            preserve_boundary: Do not add any mesh sites to the boundary.
            smooth: Number of Laplacian smoothing iterations to perform.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        films = self.films
        meshes = {}
        if not isinstance(buffer_factor, dict):
            buffer_factor = {name: buffer_factor for name in films}
        if not isinstance(buffer, dict):
            buffer = {name: buffer for name in films}
        if not isinstance(min_points, dict):
            min_points = {name: min_points for name in films}
        if not isinstance(max_edge_length, dict):
            max_edge_length = {name: max_edge_length for name in films}
        if not isinstance(smooth, dict):
            smooth = {name: smooth for name in films}
        holes_by_layer = self.polygons_by_layer("hole")
        abs_regions_by_layer = self.polygons_by_layer("abstract")
        for name, film in films.items():
            film_terminals = self.terminals.get(name)
            holes = holes_by_layer[film.layer]
            abs_regions = abs_regions_by_layer[film.layer]
            coords = [film.points]
            for poly in holes + abs_regions:
                if film.contains_points(poly.points).all():
                    coords.append(poly.points)
            if (
                film_terminals is not None
                or buffer[name] == 0
                or (buffer_factor[name] is None and buffer[name] is None)
            ):
                boundary = film.points
            else:
                boundary = Polygon(points=film.points)
                if buffer[name] is None:
                    buffer_size = buffer_factor[name] * max(boundary.extents)
                else:
                    buffer_size = buffer[name]
                buffered = boundary.polygon.buffer(
                    buffer_size,
                    join_style=getattr(geo.JOIN_STYLE, join_style),
                    single_sided=True,
                    mitre_limit=5.0,
                ).exterior.coords
                boundary = Polygon(points=buffered).resample(len(film.points)).points
                coords.append(boundary)
            points, triangles = utils.generate_mesh(
                ensure_unique(np.concatenate(coords, axis=0)),
                min_points=min_points[name],
                max_edge_length=max_edge_length[name],
                boundary=boundary,
                convex_hull=False,
                preserve_boundary=preserve_boundary or (film_terminals is not None),
                **meshpy_kwargs,
            )
            if smooth[name]:
                meshes[name] = Mesh.from_triangulation(
                    points, triangles, build_operators=False
                ).smooth(smooth[name])
            else:
                meshes[name] = Mesh.from_triangulation(points, triangles)
        self.meshes = meshes

    def boundary_vertices(self, film: str) -> np.ndarray:
        """An array of boundary vertex indices, ordered counterclockwise.

        Args:
            film: The name of the film for which to find boundary indices.

        Returns:
            An array of indices for vertices that are on the film boundary,
            ordered counterclockwise.
        """
        if self.meshes is None:
            return None
        mesh = self.meshes[film]
        points = mesh.sites
        triangles = mesh.elements
        indices = utils.boundary_vertices(points, triangles)
        if film not in self.terminals:
            return indices
        # Ensure that the indices wrap around outside of any terminals.
        for terminal in self.terminals[film]:
            boundary_points = points[indices]
            terminal_indices = terminal.contains_points(boundary_points, index=True)
            discont = np.diff(terminal_indices) != 1
            if np.any(discont):
                i_discont = np.where(discont)[0][0]
                indices = np.roll(indices, -(i_discont + 1))
                break
        return indices

    def mesh_stats_dict(self) -> Optional[Dict[str, Dict[str, Union[int, float]]]]:
        """Returns a dictionary of information about all meshes."""
        if self.meshes is None:
            return None
        return {name: mesh.stats() for name, mesh in self.meshes.items()}

    def mesh_stats(self, precision: int = 3) -> Optional[HTML]:
        """When called with in Jupyter notebook, displays
        a table of information about the mesh.

        Args:
            precision: Number of digits after the decimal for float values.

        Returns:
            An HTML table of mesh statistics.
        """
        all_stats = self.mesh_stats_dict()
        if all_stats is None:
            return None

        def make_row(*cols):
            return "<tr>" + "".join([f"<td>{c}</td>" for c in cols]) + "</tr>"

        html = ["<table>", "<tr><h2>Mesh Statistics</h2></tr>"]
        html.append(make_row("", "<b>length_units</b>", repr(self.length_units)))
        for name, stats in all_stats.items():
            for i, (key, value) in enumerate(stats.items()):
                if isinstance(value, float):
                    value = f"{value:.{precision}e}"
                if i == 0:
                    html.append(make_row(f"<b>{name!r}</b>", f"<b>{key}</b>", value))
                else:
                    html.append(make_row("", f"<b>{key}</b>", value))
        html.append("</table>")
        return HTML("".join(html))

    def mutual_inductance_matrix(
        self,
        hole_polygon_mapping: Optional[Dict[str, np.ndarray]] = None,
        units: str = "pH",
        all_iterations: bool = False,
        progress_bar: bool = False,
        **solve_kwargs,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Calculates the mutual inductance matrix :math:`\\mathbf{M}` for the Device.

        :math:`\\mathbf{M}` is defined such that element :math:`M_{ij}` is the mutual
        inductance :math:`\\Phi^f_{S_i} / I_j` between hole :math:`j` and polygon
        :math:`S_{i}`, which encloses hole :math:`i`. :math:`\\Phi^f_{S_i}` is the
        fluxoid for polygon :math:`S_i` and :math:`I_j` is the current circulating
        around hole :math:`j`.

        Args:
            hole_polygon_mapping: A dict of ``{hole_name: polygon_coordinates}``
                specifying a mapping between holes in the device and polygons
                enclosing those holes, for which the fluxoid will be calculated.
                The length of this dict, ``n_holes``, will determine the dimension
                of the square mutual inductance matrix :math:`M`.
            units: The units in which to report the mutual inductance.
            all_iterations: Whether to return mutual inductance matrices for all
                ``iterations + 1`` solutions, or just the final solution.
            progress_bar: Display a progress bar.
            solve_kwargs: Keyword arguments passed to :func:`superscreen.solve.solve`,
                e.g. ``iterations``.

        Returns:
            If all_iterations is False, returns a shape ``(n_holes, n_holes)`` mutual
            inductance matrix from the final iteration. Otherwise, returns a list of
            mutual inductance matrices, each with shape ``(n_holes, n_holes)``. The
            length of the list is ``1`` if the device has a single layer, or
            ``iterations + 1`` if the device has multiple layers.
        """
        from ..solver import factorize_model, solve

        holes = self.holes
        if hole_polygon_mapping is None:
            from ..fluxoid import make_fluxoid_polygons

            hole_polygon_mapping = make_fluxoid_polygons(self)

        n_holes = len(hole_polygon_mapping)
        for hole_name, polygon in hole_polygon_mapping.items():
            if hole_name not in holes:
                raise ValueError(f"Hole '{hole_name}' does not exist in the device.")
            if not fem.in_polygon(polygon, holes[hole_name].points).all():
                raise ValueError(
                    f"Hole '{hole_name}' is not completely contained "
                    f"within the given polygon."
                )
        solve_kwargs = solve_kwargs.copy()
        iterations = solve_kwargs.get("iterations", 1)
        solve_kwargs["current_units"] = None
        solve_kwargs["progress_bar"] = False
        # The magnitude of this current is not important
        I_circ = self.ureg("1 mA")
        if all_iterations:
            n_iter = 1 if len(self.layers) == 1 else iterations + 1
            solution_slice = slice(None)
        else:
            n_iter = 1
            solution_slice = slice(-1, None)
        mutual_inductance = np.zeros((n_iter, n_holes, n_holes))
        films_by_hole = {}
        for film, holes in self.holes_by_film().items():
            for hole in holes:
                films_by_hole[hole.name] = film
        model = None
        for j, hole_name in enumerate(
            tqdm(hole_polygon_mapping, desc="Holes", disable=(not progress_bar))
        ):
            logger.info(
                f"Evaluating {self.name!r} mutual inductance matrix "
                f"column ({j+1}/{len(hole_polygon_mapping)}), "
                f"source = {hole_name!r}."
            )
            if model is None:
                model = factorize_model(
                    device=self,
                    current_units="mA",
                    circulating_currents={hole_name: str(I_circ)},
                )
                I_circ_val = model.circulating_currents[hole_name]
            else:
                model.set_circulating_currents({hole_name: I_circ_val})
            solutions = solve(device=None, model=model, **solve_kwargs)[solution_slice]

            for n, solution in enumerate(solutions):
                logger.info(
                    f"Evaluating fluxoids for solution {n + 1}/{len(solutions)}."
                )
                for i, (name, polygon) in enumerate(hole_polygon_mapping.items()):
                    fluxoid = solution.polygon_fluxoid(
                        polygon, film=films_by_hole[name]
                    )
                    mutual_inductance[n, i, j] = (
                        (sum(fluxoid) / I_circ).to(units).magnitude
                    )
        mutual_inductance = mutual_inductance * self.ureg(units)
        # Return a list to make it clear that we are returning n_iter distinct
        # matrices. You can convert back to an ndarray using
        # np.stack(result, axis=0).
        result = [m for m in mutual_inductance]
        if not all_iterations:
            assert len(result) == 1
            result = result[0]
        return result

    def plot_polygons(
        self,
        ax: Optional[plt.Axes] = None,
        subplots: bool = False,
        legend: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot all of the Device's polygons.

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            subplots: If True, plots each film on a different subplot.
            legend: Whether to add a legend.
            figsize: matplotlib figsize, only used if ax is None.
            kwargs: Passed to ``ax.plot()`` for the polygon boundaries.

        Returns:
            Matplotlib Figure and Axes
        """
        if len(self.films) > 1 and subplots and ax is not None:
            raise ValueError(
                "Axes may not be provided if subplots is True and the device has "
                "multiple films."
            )
        if ax is None:
            if subplots:
                from ..visualization import auto_grid

                fig, axes = auto_grid(
                    len(self.films),
                    max_cols=2,
                    figsize=figsize,
                    constrained_layout=True,
                )
            else:
                fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
                axes = np.array([axes for _ in self.films])
        else:
            subplots = False
            fig = ax.get_figure()
            axes = np.array([ax for _ in self.films])
        holes_by_film = self.holes_by_film()
        terminals = self.terminals
        for ax, (name, film) in zip(axes.flat, self.films.items()):
            _ = film.plot(ax=ax, **kwargs)
            for hole in holes_by_film[name]:
                _ = hole.plot(ax=ax, **kwargs)
            if name in terminals:
                for terminal in terminals[name]:
                    _ = terminal.plot(ax=ax, **kwargs)
            if subplots:
                ax.set_title(name)
            if legend:
                ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
            units = self.ureg(self.length_units).units
            ax.set_xlabel(f"$x$ $[{units:~L}]$")
            ax.set_ylabel(f"$y$ $[{units:~L}]$")
            ax.set_aspect("equal")
        if not subplots:
            axes = axes[0]
        return fig, axes

    def plot_mesh(
        self,
        ax: Optional[plt.Axes] = None,
        subplots: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        show_sites: bool = False,
        show_edges: bool = True,
        site_color: Union[str, Sequence[float], None] = None,
        edge_color: Union[str, Sequence[float], None] = None,
        linewidth: float = 0.75,
        linestyle: str = "-",
        marker: str = ".",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot all of the Device's meshes.

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            subplots: If True, plots each film on a different subplot.
            figsize: matplotlib figsize, only used if ax is None.
            show_sites: Whether to show the mesh sites.
            show_edges: Whether to show the mesh edges.
            site_color: The color for the sites.
            edge_color: The color for the edges.
            linewidth: The line width for all edges.
            linestyle: The line style for all edges.
            marker: The marker to use for the mesh sites.

        Returns:
            Matplotlib Figure and Axes
        """
        if len(self.films) > 1 and subplots and ax is not None:
            raise ValueError(
                "Axes may not be provided if subplots is True and the device has "
                "multiple films."
            )
        if self.meshes is None:
            raise ValueError(
                "Mesh doesn't exist. Run Device.make_mesh() to generate one."
            )
        if ax is None:
            if subplots:
                from ..visualization import auto_grid

                fig, axes = auto_grid(
                    len(self.films),
                    max_cols=2,
                    figsize=figsize,
                    constrained_layout=True,
                )
            else:
                fig, axes = plt.subplots(figsize=figsize, constrained_layout=True)
                axes = np.array([axes for _ in self.films])
        else:
            subplots = False
            fig = ax.get_figure()
            axes = np.array([ax for _ in self.films])
        for i, (ax, (name, mesh)) in enumerate(zip(axes.flat, self.meshes.items())):
            sc = f"C{i}" if site_color is None else site_color
            ec = f"C{i}" if edge_color is None else edge_color
            ax = mesh.plot(
                ax=ax,
                show_sites=show_sites,
                show_edges=show_edges,
                site_color=sc,
                edge_color=ec,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
            )
            if subplots:
                ax.set_title(name)
            units = self.ureg(self.length_units).units
            ax.set_xlabel(f"$x$ $[{units:~L}]$")
            ax.set_ylabel(f"$y$ $[{units:~L}]$")
            ax.set_aspect("equal")
        if not subplots:
            axes = axes[0]
        return fig, axes

    def patches(self) -> Dict[str, Dict[str, PathPatch]]:
        """Returns a dict of ``{layer_name: {film_name: PathPatch}}``
        for visualizing the device.
        `"""
        abstract_regions = self.abstract_regions
        polygons_by_layer = self.polygons_by_layer()
        holes_by_layer = self.polygons_by_layer(polygon_type="hole")
        patches = defaultdict(dict)
        for layer, regions in polygons_by_layer.items():
            for region in regions:
                if region.name in holes_by_layer[layer]:
                    continue
                coords = region.points.tolist()
                codes = [Path.LINETO for _ in coords]
                codes[0] = Path.MOVETO
                codes[-1] = Path.CLOSEPOLY
                poly = region.polygon
                for hole in holes_by_layer[layer]:
                    if region.name not in abstract_regions and poly.contains(
                        hole.polygon
                    ):
                        hole_coords = hole.points.tolist()[::-1]
                        hole_codes = [Path.LINETO for _ in hole_coords]
                        hole_codes[0] = Path.MOVETO
                        hole_codes[-1] = Path.CLOSEPOLY
                        coords.extend(hole_coords)
                        codes.extend(hole_codes)
                patches[layer][region.name] = PathPatch(Path(coords, codes))
        return dict(patches)

    def draw(
        self,
        ax: Optional[plt.Axes] = None,
        subplots: bool = False,
        max_cols: int = 3,
        legend: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        alpha: float = 0.5,
        exclude: Optional[Union[str, List[str]]] = None,
        layer_order: str = "increasing",
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """Draws all polygons in the device as matplotlib patches.

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            subplots: If True, plots each layer on a different subplot.
            max_cols: The maximum number of columns to create if ``subplots`` is True.
            legend: Whether to add a legend.
            figsize: matplotlib figsize, only used if ax is None.
            alpha: The alpha (opacity) value for the patches (0 <= alpha <= 1).
            exclude: A polygon name or list of polygon names to exclude
                from the figure.
            layer_order: If ``"increasing"`` (``"decreasing"``) draw polygons in
                increasing (decreasing) order by layer height ``layer.z0``.

        Returns:
            Matplotlib Figre and Axes.
        """
        if len(self.layers) > 1 and subplots and ax is not None:
            raise ValueError(
                "Axes may not be provided if subplots is True and the device has "
                "multiple layers."
            )
        layer_order = layer_order.lower()
        layer_orders = ("increasing", "decreasing")
        if layer_order not in layer_orders:
            raise ValueError(
                f"Invalid layer_order: {layer_order}. "
                f"Valid layer orders are {layer_orders}."
            )
        if ax is None:
            if subplots:
                from ..visualization import auto_grid

                fig, axes = auto_grid(
                    len(self.layers),
                    max_cols=max_cols,
                    figsize=figsize,
                    constrained_layout=True,
                )
            else:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                axes = np.array([ax for _ in self.layers])
        else:
            subplots = False
            fig = ax.get_figure()
            axes = np.array([ax for _ in self.layers])
        exclude = exclude or []
        if isinstance(exclude, str):
            exclude = [exclude]
        layers = [
            layer.name for layer in sorted(self.layers.values(), key=lambda x: x.z0)
        ]
        if layer_order == "decreasing":
            layers = layers[::-1]
        patches = self.patches()
        used_axes = set()
        units = self.ureg(self.length_units).units
        x, y = self.poly_points.T
        margin = 0.1
        dx = np.ptp(x)
        dy = np.ptp(y)
        x0 = x.min() + dx / 2
        y0 = y.min() + dy / 2
        dx *= 1 + margin
        dy *= 1 + margin
        labels = []
        handles = []
        for i, (layer, ax) in enumerate(zip(layers, axes.flat)):
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlim(x0 - dx / 2, x0 + dx / 2)
            ax.set_ylim(y0 - dy / 2, y0 + dy / 2)
            ax.set_xlabel(f"$x$ $[{units:~L}]$")
            ax.set_ylabel(f"$y$ $[{units:~L}]$")
            if subplots:
                labels = []
                handles = []
            j = 0
            for name, patch in patches[layer].items():
                if name in exclude or name in self.holes:
                    continue
                patch.set_facecolor(f"C{i}")
                patch.set_alpha(alpha)
                ax.add_artist(patch)
                used_axes.add(ax)
                if j == 0:
                    labels.append(layer)
                    handles.append(patch)
                j += 1
            if subplots:
                ax.set_title(layer)
                if legend:
                    ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
        if subplots:
            for ax in fig.axes:
                if ax not in used_axes:
                    fig.delaxes(ax)
        else:
            axes = axes[0]
            if legend:
                axes.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")
        return fig, axes

    def to_hdf5(
        self,
        path_or_group: Union[os.PathLike, h5py.Group],
        save_mesh: bool = True,
        compress: bool = True,
    ) -> None:
        """Serializes the Device to and HDF5 file.

        Args:
            path_or_group: Path to an HDF5 file to create, or an open HD5F file.
            save_mesh: Whether to save the full mesh to file.
            compress: Save the minimum amount of data needed to recreate the mesh.
        """
        if isinstance(path_or_group, h5py.Group):
            save_context = nullcontext(path_or_group)
        else:
            save_context = h5py.File(path_or_group, "x")
        with save_context as h5group:
            h5group.attrs["name"] = self.name
            h5group.attrs["length_units"] = self.length_units
            h5group.attrs["solve_dtype"] = str(self.solve_dtype)
            layer_grp = h5group.create_group("layers")
            film_grp = h5group.create_group("films")
            hole_grp = h5group.create_group("holes")
            terminals_grp = h5group.create_group("terminals")
            abs_grp = h5group.create_group("abstract_regions")
            for name, layer in self.layers.items():
                layer.to_hdf5(layer_grp.create_group(name))
            for name, polygon in self.films.items():
                polygon.to_hdf5(film_grp.create_group(name))
            for name, polygon in self.holes.items():
                polygon.to_hdf5(hole_grp.create_group(name))
            for name, polygon in self.abstract_regions.items():
                polygon.to_hdf5(abs_grp.create_group(name))
            for film_name, terminals in self.terminals.items():
                grp = terminals_grp.create_group(film_name)
                for i, terminal in enumerate(terminals):
                    terminal.to_hdf5(grp.create_group(str(i)))
            if save_mesh and self.meshes:
                mesh_grp = h5group.create_group("mesh")
                for name, mesh in self.meshes.items():
                    mesh.to_hdf5(mesh_grp.create_group(name), compress=compress)

    @staticmethod
    def from_hdf5(path_or_group: Union[os.PathLike, h5py.Group]) -> "Device":
        """Loads a Device from an HDF5 file.

        Args:
            path_or_group: Path to an HDF5 file to read, or an open HD5F file.

        Returns:
            The deserialized Device.
        """
        if isinstance(path_or_group, h5py.Group):
            read_context = nullcontext(path_or_group)
        else:
            read_context = h5py.File(path_or_group, "r")
        with read_context as h5group:
            terminals = {}
            for film, grp in h5group["terminals"].items():
                terminals[film] = []
                for i in range(len(grp)):
                    terminals[film].append(Polygon.from_hdf5(grp[str(i)]))
            device = Device(
                name=h5group.attrs["name"],
                layers=[Layer.from_hdf5(grp) for grp in h5group["layers"].values()],
                films=[Polygon.from_hdf5(grp) for grp in h5group["films"].values()],
                holes=[Polygon.from_hdf5(grp) for grp in h5group["holes"].values()],
                terminals=terminals,
                abstract_regions=[
                    Polygon.from_hdf5(grp)
                    for grp in h5group["abstract_regions"].values()
                ],
                length_units=h5group.attrs["length_units"],
                solve_dtype=h5group.attrs["solve_dtype"],
            )
            if "mesh" in h5group:
                device.meshes = {
                    name: Mesh.from_hdf5(grp) for name, grp in h5group["mesh"].items()
                }
            return device

    def __repr__(self) -> str:
        # Normal tab "\t" renders a bit too big in jupyter if you ask me.
        indent = 4
        t = " " * indent
        nt = "\n" + t

        def format_list(L):
            if not L:
                return None
            items = [f"{t}{value}" for value in L]
            return "[" + nt + (", " + nt).join(items) + "," + nt + "]"

        def format_dict(D):
            if not D:
                return None
            items = [f"{t}{key!r}: {value}" for key, value in D.items()]
            return "{" + nt + (", " + nt).join(items) + "," + nt + "}"

        args = [
            f'"{self.name}"',
            f"layers={format_list(self.layers.values())}",
            f"films={format_list(self.films.values())}",
            f"holes={format_list(self.holes.values())}",
            f"terminals={format_dict(self.terminals)}",
            f"abstract_regions={format_list(self.abstract_regions.values())}",
            f'length_units="{self.length_units}"',
        ]

        return f"{self.__class__.__name__}(" + nt + (", " + nt).join(args) + ",\n)"

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Device):
            return False

        def equals_sorted(first, second):
            def key(x):
                return x.name

            return sorted(first, key=key) == sorted(second, key=key)

        return (
            self.name == other.name
            and equals_sorted(self.layers.values(), other.layers.values())
            and equals_sorted(self.films.values(), other.films.values())
            and equals_sorted(self.holes.values(), other.holes.values())
            and self.terminals == other.terminals
            and equals_sorted(
                self.abstract_regions.values(), other.abstract_regions.values()
            )
            and self.length_units == other.length_units
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Use dill for Layer objects because Layer.Lambda could be a Parameter
        state["layers"] = dill.dumps(self.layers)
        return state

    def __setstate__(self, state):
        # Use dill for Layer objects because Layer.Lambda could be a Parameter
        state["layers"] = dill.loads(state["layers"])
        self.__dict__.update(state)
