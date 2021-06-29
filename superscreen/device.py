# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
from pint import UnitRegistry
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from meshpy import triangle
import optimesh

from .parameter import Parameter


logger = logging.getLogger(__name__)

ureg = UnitRegistry()


class Layer(object):
    """A single layer of a superconducting device.

    You can provide either an effective penetration depth Lambda,
    or both a London penetration depth (lambda_london) and a layer
    thickness. Lambda and london_lambda can either be real numers or
    Parameters which compute the penetration depth as a function of
    position.

    Args:
        name: Name of the layer.
        thickness: Thickness of the superconducting film(s) located in the layer.
        london_lambda: London penetration depth of the superconducting film(s)
            located in the layer.
        z0: Vertical location of the layer.
    """

    def __init__(
        self,
        name: str,
        Lambda: Optional[Union[float, Parameter]] = None,
        london_lambda: Optional[Union[float, Parameter]] = None,
        thickness: Optional[float] = None,
        z0: float = 0,
    ):
        self.name = name
        self.thickness = thickness
        self.london_lambda = london_lambda
        self.z0 = z0
        if Lambda is None:
            if london_lambda is None or thickness is None:
                raise ValueError(
                    "You must provide either an effective penetration depth Lambda "
                    "or both a london_lambda and a thickness."
                )
            self._Lambda = None
        else:
            if london_lambda is not None or thickness is not None:
                raise ValueError(
                    "You must provide either an effective penetration depth Lambda "
                    "or both a london_lambda and a thickness (but not all three)."
                )
            self._Lambda = Lambda

    @property
    def Lambda(self) -> Union[float, Parameter]:
        """Effective penetration depth of the superconductor."""
        if self._Lambda is not None:
            return self._Lambda
        return self.london_lambda ** 2 / self.thickness

    def __repr__(self) -> str:
        Lambda = self.Lambda
        if isinstance(Lambda, (int, float)):
            Lambda = f"{Lambda:.3f}"
        d = self.thickness
        if isinstance(d, (int, float)):
            d = f"{d:.3f}"
        london = self.london_lambda
        if isinstance(london, (int, float)):
            london = f"{london:.3f}"
        return (
            f'Layer("{self.name}", Lambda={Lambda}, '
            f"thickness={d}, london_lambda={london}, z0={self.z0:.3f})"
        )


class Polygon(object):
    """A polygonal region located in a Layer.

    Args:
        name: Name of the polygon.
        layer: Name of the layer in which the polygon is located.
        points: An array of shape (n, 2) specifying the x, y coordinates of
            the polyon's vertices.
    """

    def __init__(self, name: str, *, layer: str, points: np.ndarray):
        self.name = name
        self.layer = layer
        self.points = np.asarray(points)

        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {self.points.shape}.")

    def contains_points(
        self,
        xq: Union[float, np.ndarray],
        yq: Union[float, np.ndarray],
        index: bool = False,
    ) -> Union[bool, np.ndarray]:
        """Determines whether points xq, yq lie within the polygon.

        Args:
            xq: x coordinate(s) of query point(s).
            yq: y coordinate(s) of query point(s).
            index: If True, then return the indices of the points in [xq, yq]
                that lie within the polygon. Otherwise, return a boolean array
                of the same shape as xq and yq.

        Returns:
            If index is True, returns the indices of the points in [xq, yq]
            that lie within the polygon. Otherwise, returns a boolean array
            of the same shape as xq and yq indicating whether each point
            lies within the polygon.
        """
        from .fem import in_polygon

        x, y = self.points.T
        bool_array = in_polygon(xq, yq, x, y)
        if index:
            return np.where(bool_array)[0]
        return bool_array

    def __repr__(self) -> str:
        return (
            f'Polygon("{self.name}", layer="{self.layer}", '
            f"points=ndarray[shape={self.points.shape}])"
        )


class Device(object):
    """An object representing a device composed of multiple layers of
    thin film superconductor.

    Args:
        name: Name of the device.
        layers: ``Layers`` making up the device.
        films: ``Polygons`` representing regions of superconductor.
        holes: ``Holes`` representing holes in superconducting films.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions will be meshed, and one can calculate the flux through them.
        length_units: Distance units for the coordinate system.
    """

    def __init__(
        self,
        name: str,
        *,
        layers: Union[List[Layer], Dict[str, Layer]],
        films: Union[List[Polygon], Dict[str, Polygon]],
        holes: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        abstract_regions: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        length_units: str = "um",
    ):
        self.name = name
        if isinstance(layers, dict):
            self.layers = layers
        else:
            self.layers_list = list(layers)
        if isinstance(films, dict):
            self.films = films
        else:
            self.films_list = list(films)
        if holes is None:
            holes = []
        if isinstance(holes, dict):
            self.holes = holes
        else:
            self.holes_list = list(holes)
        if abstract_regions is None:
            abstract_regions = []
        if isinstance(abstract_regions, dict):
            self.abstract_regions = abstract_regions
        else:
            self.abstract_regions_list = list(abstract_regions)
        # Make units a "read-only" attribute.
        # It should never be changed after instantiation.
        self._length_units = length_units
        self.sparse = None
        self.ureg = ureg

        self.points = None
        self.triangles = None

        self.weights = None
        self.Q = None
        self.Del2 = None

    @property
    def length_units(self):
        """Length units used for the device geometry."""
        return self._length_units

    @property
    def layers(self) -> Dict[str, Layer]:
        """Dict of ``{layer_name: layer}``"""
        return {layer.name: layer for layer in self.layers_list}

    @layers.setter
    def layers(self, layers_dict: Dict[str, Layer]) -> None:
        """Dict of ``{layer_name: layer}``"""
        if not all(isinstance(obj, Layer) for obj in layers_dict.values()):
            raise TypeError("Layers must be a dict of {layer_name: Layer}.")
        self.layers_list = list(layers_dict.values())

    @property
    def films(self) -> Dict[str, Polygon]:
        """Dict of ``{film_name: film_polygon}``"""
        return {film.name: film for film in self.films_list}

    @films.setter
    def films(self, films_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{film_name: film_polygon}``"""
        if not all(isinstance(obj, Polygon) for obj in films_dict.values()):
            raise TypeError("Films must be a dict of {film_name: Polygon}.")
        self.films_list = list(films_dict.values())

    @property
    def holes(self) -> Dict[str, Polygon]:
        """Dict of ``{hole_name: hole_polygon}``"""
        return {hole.name: hole for hole in self.holes_list}

    @holes.setter
    def holes(self, holes_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{hole_name: hole_polygon}``"""
        if not all(isinstance(obj, Polygon) for obj in holes_dict.values()):
            raise TypeError("Holes must be a dict of {hole_name: Polygon}.")
        self.holes_list = list(holes_dict.values())

    @property
    def abstract_regions(self) -> Dict[str, Polygon]:
        """Dict of ``{region_name: region_polygon}``"""
        return {region.name: region for region in self.abstract_regions_list}

    @abstract_regions.setter
    def abstract_regions(self, regions_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{region_name: region_polygon}``"""
        if not all(isinstance(obj, Polygon) for obj in regions_dict.values()):
            raise TypeError(
                "Abstract regions must be a dict of {region_name: Polygon}."
            )
        self.abstract_regions_list = list(regions_dict.values())

    @property
    def poly_points(self) -> np.ndarray:
        """Shape (n, 2) array of (x, y) coordinates of all polygons in the Device."""
        points = np.concatenate(
            [film.points for film in self.films_list]
            + [hole.points for hole in self.holes_list]
            + [region.points for region in self.abstract_regions_list]
        )
        # Remove duplicate points to avoid meshing issues.
        # If you don't do this and there are dupliate points,
        # meshpy.triangle will segfault.
        points = np.unique(points, axis=0)
        return points

    def make_mesh(
        self,
        compute_matrices: bool = True,
        sparse: bool = True,
        weight_method: str = "half_cotangent",
        min_triangles: Optional[int] = None,
        optimesh_steps: Optional[int] = None,
        optimesh_method: str = "cvt-block-diagonal",
        optimesh_tolerance: float = 1e-3,
        optimesh_verbose: bool = False,
        **meshpy_kwargs,
    ) -> None:
        """Computes or refines the triangular mesh.

        Args:
            compute_matrices: Whether to compute the field-independent matrices
                (weights, Q, Laplace operator) needed for Brandt simulations.
            sparse: Whether to use sparse matrices for weights and Laplacian.
            weight_method: Meshing scheme: either "uniform", "half_cotangent", or "inv_euclidian".
            min_triangles: Minimum number of triangles in the mesh. If None, then the
                number of triangles will be determined by meshpy_kwargs.
            optimesh_steps: Maximum number of optimesh steps. If None, then no optimization is done.
            optimesh_method: Name of the optimization method to use.
            optimesh_tolerance: Optimesh quality tolerance.
            optimesh_verbose: Whether to use verbose mode in optimesh.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        logger.info("Generating mesh...")
        # Mesh the entire convex hull of poly_points
        poly_points = self.poly_points
        hull = ConvexHull(poly_points)
        mesh_info = triangle.MeshInfo()
        mesh_info.set_points(poly_points)
        mesh_info.set_facets(hull.simplices)
        # Optimal angle is 60 degrees,
        # meshpy quality meshing default is 20 degrees
        min_angle = meshpy_kwargs.get("min_angle", 32.5)
        meshpy_kwargs["min_angle"] = min_angle
        if min_triangles is None:
            mesh = triangle.build(mesh_info=mesh_info, **meshpy_kwargs)
            points = np.array(mesh.points)
            triangles = np.array(mesh.elements)
        else:
            max_vol = 1
            num_triangles = 0
            kwargs = meshpy_kwargs.copy()
            i = 1
            while num_triangles < min_triangles:
                kwargs["max_volume"] = max_vol
                mesh = triangle.build(
                    mesh_info=mesh_info,
                    **kwargs,
                )
                points = np.array(mesh.points)
                triangles = np.array(mesh.elements)
                num_triangles = triangles.shape[0]
                logger.debug(
                    f"Iteration {i}: Made mesh with {points.shape[0]} points and "
                    f"{triangles.shape[0]} triangles using max_volume={max_vol:.2e}. "
                    f"Target number of triangles: {min_triangles}."
                )
                max_vol *= 0.9
                i += 1
        if optimesh_steps:
            logger.info(f"Optimizing mesh with {triangles.shape[0]} triangles.")
            points, triangles = optimesh.optimize_points_cells(
                points,
                triangles,
                optimesh_method,
                optimesh_tolerance,
                optimesh_steps,
                verbose=optimesh_verbose,
            )
        logger.info(
            f"Finished generating mesh with {points.shape[0]} points and "
            f"{triangles.shape[0]} triangles."
        )
        self.points = points
        self.triangles = triangles

        if compute_matrices:
            from .fem import calculate_weights, laplace_operator
            from . import brandt

            logger.info("Calculating weight matrix.")
            self.weights = calculate_weights(
                points, triangles, weight_method, sparse=sparse
            )
            logger.info("Calculating Laplace operator.")
            self.Del2 = laplace_operator(points, triangles, self.weights, sparse=sparse)
            logger.info("Calculating kernel matrix.")
            q = brandt.q_matrix(points)
            # Each layer has its own edge vector C, so each layer's kernal matrix Q
            # will have different diagonals.
            C_vectors = {}
            films = self.films
            x, y = points.T
            for layer_name in self.layers:
                films = [film for film in self.films_list if film.layer == layer_name]
                mask = np.logical_or.reduce(
                    [film.contains_points(x, y) for film in films]
                )
                C_vectors[layer_name] = brandt.C_vector(points, mask=mask)

            def Q_matrix(layer):
                return brandt.Q_matrix(q, C_vectors[layer], self.weights)

            self.Q = Q_matrix
            self.sparse = sparse

    def plot_polygons(
        self,
        ax: Optional[plt.Axes] = None,
        grid: Optional[bool] = False,
        legend: Optional[bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot all of the device's polygons.

        Keyword arguments are passed to ax.plot().

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            grid: Whether to add grid lines.
            legend: Whether to add a legend.
            figsize: matplotlib figsize, only used if ax is None.

        Returns:
            matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        for name, film in self.films.items():
            points = np.concatenate([film.points, film.points[:1]], axis=0)
            ax.plot(*points.T, label=name, **kwargs)
        for name, hole in self.holes.items():
            points = np.concatenate([hole.points, hole.points[:1]], axis=0)
            ax.plot(*points.T, label=name, **kwargs)
        for name, region in self.abstract_regions.items():
            points = np.concatenate([region.points, region.points[:1]], axis=0)
            ax.plot(*points.T, label=name, **kwargs)
        ax.set_aspect("equal")
        ax.grid(grid)
        if legend:
            ax.legend(loc="best")
        units = self.ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        return ax

    def plot_mesh(
        self,
        ax: Optional[plt.Axes] = None,
        edges: Optional[bool] = True,
        vertices: Optional[bool] = False,
        grid: Optional[bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plots the device's mesh.

        Keyword arguments are passed to ax.triplot() and ignored if edges is False.

        Args:
            ax: matplotlib axis on which to plot. If None, a new figure is created.
            edges: Whether to plot the triangle edges.
            vertices: Whether to plot the triangle vertices.
            grid: Whether to add grid lines.
            figsize: matplotlib figsize, only used if ax is None.

        Returns:
            matplotlib axis
        """
        if self.triangles is None:
            raise RuntimeError(
                "Mesh does not exist. Run device.make_mesh() to generate the mesh."
            )
        x, y = self.points.T
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if edges:
            ax.triplot(x, y, self.triangles, **kwargs)
        if vertices:
            ax.plot(x, y, "k.")
        ax.set_aspect("equal")
        ax.grid(grid)
        units = self.ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        return ax

    def __repr__(self) -> str:
        # Normal tab "\t" renders a bit too big in jupyter if you ask me.
        indent = 4
        t = " " * indent
        nt = "\n" + t

        # def format_dict(d):
        #     if not d:
        #         return None
        #     items = [f'{t}"{key}": {value}' for key, value in d.items()]
        #     return "{" + nt + (", " + nt).join(items) + "," + nt + "}"

        # args = [
        #     f'"{self.name}"',
        #     f"layers={format_dict(self.layers)}",
        #     f"films={format_dict(self.films)}",
        #     f"holes={format_dict(self.holes)}",
        #     f"abstract_regions={format_dict(self.abstract_regions)}",
        #     f'length_units="{self.length_units}"',
        # ]

        def format_list(L):
            if not L:
                return None
            items = [f"{t}{value}" for value in L]
            return "[" + nt + (", " + nt).join(items) + "," + nt + "]"

        args = [
            f'"{self.name}"',
            f"layers={format_list(self.layers_list)}",
            f"films={format_list(self.films_list)}",
            f"holes={format_list(self.holes_list)}",
            f"abstract_regions={format_list(self.abstract_regions_list)}",
            f'length_units="{self.length_units}"',
        ]

        return "Device(" + nt + (", " + nt).join(args) + ",\n)"
