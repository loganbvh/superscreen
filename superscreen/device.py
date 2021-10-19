import os
import json
import logging
from copy import deepcopy
from typing import Optional, Union, List, Tuple, Dict

import dill
import numpy as np
from pint import UnitRegistry
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.spatial import ConvexHull
from meshpy import triangle
import optimesh

from . import brandt
from . import fem
from .geometry import close_curve
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

    @Lambda.setter
    def Lambda(self, value: Union[float, Parameter]) -> None:
        """Effective penetration depth of the superconductor."""
        if self._Lambda is None:
            raise AttributeError(
                "Can't set Lambda directly. Set london_lambda and/or thickness instead."
            )
        self._Lambda = value

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

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.name == other.name
            and self.thickness == other.thickness
            and self.london_lambda == other.london_lambda
            and self.Lambda == other.Lambda
            and self.z0 == other.z0
        )

    def copy(self):
        return deepcopy(self)


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
        # Ensure that it is a closed polygon.
        self.points = close_curve(self.points)

        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {self.points.shape}.")

    @property
    def clockwise(self) -> bool:
        """Returns True if the polygon vertices are oriented clockwise."""
        # https://stackoverflow.com/a/1165943
        # https://www.element84.com/blog/
        # determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
        x = self.points[:, 0]
        y = self.points[:, 1]
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0

    @property
    def counter_clockwise(self) -> bool:
        """Returns True if the polygon vertices are oriented counter-clockwise."""
        return not self.clockwise

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
        bool_array = fem.in_polygon(xq, yq, self.points[:, 0], self.points[:, 1])
        if index:
            return np.where(bool_array)[0]
        return bool_array

    def __repr__(self) -> str:
        return (
            f'Polygon("{self.name}", layer="{self.layer}", '
            f"points=ndarray[shape={self.points.shape}])"
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Polygon):
            return False

        return (
            self.name == other.name
            and self.layer == other.layer
            and np.allclose(self.points, other.points)
        )

    def copy(self):
        return deepcopy(self)


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

    ARRAY_NAMES = (
        "points",
        "triangles",
        "mass_matrix",
        "weights",
        "Del2",
        "q",
        "C_vectors",
    )

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
        self.ureg = ureg

        self.points = None
        self.triangles = None

        self.mass_matrix = None
        self.weights = None
        self.Del2 = None
        self.q = None
        self.C_vectors = None
        self._Q_cache = {}

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
        if not (
            isinstance(layers_dict, dict)
            and all(isinstance(obj, Layer) for obj in layers_dict.values())
        ):
            raise TypeError("Layers must be a dict of {layer_name: Layer}.")
        self.layers_list = list(layers_dict.values())

    @property
    def films(self) -> Dict[str, Polygon]:
        """Dict of ``{film_name: film_polygon}``"""
        return {film.name: film for film in self.films_list}

    @films.setter
    def films(self, films_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{film_name: film_polygon}``"""
        if not (
            isinstance(films_dict, dict)
            and all(isinstance(obj, Polygon) for obj in films_dict.values())
        ):
            raise TypeError("Films must be a dict of {film_name: Polygon}.")
        self.films_list = list(films_dict.values())

    @property
    def holes(self) -> Dict[str, Polygon]:
        """Dict of ``{hole_name: hole_polygon}``"""
        return {hole.name: hole for hole in self.holes_list}

    @holes.setter
    def holes(self, holes_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{hole_name: hole_polygon}``"""
        if not (
            isinstance(holes_dict, dict)
            and all(isinstance(obj, Polygon) for obj in holes_dict.values())
        ):
            raise TypeError("Holes must be a dict of {hole_name: Polygon}.")
        self.holes_list = list(holes_dict.values())

    @property
    def abstract_regions(self) -> Dict[str, Polygon]:
        """Dict of ``{region_name: region_polygon}``"""
        return {region.name: region for region in self.abstract_regions_list}

    @abstract_regions.setter
    def abstract_regions(self, regions_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{region_name: region_polygon}``"""
        if not (
            isinstance(regions_dict, dict)
            and all(isinstance(obj, Polygon) for obj in regions_dict.values())
        ):
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
        # If you don't do this and there are duplicate points,
        # meshpy.triangle will segfault.
        points = np.unique(points, axis=0)
        return points

    def get_arrays(
        self,
        copy_arrays: bool = False,
        dense: bool = True,
    ) -> Optional[
        Dict[str, Union[Union[np.ndarray, sp.csr_matrix], Dict[str, np.ndarray]]]
    ]:
        """Returns a dict of the large arrays that belong to the device.

        Args:
            copy_arrays: Whether to copy all of the arrays or just return references
                to the existing arrays.
            dense: Whether to convert any sparse matrices to dense numpy arrays.

        Returns:
            A dict of arrays, with keys specified by ``Device.ARRAY_NAMES``,
            or None if the arrays don't exist.
        """
        arrays = {name: getattr(self, name, None) for name in self.ARRAY_NAMES}
        if all(val is None for val in arrays.values()):
            return None
        if copy_arrays:
            arrays = deepcopy(arrays)
        for name, array in arrays.items():
            if dense and sp.issparse(array):
                arrays[name] = array.toarray()
        return arrays

    def set_arrays(
        self,
        arrays: Dict[
            str, Union[Union[np.ndarray, sp.csr_matrix], Dict[str, np.ndarray]]
        ],
    ) -> None:
        """Sets the Device's large arrays from a dict like that returned
        by Device.get_arrays().

        Args:
            arrays: The dict containing the arrays to use.
        """
        # Ensure that all names and types are valid before setting any attributes.
        valid_types = {name: (np.ndarray,) for name in self.ARRAY_NAMES}
        for name in ("weights", "Del2", "mass_matrix"):
            valid_types[name] = (valid_types[name], sp.spmatrix)
        _ = valid_types.pop("C_vectors")
        C_vectors = arrays["C_vectors"]
        layer_names = set(self.layers)
        for name, array in C_vectors.items():
            if name not in layer_names:
                raise ValueError(f"Unexpected C_vector name: {name}.")
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    f"Expected type {np.ndarray} for C vector '{name}', "
                    f"but got {type(array)}."
                )
        for name, array in arrays.items():
            if name == "C_vectors":
                continue
            if name not in self.ARRAY_NAMES:
                raise ValueError(f"Unexpected array name: {name}.")
            if not isinstance(array, valid_types[name]):
                raise TypeError(
                    f"Expected type in {valid_types[name]} for array '{name}', "
                    f"but got {type(array)}."
                )
        # Finally actually set the attributes
        for name, array in arrays.items():
            setattr(self, name, array)
        self.C_vectors = C_vectors

    def copy(self, with_arrays: bool = True, copy_arrays: bool = False) -> "Device":
        """Copy this Device to create a new one.

        Args:
            with_arrays: Whether to set the large arrays on the new Device.
            copy_arrays: Whether to create copies of the large arrays, or just
                return references to the existing arrays.

        Returns:
            A new Device instance, copied from self
        """
        layers = [layer.copy() for layer in self.layers_list]
        films = [film.copy() for film in self.films_list]
        holes = [hole.copy() for hole in self.holes_list]
        abstract_regions = [region.copy() for region in self.abstract_regions_list]

        device = Device(
            self.name,
            layers=layers,
            films=films,
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=self.length_units,
        )
        if with_arrays:
            arrays = self.get_arrays(copy_arrays=copy_arrays)
            if arrays is not None:
                device.set_arrays(arrays)
        return device

    def __copy__(self) -> "Device":
        # Shallow copy (create new references to existing arrays).
        return self.copy(with_arrays=True, copy_arrays=False)

    def __deepcopy__(self, memo) -> "Device":
        # Deep copy (copy all arrays and return references to copies)
        return self.copy(with_arrays=True, copy_arrays=True)

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
            weight_method: Meshing scheme: either "uniform", "half_cotangent",
                or "inv_euclidian".
            min_triangles: Minimum number of triangles in the mesh. If None, then
                the number of triangles will be determined by meshpy_kwargs.
            optimesh_steps: Maximum number of optimesh steps. If None, then no
                optimization is done.
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
            try:
                points, triangles = optimesh.optimize_points_cells(
                    points,
                    triangles,
                    optimesh_method,
                    optimesh_tolerance,
                    optimesh_steps,
                    verbose=optimesh_verbose,
                )
            except np.linalg.LinAlgError as e:
                err = (
                    "LinAlgError encountered in optimesh. Try reducing min_triangles "
                    "or increasing the number of points in the device's important polygons."
                )
                raise RuntimeError(err) from e
        logger.info(
            f"Finished generating mesh with {points.shape[0]} points and "
            f"{triangles.shape[0]} triangles."
        )
        self.points = points
        self.triangles = triangles

        self.mass_matrix = None
        self.weights = None
        self.Del2 = None
        self._Q_cache = {}
        if compute_matrices:
            self.compute_matrices(sparse=sparse, weight_method=weight_method)

    def Q(
        self, layer: str, weights: Optional[Union[np.ndarray, sp.csr_matrix]] = None
    ) -> np.ndarray:
        """Computes the kernel matrix Q for a given layer of the device."""
        Q = self._Q_cache.get(layer, None)
        if weights is None:
            weights = self.weights
        if Q is None:
            Q = brandt.Q_matrix(self.q, self.C_vectors[layer], weights)
            self._Q_cache[layer] = Q
        return Q

    def compute_matrices(
        self, sparse: bool = True, weight_method: str = "half_cotangent"
    ) -> None:
        """Calculcates mesh weights, Laplace oeprator, and kernel functions.

        Args:
            sparse: Whether to use sparse matrices for weights and Laplacian.
            weight_method: Meshing scheme: either "uniform", "half_cotangent",
                or "inv_euclidian".
        """

        points = self.points
        triangles = self.triangles

        if points is None or triangles is None:
            raise ValueError(
                "Device mesh does not exist. Run Device.make_mesh() "
                "to generate the mesh."
            )

        logger.info("Calculating mass matrix.")
        self.mass_matrix = fem.mass_matrix(points, triangles, sparse=sparse)

        logger.info("Calculating weight matrix.")
        self.weights = fem.calculate_weights(
            points, triangles, weight_method, normalize=True, sparse=sparse
        )
        logger.info("Calculating Laplace operator.")
        self.Del2 = fem.laplace_operator(
            points,
            triangles,
            masses=self.mass_matrix,
            weight_method=weight_method,
            sparse=sparse,
        )
        logger.info("Calculating kernel matrix.")
        self.q = brandt.q_matrix(points)
        # Each layer has its own edge vector C, so each layer's kernel matrix Q
        # will have different diagonals.
        self.C_vectors = {}
        x = points[:, 0]
        y = points[:, 1]
        for layer_name in self.layers:
            films = [film for film in self.films_list if film.layer == layer_name]
            self.C_vectors[layer_name] = sum(
                brandt.C_vector(points, mask=film.contains_points(x, y))
                for film in films
            )
        self._Q_cache = {}

    def plot_polygons(
        self,
        ax: Optional[plt.Axes] = None,
        grid: bool = False,
        legend: bool = True,
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
            _, ax = plt.subplots(figsize=figsize)
        for name, film in self.films.items():
            ax.plot(*film.points.T, label=name, **kwargs)
        for name, hole in self.holes.items():
            ax.plot(*hole.points.T, label=name, **kwargs)
        for name, region in self.abstract_regions.items():
            ax.plot(*region.points.T, label=name, **kwargs)
        ax.set_aspect("equal")
        ax.grid(grid)
        if legend:
            ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        units = self.ureg(self.length_units).units
        ax.set_xlabel(f"$x$ $[{units:~L}]$")
        ax.set_ylabel(f"$y$ $[{units:~L}]$")
        return ax

    def plot_mesh(
        self,
        ax: Optional[plt.Axes] = None,
        edges: bool = True,
        vertices: bool = False,
        grid: bool = True,
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
        x = self.points[:, 0]
        y = self.points[:, 1]
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
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

    def to_file(
        self, directory: str, save_mesh: bool = True, compressed: bool = True
    ) -> None:
        """Serializes the Device to disk.

        Args:
            directory: The name of the directory in which to save the Device
                (must either be empty or not yet exist).
            save_mesh: Whether to save the full mesh to file.
            compressed: Whether to use numpy.savez_compressed rather than numpy.savez
                when saving the mesh.
        """
        from .io import NumpyJSONEncoder

        if os.path.isdir(directory) and len(os.listdir(directory)):
            raise IOError(f"Directory '{directory}' already exists and is not empty.")
        os.makedirs(directory, exist_ok=True)

        # Serialize films, holes, and abstract_regions to JSON
        polygons = {"device_name": self.name, "length_units": self.length_units}
        for poly_type in ["films", "holes", "abstract_regions"]:
            polygons[poly_type] = {}
            for name, poly in getattr(self, poly_type).items():
                polygons[poly_type][name] = {
                    "layer": poly.layer,
                    "points": poly.points,
                }
        with open(os.path.join(directory, "polygons.json"), "w") as f:
            json.dump(polygons, f, indent=4, cls=NumpyJSONEncoder)

        # Serialize layers to JSON.
        # If Lambda or london_lambda is a Parameter, then we will
        # pickle it using dill.
        layers = {"device_name": self.name, "length_units": self.length_units}
        for name, layer in self.layers.items():
            layers[name] = {"z0": layer.z0, "thickness": layer.thickness}
            if isinstance(layer._Lambda, (int, float)):
                layers[name]["Lambda"] = layer._Lambda
                layers[name]["london_lambda"] = None
            elif isinstance(layer.london_lambda, (int, float)):
                layers[name]["Lambda"] = None
                layers[name]["london_lambda"] = layer.london_lambda
            else:
                if isinstance(layer._Lambda, Parameter):
                    # Lambda is defined as a Parameter and london_lambda is None
                    assert layer.london_lambda is None
                    param = layer._Lambda
                    key = "Lambda"
                    null_key = "london_lambda"
                else:
                    # london_lambda is defined as a Parameter and _Lambda is None
                    assert layer._Lambda is None
                    param = layer.london_lambda
                    key = "london_lambda"
                    null_key = "Lambda"
                dill_fname = f"{layer.name}_{param.func.__name__}.dill"
                layers[name][null_key] = None
                layers[name][key] = {"path": dill_fname}
                with open(os.path.join(directory, dill_fname), "wb") as f:
                    dill.dump(param, f)

        with open(os.path.join(directory, "layers.json"), "w") as f:
            json.dump(layers, f, indent=4, cls=NumpyJSONEncoder)

        if save_mesh:
            # Serialize mesh, if it exists.
            save_npz = np.savez_compressed if compressed else np.savez
            if self.points is not None:
                save_npz(
                    os.path.join(directory, "mesh.npz"),
                    points=self.points,
                    triangles=self.triangles,
                )

    @classmethod
    def from_file(cls, directory: str, compute_matrices: bool = True) -> "Device":
        """Creates a new Device from one serialized to disk.

        Args:
            directory: The directory from which to load the device.
            compute_matrices: Whether to compute the field-independent
                matrices for the device if the mesh already exists.

        Returns:
            The loaded Device instance
        """
        from .io import json_numpy_obj_hook

        # Load all polygons (films, holes, abstract_regions)
        with open(os.path.join(directory, "polygons.json"), "r") as f:
            polygons_json = json.load(f, object_hook=json_numpy_obj_hook)

        device_name = polygons_json.pop("device_name")
        length_units = polygons_json.pop("length_units")
        films = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["films"].items()
        }
        holes = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["holes"].items()
        }
        abstract_regions = {
            name: Polygon(name, **kwargs)
            for name, kwargs in polygons_json["abstract_regions"].items()
        }

        # Load all layers
        with open(os.path.join(directory, "layers.json"), "r") as f:
            layers_json = json.load(f, object_hook=json_numpy_obj_hook)

        device_name = layers_json.pop("device_name")
        length_units = layers_json.pop("length_units")
        for name, layer_dict in layers_json.items():
            # Check whether either Lambda or london_lambda is a Parameter
            # that was pickled.
            if isinstance(layer_dict["Lambda"], dict):
                assert layer_dict["london_lambda"] is None
                path = layer_dict["Lambda"]["path"]
                with open(os.path.join(directory, path), "rb") as f:
                    Lambda = dill.load(f)
                layers_json[name]["Lambda"] = Lambda
            elif isinstance(layer_dict["london_lambda"], dict):
                assert layer_dict["Lambda"] is None
                path = layer_dict["london_lambda"]["path"]
                with open(os.path.join(directory, path), "rb") as f:
                    london_lambda = dill.load(f)
                layers_json[name]["london_lambda"] = london_lambda

        layers = {name: Layer(name, **kwargs) for name, kwargs in layers_json.items()}

        device = cls(
            device_name,
            layers=layers,
            films=films,
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=length_units,
        )

        # Load the mesh if it exists
        if "mesh.npz" in os.listdir(directory):
            with np.load(os.path.join(directory, "mesh.npz")) as mesh:
                points = mesh["points"]
                triangles = mesh["triangles"]
            device.points = points
            device.triangles = triangles
            if compute_matrices:
                device.compute_matrices()

        return device

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

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Device):
            return False

        return (
            self.name == other.name
            and self.layers_list == other.layers_list
            and self.films_list == other.films_list
            and self.holes_list == other.holes_list
            and self.abstract_regions_list == other.abstract_regions_list
            and self.length_units == other.length_units
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Pickle can't handle a pint.UnitRegistry.
        del state["ureg"]
        # Use dill for Layer objects because Layer.Lambda could be a Parameter
        state["layers_list"] = dill.dumps(self.layers_list)
        return state

    def __setstate__(self, state):
        # Pickle can't handle a pint.UnitRegistry.
        state["ureg"] = ureg
        # Use dill for Layer objects because Layer.Lambda could be a Parameter
        state["layers_list"] = dill.loads(state["layers_list"])
        self.__dict__.update(state)
