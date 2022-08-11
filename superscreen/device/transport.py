from typing import List, Dict, Union, Optional

import numpy as np
from scipy import integrate
import scipy.linalg as la

from .components import Layer, Polygon
from .device import Device


def stream_from_current_density(points, J):
    # (0, 0, 1) X (Jx, Jy, 0) == (-Jy, Jx, 0)
    zhat_cross_J = J[:, [1, 0]]
    zhat_cross_J[:, 0] *= -1
    dl = np.diff(points, axis=0, prepend=[points[0]])
    integrand = np.sum(zhat_cross_J * dl, axis=1)
    return integrate.cumulative_trapezoid(integrand, initial=0)


def unit_vector(vector):
    """Normalizes ``vector``."""
    return vector / la.norm(vector, axis=-1)[:, np.newaxis]


def edge_vectors(points):
    dr = np.diff(points, axis=0)
    lengths = np.linalg.norm(dr, axis=1)[:, np.newaxis]
    normals = np.cross(dr, [0, 0, 1])
    unit_normals = unit_vector(normals)
    return np.sum(lengths), unit_normals[:, :2]


def terminal_current_density(points, current):
    total_length, vectors = edge_vectors(points)
    return current * vectors / total_length


def stream_from_terminal_current(points, current):
    J = terminal_current_density(points, current)
    J = np.append(J, [J[-1]], axis=0)
    return stream_from_current_density(points, J)


class TransportDevice(Device):

    """An object representing a device composed of multiple layers of
    thin film superconductor.

    Args:
        name: Name of the device.
        layer: The ``Layer`` making up the device.
        film: The ``Polygon`` representing superconducting film.
        holes: ``Holes`` representing holes in superconducting films.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions will be meshed, and one can calculate the flux through them.
        length_units: Distance units for the coordinate system.
        solve_dtype: The float data type to use when solving the device.
    """

    def __init__(
        self,
        name: str,
        *,
        layer: Layer,
        film: Polygon,
        source_terminals: Optional[List[Polygon]] = None,
        drain_terminals: Optional[List[Polygon]] = None,
        holes: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        abstract_regions: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        length_units: str = "um",
        solve_dtype: Union[str, np.dtype] = "float64",
    ):
        super().__init__(
            name,
            layers=[layer],
            films=[film],
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=length_units,
            solve_dtype=solve_dtype,
        )
        self.source_terminals = source_terminals or []
        self.drain_terminals = drain_terminals or []
        all_terminals = self.source_terminals + self.drain_terminals
        for terminal in all_terminals:
            if terminal.name is None:
                raise ValueError("All current terminals must have a unique name.")
        num_terminals = len(all_terminals)
        if len(set(terminal.name for terminal in all_terminals)) != num_terminals:
            raise ValueError("All current terminals must have a unique name.")

    @property
    def layers(self) -> Dict[str, Layer]:
        """Dict of ``{layer_name: layer}``"""
        return {layer.name: layer for layer in self.layers_list}

    @layers.setter
    def layers(self, layers_dict: Dict[str, Layer]) -> None:
        """Dict of ``{layer_name: layer}``"""
        if len(layers_dict) > 1:
            raise ValueError("A TransportDevice can have only one layer.")
        if not (
            isinstance(layers_dict, dict)
            and all(isinstance(obj, Layer) for obj in layers_dict.values())
        ):
            raise TypeError("Layers must be a dict of {layer_name: Layer}.")
        self.layers_list = list(layers_dict.values())

    @property
    def layer(self) -> Layer:
        return self.layers_list[0]

    @layer.setter
    def layer(self, layer: Layer) -> None:
        self.layers = {layer.name: layer}

    @property
    def films(self) -> Dict[str, Polygon]:
        """Dict of ``{film_name: film_polygon}``"""
        return {film.name: film for film in self._films_list}

    @films.setter
    def films(self, films_dict: Dict[str, Polygon]) -> None:
        """Dict of ``{film_name: film_polygon}``"""
        if len(films_dict) > 1:
            raise ValueError("A TransportDevice can have only one film.")
        if not (
            isinstance(films_dict, dict)
            and all(isinstance(obj, Polygon) for obj in films_dict.values())
        ):
            raise TypeError("Films must be a dict of {film_name: Polygon}.")
        for name, polygon in films_dict.items():
            polygon.name = name
        self._films_list = list(self._validate_polygons(films_dict.values(), "film"))

    @property
    def film(self) -> Layer:
        return self._films_list[0]

    @film.setter
    def film(self, film: Polygon) -> None:
        self.films = {film.name: film}

    def make_mesh(
        self,
        compute_matrices: bool = True,
        weight_method: str = "half_cotangent",
        min_points: Optional[int] = None,
        **meshpy_kwargs,
    ) -> None:
        """Generates and optimizes the triangular mesh.

        Args:
            compute_matrices: Whether to compute the field-independent matrices
                (weights, Q, Laplace operator) needed for Brandt simulations.
            weight_method: Weight methods for computing the Laplace operator:
                one of "uniform", "half_cotangent", or "inv_euclidian".
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs and the
                number of vertices in the underlying polygons.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        return super().make_mesh(
            min_points=min_points,
            weight_method=weight_method,
            compute_matrices=compute_matrices,
            convex_hull=False,
            optimesh_steps=None,
            **meshpy_kwargs,
        )

    def _check_current_mapping(self, current_mapping: Dict[str, float]) -> None:
        terminal_names = {t.name for t in self.source_terminals + self.drain_terminals}
        if terminal_names != set(current_mapping):
            raise ValueError(
                "The current mapping must have one entry for every current terminal."
            )
        source_currents = [current_mapping[t.name] for t in self.source_terminals]
        drain_currents = [current_mapping[t.name] for t in self.drain_terminals]
        if sum(source_currents) != sum(drain_currents):
            raise ValueError(
                "The sum of source currents must be equal to the sum of drain currents."
            )
