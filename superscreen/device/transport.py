from typing import Dict, List, Optional, Union

import numpy as np
from scipy import integrate

from ..geometry import path_vectors
from . import utils
from .device import Device
from .layer import Layer
from .mesh import Mesh
from .polygon import Polygon


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
    length, unit_normals = path_vectors(points)
    J = current * unit_normals / length
    g = stream_from_current_density(points, J)
    return g * current / g[-1]


class TransportDevice(Device):
    """An object representing a single-layer, single-film  device to which
    a source-drain current bias can be applied.

    There can be multiple source terminals, but only a single drain terminal.
    The drain terminal acts as a sink for currents from all source terminals.

    Args:
        name: Name of the device.
        layer: The ``Layer`` making up the device.
        film: The ``Polygon`` representing superconducting film.
        source_terminals: A list of Polygons representing source terminals for
            a transport current. Any mesh vertices that are on the boundary of the film
            and lie inside a source terminal will have current source boundary
            conditions.
        drain_terminal: Polygon representing the sole current drain (or output) terminal
            of the device.
        holes: ``Holes`` representing holes in superconducting films.
        abstract_regions: ``Polygons`` representing abstract regions in a device.
            Abstract regions will be meshed, and one can calculate the flux through them.
        length_units: Distance units for the coordinate system.
        solve_dtype: The float data type to use when solving the device.
    """

    POLYGONS = Device.POLYGONS + ("terminals",)

    def __init__(
        self,
        name: str,
        *,
        layer: Layer,
        film: Polygon,
        source_terminals: Optional[List[Polygon]] = None,
        drain_terminal: Optional[Polygon] = None,
        holes: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        abstract_regions: Optional[Union[List[Polygon], Dict[str, Polygon]]] = None,
        length_units: str = "um",
        solve_dtype: Union[str, np.dtype] = "float64",
    ):
        self.source_terminals = source_terminals or []
        self.drain_terminal = drain_terminal
        if self.source_terminals and self.drain_terminal is None:
            raise ValueError("Cannot have source terminals without a drain terminal.")
        if self.drain_terminal and not self.source_terminals:
            raise ValueError("Cannot have a drain terminal without source terminals.")
        all_terminals = list(self.terminals.values())
        for terminal in all_terminals:
            if terminal.name is None:
                raise ValueError("All current terminals must have a unique name.")
            terminal.mesh = False
            terminal.layer = film.layer
        num_terminals = len(all_terminals)
        if len(set(terminal.name for terminal in all_terminals)) != num_terminals:
            raise ValueError("All current terminals must have a unique name.")

        super().__init__(
            name,
            layers=[layer],
            films=[film],
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=length_units,
            solve_dtype=solve_dtype,
        )

    @property
    def terminals(self) -> Dict[str, Polygon]:
        if self.drain_terminal is None:
            return {}
        return {t.name: t for t in self.source_terminals + [self.drain_terminal]}

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

    @property
    def film(self) -> Layer:
        return self._films_list[0]

    @property
    def mesh(self) -> Mesh:
        return self.meshes[self._films_list[0].name]

    def copy(
        self, with_mesh: bool = True, copy_mesh: bool = False
    ) -> "TransportDevice":
        """Copy this Device to create a new one.

        Args:
            with_mesh: Whether to shallow copy the ``meshes`` dictionary.
            copy_mesh: Whether to deepcopy the arrays defining the mesh.

        Returns:
            A new Device instance, copied from self
        """
        source_terminals = [term.copy() for term in self.source_terminals]
        drain_terminal = self.drain_terminal
        if drain_terminal is not None:
            drain_terminal = drain_terminal.copy()
        holes = [hole.copy() for hole in self.holes.values()]
        abstract_regions = [region.copy() for region in self.abstract_regions.values()]

        device = TransportDevice(
            self.name,
            layer=self.layer.copy(),
            film=self.film.copy(),
            source_terminals=source_terminals,
            drain_terminal=drain_terminal,
            holes=holes,
            abstract_regions=abstract_regions,
            length_units=self.length_units,
        )
        if with_mesh and self.meshes is not None:
            if copy_mesh:
                device.meshes = {
                    name: mesh.copy() for name, mesh in self.meshes.items()
                }
            else:
                device.meshes = self.meshes
        return device

    def make_mesh(
        self,
        min_points: Union[int, None] = None,
        max_edge_length: Union[float, None] = None,
        smooth: int = 0,
        **meshpy_kwargs,
    ) -> None:
        """Generates the triangular mesh for the film and stores it in ``self.mesh``.

        Args:
            min_points: Minimum number of vertices in the mesh. If None, then
                the number of vertices will be determined by meshpy_kwargs and the
                number of vertices in the underlying polygons.
            max_edge_length: The maximum distance between vertices in the resulting mesh.
            smooth: Number of Laplacian smoothing iterations to perform.
            **meshpy_kwargs: Passed to meshpy.triangle.build().
        """
        return super().make_mesh(
            min_points=min_points,
            max_edge_length=max_edge_length,
            smooth=smooth,
            **meshpy_kwargs,
        )

    def boundary_vertices(self) -> np.ndarray:
        """An array of boundary vertex indices, ordered counterclockwise.

        Returns:
            An array of indices for vertices that are on the device boundary,
            ordered counterclockwise.
        """
        mesh = self.mesh
        if mesh is None:
            return None
        points = mesh.sites
        triangles = mesh.elements
        indices = utils.boundary_vertices(points, triangles)
        indices_list = indices.tolist()
        # Ensure that the indices wrap around outside of any terminals.
        boundary = points[indices]
        for term in self.terminals.values():
            boundary = points[indices]
            term_ix = indices[term.contains_points(boundary)]
            discont = np.diff(term_ix) != 1
            if np.any(discont):
                i_discont = indices_list.index(term_ix[np.where(discont)[0][0]])
                indices = np.roll(indices, -(i_discont + 2))
                break
        return indices

    def _check_current_mapping(self, current_mapping: Dict[str, float]) -> None:
        source_names = {t.name for t in self.source_terminals}
        if not set(current_mapping).issubset(source_names):
            raise ValueError(
                "The current mapping must have one entry for every current terminal."
            )
