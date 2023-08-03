import itertools
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.tri import Triangulation

from ..distance import q_matrix
from ..fem import gradient_vertices, laplace_operator
from . import utils
from .edge_mesh import EdgeMesh


class Mesh:
    """A triangular mesh of a simply- or multiply-connected polygon.

    .. tip::

        Use :meth:`Mesh.from_triangulation` to create a new mesh from a triangulation.

    Args:
        sites: The (x, y) coordinates of the mesh vertices.
        elements: A list of triplets that correspond to the indices of he vertices that
            form a triangle. [[0, 1, 2], [0, 1, 3]] corresponds to a triangle
            connecting vertices 0, 1, and 2 and another triangle connecting vertices
            0, 1, and 3.
        boundary_indices: Indices corresponding to the boundary.
        vertex_areas: The areas corresponding to the sites or vertices.
        triangle_areas: The areas of the triangular mesh elements.
        edge_mesh: The edge mesh.
        build_operators: Whether to build the :class:`superscreen.device.MeshOperators`
            for the mesh.
    """

    def __init__(
        self,
        sites: Sequence[Tuple[float, float]],
        elements: Sequence[Tuple[int, int, int]],
        boundary_indices: Sequence[int],
        vertex_areas: Sequence[float],
        triangle_areas: Sequence[float],
        edge_mesh: EdgeMesh,
        build_operators: bool = True,
    ):
        self.sites = np.asarray(sites).squeeze()
        self.elements = np.asarray(elements, dtype=np.int64)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        self.vertex_areas = np.asarray(vertex_areas)
        self.triangle_areas = np.asarray(triangle_areas)
        self.edge_mesh = edge_mesh
        self.operators: Optional[MeshOperators] = None
        self._triangulation: Optional[Triangulation] = None
        if build_operators:
            self.operators = MeshOperators.from_mesh(self)

    @property
    def triangulation(self) -> Triangulation:
        """Matplotlib triangulation of the mesh."""
        if self._triangulation is None:
            self._triangulation = Triangulation(
                self.sites[:, 0], self.sites[:, 1], self.elements
            )
        return self._triangulation

    def stats(self) -> Dict[str, Union[int, float]]:
        """Returns a dictionary of information about the mesh."""
        edge_lengths = self.edge_mesh.edge_lengths
        vertex_areas = self.vertex_areas

        def _min(arr):
            if arr is not None:
                return arr.min()

        def _max(arr):
            if arr is not None:
                return arr.max()

        # def _mean(arr):
        #     if arr is not None:
        #         return arr.mean()

        return dict(
            num_sites=len(self.sites),
            num_elements=len(self.elements),
            min_edge_length=_min(edge_lengths),
            max_edge_length=_max(edge_lengths),
            # mean_edge_length=_mean(edge_lengths),
            min_vertex_area=_min(vertex_areas),
            max_vertex_area=_max(vertex_areas),
            # mean_vertex_area=_mean(vertex_areas),
        )

    def closest_site(self, xy: Tuple[float, float]) -> int:
        """Returns the index of the mesh site closest to ``(x, y)``.

        Args:
            xy: A shape ``(2, )`` or ``(2, 1)`` sequence of floats, ``(x, y)``.

        Returns:
            The index of the mesh site closest to ``(x, y)``.
        """
        return np.argmin(np.linalg.norm(self.sites - np.atleast_2d(xy), axis=1))

    @staticmethod
    def from_triangulation(
        sites: Sequence[Tuple[float, float]],
        elements: Sequence[Tuple[int, int, int]],
        build_operators: bool = True,
    ) -> "Mesh":
        """Create a triangular mesh from the coordinates of the triangle vertices
        and a list of indices corresponding to the vertices that connect to triangles.

        Args:
            sites: The (x, y) coordinates of the mesh sites.
            elements: A list of triplets that correspond to the indices of the vertices
                that form a triangle.   E.g. [[0, 1, 2], [0, 1, 3]] corresponds to a
                triangle connecting vertices 0, 1, and 2 and another triangle
                connecting vertices 0, 1, and 3.
            build_operators: Whether to build the :class:`superscreen.device.MeshOperators`
                for the mesh.

        Returns:
            A new :class:`tdgl.finite_volume.Mesh` instance
        """
        sites = np.asarray(sites).squeeze()
        elements = np.asarray(elements).squeeze()
        if sites.ndim != 2 or sites.shape[1] != 2:
            raise ValueError(
                f"The site coordinates must have shape (n, 2), got {sites.shape!r}"
            )
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError(
                f"The elements must have shape (m, 3), got {elements.shape!r}."
            )
        boundary_indices = Mesh.find_boundary_indices(elements)
        edge_mesh = EdgeMesh.from_mesh(sites, elements)
        triangle_areas = utils.triangle_areas(sites, elements)
        vertex_areas = utils.vertex_areas(sites, elements, tri_areas=triangle_areas)
        return Mesh(
            sites=sites,
            elements=elements,
            boundary_indices=boundary_indices,
            edge_mesh=edge_mesh,
            vertex_areas=vertex_areas,
            triangle_areas=triangle_areas,
            build_operators=build_operators,
        )

    @staticmethod
    def find_boundary_indices(elements: np.ndarray) -> np.ndarray:
        """Find the boundary vertices.

        Args:
            elements: The triangular elements.

        Returns:
            An array of site indices corresponding to the boundary.
        """
        edges, is_boundary = utils.get_edges(elements)
        # Get the boundary edges and all boundary points
        boundary_edges = edges[is_boundary]
        return np.unique(boundary_edges.ravel())

    def smooth(self, iterations: int, build_operators: bool = True) -> "Mesh":
        """Perform Laplacian smoothing of the mesh, i.e., moving each interior vertex
        to the arithmetic average of its neighboring points.

        Args:
            iterations: The number of smoothing iterations to perform.
            build_operators: Whether to build the :class:`superscreen.device.MeshOperators`
                for the mesh.

        Returns:
            A new Mesh with relaxed vertex positions.
        """
        mesh = self
        elements = mesh.elements
        edges, _ = utils.get_edges(elements)
        n = len(mesh.sites)
        shape = (n, 2)
        boundary = mesh.boundary_indices
        for i in range(iterations):
            sites = mesh.sites
            num_neighbors = np.bincount(edges.ravel(), minlength=shape[0])

            new_sites = np.zeros(shape)
            vals = sites[edges[:, 1]].T
            new_sites += np.array(
                [np.bincount(edges[:, 0], val, minlength=n) for val in vals]
            ).T
            vals = sites[edges[:, 0]].T
            new_sites += np.array(
                [np.bincount(edges[:, 1], val, minlength=n) for val in vals]
            ).T
            new_sites /= num_neighbors[:, np.newaxis]
            # reset boundary points
            new_sites[boundary] = sites[boundary]
            mesh = Mesh.from_triangulation(
                new_sites,
                elements,
                build_operators=(build_operators and (i == (iterations - 1))),
            )
        return mesh

    def plot(
        self,
        ax: Union[plt.Axes, None] = None,
        show_sites: bool = False,
        show_edges: bool = True,
        site_color: Union[str, Sequence[float], None] = None,
        edge_color: Union[str, Sequence[float], None] = "k",
        linewidth: float = 0.75,
        linestyle: str = "-",
        marker: str = ".",
    ) -> plt.Axes:
        """Plot the mesh.

        Args:
            ax: A :class:`plt.Axes` instance on which to plot the mesh.
            show_sites: Whether to show the mesh sites.
            show_edges: Whether to show the mesh edges.
            site_color: The color for the sites.
            edge_color: The color for the edges.
            linewidth: The line width for all edges.
            linestyle: The line style for all edges.
            marker: The marker to use for the mesh sites.

        Returns:
            The resulting :class:`plt.Axes`
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.set_aspect("equal")
        x, y = self.sites.T
        tri = self.elements
        if show_edges:
            ax.triplot(x, y, tri, color=edge_color, ls=linestyle, lw=linewidth)
        if show_sites:
            ax.plot(x, y, marker=marker, ls="", color=site_color)
        return ax

    def to_hdf5(self, h5group: h5py.Group, compress: bool = True) -> None:
        """Save the mesh to a :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` into which to store the mesh.
            compress: If ``True``, store only the sites and elements.
        """
        h5group["sites"] = self.sites
        h5group["elements"] = self.elements
        if not compress:
            h5group["boundary_indices"] = self.boundary_indices
            h5group["vertex_areas"] = self.vertex_areas
            h5group["triangle_areas"] = self.triangle_areas
            self.edge_mesh.to_hdf5(h5group.create_group("edge_mesh"))

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "Mesh":
        """Load a mesh from an HDF5 file.

        Args:
            h5group: The HDF5 group to load the mesh from.

        Returns:
            The loaded mesh.
        """
        if not ("sites" in h5group and "elements" in h5group):
            raise IOError("Could not load mesh due to missing data.")

        if Mesh.is_restorable(h5group):
            return Mesh(
                sites=np.array(h5group["sites"]),
                elements=np.array(h5group["elements"], dtype=np.int64),
                boundary_indices=np.array(h5group["boundary_indices"], dtype=np.int64),
                vertex_areas=np.array(h5group["vertex_areas"]),
                triangle_areas=np.array(h5group["triangle_areas"]),
                edge_mesh=EdgeMesh.from_hdf5(h5group["edge_mesh"]),
            )
        # Recreate mesh from triangulation data if not all data is available
        return Mesh.from_triangulation(
            sites=np.array(h5group["sites"]).squeeze(),
            elements=np.array(h5group["elements"]),
        )

    @staticmethod
    def is_restorable(h5group: h5py.Group) -> bool:
        """Returns ``True`` if the :class:`h5py.Group` contains all of the data
        necessary to create a :class:`Mesh` without re-computing any values.

        Args:
            h5group: The :class:`h5py.Group` to check.

        Returns:
            Whether the mesh can be restored from the given group.
        """
        return (
            "sites" in h5group
            and "elements" in h5group
            and "boundary_indices" in h5group
            and "vertex_areas" in h5group
            and "triangle_areas" in h5group
            and "edge_mesh" in h5group
        )

    def copy(self) -> "Mesh":
        mesh = Mesh(
            sites=self.sites.copy(),
            elements=self.elements.copy(),
            boundary_indices=self.boundary_indices.copy(),
            vertex_areas=self.vertex_areas.copy(),
            triangle_areas=self.triangle_areas.copy(),
            edge_mesh=self.edge_mesh.copy(),
        )
        if self.operators is not None:
            mesh.operators = self.operators.copy()
        return mesh


class MeshOperators:
    """A container for the finite element operators for a :class:`superscreen.Mesh`.

    Args:
        weights: The mesh weights or effective vertex areas, shape ``(n, )``
        Q: The kernel matrix ``Q``, shape ``(n, n)``
        gradient_x: The x-gradient matrix, shape ``(n, n)``
        gradient_y: The y-gradient matrix, shape ``(n, n)``
        laplacian: The mesh Laplacian, shape ``(n, n)``
    """

    def __init__(
        self,
        *,
        weights: np.ndarray,
        Q: np.ndarray,
        gradient_x: sp.spmatrix,
        gradient_y: sp.spmatrix,
        laplacian: sp.spmatrix,
    ):
        self.weights = weights
        self.Q = Q
        self.gradient_x = gradient_x
        self.gradient_y = gradient_y
        self.laplacian = laplacian

    @staticmethod
    def from_mesh(mesh: Mesh) -> "MeshOperators":
        """Construct a :class:`superscreen.device.MeshOperators` instance
        from a :class:`superscreen.Mesh`.

        Args:
            mesh: The :class:`superscreen.Mesh`

        Returns:
            A new :class:`superscreen.device.MeshOperators` instance
        """
        sites = mesh.sites
        elements = mesh.elements
        weights = mesh.vertex_areas
        Q = MeshOperators.Q_matrix(sites, weights)
        gradient_x, gradient_y = gradient_vertices(
            sites, elements, areas=mesh.triangle_areas
        )
        # gradient_edges = gradient_edges(
        #     sites, mesh.edge_mesh.edges, mesh.edge_mesh.edge_lengths
        # )
        laplacian = laplace_operator(sites, elements, weights)
        return MeshOperators(
            weights=weights,
            Q=Q,
            gradient_x=gradient_x,
            gradient_y=gradient_y,
            laplacian=laplacian,
        )

    def copy(self) -> "MeshOperators":
        """Returns a deep copy."""
        return deepcopy(self)

    @staticmethod
    def C_vector(points: np.ndarray) -> np.ndarray:
        """Computes the edge vector, C:

        .. math::
            C_i &= \\frac{1}{4\\pi}\\sum_{p,q=\\pm1}\\sqrt{(\\Delta x - px_i)^{-2}
                + (\\Delta y - qy_i)^{-2}}\\\\
            \\Delta x &= \\frac{1}{2}(\\mathrm{max}(x) - \\mathrm{min}(x))\\\\
            \\Delta y &= \\frac{1}{2}(\\mathrm{max}(y) - \\mathrm{min}(y))

        See Eq. 12 in [Brandt-PRB-2005]_, Eq. 16 in [Kirtley-RSI-2016]_,
        and Eq. 15 in [Kirtley-SST-2016]_.

        Args:
            points: Shape (n, 2) array of x, y coordinates of vertices.

        Returns:
            Shape (n, ) array, Ci
        """
        x = points[:, 0]
        y = points[:, 1]
        x = x - x.mean()
        y = y - y.mean()
        a = np.ptp(x) / 2
        b = np.ptp(y) / 2
        with np.errstate(divide="ignore"):
            C = sum(
                np.sqrt((a - p * x) ** (-2) + (b - q * y) ** (-2))
                for p, q in itertools.product((-1, 1), repeat=2)
            )
        C[np.isinf(C)] = 1e30
        C /= 4 * np.pi
        return C

    @staticmethod
    def Q_matrix(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Computes the kernel matrix, Q:

        .. math::

            Q_{ij} = (\\delta_{ij}-1)q_{ij}
            + \\delta_{ij}\\frac{1}{w_{ij}}\\left(C_i
            + \\sum_{l\\neq i}q_{il}w_{il}\\right)

        See Eq. 10 in [Brandt-PRB-2005]_, Eq. 11 in [Kirtley-RSI-2016]_,
        and Eq. 11 in [Kirtley-SST-2016]_.

        Args:
            points: Shape ``(n, 2)`` array of mesh sites.
            weights: Shape ``(n, )`` weight vector.

        Returns:
            Shape (n, n) array, Qij
        """
        q = q_matrix(points)
        C = MeshOperators.C_vector(points)
        diag = -(C + np.einsum("ij, j -> i", q, weights)) / weights
        np.fill_diagonal(q, diag)
        return -q
