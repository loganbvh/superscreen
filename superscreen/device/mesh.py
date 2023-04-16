import itertools
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from ..fem import cdist_batched, gradient_edges, gradient_vertices, laplace_operator
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
        areas: The areas corresponding to the sites.
        edge_mesh: The edge mesh.
    """

    def __init__(
        self,
        sites: Sequence[Tuple[float, float]],
        elements: Sequence[Tuple[int, int, int]],
        boundary_indices: Sequence[int],
        vertex_areas: Optional[Sequence[float]] = None,
        triangle_areas: Optional[Sequence[float]] = None,
        edge_mesh: Optional[EdgeMesh] = None,
        build_operators: bool = True,
    ):
        self.sites = np.asarray(sites).squeeze()
        # Setting dtype to int64 is important when running on Windows.
        # Using default dtype uint64 does not work as Scipy indices in some
        # instances.
        self.elements = np.asarray(elements, dtype=np.int64)
        self.boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
        if vertex_areas is not None:
            vertex_areas = np.asarray(vertex_areas)
        self.vertex_areas = vertex_areas
        if triangle_areas is not None:
            triangle_areas = np.asarray(triangle_areas)
        self.triangle_areas = triangle_areas
        self.edge_mesh = edge_mesh
        if not build_operators or self.edge_mesh is None:
            self.operators = None
        else:
            self.operators = MeshOperators(self)

    def stats(self) -> Dict[str, Union[int, float]]:
        """Returns a dictionary of information about the mesh."""
        edge_lengths = None
        if self.edge_mesh is not None:
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
        edge_mesh = triangle_areas = vertex_areas = None
        if build_operators:
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
        return np.unique(boundary_edges.flatten())

    def smooth(self, iterations: int) -> "Mesh":
        """Perform Laplacian smoothing of the mesh, i.e., moving each interior vertex
        to the arithmetic average of its neighboring points.

        Args:
            iterations: The number of smoothing iterations to perform.

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
                new_sites, elements, build_operators=(i == (iterations - 1))
            )
        return mesh

    def plot(
        self,
        ax: Union[plt.Axes, None] = None,
        show_sites: bool = True,
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
            marker: The marker to use for the mesh sites and Voronoi centroids.

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

    def to_dict(
        self,
    ) -> Dict[str, Union[np.ndarray, Dict[str, Union[np.ndarray, sp.spmatrix]]]]:
        mesh_as_dict = dict(
            sites=self.sites,
            elements=self.elements,
            boundary_indices=self.boundary_indices,
            vertex_areas=self.vertex_areas,
            triangle_areas=self.triangle_areas,
            edge_mesh=self.edge_mesh.to_dict(),
        )
        if self.operators is not None:
            mesh_as_dict["operators"] = self.operators.to_dict()
        return mesh_as_dict

    @staticmethod
    def from_dict(
        mesh_as_dict: Dict[
            str, Union[np.ndarray, Dict[str, Union[np.ndarray, sp.spmatrix]]]
        ]
    ) -> "Mesh":
        mesh_as_dict = mesh_as_dict.copy()
        operators_dict = mesh_as_dict.pop("operators", None)
        edge_mesh_dict = mesh_as_dict.pop("edge_mesh")
        mesh_as_dict["edge_mesh"] = EdgeMesh(**edge_mesh_dict)
        mesh = Mesh(**mesh_as_dict)
        if operators_dict:
            mesh.operators = MeshOperators(**operators_dict)
        return mesh


class MeshOperators:
    def __init__(self, mesh: Mesh):
        sites = mesh.sites
        elements = mesh.elements
        self.weights = mesh.vertex_areas
        self.Q = MeshOperators.Q_matrix(
            MeshOperators.q_matrix(sites),
            MeshOperators.C_vector(sites),
            self.weights,
        )
        self.gradient_x, self.gradient_y = gradient_vertices(
            sites, elements, triangle_areas=mesh.triangle_areas
        )
        self.gradient_edges = gradient_edges(
            sites, mesh.edge_mesh.edges, mesh.edge_mesh.edge_lengths
        )
        self.laplacian = laplace_operator(sites, elements, self.weights)

    def to_dict(self) -> Dict[str, Union[np.ndarray, sp.spmatrix]]:
        return dict(
            weights=self.weights,
            Q=self.Q,
            gradient_x=self.gradient_x,
            gradient_y=self.gradient_y,
            gradient_edges=self.gradient_edges,
            laplacian=self.laplacian,
        )

    def copy(self) -> "MeshOperators":
        return deepcopy(self)

    @staticmethod
    def q_matrix(
        points: np.ndarray,
        dtype: Optional[Union[str, np.dtype]] = None,
        batch_size: int = 100,
    ) -> np.ndarray:
        """Computes the denominator matrix, q:

        .. math::

            q_{ij} = \\frac{1}{4\\pi|\\vec{r}_i-\\vec{r}_j|^3}

        See Eq. 7 in [Brandt-PRB-2005]_, Eq. 8 in [Kirtley-RSI-2016]_,
        and Eq. 8 in [Kirtley-SST-2016]_.

        Args:
            points: Shape (n, 2) array of x,y coordinates of vertices.
            dtype: Output dtype.
            batch_size: Size of batches in which to compute the distance matrix.

        Returns:
            Shape (n, n) array, qij
        """
        # Euclidean distance between points
        distances = cdist_batched(
            points, points, batch_size=batch_size, metric="euclidean"
        )
        if dtype is not None:
            distances = distances.astype(dtype, copy=False)
        with np.errstate(divide="ignore"):
            q = 1 / (4 * np.pi * distances**3)
        np.fill_diagonal(q, np.inf)
        return q.astype(dtype, copy=False)

    @staticmethod
    def C_vector(
        points: np.ndarray,
        dtype: Optional[Union[str, np.dtype]] = None,
    ) -> np.ndarray:
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
            dtype: Output dtype.

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
        if dtype is not None:
            C = C.astype(dtype, copy=False)
        return C

    @staticmethod
    def Q_matrix(
        q: np.ndarray,
        C: np.ndarray,
        weights: np.ndarray,
        dtype: Optional[Union[str, np.dtype]] = None,
    ) -> np.ndarray:
        """Computes the kernel matrix, Q:

        .. math::

            Q_{ij} = (\\delta_{ij}-1)q_{ij}
            + \\delta_{ij}\\frac{1}{w_{ij}}\\left(C_i
            + \\sum_{l\\neq i}q_{il}w_{il}\\right)

        See Eq. 10 in [Brandt-PRB-2005]_, Eq. 11 in [Kirtley-RSI-2016]_,
        and Eq. 11 in [Kirtley-SST-2016]_.

        Args:
            q: Shape (n, n) matrix qij.
            C: Shape (n, ) vector Ci.
            weights: Shape (n, ) weight vector.
            dtype: Output dtype.

        Returns:
            Shape (n, n) array, Qij
        """
        if sp.issparse(weights):
            weights = weights.diagonal()
        # q[i, i] are np.inf, but Q[i, i] involves a sum over only the
        # off-diagonal elements of q, so we can just set q[i, i] = 0 here.
        q = q.copy()
        np.fill_diagonal(q, 0)
        Q = -q
        np.fill_diagonal(Q, (C + np.einsum("ij, j -> i", q, weights)) / weights)
        if dtype is not None:
            Q = Q.astype(dtype, copy=False)
        return Q
