import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from matplotlib.path import Path


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculates the area of each triangle.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    xy = points[triangles]
    # s1 = xy[:, 2, :] - xy[:, 1, :]
    # s2 = xy[:, 0, :] - xy[:, 2, :]
    # s3 = xy[:, 1, :] - xy[:, 0, :]
    # which can be simplified to
    # s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]  # 3D
    s = xy[:, [2, 0]] - xy[:, [1, 2]]  # 2D
    a = np.linalg.det(s)
    return a * 0.5


def in_polygon(
    poly_points: np.ndarray,
    query_points: np.ndarray,
    radius: float = 0,
) -> Union[bool, np.ndarray]:
    """Returns a boolean array indicating which points in ``query_points``
    lie inside the polygon defined by ``poly_points``.

    Args:
        poly_points: Shape ``(m, 2)`` array of polygon vertex coordinates.
        query_points: Shape ``(n, 2)`` array of "query points".
        radius: See :meth:`matplotlib.path.Path.contains_points`.

    Returns:
        A shape ``(n, )`` boolean array indicating which ``query_points``
        lie inside the polygon. If only a single query point is given, then
        a single boolean value is returned.
    """
    query_points, poly_points = np.atleast_2d(query_points, poly_points)
    path = Path(poly_points)
    bool_array = path.contains_points(query_points, radius=radius).squeeze()
    if len(bool_array.shape) == 0:
        bool_array = bool_array.item()
    return bool_array


def centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns x, y coordinates for triangle centroids (centers of mass).

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.

    Returns:
        Shape (m, 2) array of triangle centroid (center of mass) coordinates
    """
    return points[triangles].sum(axis=1) / 3


def adjacency_matrix(
    triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, sp.csr_array]:
    """Computes the adjacency matrix for a given set of triangles.

    Args:
        triangles: Shape (m, 3) array of triangle indices
        sparse: Whether to return a sparse array or numpy ndarray.

    Returns:
        Shape (n, n) adjacency matrix, where n = triangles.max() + 1

    """
    # shape (m * 3, 2) array of graph edges
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    row, col = edges[:, 0], edges[:, 1]
    nrow, ncol = row.max() + 1, col.max() + 1
    data = np.ones_like(row, dtype=int)
    # This is the (data, (row_ind, col_ind)) format for csr_array,
    # meaning that adj[row_ind[k], col_ind[k]] = data[k]
    adj = sp.csr_array((data, (row, col)), shape=(nrow, ncol))
    # Undirected graph -> symmetric adjacency matrix
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    if sparse:
        return adj
    return adj.toarray()


def adj_directed_tri_indices(triangles: np.ndarray, num_sites: int) -> sp.csc_array:
    """Construct the directed adjacency matrix.

    Each element (i, j) represents an edge in the mesh, and the value at (i, j)
    is 1 + the index of a triangle containing that edge.

    Args:
        triangles: The triangle indices, shape ``(m, 3)``
        num_sites: The number of sites in the mesh

    Returns:
        A directed adjacency matrix containing triangle indices + 1
    """
    t0 = triangles[:, 0]
    t1 = triangles[:, 1]
    t2 = triangles[:, 2]
    i = np.column_stack([t0, t1, t2]).ravel()
    j = np.column_stack([t1, t2, t0]).ravel()
    # store triangle index + 1 (zero means no edge connecting i and j)
    data = np.repeat(np.arange(1, triangles.shape[0] + 1), 3)
    return sp.csc_array((data, (i, j)), shape=(num_sites, num_sites))


def weights_inv_euclidean(
    points: np.ndarray, triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, sp.lil_array]:
    """Weights edges by the inverse Euclidean distance of the edge lengths.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py
    N = points.shape[0]
    if sparse:
        # Use lil_array for operations that change matrix sparsity
        weights = sp.lil_array((N, N), dtype=float)
    else:
        weights = np.zeros((N, N), dtype=float)

    # Compute the three vectors of each triangle and their norms
    vec10 = points[triangles[:, 1]] - points[triangles[:, 0]]
    vec20 = points[triangles[:, 2]] - points[triangles[:, 0]]
    vec21 = points[triangles[:, 2]] - points[triangles[:, 1]]
    n10 = la.norm(vec10, axis=1)
    n20 = la.norm(vec20, axis=1)
    n21 = la.norm(vec21, axis=1)
    # Fill in the weight matrix
    weights[triangles[:, 0], triangles[:, 1]] = 1 / n10
    weights[triangles[:, 1], triangles[:, 0]] = 1 / n10
    weights[triangles[:, 0], triangles[:, 2]] = 1 / n20
    weights[triangles[:, 2], triangles[:, 0]] = 1 / n20
    weights[triangles[:, 2], triangles[:, 1]] = 1 / n21
    weights[triangles[:, 1], triangles[:, 2]] = 1 / n21

    return weights


def weights_half_cotangent(
    points: np.ndarray, triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, sp.lil_array]:
    """Weights edges by half of the cotangent of the two angles opposite the edge.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py
    N = points.shape[0]
    if sparse:
        # Use lil_array for operations that change matrix sparsity
        weights = sp.lil_array((N, N), dtype=float)
    else:
        weights = np.zeros((N, N), dtype=float)

    # First vertex
    vec1 = points[triangles[:, 1]] - points[triangles[:, 0]]
    vec2 = points[triangles[:, 2]] - points[triangles[:, 0]]
    w = 0.5 / np.tan(
        np.arccos(
            np.sum(vec1 * vec2, axis=1)
            / (la.norm(vec1, axis=1) * la.norm(vec2, axis=1))
        )
    )
    weights[triangles[:, 1], triangles[:, 2]] += w
    weights[triangles[:, 2], triangles[:, 1]] += w

    # Second vertex
    vec1 = points[triangles[:, 0]] - points[triangles[:, 1]]
    vec2 = points[triangles[:, 2]] - points[triangles[:, 1]]
    w = 0.5 / np.tan(
        np.arccos(
            np.sum(vec1 * vec2, axis=1)
            / (la.norm(vec1, axis=1) * la.norm(vec2, axis=1))
        )
    )
    weights[triangles[:, 0], triangles[:, 2]] += w
    weights[triangles[:, 2], triangles[:, 0]] += w

    # Third vertex
    vec1 = points[triangles[:, 0]] - points[triangles[:, 2]]
    vec2 = points[triangles[:, 1]] - points[triangles[:, 2]]
    w = 0.5 / np.tan(
        np.arccos(
            np.sum(vec1 * vec2, axis=1)
            / (la.norm(vec1, axis=1) * la.norm(vec2, axis=1))
        )
    )
    weights[triangles[:, 0], triangles[:, 1]] += w
    weights[triangles[:, 1], triangles[:, 0]] += w

    return weights


def calculate_weights(
    points: np.ndarray,
    triangles: np.ndarray,
    method: str,
    sparse: bool = True,
) -> Union[np.ndarray, sp.csr_array]:
    """Returns the weight matrix, calculated using the specified method.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        method: Method for calculating the weights. One of: "uniform",
            "inv_euclidean", or "half_cotangent".
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    method = method.lower()
    if method == "uniform":
        # Uniform weights are just the adjacency matrix.
        return adjacency_matrix(triangles, sparse=sparse).astype(float)
    if method == "inv_euclidean":
        return weights_inv_euclidean(points, triangles, sparse=sparse)
    if method == "half_cotangent":
        return weights_half_cotangent(points, triangles, sparse=sparse)
    raise ValueError(
        f"Unknown method ({method}). "
        f"Supported methods are 'uniform', 'inv_euclidean', and 'half_cotangent'."
    )


def laplace_operator(
    points: np.ndarray,
    triangles: np.ndarray,
    masses: Optional[np.ndarray] = None,
    weight_method: Literal[
        "uniform", "half_cotangent", "inv_euclidean"
    ] = "half_cotangent",
) -> sp.csr_array:
    """Laplacian operator for the mesh (sometimes called
    Laplace-Beltrami operator).

    The Laplacian operator is defined as ``inv(M) @ L``,
    where M is the mass matrix and L is the Laplacian matrix.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        masses: Pre-computed mass matrix, i.e., the vertex areas.
        weight_method: Method for calculating the weights. One of: "uniform",
            "inv_euclidean", or "half_cotangent".

    Returns:
        Shape (n, n) Laplacian operator
    """
    # See: http://rodolphe-vaillant.fr/?e=20
    # See: http://ddg.cs.columbia.edu/SGP2014/LaplaceBeltrami.pdf
    if masses is None:
        from .device.utils import vertex_areas

        masses = vertex_areas(points, triangles)
    L = calculate_weights(points, triangles, weight_method, sparse=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Changing the sparsity structure")
        L.setdiag(0)
        L.setdiag(-L.sum(axis=1))
        L = L.tocsr()
    laplacian = sp.diags(1 / masses, format="csr") @ L
    return laplacian


def gradient_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    areas: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_array, sp.csr_array]:
    """Returns the triangle gradient operators ``Gx`` and ``Gy``.

    Given a mesh with ``n`` vertices and ``m`` triangles and a scalar field ``f``
    defined at the mesh vertices, ``Gx`` and ``Gy`` are shape ``(m, n)`` matrices
    such that ``Gx @ f`` and ``Gy @ f`` compute the gradients of ``f`` along
    x and y, evaluated at the triangle centroids.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices
        _areas: Shape (m, ) array of triangle areas

    Returns:
        x and y gradient matrices, both of which have shape ``(m, n)``
    """
    if areas is None:
        areas = triangle_areas(points, triangles)
    # Shape (triangles.shape[0], 3, 2)
    xy = points[triangles]
    edges = np.roll(xy, 2, axis=1) - np.roll(xy, 1, axis=1)
    # Rotate edges clockwise by 90 degrees:
    # +x --> -y, +y --> +x
    edges_rot = np.empty_like(edges)
    edges_rot[:, :, 0] = +edges[:, :, 1]
    edges_rot[:, :, 1] = -edges[:, :, 0]

    tri_data = edges_rot / (2 * areas[:, np.newaxis, np.newaxis])
    tri_data = tri_data.reshape(-1, 2).T
    shape = (triangles.shape[0], points.shape[0])
    # Row indices: [0, 0, 0, 1, 1, 1, ...]
    row_ind = np.array([[i] * 3 for i in range(len(triangles))]).ravel()
    # Column indices: [t[0,0], t[0,1], t[0,2], t[1,0], t[1,1], t[1,2], ...]
    col_ind = triangles.ravel()
    Gx = sp.csr_array(
        (tri_data[0], (row_ind, col_ind)),
        shape=shape,
        dtype=float,
    )
    Gy = sp.csr_array(
        (tri_data[1], (row_ind, col_ind)),
        shape=shape,
        dtype=float,
    )
    return Gx, Gy


def gradient_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    areas: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_array, sp.csr_array]:
    """Returns the vertex gradient operators ``gx`` and ``gy``.

    Given a mesh with ``n`` vertices and ``m`` triangles and a scalar field ``f``
    defined at the mesh vertices, ``gx`` and ``gy`` are shape ``(n, n)`` matrices
    such that ``gx @ f`` and ``gy @ f`` compute the gradients of ``f`` along
    x and y, evaluated at the vertices.

    The vertex gradient operators are calculated by averaging the the triangle
    gradient operators over all triangles adjacent to each vertex.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices
        areas: Shape (m, ) array of triangle areas

    Returns:
        x and y gradient matrices, both of which have shape ``(n, n)``
    """
    if areas is None:
        areas = triangle_areas(points, triangles)
    n = len(points)
    Gx, Gy = gradient_triangles(points, triangles, areas=areas)
    Gx = Gx.tolil()
    Gy = Gy.tolil()
    gx = sp.lil_array((n, n), dtype=float)
    gy = sp.lil_array((n, n), dtype=float)
    # This loop is difficult to vectorize because different vertices
    # have different numbers of adjacent triangles.
    adj_tri = adj_directed_tri_indices(triangles, n).tolil()
    for i in range(n):
        # Triangles adjacent to site i
        adj = np.array(adj_tri.data[i]) - 1
        # Weight each triangle adjacent to vertex i by its angle at the vertex.
        vec1 = points[triangles[adj, 1]] - points[triangles[adj, 0]]
        vec2 = points[triangles[adj, 2]] - points[triangles[adj, 0]]
        weights = np.arccos(
            np.einsum("ij, ij -> i", vec1, vec2)
            / (la.norm(vec1, axis=1) * la.norm(vec2, axis=1))
        )
        weights /= weights.sum()
        gx[i, :] = np.einsum("i, ij -> j", weights, Gx[adj, :].toarray())
        gy[i, :] = np.einsum("i, ij -> j", weights, Gy[adj, :].toarray())
    return gx.asformat("csr"), gy.asformat("csr")


# def gradient_edges(
#     points: np.ndarray,
#     edges: np.ndarray,
#     edge_lengths: np.ndarray,
# ) -> sp.csr_array:
#     """Build the gradient for a function living on the sites onto the edges.

#     Args:
#         points: Mesh vertex positions
#         edges: Mesh edge indices.
#         edge_lengths: Mesh edge lengths.

#     Returns:
#         The gradient matrix.
#     """
#     edge_indices = np.arange(len(edges))
#     weights = 1 / edge_lengths
#     rows = np.concatenate([edge_indices, edge_indices])
#     cols = np.concatenate([edges[:, 1], edges[:, 0]])
#     values = np.concatenate([weights, -weights])
#     return sp.csr_array((values, (rows, cols)), shape=(len(edges), len(points)))
