import warnings
from typing import Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from matplotlib.path import Path


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


def areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
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


def adjacency_matrix(
    triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, sp.csr_matrix]:
    """Computes the adjacency matrix for a given set of triangles.

    Args:
        triangles: Shape (m, 3) array of triangle indices
        sparse: Whether to return a sparse matrix or numpy ndarray.

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
    # This is the (data, (row_ind, col_ind)) format for csr_matrix,
    # meaning that adj[row_ind[k], col_ind[k]] = data[k]
    adj = sp.csr_matrix((data, (row, col)), shape=(nrow, ncol))
    # Undirected graph -> symmetric adjacency matrix
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    if sparse:
        return adj
    return adj.toarray()


def weights_inv_euclidean(
    points: np.ndarray, triangles: np.ndarray, sparse: bool = True
) -> Union[np.ndarray, sp.lil_matrix]:
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
        # Use lil_matrix for operations that change matrix sparsity
        weights = sp.lil_matrix((N, N), dtype=float)
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
) -> Union[np.ndarray, sp.lil_matrix]:
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
        # Use lil_matrix for operations that change matrix sparsity
        weights = sp.lil_matrix((N, N), dtype=float)
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
) -> Union[np.ndarray, sp.csr_matrix]:
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


def mass_matrix(
    points: np.ndarray,
    triangles: np.ndarray,
    sparse: bool = False,
) -> Union[np.ndarray, sp.csc_matrix]:
    """The mass matrix defines an effective area for each vertex.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) sparse mass matrix or shape (n,) vector of diagonals.
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py
    N = points.shape[0]
    if sparse:
        mass = sp.lil_matrix((N, N), dtype=float)
    else:
        mass = np.zeros((N, N), dtype=float)

    tri_areas = areas(points, triangles)

    for a, t in zip(tri_areas / 3, triangles):
        mass[t[0], t[0]] += a
        mass[t[1], t[1]] += a
        mass[t[2], t[2]] += a

    if sparse:
        # Use csc_matrix because we will eventually invert the mass matrix,
        # and csc is efficient for inversion.
        return mass.tocsc()
    return mass.diagonal()


def laplace_operator(
    points: np.ndarray,
    triangles: np.ndarray,
    masses: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    weight_method: str = "half_cotangent",
    sparse: bool = True,
) -> Union[np.ndarray, sp.csr_matrix]:
    """Laplacian operator for the mesh (sometimes called
    Laplace-Beltrami operator).

    The Laplacian operator is defined as ``inv(M) @ L``,
    where M is the mass matrix and L is the Laplacian matrix.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        masses: Pre-computed mass matrix: shape (n, n) sparse diagonal matrix
            or shape (n,) array of diagonals.
        weight_method: Method for calculating the weights. One of: "uniform",
            "inv_euclidean", or "half_cotangent".
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) Laplacian operator
    """
    # See: http://rodolphe-vaillant.fr/?e=20
    # See: http://ddg.cs.columbia.edu/SGP2014/LaplaceBeltrami.pdf
    if masses is None:
        masses = mass_matrix(points, triangles)
    if sp.issparse(masses):
        masses = masses.diagonal()
    L = calculate_weights(points, triangles, weight_method, sparse=True)
    with warnings.catch_warnings():
        # scipy.sparse throws a warning here
        warnings.filterwarnings("ignore", message="Changing the sparsity structure")
        L.setdiag(0)
        w_sum = np.atleast_2d(L.sum(axis=1))
        L.setdiag(-w_sum)
        L = L.tocsr()
    Del2 = sp.diags(1 / masses, format="csr") @ L
    if not sparse:
        Del2 = Del2.toarray()
    return Del2


def gradient_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_areas: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Returns the triangle gradient operators ``Gx`` and ``Gy``.

    Given a mesh with ``n`` vertices and ``m`` triangles and a scalar field ``f``
    defined at the mesh vertices, ``Gx`` and ``Gy`` are shape ``(m, n)`` matrices
    such that ``Gx @ f`` and ``Gy @ f`` compute the gradients of ``f`` along
    x and y, evaluated at the triangle centroids.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices
        triangle_areas: Shape (m, ) array of triangle areas

    Returns:
        x and y gradient matrices, both of which have shape ``(m, n)``
    """
    if triangle_areas is None:
        triangle_areas = areas(points, triangles)
    # Shape (triangles.shape[0], 3, 2)
    xy = points[triangles]
    edges = np.roll(xy, 2, axis=1) - np.roll(xy, 1, axis=1)
    # Rotate edges clockwise by 90 degrees:
    # +x --> -y, +y --> +x
    edges_rot = np.empty_like(edges)
    edges_rot[:, :, 0] = +edges[:, :, 1]
    edges_rot[:, :, 1] = -edges[:, :, 0]

    tri_data = edges_rot / (2 * triangle_areas[:, np.newaxis, np.newaxis])
    tri_data = tri_data.reshape(-1, 2).T
    shape = (triangles.shape[0], points.shape[0])
    # Row indices: [0, 0, 0, 1, 1, 1, ...]
    row_ind = np.array([[i] * 3 for i in range(len(triangles))]).ravel()
    # Column indices: [t[0,0], t[0,1], t[0,2], t[1,0], t[1,1], t[1,2], ...]
    col_ind = triangles.ravel()
    Gx = sp.csr_matrix(
        (tri_data[0], (row_ind, col_ind)),
        shape=shape,
        dtype=float,
    )
    Gy = sp.csr_matrix(
        (tri_data[1], (row_ind, col_ind)),
        shape=shape,
        dtype=float,
    )
    return Gx, Gy


def gradient_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    triangle_areas: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
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
        triangle_areas: Shape (m, ) array of triangle areas

    Returns:
        x and y gradient matrices, both of which have shape ``(n, n)``
    """
    if triangle_areas is None:
        triangle_areas = areas(points, triangles)
    n = points.shape[0]
    Gx, Gy = gradient_triangles(points, triangles, triangle_areas=triangle_areas)
    # Use numpy arrays for fast slicing even though the operators are sparse.
    Gx = Gx.toarray()
    Gy = Gy.toarray()
    gx = np.zeros((n, n), dtype=float)
    gy = np.zeros((n, n), dtype=float)
    # This loop is difficult to vectorize because different vertices
    # have different numbers of adjacent triangles.
    for i in range(n):
        adjacent_triangles, _ = np.where(np.isin(triangles, i))
        t = adjacent_triangles
        # Weight each triangle adjacent to vertex i by its angle at the vertex.
        vec1 = points[triangles[t, 1]] - points[triangles[t, 0]]
        vec2 = points[triangles[t, 2]] - points[triangles[t, 0]]
        weights = np.arccos(
            np.sum(vec1 * vec2, axis=1)
            / (la.norm(vec1, axis=1) * la.norm(vec2, axis=1))
        )
        weights /= weights.sum()
        assert (weights > 0).all()
        gx[i, :] = np.einsum("i, ij -> j", weights, Gx[t, :])
        gy[i, :] = np.einsum("i, ij -> j", weights, Gy[t, :])
    return sp.csr_matrix(gx), sp.csr_matrix(gy)
