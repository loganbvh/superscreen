# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

from typing import Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from matplotlib.path import Path


def in_polygon(
    xq: Union[float, np.ndarray],
    yq: Union[float, np.ndarray],
    xp: Union[float, np.ndarray],
    yp: Union[float, np.ndarray],
) -> Union[bool, np.ndarray]:
    """Returns a boolean array of the same shape as ``xq`` and ``yq`` indicating
    whether each point ``xq[i], yq[i]`` lies within the polygon defined by
    ``xp`` and ``yp``. If ``xq`` and ``yq`` are scalars,
    then a single boolean is returned.

    Args:
        xq: x-coordinates of "query" points
        yq: y-coordinates of "query" points
        xp: x-coordinates of polygon
        yp: y-coordinates of polygon
    """
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    shape = xq.shape
    xq = np.asarray(xq).reshape(-1)
    yq = np.asarray(yq).reshape(-1)
    xp = np.asarray(xp).reshape(-1)
    yp = np.asarray(yp).reshape(-1)
    q = list(zip(xq, yq))
    p = Path(list(zip(xp, yp)))
    bool_array = p.contains_points(q).reshape(shape).squeeze()
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
        triangles: Shape (m, 3) array of triangles indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    a = np.zeros(triangles.shape[0], dtype=float)
    for i, t in enumerate(triangles):
        xy = points[t]
        # s1 = xy[2, :] - xy[1, :]
        # s2 = xy[0, :] - xy[2, :]
        # s3 = xy[1, :] - xy[0, :]
        # which can be simplified to
        # s = xy[[2, 0, 1]] - xy[[1, 2, 0]]  # 3D
        s = xy[[2, 0]] - xy[[1, 2]]  # 2D
        # a should be positive if triangles are CCW arranged
        a[i] = la.det(s)
    return a * 0.5


def adjacency_matrix(
    triangles: np.ndarray, sparse: bool = False
) -> Union[np.ndarray, sp.csr_matrix]:
    """Computes the adjacency matrix for a given set of triangles.

    Args:
        triangles: Shape (m, 3) array of triangles indices
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) adjacency matrix, where n = triangles.max() + 1

    """
    A = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    # This is the (data, (row_ind, col_ind)) format for csr_matrix,
    # meaning that adj[row_ind[k], col_ind[k]] = data[k]
    adj = sp.csr_matrix((np.ones(A.shape[0]), (A[:, 0], A[:, 1])))
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    if sparse:
        return adj
    return adj.toarray()


def calculcate_weights(
    points: np.ndarray,
    triangles: np.ndarray,
    method: str,
    sparse: bool = False,
) -> Union[np.ndarray, sp.csr_matrix]:
    """Returns the weight matrix, calculated using the specified method.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangle indices.
        method: "uniform", "inv_euclidean", or "half_cotangent".
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    method = method.lower()
    if method == "uniform":
        # Uniform weights are just the adjacency matrix.
        adj = adjacency_matrix(triangles, sparse=sparse)
        weights = adj.astype(float)
    elif method == "inv_euclidean":
        weights = weights_inv_euclidean(points, triangles, sparse=sparse)
    elif method == "half_cotangent":
        weights = weights_half_cotangent(points, triangles, sparse=sparse)
    else:
        raise ValueError(
            f"Unknown method ({method}). "
            f"Supported methods are 'uniform', 'inv_euclidean', and 'half_cotangent'."
        )
    # normalize row-by-row
    if sp.issparse(weights):
        # weights / weights.sum(axis=1) returns np.matrix,
        # so convert back to lil.
        weights = sp.lil_matrix(weights / weights.sum(axis=1))
        weights.setdiag(1.0)
        weights = weights.tocsr()
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum(axis=1)[:, np.newaxis]
        np.fill_diagonal(weights, 1.0)
    return weights


def weights_inv_euclidean(
    points: np.ndarray, triangles: np.ndarray, sparse: bool = False
) -> Union[np.ndarray, sp.lil_matrix]:
    """Weights edges by the inverse Euclidean distance of the edge lengths.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangles indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py

    if sparse:
        # Use lil_matrix for operations that change matrix sparsity
        weights = sp.lil_matrix((points.shape[0], points.shape[0]), dtype=float)
    else:
        weights = np.zeros((points.shape[0], points.shape[0]), dtype=float)

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
    points: np.ndarray, triangles: np.ndarray, sparse: bool = False
) -> Union[np.ndarray, sp.lil_matrix]:
    """Weights edges by half of the cotangent of the two angles opposite the edge.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangles indices.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) matrix of vertex weights
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py

    weights = np.zeros((points.shape[0], points.shape[0]), dtype=float)

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

    if sparse:
        # Convert to sparse after building the matrix (it is faster in this case).
        weights = sp.lil_matrix(weights)

    return weights


def mass_matrix(
    points: np.ndarray,
    triangles: np.ndarray,
    diagonal: bool = True,
    sparse: bool = False,
) -> Union[np.ndarray, sp.csc_matrix]:
    """The mass matrix defines an effective area for each vertex.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangles indices.
        diagonal: Whether to return a diagonal mass matrix.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) mass matrix
    """
    # Adapted from spharaphy.TriMesh:
    # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
    # https://gitlab.com/uwegra/spharapy/-/blob/master/spharapy/trimesh.py

    if sparse:
        mass = sp.lil_matrix((points.shape[0], points.shape[0]), dtype=float)
    else:
        mass = np.zeros((points.shape[0], points.shape[0]), dtype=float)

    tri_areas = areas(points, triangles)

    if diagonal:
        for a, t in zip(tri_areas / 3, triangles):
            mass[t[0], t[0]] += a
            mass[t[1], t[1]] += a
            mass[t[2], t[2]] += a
    else:
        for a, t in zip(tri_areas / 6, triangles):
            # Add A / 12 to every edge in t
            a2 = a / 2
            mass[t[0], t[1]] += a2
            mass[t[1], t[0]] = mass[t[0], t[1]]
            mass[t[0], t[2]] += a2
            mass[t[2], t[0]] = mass[t[0], t[2]]
            mass[t[1], t[2]] += a2
            mass[t[2], t[1]] = mass[t[1], t[2]]

            # Add A / 6 to every point in t
            mass[t[0], t[0]] += a
            mass[t[1], t[1]] += a
            mass[t[2], t[2]] += a

    if sparse:
        # Use csc_matrix because we will eventually invert the mass matrix,
        # and csc is efficient for inversion.
        mass = mass.tocsc()

    return mass


def laplacian_operator(
    points: Union[np.ndarray, sp.csr_matrix],
    triangles: Union[np.ndarray, sp.csr_matrix],
    weights: Union[np.ndarray, sp.csr_matrix],
    sparse: bool = False,
) -> Union[np.ndarray, sp.csr_matrix]:
    """Laplacian operator for the mesh (sometimes called
    Laplace-Beltrami operator).

    The Laplacian operator is defined as ``inv(M) @ L``,
    where M is the mass matrix and L is the Laplacian matrix.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices.
        triangles: Shape (m, 3) array of triangles indices.
        weights: Shape (n, n) array of vertex weights.
        sparse: Whether to return a sparse matrix or numpy ndarray.

    Returns:
        Shape (n, n) Laplacian operator
    """
    # See: http://rodolphe-vaillant.fr/?e=20
    # See: http://ddg.cs.columbia.edu/SGP2014/LaplaceBeltrami.pdf
    M = mass_matrix(points, triangles, sparse=sparse)
    if sparse:
        if not sp.isspmatrix_csc(M):
            M = M.tocsc()
        if not sp.isspmatrix_csr(weights):
            weights = sp.csr_matrix(weights)
        L = weights - sp.diags(np.asarray(weights.sum(axis=1)).squeeze(), format="csr")
        # * is matrix multiplication for sparse matrices
        Del2 = (sp.linalg.inv(M) * L).tocsr()
    else:
        if sp.issparse(weights):
            weights = weights.toarray()
        L = weights - np.diag(weights.sum(axis=1))
        Del2 = la.inv(M) @ L
    return Del2
