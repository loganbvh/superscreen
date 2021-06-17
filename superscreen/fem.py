from typing import Union

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from matplotlib.path import Path


def in_polygon(
    xq: Union[float, np.ndarray],
    yq: Union[float, np.ndarray],
    xp: Union[float, np.ndarray],
    yp: Union[float, np.ndarray],
) -> Union[bool, np.ndarray]:
    """Checks whether query points (xq, yq) lie in the polyon
    defined by points (xp, yp).
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
    """Returns x,y coordinates for triangle centroids."""
    return np.array([np.sum(points[t], axis=0) / 3 for t in triangles])


def adjacency_matrix(
    triangles: np.ndarray, sparse: bool = False
) -> Union[np.ndarray, csr_matrix]:
    """Computes the adjacency matrix for a given set of triangles."""
    A = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    adj = csr_matrix((np.ones(A.shape[0]), (A[:, 0], A[:, 1])))
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    if sparse:
        return adj
    return adj.toarray()


def area(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Calculate the area of each triangle.

    Args:
        points (np.ndarray): shape (n,2) array of x,y coordinates of nodes
        tris (np.ndarray) shape (m,3) array of triangles

    Returns:
        np.ndarray: shape (m,) array of triangle areas.
    """
    a = np.zeros(tris.shape[0])
    for i, e in enumerate(tris):
        xy = points[e]
        # s1 = xy[2, :] - xy[1, :]
        # s2 = xy[0, :] - xy[2, :]
        # s3 = xy[1, :] - xy[0, :]
        # which can be simplified to
        # s = xy[[2, 0, 1]] - xy[[1, 2, 0]]
        s = xy[[2, 0]] - xy[[1, 2]]
        # a should be positive if triangles are CCW arranged
        a[i] = la.det(s)
    return a * 0.5


def calculcate_weights(
    points: np.ndarray, triangles: np.ndarray, method: str
) -> np.ndarray:
    method = method.lower()
    if method == "uniform":
        adj = adjacency_matrix(triangles)
        weights = (adj != 0).astype(float)
    elif method == "inv_euclidean":
        weights = weights_inv_euclidean(points, triangles)
    elif method == "half_cotangent":
        weights = weights_half_cotangent(points, triangles)
    else:
        raise ValueError(
            f"Unknown method ({method}). "
            f"Supported methods are 'uniform', 'inv_euclidean', and 'half_cotangent'."
        )
    # normalize row-by-row
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(weights, 1.0)
    return weights


def weights_inv_euclidean(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    weights = np.zeros((points.shape[0], points.shape[0]), dtype=float)

    for t in triangles:
        # compute the three vectors of the triangle
        vec10 = points[t[1]] - points[t[0]]
        vec20 = points[t[2]] - points[t[0]]
        vec21 = points[t[2]] - points[t[1]]

        # fill in the weights in the weight matrix
        n10 = la.norm(vec10)
        n20 = la.norm(vec20)
        n21 = la.norm(vec21)
        weights[t[0], t[1]] = 1 / n10
        weights[t[1], t[0]] = 1 / n10
        weights[t[0], t[2]] = 1 / n20
        weights[t[2], t[0]] = 1 / n20
        weights[t[2], t[1]] = 1 / n21
        weights[t[1], t[2]] = 1 / n21
    return weights


def weights_half_cotangent(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    weights = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    for t in triangles:
        # compute the directional vectors at the 1st vertex
        vec1 = points[t[1]] - points[t[0]]
        vec2 = points[t[2]] - points[t[0]]

        # compute the weight of the edge 0.5 * cot
        w = 0.5 * (
            1.0
            / np.tan(np.arccos(np.dot(vec1, vec2) / (la.norm(vec1) * la.norm(vec2))))
        )

        weights[t[1], t[2]] += w
        weights[t[2], t[1]] += w

        # compute the directional vectors at the 2nd vertex
        vec1 = points[t[0]] - points[t[1]]
        vec2 = points[t[2]] - points[t[1]]

        # compute the weight of the edge 0.5 * cot
        w = 0.5 * (
            1.0
            / np.tan(np.arccos(np.dot(vec1, vec2) / (la.norm(vec1) * la.norm(vec2))))
        )

        weights[t[0], t[2]] += w
        weights[t[2], t[0]] += w

        # compute the directional vectors at the 3rd vertex
        vec1 = points[t[0]] - points[t[2]]
        vec2 = points[t[1]] - points[t[2]]

        # compute the weight of the edge 0.5 * cot
        w = 0.5 * (
            1.0
            / np.tan(np.arccos(np.dot(vec1, vec2) / (la.norm(vec1) * la.norm(vec2))))
        )

        weights[t[0], t[1]] += w
        weights[t[1], t[0]] += w
    return weights


def mass_matrix(points: np.ndarray, triangles: np.ndarray, diagonal: bool = True):

    mass = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    areas = area(points, triangles)

    if diagonal:
        for A, t in zip(areas, triangles):
            mass[t[0], t[0]] += A / 3
            mass[t[1], t[1]] += A / 3
            mass[t[2], t[2]] += A / 3
    else:
        for A, t in zip(areas, triangles):
            # Add A / 12 to every edge in t
            mass[t[0], t[1]] += A / 12
            mass[t[1], t[0]] = mass[t[0], t[1]]
            mass[t[0], t[2]] += A / 12
            mass[t[2], t[0]] = mass[t[0], t[2]]
            mass[t[1], t[2]] += A / 12
            mass[t[2], t[1]] = mass[t[1], t[2]]

            # Add A / 6 to every point in t
            mass[t[0], t[0]] += A / 6
            mass[t[1], t[1]] += A / 6
            mass[t[2], t[2]] += A / 6

    return mass


def laplacian_operator(
    points: np.ndarray, triangles: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    # Laplacian matrix
    L = weights - np.diag(weights.sum(axis=1))
    M = mass_matrix(points, triangles)
    # Laplacian operator
    Del2 = la.inv(M) @ L
    return Del2
