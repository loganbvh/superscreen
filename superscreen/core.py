import logging
from re import L
import warnings
from typing import Type, Union, Callable, Optional, Dict, Tuple

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from matplotlib.path import Path
import matplotlib.tri as mtri

from .device import Device

logger = logging.getLogger(__name__)


class BrandtSolution(object):
    def __init__(
        self,
        *,
        device: Device,
        streams: Dict[str, np.ndarray],
        fields: Dict[str, np.ndarray],
        response_fields: Dict[str, np.ndarray],
        applied_field: Callable,
        circulating_currents: Optional[Dict[str, float]] = None,
    ):
        self.device = device
        self.streams = streams
        self.fields = fields
        self.response_fields = response_fields
        self.applied_field = applied_field
        self.circulating_currents = circulating_currents

    def grid_data(
        self,
        dataset: Optional[str] = "fields",
        grid_shape=Union[int, Tuple[int, int]],
        method: Optional[str] = "cubic",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        valid_data = ("streams", "fields", "response_fields")
        if dataset not in valid_data:
            raise ValueError(f"Expected one of {', '.join(valid_data)}, not {dataset}.")
        datasets = getattr(self, dataset)

        points = self.device.mesh_points
        x, y = points.T
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise TypeError(
                f"Expected a tuple of length 2, but got {grid_shape} ({type(grid_shape)})."
            )

        xs = np.linspace(xmin, xmax, grid_shape[1])
        ys = np.linspace(ymin, ymax, grid_shape[0])

        xgrid, ygrid = np.meshgrid(xs, ys)
        zgrids = {}
        for name, array in datasets.items():
            zgrid = griddata(points, array, (xgrid, ygrid), method=method, **kwargs)
            zgrids[name] = zgrid

        return xgrid, ygrid, zgrids

    def current_density(
        self,
        grid_shape=Union[int, Tuple[int, int]],
        method: Optional[str] = "cubic",
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        xgrid, ygrid, streams = self.grid_data(
            dataset="streams", grid_shape=grid_shape, method=method, **kwargs
        )

        Js = {}
        for name, g in streams.items():
            # J = [dg/dy, -dg/dx]
            # y is axis 0 (rows), x is axis 1 (columns)
            gy, gx = np.gradient(g, ygrid, xgrid)
            Js[name] = np.array([gy, -gx])

        return xgrid, ygrid, Js


def brandt_layer(
    device: Device,
    layer: str,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, float]] = None,
    check_inversion: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    circulating_currents = circulating_currents or {}

    points = device.mesh_points

    if device.adj is None:
        device.make_mesh(compute_arrays=True)

    weights = device.weights
    Q = device.Q
    Del2 = device.Del2
    x, y = points.T

    films = [name for name, film in device.films.items() if film.layer == layer]
    Lambda = device.layers[layer].Lambda

    Hz_applied = applied_field(points[:, 0], points[:, 1])
    Ha_eff = np.zeros_like(Hz_applied)

    g = np.zeros(len(x))
    for name in films:
        film = device.films[name]
        # Form system A @ gf = h (film only)
        ix1d = film.contains_points(x, y, index=True)
        ix2d = np.ix_(ix1d, ix1d)
        A = Q[ix2d] * weights[ix2d] - Lambda * Del2[ix2d]
        h = Hz_applied[ix1d] + Ha_eff[ix1d]
        # lu_solve seems to be slightly faster than gf = la.inv(A) @ h,
        # slightly faster than gf = la.solve(A, h),
        # and much faster than gf = la.pinv(A) @ h.
        lu, piv = la.lu_factor(A)
        gf = la.lu_solve((lu, piv), h)
        g[ix1d] = gf
        if check_inversion:
            # Validate solution
            errors = (A @ gf) - h
            if not np.allclose(errors, 0):
                warnings.warn(
                    f"Unable to solve for stream matrix, maximum error {np.abs(errors).max():.3e}."
                )

    for name, current in circulating_currents.items():
        hole = device.holes[name]
        ix1d = hole.contains_points(x, y, index=True)
        g[ix1d] = current  # g[hole] = I_circ

    response_field = -(Q * weights) @ g
    total_field = Hz_applied + response_field

    return g, total_field, response_field


def brandt_layers(
    device: Device,
    applied_field: Callable,
    coupled: Optional[bool] = True,
    circulating_currents: Optional[Dict[str, float]] = None,
    check_inversion: Optional[bool] = True,
) -> BrandtSolution:

    points = device.mesh_points
    x, y = points.T
    triangles = device.triangles

    streams = {}
    fields = {}
    response_fields = {}

    for name, layer in device.layers.items():
        logging.info(f"Calculating {name} response to applied field.")
        Hz_func = lambda x, y: applied_field(x, y, layer.z0)
        g, total_field, response_field = brandt_layer(
            device,
            name,
            Hz_func,
            circulating_currents=circulating_currents,
            check_inversion=check_inversion,
        )
        streams[name] = g
        fields[name] = total_field
        response_fields[name] = response_field

    if coupled:
        xt, yt = centroids(points, triangles).T
        # Calculcate the response fields at each layer from every other layer.
        other_responses = {}
        for name, layer in device.layers.items():
            Hzr = np.zeros(points.shape[0])
            for other_name, other_layer in device.layers.items():
                if name == other_name:
                    continue
                logger.info(f"Calculcating response field at {name} from {other_name}.")
                h_interp = mtri.LinearTriInterpolator(
                    device.mesh, response_fields[other_name]
                )
                h = h_interp(xt, yt)
                areas = area(points, triangles)
                dz = layer.z0 - other_layer.z0
                for i in range(points.shape[0]):
                    rho = np.sqrt((x[i] - xt) ** 2 + (y[i] - yt) ** 2)
                    q = (2 * dz ** 2 - rho ** 2) / (
                        4 * np.pi * (dz ** 2 + rho ** 2) ** (5 / 2)
                    )
                    Hzr[i] += np.sign(dz) * np.sum(areas * h * q)
            other_responses[name] = Hzr

        streams = {}
        fields = {}
        response_fields = {}
        # Solve again with the response fields from all layers
        for name, layer in device.layers.items():
            logging.info(
                f"Calculating {name} response to applied field and "
                "response field from other layers."
            )
            Hz_func = lambda x, y: applied_field(x, y, layer.z0) + other_responses[name]
            g, total_field, response_field = brandt_layer(
                device,
                name,
                Hz_func,
                circulating_currents=circulating_currents,
                check_inversion=check_inversion,
            )
            streams[name] = g
            fields[name] = total_field
            response_fields[name] = response_field

    solution = BrandtSolution(
        device=device,
        streams=streams,
        fields=fields,
        response_fields=response_fields,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
    )

    return solution


def in_polygon(
    xq: Union[float, np.ndarray],
    yq: Union[float, np.ndarray],
    xp: Union[float, np.ndarray],
    yp: Union[float, np.ndarray],
) -> Union[bool, np.ndarray[bool]]:
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


def centroids(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Returns x,y coordinates for triangle centroids."""
    return np.array([np.sum(points[ix], axis=0) / 3 for ix in simplices])


def compute_adj(triangles: np.ndarray) -> np.ndarray:
    A = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    adj = csr_matrix((np.ones(A.shape[0]), (A[:, 0], A[:, 1])))
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    return np.asarray(adj.todense())


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


def uniform_weights(adj: np.ndarray) -> np.ndarray:
    """Computes the weight matrix using uniform weighting."""
    weights = (adj != 0).astype(float)
    # Normalize row-by-row
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(weights, 1.0)
    return weights


# def uniform_weights_for_loop(adj):
#     n = adj.shape[0]
#     weights = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):  # weights is symmetric, so only need
#             # compute upper triangular part
#             if adj[i, j]:  # Check if edge is in graph
#                 weights[i, j] = 1.0
#     # Symmetrize weights
#     weights = weights + weights.T
#     # Normalize weights
#     for i in range(n):
#         weights[i, :] = weights[i, :] / sum(weights[i, :])
#     for i in range(n):
#         weights[i, i] = 1.0  # Fill in diagonal terms wii
#     return weights


def q_matrix(points: np.ndarray) -> np.ndarray:
    """Computes the denominator matrix, q."""
    distances = cdist(points, points)  # Euclidian distance between points
    q = np.zeros_like(distances)
    # Diagonals of distances are zero by definition, so q[i,i] will diverge
    nz = np.nonzero(distances)
    q[nz] = 1 / (4 * np.pi * distances[nz] ** 3)
    np.fill_diagonal(q, np.inf)  # diagonal elements diverge
    return q


# def q_matrix_for_loop(points):
#     n = points.shape[0]
#     q = np.zeros((n, n))
#     w_q = 1.0 / (4 * np.pi)  # Weight factor
#     for i in range(n):
#         for j in range(i + 1, n):  # Q is symmetric, so only need
#             # compute upper triangular part
#             n_rij = np.sqrt(
#                 (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
#             )
#             if n_rij == 0:
#                 raise ValueError("Coincident points in mesh!")
#             q[i, j] = w_q / (n_rij ** 3)
#     # Symmetrize q
#     q = q + q.T
#     for i in range(n):
#         q[i, i] = np.inf  # Fill in diagonal terms qii = inf
#     return q


def C_vector(points: np.ndarray) -> np.ndarray:
    """Computes the edge vector, C."""
    xmax = points[:, 0].max()
    xmin = points[:, 0].min()
    ymax = points[:, 1].max()
    ymin = points[:, 1].min()
    C = np.zeros(points.shape[0])
    for i, (x, y) in enumerate(points):
        with np.errstate(divide="ignore"):
            C_i = (
                np.sqrt((xmax - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmax - x) ** (-2) + (ymin - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymin - y) ** (-2))
            )
        C[i] = C_i
    C[np.isinf(C)] = 1e30
    return C / (4 * np.pi)


def laplacian(weights: np.ndarray, method: str = "uniform") -> np.ndarray:
    if method == "uniform":
        Del2 = np.triu(weights)
        Del2 = Del2 + Del2.T
        np.fill_diagonal(Del2, -1.0)
    else:
        raise NotImplementedError(f'Laplacian method "{method}" not implemented.')
    return Del2


# def laplacian_for_loop(weights, method="uniform"):
#     if method != "uniform":
#         raise NotImplementedError('Laplacian method "{method}" not implemented.')
#     n = weights.shape[0]
#     Del2 = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):  # Del2 is symmetric, so only need
#             # compute upper triangular part
#             Del2[i, j] = weights[i, j]
#     # Symmetrize Del2
#     Del2 = Del2 + Del2.T
#     for i in range(n):
#         Del2[i, i] = -1.0  # Fill in diagonal terms Del2ii = -1
#     return Del2


def Q_matrix(
    q: np.ndarray, C: np.ndarray, weights: np.ndarray, copy_q: bool = True
) -> np.ndarray:
    """Computes the kernel matrix, Q."""
    if copy_q:
        q = q.copy()
    # q[i, i] are np.inf, but Q[i, i] involves a sum over the off-diagonal
    # elements of q, so we can just set q[i, i] = 0 here.
    np.fill_diagonal(q, 0)
    Q = -np.triu(q)
    Q = Q + Q.T
    np.fill_diagonal(Q, (C + np.sum(q * weights, axis=1)) / np.diag(weights))
    return Q


# def Q_matrix_for_loop(q, C, weights):
#     n = weights.shape[0]
#     Q = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i + 1, n):  # Q is symmetric, so we only need
#             # compute upper triangular part
#             Q[i, j] = -q[i, j]
#     # Symmetrize Q
#     Q = Q + Q.T
#     Q_ij = 0
#     # Fill in diagonal terms Qii
#     for i in range(n):
#         for k in range(n):
#             if k != i:
#                 Q_ij = Q_ij + q[i, k] * weights[i, k]
#         # weights[i,i] cancels out, so can be arbitrary
#         Q[i, i] = (Q_ij + C[i]) / weights[i, i]
#         Q_ij = 0
#     return Q
