import os, sys
import time
import json
import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse

# results_dir = os.path.join(os.path.dirname(__file__), "results")
# log_dir = os.path.join(os.path.dirname(__file__), "logs")

# class PathMaker(object):
#     def __init__(self, directory="results"):
#         self.basedir = directory

#     @property
#     def dir(self):
#         path = os.path.join(
#             os.path.dirname(__file__), self.basedir, time.strftime("%y%m%d")
#         )
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         return path

#     @property
#     def image_dir(self):
#         path = os.path.join(self.dir, "images")
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         return path


# def right_now(fmt="%y%m%d_%H%M%S"):
#     return time.strftime(fmt)


# def sim2pts(pts, sim, sim_values):
#     """Interpolate values defined on mesh simplices to values on nodes.

#     The areas/volumes are used as weights.
#     f_n = (sum_e r_e*S_e) / (sum_e S_e)
#     where r_e is the value on triangles who share the node n, S_e is the area of triangle e.
#     This function is similar to pdeprtni in MATLAB pde.
#     https://github.com/ElsevierSoftwareX/SOFTX_2018_114/blob/master/pyeit/eit/interp2d.py
#     """
#     N = pts.shape[0]
#     M, dim = sim.shape
#     # calculate the weights
#     # triangle/tetrahedron must be CCW (recommended), then a is positive
#     if dim == 3:
#         weight_func = tri_area
#     elif dim == 4:
#         weight_func = tet_volume
#     weights = weight_func(pts, sim)
#     # build tri->pts matrix, could be accelerated using sparse matrix
#     row = np.ravel(sim)
#     col = np.repeat(np.arange(M), dim)  # [0, 0, 0, 1, 1, 1, ...]
#     data = np.repeat(weights, dim)
#     e2n_map = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, M)).tocsr()
#     # map values from elements to nodes
#     # and re-weight by the sum of the areas/volumes of adjacent elements
#     f = e2n_map.dot(sim_values)
#     w = np.sum(e2n_map.toarray(), axis=1)
#     return f / w


# def pts2sim(sim, pts_values):
#     """
#     Given values on nodes, calculate interpolated values on simplices.
#     This function was tested and equivalent to MATLAB 'pdeintrp'
#     except for the shapes of 'pts' and 'tri'.
#     https://github.com/ElsevierSoftwareX/SOFTX_2018_114/blob/master/pyeit/eit/interp2d.py

#     Args:
#     sim (np.ndarray): shape (m,3) or (m,4) array, elements or simplices
#         triangles denote connectivity [[i, j, k]]
#         tetrahedrons denote connectivity [[i, j, m, n]]
#     pts_values (np.ndarray): shape (n,1) array of values on nodes, real/complex valued

#     Returns:
#         (np.ndarray): shape (m,1) array of values on simplices, real/complex valued
#     """
#     # averaged over 3 nodes of a triangle
#     el_value = np.mean(pts_values[sim], axis=1)
#     return el_value


# def tet_volume(pts, sim):
#     """Calculate the area of each tetrahedron.

#     Args:
#         pts (np.ndarray): shape (n,3) array of x,y,z coordinates of nodes
#         sim (np.ndarray) shape (m,4) array of tetrahedron simplices

#     Returns:
#         (np.ndarray): shape (m,) array of tetrahedron volumes.
#     """
#     v = np.zeros(np.shape(sim)[0])
#     for i, e in enumerate(sim):
#         xyz = pts[e]
#         s = xyz[[2, 3, 0]] - xyz[[1, 2, 3]]
#         # a should be positive if triangles are CCW arranged
#         v[i] = scipy.linalg.det(s)
#     return v / 6.0


def auto_range_iqr(data_array, cutoff_percentile=90):
    """Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75% and 25% of the distribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    https://github.com/QCoDeS/Qcodes/blob/master/qcodes/utils/plotting.py.

    Args:
        data_array (np.ndarray): Array of arbitrary dimension containing the statistical data.
        cutoff_percentile (float | tuple[float]): Percentile of data that may maximally be clipped
            on both sides of the distribution. If given a tuple (a,b) the percentile limits will be a and 100-b.

    Returns:
        (tuple[float]): (vmin, vmax)
    """
    if isinstance(cutoff_percentile, tuple):
        t = cutoff_percentile[0]
        b = cutoff_percentile[1]
    else:
        t = cutoff_percentile
        b = cutoff_percentile
    z = data_array.flatten()
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    zrange = zmax - zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [b, 75, 25, 100 - t])
    IQR = q3 - q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if zrange == 0.0 or IQR / zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5 * IQR, zmin)
        vmax = min(q3 + 1.5 * IQR, zmax)
        # do not clip more than cutoff_percentile:
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # scalar complex values only
        elif isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, np.void):
            return None

        # float, int, etc.
        elif isinstance(obj, np.generic):
            return obj.item()

        return super().default(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and set(dct.keys()) == {"real", "imag"}:
        return complex(dct["real"], dct["imag"])
    return dct
