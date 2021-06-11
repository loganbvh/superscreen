import warnings
from typing import Callable, Optional, Dict, Tuple

import numpy as np
import scipy.linalg as la
import matplotlib.tri as mtri

from .device import Device
from . import core


class BrandtSolution(object):
    def __init__(
        self,
        *,
        device: Device,
        streams: Dict[str, np.ndarray],
        fields: Dict[str, np.ndarray],
        response_fields: Dict[str, np.ndarray],
        Hz_func: Callable,
        circulating_currents: Optional[Dict[str, float]] = None,
    ):
        self.device = device
        self.streams = streams
        self.fields = fields
        self.response_fields = response_fields
        self.Hz_func = Hz_func
        self.circulating_currents = circulating_currents


def brandt_layer(device, layer, Hz_func, circulating_currents=None):
    
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
    
    Hz_applied = Hz_func(points[:, 0], points[:, 1])
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
        # Validate solution
        errors = (A @ gf) - h
        if not np.allclose(errors, 0):
            warnings.warn(
                "Unable to solve for stream matrix, maximum error {:.3e}".format(
                    np.abs(errors).max())
            )
            
    for name, current in circulating_currents.items():
        hole = device.holes[name]
        ix1d = hole.contains_points(x, y, index=True)
        g[ix1d] = current  # g[hole] = I_circ

    response_field = -(Q * weights) @ g
    total_field = Hz_applied + response_field

    return g, total_field, response_field


def brandt_layers(device, Hz_func, coupled=True, circulating_currents=None):

    points = device.mesh_points
    x, y = points.T
    triangles = device.triangles

    streams = {}
    fields = {}
    response_fields = {}

    for name, layer in device.layers.items():
        print("Solving layer", name)
        Hz0_func = lambda x, y: Hz_func(x, y, layer.z0)
        g, total_field, response_field = brandt_layer(
            device,
            name,
            Hz0_func,
            circulating_currents=circulating_currents
        )
        streams[name] = g
        fields[name] = total_field
        response_fields[name] = response_field
        
    if coupled:
        xt, yt = core.centroids(points, triangles).T
        # Calculcate the response fields at each layer from every other layer.
        other_responses = {}
        for name, layer in device.layers.items():
            print("Calculcating response field at {} from".format(name))
            Hzr = np.zeros(points.shape[0])
            for other_name, other_layer in device.layers.items():
                if name == other_name:
                    continue
                print("\t" + other_name)
                h_interp = mtri.LinearTriInterpolator(device.mesh, response_fields[other_name])
                h = h_interp(xt, yt)
                areas = core.area(points, triangles)
                dz = layer.z0 - other_layer.z0
                for i in range(points.shape[0]):
                    rho =  np.sqrt((x[i] - xt) ** 2 + (y[i] - yt) ** 2)
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
            print("Solving layer again", name)
            Hz0_func = lambda x, y: Hz_func(x, y, layer.z0) + other_responses[name]
            g, total_field, response_field = brandt_layer(
                device,
                name,
                Hz0_func,
                circulating_currents=circulating_currents
            )
            streams[name] = g
            fields[name] = total_field
            response_fields[name] = response_field

    return streams, fields, response_fields


# class BrandtModel(object):
#     def __init__(self, device_layout, mesh_refinements=3, progbar=True):
#         """Base class for multi-layer supercondcting film screening simulations following Brandt.
#         arXiv:cond-mat/0506144 (10.1103/PhysRevB.72.024529)

#         Adapted from John Kirtley's MATLAB code used for arXiv:1607.03950 (10.1088/0953-2048/29/12/124001).
#         Equation numbers in comments correspond to arXiv versions of the above papers.

#         Args:
#             device_layout (:class:`layouts.DeviceLayout`): DeviceLayout representing the device you want to model.
#             mesh_refinements (optional, int): Number of times to refine the mesh of device_layout. Default: 3.
#             progbar (optional, bool): Whether to show tqdm progress bar for calculations.
#         """
#         self.device = device_layout
#         self.device.make_mesh(mesh_refinements)
#         self.areas = None
#         self._applied_Hz = {}
#         self.response_Hz = {}
#         self.total_Hz = {}
#         self.Ks = {}
#         self.streams = {}
#         self.applied_flux = {}
#         self.total_flux = {}
#         self.progbar = progbar
#         self._weights = None
#         self._adj = None

#     @property
#     def adj(self):
#         if self._adj is None:
#             self._adj = core.compute_adj(self.simplices)
#         return self._adj

#     @property
#     def weights(self):
#         """Mesh vertex weights."""
#         # return (self.device.x_width * self.device.y_width) / len(self.points)
#         if self._weights is None:
#             self._weights = core.uniform_weights(self.adj)
#         return self._weights

#     @property
#     def points(self):
#         """x, y coordinates of mesh nodes."""
#         return self.device.dtri.points

#     @property
#     def simplices(self):
#         """Simplices mesh triangles. Coordinates of mesh triangles are self.points[self.simplices]"""
#         return self.device.dtri.simplices

#     @property
#     def applied_Hz(self):
#         """Dict of {layer_name: applied out-of-plane field} for all superconducting layers in device_layout.
#         The units here should be Phi0 / (mesh distance units)**2, e.g. Phi0/um**2.
#         """
#         return self._applied_Hz

#     @property
#     def device_origin(self):
#         return self.device.origin

#     @device_origin.setter
#     def device_origin(self, new_origin):
#         self.device.origin = new_origin
#         self.reset_origin_dependent_values()

#     def set_applied_Hz(self):
#         """Method to initialize the applied field. Must be implemented by subclasses.
#         An example for a uniform field of 0.1 Phi0/um**2 is given below.

#         n = len(self.points)
#         Hz = 0.1 # [Phi0 / um**2]
#         self._applied_Hz = {layer: Hz * np.ones(n) for layer in self.device.heights}
#         self.total_Hz = {layer: Hz * np.ones(n) for layer in self.device.heights}
#         """
#         raise NotImplementedError(
#             "set_applied_Hz must be implemented by subclasses of BrandtModel."
#         )

#     def calculate_C(self):
#         r"""Brandt [12], Kirtley [15].
#         :math:`C(x,y)=\sum_{p,q\in\pm1}[(a-px)^{-2} + (b-qy)^{-2}]^{1/2}`

#         TODO: Implement line integral instead of rectangular approximation.
#         """
#         self.C = core.C_vector(self.points)
#         return self.C

#         # print('Calculating C matrix...')
#         # Rectangular approximation instead of line integral
#         # xw, yw = self.device.x_width, self.device.y_width

#         # def Cfunc(x, y):
#         #     s = np.sum(
#         #         [
#         #             np.sqrt((xw - p * x) ** (-2) + (yw - q * y) ** (-2))
#         #             for p in (-1, 1)
#         #             for q in (-1, 1)
#         #         ]
#         #     )
#         #     return s / (4 * np.pi)

#         # self.C = np.array(
#         #     [
#         #         Cfunc(x, y)
#         #         for x, y in tqdm(
#         #             self.points, desc="Calculating C matrix", disable=(not self.progbar)
#         #         )
#         #     ]
#         # )

#     # def calculate_C(self):
#     #     points = self.points
#     #     sim = self.simplices
#     #     centers = centroids(points, sim)
#     #     rho = scipy.spatial.distance.cdist(centers, centers)
#     #     r3 = rho**(-3)
#     #     r3[r3 == np.inf] = np.nan
#     #     if self.areas is None:
#     #         self.areas = tri_area(points, sim)
#     #     C = np.nansum(self.areas * r3, axis=1) / (4 * np.pi)
#     #     self.C = sim2pts(points, sim, C)

#     def calculate_Q(self):
#         r"""Brandt [7,10], Kirtley [8,11].
#         :math:`Q_{i\neq j}=\frac{-1}{4\pi|\vec{r}_j-\vec{r}_j|^3}=-q_{ij}`
#         :math:`Q_{ij}=(\delta_{ij}-1)q_{ij}+\delta_{ij}\left(\sum_{l\neq i}q_{il}w_l+C_i\right)/w_j`
#         """
#         points = self.points
#         C = core.C_vector(points)
#         q = core.q_matrix(points)
#         w = self.weights
#         self.Q = core.Q_matrix(q, C, w)
#         return self.Q
#         # points = self.points
#         # rij = scipy.spatial.distance.cdist(points, points)
#         # qij = np.zeros_like(rij)
#         # nz = np.nonzero(rij)
#         # qij[nz] = 1 / (4 * np.pi * np.abs(rij[nz]) ** 3)
#         # np.fill_diagonal(qij, 0)  # diagonal elements diverge
#         # Qij = -qij
#         # # np.fill_diagonal(Qij, 0)
#         # # n = Qij.shape[0]
#         # # diag = np.zeros(n)
#         # # for i in tqdm(range(n), desc='Calculating Q matrix', disable=(not self.progbar)):
#         # #     s = 0
#         # #     for l in range(n):
#         # #         if i != l:
#         # #             s += qij[i,l] * self.w
#         # # diag[i] = (s + self.C[i]) / self.w
#         # diag = np.sum(qij, axis=1) + self.C / self.w
#         # self.Qs = Qij + np.diag(diag)

#     def calculate_Laplacian(self):
#         r"""Approximates the Laplace-Beltrami operator on the mesh, following arXiv:math/0503219.
#         As noted in Kirtley, this formulation only holds exactly for a square lattice. Kirtley [16].
#         :math:`\nabla^2_{ij}=\frac{1}{w}\sum_j^{N_i}(\delta_{ij}-\delta_ii)`
#         TODO: Implement a less approximate Laplacian, e.g. with half-cotangent weighting.
#         """
#         # n = self.points.shape[0]
#         # laplacian = np.zeros((n, n))
#         # indptr, indices = self.device.dtri.vertex_neighbor_vertices
#         # w = self.w
#         # for i in tqdm(
#         #     range(n), desc="Calculating Laplacian", disable=(not self.progbar)
#         # ):
#         #     neighbors = indices[indptr[i] : indptr[i + 1]]
#         #     for n in neighbors:
#         #         laplacian[i, i] -= 1 / w
#         #         laplacian[i, n] += 1 / w
#         # self.Del2 = laplacian

#         # Smarter Laplace-Beltrami, which I haven't figured out yet
#         # https://spharapy.readthedocs.io/en/latest/modules/trimesh.html
#         # from spharapy import trimesh
#         # weight_mode = 'half_cotangent'
#         # sim = self.simplices
#         # vertices = self.points
#         # # trimesh requires three dimensions
#         # vertices = np.append(vertices, np.zeros((len(vertices),1)), axis=1)
#         # tm = trimesh.TriMesh(sim, vertices)
#         # self.wij = tm.weightmatrix(mode=weight_mode)
#         # self.Del2 = tm.laplacianmatrix(mode=weight_mode) / self.wij

#         self.Del2 = core.laplacian(self.weights)
#         return self.Del2

#     def calculate_polygon_K(self, polygon):
#         r"""Calculates the K matrix for a given polygon in device_layout.
#         Brandt [11], Kirtley [14].
#         :math:`K_{ij}^\Lambda = (Q_{ij}w_j-\Lambda\nabla^2_{ij})^{-1}`
#         """
#         x, y = self.points.T
#         Kij = np.zeros((len(x), len(x)))
#         film = self.device.inpolygon(polygon, x, y)
#         london = self.device.lambdas[self.device.layers[polygon]]
#         d = self.device.thicknesses[self.device.layers[polygon]]
#         Pearl = london ** 2 / d
#         ix = np.ix_(film, film)
#         w = self.w
#         Kij[ix] = np.linalg.inv(self.Qs[ix] * w - Pearl * self.Del2[ix])
#         # w = self.wij
#         # Kij[ix] = np.linalg.inv(self.Qs[ix] * w[ix] - Pearl * self.Del2[ix])
#         self.Ks[polygon] = Kij

#     def calculate_polygon_stream(self, polygon):
#         """Calculcates the stream functions for a given polygon in device_layout given some total_Hz
#         at the layer in which polygon resides.
#         Brandt [16], Kirtley [13].
#         :math:`g_i=-\sum_j K_{ij}^\Lambda H_{z,i}`
#         """
#         if polygon not in self.Ks:
#             self.calculate_polygon_K(polygon)
#         Kij = self.Ks[polygon]
#         layer = self.device.layers[polygon]
#         Hz = self.total_Hz[layer]
#         self.streams[polygon] = -np.dot(Kij, Hz).squeeze()

#     def response_field_from_polygon_at_layer(self, polygon, layer):
#         r"""Calculates the response field from a given polygon at the z position of a given layer.
#         Brandt [1,2,4], Kirtley [5,6,7]
#         :math:`H_z(\vec{r})=H_z(\vec{r})+\int_S d^2r'Q(\vec{r},\vec{r}')g(\vec{r})`
#         :math:`z\neq0:\;Q(\vec{r},\vec{r}')=\lim_{z\to0}\frac{2z^2-\rho^2}{4\pi(z^2 + \rho^2)^{5/2}}`
#         :math:`z=0:\;H_{z,i} = H_{a,i} + \sum_j Q_{ij} w_j g_j`
#         """
#         g = self.streams[polygon]
#         points = self.points
#         dz = (
#             self.device.heights[layer]
#             - self.device.heights[self.device.layers[polygon]]
#         )
#         if dz == 0:
#             w = self.w
#             # w = self.wij
#             Hz = np.zeros(len(points))
#             msg = "Calculating response at {} from {}".format(layer, polygon)
#             for i in tqdm(range(len(points)), desc=msg, disable=(not self.progbar)):
#                 Hz[i] = w * np.sum(self.Qs[i, :] * g)
#                 # Hz[i] = np.sum(w[i,:] * self.Qs[i,:] * g)
#             # Hz = w * np.sum(self.Qs * g, axis=1)
#         else:
#             sim = self.simplices
#             Hzr = np.zeros(len(sim))
#             xt, yt = centroids(points, sim).T
#             ginterp = pts2sim(sim, g)
#             if self.areas is None or len(self.areas) != len(ginterp):
#                 self.areas = tri_area(points, sim)
#             areas = self.areas
#             msg = "Calculating response at {} from {}".format(layer, polygon)
#             for i in tqdm(range(len(sim)), desc=msg, disable=(not self.progbar)):
#                 rhos = np.sqrt((xt[i] - xt) ** 2 + (yt[i] - yt) ** 2)
#                 f = (2 * dz ** 2 - rhos ** 2) / (
#                     4 * np.pi * (dz ** 2 + rhos ** 2) ** (5 / 2)
#                 )
#                 Hzr[i] = np.nansum(areas * ginterp * f)
#             # pts = np.stack([xt, yt], axis=1)
#             # if self.tri_rhos is None:
#             #     self.tri_rhos = scipy.spatial.distance_matrix(pts, pts)
#             # rhos = self.tri_rhos
#             # f = (2 * dz**2 - rhos**2) / (4*np.pi * (dz**2 + rhos**2)**(5/2))
#             # Hzr = np.nansum(areas * ginterp * f, axis=1)
#             Hz = sim2pts(points, sim, Hzr)
#         self.response_Hz[(layer, polygon)] = Hz
#         # add this response Hz to total_Hz, ignoring NaNs
#         self.total_Hz[layer] = np.nansum(np.stack((self.total_Hz[layer], Hz)), axis=0)

#     def calculate_flux(self):
#         """Integrate the flux penetrating each layer, and each region in self.device.flux_regions."""
#         points = self.points
#         sim = self.simplices
#         if self.areas is None:
#             self.areas = tri_area(points, sim)
#         areas = self.areas
#         xt, yt = centroids(points, sim).T
#         for name, (poly, layer) in self.device.flux_regions.items():
#             x, y = poly.T
#             in_points = inpolygon(xt, yt, x, y)
#             ix = np.ix_(in_points)
#             # unscreened flux
#             Hz = pts2sim(sim, self.applied_Hz[layer])
#             self.applied_flux[name] = np.sum(Hz[ix] * areas[ix])
#             self.applied_flux[layer] = np.sum(Hz * areas)
#             # total flux
#             Hz = pts2sim(sim, self.total_Hz[layer])
#             self.total_flux[name] = np.sum(Hz[ix] * areas[ix])
#             self.total_flux[layer] = np.sum(Hz * areas)
#         for layer in self.device.heights:
#             if layer not in self.total_flux:
#                 # unscreened flux
#                 Hz = pts2sim(sim, self.applied_Hz[layer])
#                 self.applied_flux[layer] = np.sum(Hz * areas)
#                 # total flux
#                 Hz = pts2sim(sim, self.total_Hz[layer])
#                 self.total_flux[layer] = np.sum(Hz * areas)

#     def reset_origin_dependent_values(self):
#         """Reset any values that are invalid if we change the origin of self.device."""
#         self.areas = None
#         self._applied_Hz = {}
#         self.response_Hz = {}
#         self.total_Hz = {}
#         self.streams = {}
#         self.applied_flux = {}
#         self.total_flux = {}

#     def prepare_model(self):
#         """Create matrices that don't depend on applied field or the origin of self.device."""
#         self.calculate_C()
#         self.calculate_Q()
#         self.calculate_Laplacian()

#     def run_all(self):
#         if not hasattr(self, "Del2"):
#             self.prepare_model()
#         self.set_applied_Hz()
#         for p in self.device.polygons:
#             self.calculate_polygon_K(p)
#         for layer in self.device.heights:
#             for polygon in self.device.polygons:
#                 self.calculate_polygon_stream(polygon)
#                 self.response_field_from_polygon_at_layer(polygon, layer)
#         self.calculate_flux()

#     def grid_streams(self, n_interp=500):
#         x, y = self.points.T
#         xs, ys = np.linspace(x.min(), x.max(), n_interp), np.linspace(
#             y.min(), y.max(), n_interp
#         )
#         xx, yy = np.meshgrid(xs, ys)
#         streams = {}
#         for name, arr in self.streams.items():
#             streams[name] = griddata(
#                 self.points, arr.squeeze(), (xx, yy), method="linear"
#             )
#         return xs, ys, streams

#     def grid_fields(self, field="total", n_interp=500):
#         assert field.lower() in ("total", "applied", "response")
#         if field.lower() in ("total", "applied"):
#             fields = getattr(self, "{}_Hz".format(field.lower()))
#         else:
#             fields = {
#                 name: arr.copy() - self.applied_Hz[name]
#                 for name, arr in self.total_Hz.items()
#             }
#         x, y = self.points.T
#         xs, ys = np.linspace(x.min(), x.max(), n_interp), np.linspace(
#             y.min(), y.max(), n_interp
#         )
#         xx, yy = np.meshgrid(xs, ys)
#         result = {}
#         for name, arr in fields.items():
#             result[name] = griddata(
#                 self.points, arr.squeeze(), (xx, yy), method="linear"
#             )
#         return xs, ys, result

#     def current_density(self, n_interp=500):
#         """In some units..."""
#         if not self.streams:
#             return
#         xs, ys, streams = self.grid_streams(n_interp=n_interp)
#         Js = {}
#         for name, arr in tqdm(
#             streams.items(), desc="Calculating Js for {}".format(name)
#         ):
#             dx, dy = np.gradient(arr, xs, ys)
#             Js[name] = np.stack([dy, -dx], axis=1)
#         return xs, ys, Js

#     def plot_streams(self, n_interp=500, plot_polygons=True, cutoff_percentile=0.1):
#         """Plot the stream functions for all superconducting layers.
#         This interpolates data on the mesh to data on a regular grid for plotting.

#         Args:
#             n_intrp (optional, int): Desired number of x and y points for interpolated data.
#             plot_polygons (optional, bool): Whether to overlay the outlines of the device polygons.
#             cutoff_percentile (optional, float): Cutoff percentile for color scale. See utils.auto_range_iqr.
#         """
#         if not self.streams:
#             return
#         fig, axes = plt.subplots(1, len(self.streams), sharex=True, sharey=True)
#         # x, y = self.points.T
#         # xs, ys = np.linspace(x.min(), x.max(), n_interp), np.linspace(y.min(), y.max(), n_interp)
#         # xx, yy = np.meshgrid(xs, ys)
#         xs, ys, streams = self.grid_streams(n_interp=n_interp)
#         for name, ax in zip(streams, axes):
#             # for name, ax in zip(self.streams, axes):
#             # z = self.streams[name].squeeze()
#             # gd = griddata(self.points, z, (xx, yy), method='linear')
#             gd = streams[name]
#             lims = auto_range_iqr(gd, cutoff_percentile=cutoff_percentile)
#             im = ax.pcolormesh(xs, ys, gd)
#             im.set_clim(lims)
#             fig.colorbar(im, orientation="horizontal", pad=0.1, ax=ax)
#             ax.set_aspect("equal")
#             ax.set_title(name)
#             if plot_polygons:
#                 self.device.plot_polygons(ax=ax, grid=False, legend=False)
#         fig.suptitle("Stream functions")
#         fig.tight_layout()
#         return fig, axes

#     def plot_fields(
#         self, field="total", n_interp=500, plot_polygons=True, cutoff_percentile=0.1
#     ):
#         """Plot the out-of-plane fields for all superconducting layers.
#         This interpolates data on the mesh to data on a regular grid for plotting.

#         Args:
#             field (optional, {'total', 'applied', 'response'}): Which fields you want to plot.
#             n_intrp (optional, int): Desired number of x and y points for interpolated data.
#             plot_polygons (optional, bool): Whether to overlay the outlines of the device polygons.
#             cutoff_percentile (optional, float): Cutoff percentile for color scale. See utils.auto_range_iqr.
#         """
#         if not self.total_Hz:
#             return
#         assert field.lower() in ("total", "applied", "response")
#         if field.lower() in ("total", "applied"):
#             fields = getattr(self, "{}_Hz".format(field.lower()))
#         else:
#             fields = {
#                 name: arr.copy() - self.applied_Hz[name]
#                 for name, arr in self.total_Hz.items()
#             }
#         fig, axes = plt.subplots(1, len(fields), sharex=True, sharey=True)
#         # x, y = self.points.T
#         # xs, ys = np.linspace(x.min(), x.max(), n_interp), np.linspace(y.min(), y.max(), n_interp)
#         # xx, yy = np.meshgrid(xs, ys)
#         xs, ys, fields = self.grid_fields(field=field, n_interp=n_interp)
#         # for name, ax in zip(fields, axes):
#         #     z = fields[name].squeeze()
#         #     gd = griddata(self.points, z, (xx, yy), method='linear')
#         for name, ax in zip(fields, axes):
#             gd = fields[name]
#             lims = auto_range_iqr(gd, cutoff_percentile=cutoff_percentile)
#             im = ax.pcolormesh(xs, ys, gd)
#             im.set_clim(lims)
#             fig.colorbar(im, orientation="horizontal", pad=0.1, ax=ax)
#             ax.set_aspect("equal")
#             ax.set_title(name)
#             if plot_polygons:
#                 self.device.plot_polygons(ax=ax, grid=False, legend=False)
#         fig.suptitle("{} field".format(field.title()))
#         fig.tight_layout()
#         return fig, axes

#     def save_results(self, path=None, fmt="h5"):
#         from utils import PathMaker

#         path = PathMaker(directory="results").dir
#         fn = "{}_brandt_results".format(time.strftime("%y%m%d_%H%M%S"))
#         fpath = os.path.join(path, fn)
#         data_to_save = {
#             "points": self.points,
#             "simplices": self.simplices,
#             "polygons": self.device.polygons,
#             "layers": self.device.layers,
#             "C": self.C,
#             "Qs": self.Qs,
#             "Ks": self.Ks,
#             "Del2": self.Del2,
#             "applied_Hz": self.applied_Hz,
#             "response_Hz": {
#                 str(k): v for k, v in self.response_Hz.items
#             },  # has tuples as keys
#             "total_Hz": self.total_Hz,
#             "total_flux": self.total_flux,
#             "applied_flux": self.applied_flux,
#         }
#         if fmt == "h5":
#             fpath += ".h5"
#             import h5py
#             from utils import set_h5_attrs

#             with h5py.File(fpath, "w") as df:
#                 set_h5_attrs(df, data_to_save)
#         elif fmt == "mat":
#             fpath += ".mat"
#             from scipy.io import savemat

#             savemat(fpath, data_to_save)
#         elif fmt == "pickle":
#             import pickle

#             fpath += ".p"
#             data_to_save["response_Hz"] = self.response_Hz
#             with open(fpath, "wb") as f:
#                 pickle.dump(data_to_save, f)
#         else:
#             raise ValueError("Unsupported file format: {}".format(fmt))


# class BrandtVortexModel(BrandtModel):
#     def __init__(self, device_layout, mesh_refinements=3, progbar=True, vortices=None):
#         """BrandtModel where the only applied field comes from vortices in the x-y plane.

#         Args:
#             device_layout (layout.DeviceLayout): DeviceLayout representing the device you want to model.
#             mesh_refinements (optional, int): Number of times to refine the mesh of device_layout. Default: 3.
#             vortices (optional, dict): Dict of {(vortex x, vortex y): number of Phi0} for all vortices in the model.
#                 Default: {(0,0): 1}, a single vortex at the origin.
#         """
#         self.vortices = vortices or {
#             (0, 0): 1
#         }  # {vortex_position: number of flux quanta}
#         super().__init__(
#             device_layout, mesh_refinements=mesh_refinements, progbar=progbar
#         )

#     def set_applied_Hz(self):
#         self.applied_Hz_from_vortices()

#     def Hz_from_vortex(self, x, y, z, xv, yv, nphi0=1):
#         """z-component of the field from a vortex (point source) located at (xv, yv, 0) in units of Phi0 / (xyz unit)**2.

#         Args:
#             x, y, z (float | np.ndarray(float)): x, y, and z position at which to evaluate the field.
#             xv, yv (float): Vortex position in the x-y plane.
#             nphi0 (optional, int | float): Number of Phi0 located at (xv,yv), in case of double or half Phi0 vortices ;).
#         """
#         xp = x - xv
#         yp = y - yv
#         Hz0 = z / (xp ** 2 + yp ** 2 + z ** 2) ** (3 / 2) / (2 * np.pi)
#         return nphi0 * Hz0

#     def applied_Hz_from_vortices(self):
#         """Calculates the field at all device layers due to self.vortices. Units are Phi0/um**2 assuming
#         mesh dimensions are in um.
#         """
#         self.total_Hz = {}
#         # Units: Phi0 / um**2
#         x, y = self.points.T
#         z0 = self.device.origin[-1]
#         for layer, z in self.device.heights.items():
#             Hz = np.zeros_like(x)
#             for (xv, yv), phi0s in self.vortices.items():
#                 Hz += self.Hz_from_vortex(x, y, z + z0, xv, yv, nphi0=phi0s)
#             self._applied_Hz[layer] = Hz
#             self.total_Hz[layer] = Hz.copy()


# class BrandtConstantFieldModel(BrandtModel):
#     def __init__(self, device_layout, Hz, mesh_refinements=3, progbar=True):
#         """BrandtModel with a constant applied z-field.

#         Args:
#             device_layout (layout.DeviceLayout): DeviceLayout representing the device you want to model.
#             Hz (float): z-component of the applied field, in units of Phi0/um**2 (2.068 mT, 20.68 gauss).
#             mesh_refinements (optional, int): Number of times to refine the mesh of device_layout. Default: 3.
#             progbar (optional, bool): Whether to display tqdm progress bar. Default: True.
#         """
#         self.Hz = float(Hz)
#         super().__init__(
#             device_layout, mesh_refinements=mesh_refinements, progbar=progbar
#         )

#     def set_applied_Hz(self):
#         n = len(self.points)
#         Hz = self.Hz  # [Phi0 / um**2]
#         self._applied_Hz = {layer: Hz * np.ones(n) for layer in self.device.heights}
#         self.total_Hz = {layer: Hz * np.ones(n) for layer in self.device.heights}
