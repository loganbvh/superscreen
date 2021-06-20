# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import logging
import warnings
from typing import Union, Callable, Optional, Dict, Tuple, List

import pint
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from matplotlib.tri import Triangulation, LinearTriInterpolator

from .device import Device
from .fem import areas, centroids
from .parameter import Constant

logger = logging.getLogger(__name__)

lambda_str = "\u03bb"
Lambda_str = "\u039b"


def q_matrix(points: np.ndarray) -> np.ndarray:
    """Computes the denominator matrix, q.

    Eq. 7 in [Brandt.], Eq. 8 in [Kirtley1], Eq. 8 in [Kirtley2].

    Args:
        points: Shape (n, 2) array of x,y coordinates of vertices

    Returns:
        Shape (n, n) array qij
    """
    # euclidean distance between points
    distances = cdist(points, points, metric="euclidean")
    q = np.zeros_like(distances)
    # Diagonals of distances are zero by definition, so q[i,i] will diverge
    nz = np.nonzero(distances)
    q[nz] = 1 / (4 * np.pi * distances[nz] ** 3)
    np.fill_diagonal(q, np.inf)  # diagonal elements diverge
    return q


def C_vector(points: np.ndarray) -> np.ndarray:
    """Computes the edge vector, C.

    Eq. 12 in [Brandt.], Eq. 16 in [Kirtley1], Eq. 15 in [Kirtley2].

    Args:
        points: Shape (n, 2) array of x,y coordinates of vertices

    Returns:
        Shape (n, ) array Ci
    """
    xmax = points[:, 0].max()
    xmin = points[:, 0].min()
    ymax = points[:, 1].max()
    ymin = points[:, 1].min()
    C = np.zeros(points.shape[0])
    with np.errstate(divide="ignore"):
        for i, (x, y) in enumerate(points):
            C[i] = (
                np.sqrt((xmax - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmax - x) ** (-2) + (ymin - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymin - y) ** (-2))
            )
    C[np.isinf(C)] = 1e30
    return C / (4 * np.pi)


def Q_matrix(
    q: np.ndarray, C: np.ndarray, weights: np.ndarray, copy_q: bool = True
) -> np.ndarray:
    """Computes the kernel matrix, Q.

    Eq. 10 in [Brandt.], Eq. 11 in [Kirtley1], Eq. 11 in [Kirtley2].

    Args:
        points: Shape (n, 2) array of x,y coordinates of vertices

    Returns:
        Shape (n, n) array Qij
    """
    if copy_q:
        q = q.copy()
    if not isinstance(weights, np.ndarray):
        # Convert sparse matrix to array
        weights = weights.toarray()
    # q[i, i] are np.inf, but Q[i, i] involves a sum over only the
    # off-diagonal elements of q, so we can just set q[i, i] = 0 here.
    np.fill_diagonal(q, 0)
    Q = -np.triu(q)
    Q = Q + Q.T
    np.fill_diagonal(Q, (C + np.sum(q * weights, axis=1)) / np.diag(weights))
    return Q


def field_conversion(
    field_units: str,
    current_units: str,
    length_units: str = "m",
    ureg: Optional[pint.UnitRegistry] = None,
) -> pint.Quantity:
    """Returns a conversion factor from ``field_units`` to ``current_units / length_units``.

    Args:
        field_units: Magnetic field/flux unit to convert, having dimensionality
            either of magnetic field ``H`` (e.g. A / m or Oe) or of
            magnetic flux density ``B = mu0 * H`` (e.g. Tesla or Gauss).
        current_units: Current unit to use for the conversion.
        length_units: Lenght/distance unit to use for the conversion.
        ureg: pint UnitRegistry to use for the conversion. If None is provided,
            a new UnitRegistry is created.

    Returns:
        Conversion factor as a ``pint.Quantity``. ``conversion_factor.magnitude``
        gives you the numerical value of the conversion factor.
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    field = ureg(field_units)
    target_units = f"{current_units} / {length_units}"
    try:
        field = field.to(target_units)
    except pint.DimensionalityError:
        # field_unit is flux density B = mu0 * H
        field = (field / ureg("mu0")).to(target_units)
    return field / ureg(field_units)


def brandt_layer(
    *,
    device: Device,
    layer: str,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, float]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the stream function and magnetic field within a single layer of a ``Device``.

    Args:
        device: The Device to simulate.
        layer: Name of the layer to analyze.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y coordinates.
        circulating_currents: A dict of {hole_name: hole_current}. Default: {}.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be converted
            to units of [current_units / device.units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.

    Returns:
        stream function, total field, film response field
    """
    circulating_currents = circulating_currents or {}

    if device.weights is None:
        device.make_mesh(compute_arrays=True)

    if not device._mesh_is_valid:
        raise RuntimeError(
            "Device mesh is not valid. Run device.make_mesh() to generate the mesh."
        )

    weights = device.weights
    Del2 = device.Del2
    # We need to do slicing, etc., so its easiest to just use numpy arrays
    if sp.issparse(weights):
        weights = weights.toarray()
    if sp.issparse(Del2):
        Del2 = Del2.toarray()
    Q = device.Q
    points = device.points
    x, y = points.T
    london_lambda = device.layers[layer].london_lambda
    d = device.layers[layer].thickness
    Lambda = device.layers[layer].Lambda

    if isinstance(london_lambda, (int, float)) and london_lambda <= d:
        warnings.warn(
            f"Layer '{layer}': The film thickness, d = {d:.4f} {device.units}, "
            f"is greater than or equal to the London penetration depth "
            f"({lambda_str} = {london_lambda:.4f} {device.units}), resulting "
            f"in an effective penetration depth {Lambda_str} = {Lambda:.4f} "
            f"{device.units} <= {lambda_str}. The assumption that the current density "
            f"is nearly constant over the thickness of the film may not be valid. "
        )

    film_names = [name for name, film in device.films.items() if film.layer == layer]

    Hz_applied = applied_field(points[:, 0], points[:, 1])
    field_conversion_factor = field_conversion(
        field_units,
        current_units,
        length_units=device.units,
        ureg=device.ureg,
    )
    logger.info(
        f"Converting applied field in units of {field_units} "
        f"to units of {current_units} / {device.units}. "
        f"Conversion factor: {field_conversion_factor}."
    )
    Hz_applied *= field_conversion_factor.magnitude
    if isinstance(Lambda, (int, float)):
        # Make Lambda a callable
        Lambda = Constant(Lambda)
    Lambda = Lambda(points[:, 0], points[:, 1])

    # Identify holes in the superconductor
    hole_indices = {}
    in_hole = np.zeros(len(x), dtype=int)
    for name, hole in device.holes.items():
        ix = hole.contains_points(x, y)
        hole_indices[name] = np.where(ix)[0]
        in_hole = np.logical_or(in_hole, ix)

    # Set the boundary conditions for all holes:
    # 1. g[hole] = I_circ_hole
    # 2. Effective field associated with I_circ_hole
    # See Section II(a) in [Brandt], Eqs. 18-19 in [Kirtley1],
    # and Eqs 17-18 in [Kirtley2].
    g = np.zeros_like(x)
    Ha_eff = np.zeros_like(Hz_applied)
    for name, current in circulating_currents.items():
        hole = hole_indices[name]
        g[hole] = current  # g[hole] = I_circ
        # Effective field associated with the circulating currents:
        A = Q[:, hole] * weights[:, hole] - Lambda[hole] * Del2[:, hole]
        # g is in [current_units], Lambda is in [device.units],
        # and Del2 is in [device.units ** (-2)], so
        # Ha_eff has units of [current_unit / device.units]
        Ha_eff += A @ g[hole]

    # Now solve for the stream function inside the superconducting films
    for name in film_names:
        film = device.films[name]
        # We want all points that are in a film and not in a hole.
        ix1d = np.logical_and(film.contains_points(x, y), np.logical_not(in_hole))
        ix1d = np.where(ix1d)[0]
        ix2d = np.ix_(ix1d, ix1d)
        # Form the linear system for the film:
        # (Q * w - Lambda * Del2) @ gf = A @ gf = h
        # Eqs. 15-17 in [Brandt], Eqs 12-14 in [Kirtley1], Eqs. 12-14 in [Kirtley2].
        A = Q[ix2d] * weights[ix2d] - Lambda[ix1d] * Del2[ix2d]
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
                    f"Unable to solve for stream function in {layer} ({name}), "
                    f"maximum error {np.abs(errors).max():.3e}."
                )
    # Eq. 7 in [Kirtley1], Eq. 7 in [Kirtley2]
    screening_field = -(Q * weights) @ g
    total_field = Hz_applied + screening_field

    return g, total_field, screening_field


def solve(
    *,
    device: Device,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, float]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
    coupled: Optional[bool] = True,
    iterations: Optional[int] = 1,
) -> List["BrandtSolution"]:
    """Computes the stream functions and magnetic fields for all layers in a ``Device``.

    The simulation strategy is:

    1. Compute the stream functions and fields for each layer given
    only the applied field.

    2. If coupled is True and there are multiple layers, then for each layer,
    calculcate the screening field from all other layers and recompute the
    stream function and fields based on the sum of the applied field
    and the screening fields from all other layers.

    3. If iterations > 1, then repeat step 2 (iterations - 1) times.


    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        circulating_currents: A dict of {hole_name: hole_current}. Default: {}.
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be converted
            to units of [current_units / device.units].
        coupled: Whether to account for the interactions between different layers
            (e.g. shielding).
        iterations: Number of times to compute the interactions between layers
            (iterations is ignored if coupled is False).

    Returns:
        A list of BrandtSolutions of length 1 if coupled is False,
        or length (iterations + 1) if coupled is True.
    """
    points = device.points
    triangles = device.triangles

    solutions = []

    streams = {}
    fields = {}
    screening_fields = {}

    field_conversion_factor = 1 / field_conversion(
        field_units,
        current_units,
        length_units=device.units,
        ureg=device.ureg,
    )
    logger.info(
        f"Conversion factor to recover fields in units of "
        f"{field_units}: {field_conversion_factor}."
    )

    # Compute the stream functions and fields for each layer
    # given only the applied field.
    for name, layer in device.layers.items():
        logger.info(f"Calculating {name} response to applied field.")

        def layer_field(x, y):
            return applied_field(x, y, layer.z0)

        g, total_field, screening_field = brandt_layer(
            device=device,
            layer=name,
            applied_field=layer_field,
            circulating_currents=circulating_currents,
            field_units=field_units,
            current_units=current_units,
            check_inversion=check_inversion,
        )
        streams[name] = g
        fields[name] = total_field
        screening_fields[name] = screening_field

    solution = BrandtSolution(
        device=device,
        streams=streams,
        fields={
            layer: field * field_conversion_factor.magnitude
            for layer, field in fields.items()
        },
        screening_fields={
            layer: screening_field * field_conversion_factor.magnitude
            for layer, screening_field in screening_fields.items()
        },
        applied_field=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
    )
    solutions.append(solution)

    if coupled:
        if iterations < 1:
            raise ValueError(f"Iterations ({iterations}) cannot be less than 1.")

        tri_points = centroids(points, triangles)
        tri_areas = areas(points, triangles)
        # Compute ||(x, y) - (xt, yt)||^2
        rho2 = cdist(points, tri_points, metric="sqeuclidean")
        mesh = Triangulation(*points.T, triangles=triangles)
        for i in range(iterations):
            # Calculcate the screening fields at each layer from every other layer
            other_screening_fields = {}
            for name, layer in device.layers.items():
                Hzr = np.zeros(points.shape[0], dtype=float)
                for other_name, other_layer in device.layers.items():
                    if name == other_name:
                        continue
                    logger.info(
                        f"Calculcating response field at {name} "
                        f"from {other_name} ({i+1}/{iterations})."
                    )
                    dz = other_layer.z0 - layer.z0
                    # Interpolate other_layer's stream function to the triangle coordinates
                    # so that we can do a surface integral using triangle areas.
                    g_interp = LinearTriInterpolator(mesh, streams[other_name])
                    g = g_interp(*tri_points.T)
                    # Calculate the dipole kernel and integrate
                    # Eqs. 1-2 in [Brandt], Eqs. 5-6 in [Kirtley1], Eqs. 5-6 in [Kirtley2].
                    q = (2 * dz ** 2 - rho2) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                    Hzr += np.sign(dz) * np.sum(tri_areas * q * g, axis=1)
                other_screening_fields[name] = Hzr

            # Solve again with the response fields from all layers
            streams = {}
            fields = {}
            screening_fields = {}
            for name, layer in device.layers.items():
                logger.info(
                    f"Calculating {name} response to applied field and "
                    f"response field from other layers ({i+1}/{iterations})."
                )

                def layer_field(x, y):
                    return applied_field(x, y, layer.z0) + other_screening_fields[name]

                g, total_field, screening_field = brandt_layer(
                    device=device,
                    layer=name,
                    applied_field=layer_field,
                    circulating_currents=circulating_currents,
                    field_units=field_units,
                    current_units=current_units,
                    check_inversion=check_inversion,
                )
                streams[name] = g
                fields[name] = total_field
                screening_fields[name] = screening_field

            solution = BrandtSolution(
                device=device,
                streams=streams,
                fields={
                    layer: field * field_conversion_factor.magnitude
                    for layer, field in fields.items()
                },
                screening_fields={
                    layer: screening_field * field_conversion_factor.magnitude
                    for layer, screening_field in screening_fields.items()
                },
                applied_field=applied_field,
                field_units=field_units,
                current_units=current_units,
                circulating_currents=circulating_currents,
            )
            solutions.append(solution)

    return solutions


class BrandtSolution(object):
    """A container for the calculated stream functions and fields,
    with some convenient data processing methods.
    """

    def __init__(
        self,
        *,
        device: Device,
        streams: Dict[str, np.ndarray],
        fields: Dict[str, np.ndarray],
        screening_fields: Dict[str, np.ndarray],
        applied_field: Callable,
        field_units: str,
        current_units: str,
        circulating_currents: Optional[Dict[str, float]] = None,
    ):
        self.device = device
        self.streams = streams
        self.fields = fields
        self.screening_fields = screening_fields
        self.applied_field = applied_field
        self.field_units = field_units
        self.current_units = current_units
        self.circulating_currents = circulating_currents

    def grid_data(
        self,
        dataset: str,
        grid_shape: Union[int, Tuple[int, int]],
        layers: Optional[Union[str, List[str]]] = None,
        method: Optional[str] = "cubic",
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Interpolates results from the triangular mesh to a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            dataset: Name of the dataset to interpolate
                (one of "streams", "fields", or "screening_fields").
            grid_shape: Shape of the desired rectangular grid. If a single integer N is given,
                then the grid will be square, shape = (N, N).
            layers: Name(s) of the layer(s) for which to interpolate results.
            method: Interpolation method to use (see scipy.interpolate.griddata).
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated data for each layer
        """
        valid_data = ("streams", "fields", "screening_fields")
        if dataset not in valid_data:
            raise ValueError(f"Expected one of {', '.join(valid_data)}, not {dataset}.")
        datasets = getattr(self, dataset)

        if isinstance(layers, str):
            layers = [layers]
        if layers is None:
            layers = list(self.device.layers)
        else:
            for layer in layers:
                if layer not in self.device.layers:
                    raise ValueError(f"Unknown layer, {layer}.")

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape, grid_shape)
        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise TypeError(
                f"Expected a tuple of length 2, but got {grid_shape} ({type(grid_shape)})."
            )

        points = self.device.points
        x, y = points.T
        xgrid, ygrid = np.meshgrid(
            np.linspace(x.min(), x.max(), grid_shape[1]),
            np.linspace(y.min(), y.max(), grid_shape[0]),
        )
        zgrids = {}
        for name, array in datasets.items():
            if name in layers:
                zgrid = griddata(points, array, (xgrid, ygrid), method=method, **kwargs)
                zgrids[name] = zgrid
        if with_units:
            xgrid = xgrid * self.device.ureg(self.device.units)
            ygrid = ygrid * self.device.ureg(self.device.units)
            if dataset in ("fields", "screening_fields"):
                units = self.field_units
            else:
                units = self.current_units
            zgrids = {
                layer: data * self.device.ureg(units) for layer, data in zgrids.items()
            }
        return xgrid, ygrid, zgrids

    def current_density(
        self,
        grid_shape=Union[int, Tuple[int, int]],
        layers: Optional[Union[str, List[str]]] = None,
        method: Optional[str] = "cubic",
        with_units: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Computes the current density ``J = [dg/dy, -dg/dx]`` on a rectangular grid.

        Keyword arguments are passed to scipy.interpolate.griddata().

        Args:
            grid_shape: Shape of the desired rectangular grid. If a single integer N is given,
                then the grid will be square, shape = (N, N).
            layers: Name(s) of the layer(s) for which to interpolate current density.
            method: Interpolation method to use (see scipy.interpolate.griddata).
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated current density for each layer
        """
        xgrid, ygrid, streams = self.grid_data(
            dataset="streams",
            layers=layers,
            grid_shape=grid_shape,
            method=method,
            with_units=with_units,
            **kwargs,
        )
        Js = {}
        for name, g in streams.items():
            # J = [dg/dy, -dg/dx]
            # y is axis 0 (rows), x is axis 1 (columns)
            gy, gx = np.gradient(g, ygrid[:, 0], xgrid[0, :])
            Js[name] = np.array([gy, -gx])
        return xgrid, ygrid, Js

    def polygon_flux(
        self,
        polygons: Optional[Union[str, List[str]]] = None,
        with_units: bool = False,
    ) -> Dict[str, float]:
        """Compute the flux through all polygons (films, holes, and flux regions)
        by integrating the calculated fields.

        Args:
            polygons: Name(s) of the polygon(s) for which to compute the flux.
                Default: All polygons.
            with_units: Whether to a dict of pint.Quantities with units attached.

        Returns:
            dict of flux for each polygon in units of ``[self.field_units * device.units**2]``.
        """
        films = list(self.device.films)
        holes = list(self.device.holes)
        flux_regions = list(self.device.flux_regions)
        all_polygons = films + holes + flux_regions

        if isinstance(polygons, str):
            polygons = [polygons]
        if polygons is None:
            polygons = all_polygons
        else:
            for poly in polygons:
                if poly not in all_polygons:
                    raise ValueError(f"Unknown polygon, {poly}.")

        if with_units:
            ureg = self.device.ureg
            units = ureg(self.field_units) * ureg(self.device.units) ** 2
        else:
            units = 1

        points = self.device.points
        triangles = self.device.triangles

        tri_areas = areas(points, triangles)
        xt, yt = centroids(points, triangles).T
        mesh = Triangulation(*points.T, triangles=triangles)
        flux = {}
        for name in polygons:
            if name in films:
                poly = self.device.films[name]
            elif name in holes:
                poly = self.device.holes[name]
            else:
                poly = self.device.flux_regions[name]
            h_interp = LinearTriInterpolator(mesh, self.fields[poly.layer])
            field = h_interp(xt, yt)
            ix = poly.contains_points(xt, yt, index=True)
            flux[name] = np.sum(field[ix] * tri_areas[ix]) * units
        return flux
