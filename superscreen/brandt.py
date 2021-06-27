# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import logging
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
    """Computes the denominator matrix, q:

    .. math::

        q_{ij} = \\frac{1}{4\\pi|\\vec{r}_i-\\vec{r}_j|^3}

    See Eq. 7 in [Brandt-PRB-2005]_, Eq. 8 in [Kirtley-RSI-2016]_,
    and Eq. 8 in [Kirtley-SST-2016]_.

    Args:
        points: Shape (n, 2) array of x,y coordinates of vertices

    Returns:
        Shape (n, n) array, qij
    """
    # Euclidean distance between points
    distances = cdist(points, points, metric="euclidean")
    q = np.zeros_like(distances)
    # Diagonals of distances are zero by definition, so q[i,i] will diverge
    nz = np.nonzero(distances)
    q[nz] = 1 / (4 * np.pi * distances[nz] ** 3)
    np.fill_diagonal(q, np.inf)
    return q


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
        points: Shape (n, 2) array of x,y coordinates of vertices

    Returns:
        Shape (n, ) array, Ci
    """
    x, y = points.T
    a = np.ptp(x) / 2
    b = np.ptp(y) / 2
    with np.errstate(divide="ignore"):
        C = (
            np.sqrt((a - x) ** (-2) + (b - y) ** (-2))
            + np.sqrt((a + x) ** (-2) + (b - y) ** (-2))
            + np.sqrt((a - x) ** (-2) + (b + y) ** (-2))
            + np.sqrt((a + x) ** (-2) + (b + y) ** (-2))
        )
    C[np.isinf(C)] = 1e30
    return C / (4 * np.pi)


def Q_matrix(q: np.ndarray, C: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Computes the kernel matrix, Q:

    .. math::

        Q_{ij} = (\\delta_{ij}-1)q_{ij}
        + \\delta_{ij}\\frac{1}{w_{ij}}\\left(C_i
        + \\sum_{l\\neq i}q_{il}w_{il}\\right)

    See Eq. 10 in [Brandt-PRB-2005]_, Eq. 11 in [Kirtley-RSI-2016]_,
    and Eq. 11 in [Kirtley-SST-2016]_.

    Args:
        q: Shape (n, n) matrix qij
        C: Shape (n, ) vector Ci
        weights: Shape (n, n) weight matrix

    Returns:
        Shape (n, n) array, Qij
    """
    if not isinstance(weights, np.ndarray):
        # Convert sparse matrix to array
        weights = weights.toarray()
    # q[i, i] are np.inf, but Q[i, i] involves a sum over only the
    # off-diagonal elements of q, so we can just set q[i, i] = 0 here.
    q = q.copy()
    np.fill_diagonal(q, 0)
    Q = -q
    np.fill_diagonal(Q, (C + np.sum(q * weights, axis=1)) / np.diag(weights))
    return Q


def convert_field(
    value: Union[np.ndarray, float, str, pint.Quantity],
    new_units: Union[str, pint.Unit],
    old_units: Optional[Union[str, pint.Unit]] = None,
    ureg: Optional[pint.UnitRegistry] = None,
    magnitude: bool = False,
) -> Union[pint.Quantity, np.ndarray, float]:
    """Converts a value between different field units, either magnetic field H
    [current] / [length] or flux density B = mu0 * H [mass] / ([curret] [time]^2)).

    Args:
        value: The value to convert. It can either be a numpy array (no units),
            a float (no units), a string like "1 uA/um", or a scalar or array
            ``pint.Quantity``. If value is not a string wiht units or a ``pint.Quantity``,
            then old_units must specify the units of the float or array.
        new_units: The units to convert to.
        old_units: The old units of ``value``. This argument is required if ``value``
            is not a string with units or a ``pint.Quantity``.
        ureg: The ``pint.UnitRegistry`` to use for conversion. If None is given,
            a new instance is created.
        magnitude: Whether to return just the magnitude instead of a full ``pint.Quantity``

    Returns:
        The converted value, either a pint.Quantity (scalar or array with units),
        or an array or float without units, depending on the ``magnitude`` argument.
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    if isinstance(value, str):
        value = ureg(value)
    if isinstance(value, pint.Quantity):
        old_units = value.units
    if old_units is None:
        raise ValueError(
            "Old units must be specified if value is not a string or pint.Quantity."
        )
    if isinstance(old_units, str):
        old_units = ureg(old_units)
    if isinstance(new_units, str):
        new_units = ureg(new_units)
    if not isinstance(value, pint.Quantity):
        value = value * old_units
    if new_units.dimensionality == old_units.dimensionality:
        value = value.to(new_units)
    elif "[length]" in old_units.dimensionality:
        # value is H in units with dimensionality [current] / [length]
        # and we want B = mu0 * H
        value = (value * ureg("mu0")).to(new_units)
    else:
        # value is B = mu0 * H in units with dimensionality
        # [mass] / ([current] [time]^2) and we want H = B / mu0
        value = (value / ureg("mu0")).to(new_units)
    if magnitude:
        value = value.magnitude
    return value


def field_conversion_factor(
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
        # field_units is flux density B = mu0 * H
        field = (field / ureg("mu0")).to(target_units)
    return field / ureg(field_units)


def brandt_layer(
    *,
    device: Device,
    layer: str,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    current_units: str = "uA",
    check_inversion: bool = True,
    check_lambda: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the stream function and magnetic field within a single layer of a ``Device``.

    Args:
        device: The Device to simulate.
        layer: Name of the layer to analyze.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        check_lambda: Whether to generate a warning if Lambda <= london_lambda.

    Returns:
        stream function, total field, film screening field
    """
    circulating_currents = circulating_currents or {}

    if device.weights is None:
        device.make_mesh(compute_arrays=True)

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

    if check_lambda:
        if isinstance(london_lambda, (int, float)) and london_lambda <= d:
            logger.warn(
                f"Layer '{layer}': The film thickness, d = {d:.4f} {device.length_units}"
                f", is greater than or equal to the London penetration depth "
                f"({lambda_str} = {london_lambda:.4f} {device.length_units}), resulting "
                f"in an effective penetration depth {Lambda_str} = {Lambda:.4f} "
                f"{device.length_units} <= {lambda_str}. The assumption that the current density "
                f"is nearly constant over the thickness of the film may not be valid. "
            )

    film_names = [name for name, film in device.films.items() if film.layer == layer]
    hole_names = [name for name, hole in device.holes.items() if hole.layer == layer]

    # Units for field are {current_units} / {device.length_units}
    Hz_applied = applied_field(points[:, 0], points[:, 1])

    if isinstance(Lambda, (int, float)):
        # Make Lambda a callable
        Lambda = Constant(Lambda)
    Lambda = Lambda(points[:, 0], points[:, 1])

    # Identify holes in the superconductor
    hole_indices = {}
    in_hole = {name: np.zeros(len(x), dtype=bool) for name in device.layers}
    holes = device.holes
    for name in hole_names:
        hole = holes[name]
        ix = hole.contains_points(x, y)
        hole_indices[name] = np.where(ix)[0]
        in_hole[hole.layer] = np.logical_or(in_hole[hole.layer], ix)

    # Set the boundary conditions for all holes:
    # 1. g[hole] = I_circ_hole
    # 2. Effective field associated with I_circ_hole
    # See Section II(a) in [Brandt], Eqs. 18-19 in [Kirtley1],
    # and Eqs 17-18 in [Kirtley2].
    g = np.zeros_like(x)
    Ha_eff = {name: np.zeros_like(Hz_applied) for name in device.layers}
    holes = device.holes
    for name in hole_names:
        current = circulating_currents.get(name, None)
        if current is None:
            continue

        if isinstance(current, str):
            current = device.ureg(current)
        if isinstance(current, pint.Quantity):
            current = current.to(current_units).magnitude

        hole = hole_indices[name]
        g[hole] = current  # g[hole] = I_circ
        # Effective field associated with the circulating currents:
        # current is in [current_units], Lambda is in [device.length_units],
        # and Del2 is in [device.length_units ** (-2)], so
        # Ha_eff has units of [current_unit / device.length_units]
        Ha_eff[holes[name].layer] += -current * (
            Q[:, hole] * weights[:, hole] - Lambda[:, np.newaxis] * Del2[:, hole]
        ).sum(axis=1)

    # Now solve for the stream function inside the superconducting films
    for name in film_names:
        film = device.films[name]
        # We want all points that are in a film and not in a hole.
        ix1d = np.logical_and(
            film.contains_points(x, y), np.logical_not(in_hole[film.layer])
        )
        ix1d = np.where(ix1d)[0]
        ix2d = np.ix_(ix1d, ix1d)
        # Form the linear system for the film:
        # -(Q * w - Lambda * Del2) @ gf = A @ gf = h
        # Eqs. 15-17 in [Brandt], Eqs 12-14 in [Kirtley1], Eqs. 12-14 in [Kirtley2].
        A = -(Q[ix2d] * weights[ix2d] - Lambda[ix1d] * Del2[ix2d])
        h = Hz_applied[ix1d] - Ha_eff[film.layer][ix1d]
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
                logger.warn(
                    f"Unable to solve for stream function in {layer} ({name}), "
                    f"maximum error {np.abs(errors).max():.3e}."
                )
    # Eq. 7 in [Kirtley1], Eq. 7 in [Kirtley2]
    screening_field = (Q * weights) @ g
    total_field = Hz_applied + screening_field

    return g, total_field, screening_field


def solve(
    *,
    device: Device,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
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
    calculate the screening field from all other layers and recompute the
    stream function and fields based on the sum of the applied field
    and the screening fields from all other layers.

    3. If iterations > 1, then repeat step 2 (iterations - 1) times.


    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
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

    field_conversion = field_conversion_factor(
        field_units,
        current_units,
        length_units=device.length_units,
        ureg=device.ureg,
    )
    logger.debug(
        f"Conversion factor from {device.ureg(field_units).units:~P} to "
        f"{device.ureg(current_units) / device.ureg(device.length_units):~P}: "
        f"{field_conversion:~P}."
    )
    field_conversion_magnitude = field_conversion.magnitude
    # Compute the stream functions and fields for each layer
    # given only the applied field.
    for name, layer in device.layers.items():
        logger.info(f"Calculating {name} response to applied field.")

        def layer_field(x, y):
            # Units: current_units / device.length_units
            return applied_field(x, y, layer.z0) * field_conversion_magnitude

        g, total_field, screening_field = brandt_layer(
            device=device,
            layer=name,
            applied_field=layer_field,
            circulating_currents=circulating_currents,
            current_units=current_units,
            check_inversion=check_inversion,
        )
        # Units: current_units
        streams[name] = g
        # Units: current_units / device.length_units
        fields[name] = total_field
        screening_fields[name] = screening_field

    solution = BrandtSolution(
        device=device,
        streams=streams,
        fields={
            layer: field / field_conversion_magnitude for layer, field in fields.items()
        },
        screening_fields={
            layer: screening_field / field_conversion_magnitude
            for layer, screening_field in screening_fields.items()
        },
        applied_field=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
    )
    solutions.append(solution)

    if coupled and len(device.layers) > 1:
        if iterations < 1:
            raise ValueError(f"Iterations ({iterations}) cannot be less than 1.")

        tri_points = centroids(points, triangles)
        tri_areas = areas(points, triangles)
        # Compute ||(x, y) - (xt, yt)||^2
        rho2 = cdist(points, tri_points, metric="sqeuclidean")
        # Cache the calculated kernel matrices.
        kernels = {}
        for i in range(iterations):
            # Calculate the screening fields at each layer from every other layer
            other_screening_fields = {}
            for name, layer in device.layers.items():
                Hzr = np.zeros(points.shape[0], dtype=float)
                for other_name, other_layer in device.layers.items():
                    if name == other_name:
                        continue
                    logger.info(
                        f"Calculating screening field at {name} "
                        f"from {other_name} ({i+1}/{iterations})."
                    )
                    dz = other_layer.z0 - layer.z0
                    # Average stream function over all vertices in each triangle to
                    # estimate the stream function value at the centroid of the triangle.
                    g = streams[other_name][triangles].mean(axis=1)
                    # Calculate the dipole kernel and integrate
                    # Eqs. 1-2 in [Brandt], Eqs. 5-6 in [Kirtley1], Eqs. 5-6 in [Kirtley2].
                    q = kernels.get((name, other_name), None)
                    if q is None:
                        q = (2 * dz ** 2 - rho2) / (
                            4 * np.pi * (dz ** 2 + rho2) ** (5 / 2)
                        )
                        kernels[(name, other_name)] = q
                    Hzr += np.sum(tri_areas * q * g, axis=1)
                other_screening_fields[name] = Hzr

            # Solve again with the screening fields from all layers
            streams = {}
            fields = {}
            screening_fields = {}
            for name, layer in device.layers.items():
                logger.info(
                    f"Calculating {name} response to applied field and "
                    f"screening field from other layers ({i+1}/{iterations})."
                )

                def layer_field(x, y):
                    # Units: current_units / device.length_units
                    return (
                        applied_field(x, y, layer.z0) * field_conversion_magnitude
                        + other_screening_fields[name]
                    )

                g, total_field, screening_field = brandt_layer(
                    device=device,
                    layer=name,
                    applied_field=layer_field,
                    circulating_currents=circulating_currents,
                    current_units=current_units,
                    check_inversion=check_inversion,
                    check_lambda=False,
                )
                streams[name] = g
                fields[name] = total_field
                screening_fields[name] = screening_field

            solution = BrandtSolution(
                device=device,
                streams=streams,
                fields={
                    layer: field / field_conversion_magnitude
                    for layer, field in fields.items()
                },
                screening_fields={
                    layer: screening_field / field_conversion_magnitude
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

    Args:
        device: The ``Device`` that was solved
        streams: A dict of ``{layer_name: stream_function}``
        fields: A dict of ``{layer_name: total_field}``
        screening_fields: A dict of ``{layer_name: screening_field}``
        applied_field: The function defining the applied field
        field_units: Units of the applied field
        current_units: Units used for current quantities.
        circulating_currents: A dict of ``{hole_name: circulating_current}``
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
        circulating_currents: Optional[
            Dict[str, Union[float, str, pint.Quantity]]
        ] = None,
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
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
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
            xgrid = xgrid * self.device.ureg(self.device.length_units)
            ygrid = ygrid * self.device.ureg(self.device.length_units)
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
        grid_shape: Union[int, Tuple[int, int]] = (200, 200),
        layers: Optional[Union[str, List[str]]] = None,
        method: Optional[str] = "cubic",
        units: Optional[str] = None,
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
            units: The desired units for the current density. Defaults to
                ``self.current_units / self.device.length_units``.
            with_units: Whether to return arrays of pint.Quantities with units attached.

        Returns:
            x grid, y grid, dict of interpolated current density for each layer
        """
        xgrid, ygrid, streams = self.grid_data(
            dataset="streams",
            layers=layers,
            grid_shape=grid_shape,
            method=method,
            with_units=True,
            **kwargs,
        )
        units = units or f"{self.current_units} / {self.device.length_units}"
        Js = {}
        for name, g in streams.items():
            # J = [dg/dy, -dg/dx]
            # y is axis 0 (rows), x is axis 1 (columns)
            dg_dy, dg_dx = np.gradient(
                g.magnitude, ygrid[:, 0].magnitude, xgrid[0, :].magnitude
            )
            J = (np.array([dg_dy, -dg_dx]) * g.units / xgrid.units).to(units)
            if not with_units:
                J = J.magnitude
            Js[name] = J
        if not with_units:
            xgrid = xgrid.magnitude
            ygrid = ygrid.magnitude
        return xgrid, ygrid, Js

    def polygon_flux(
        self,
        polygons: Optional[Union[str, List[str]]] = None,
        units: Optional[Union[str, pint.Unit]] = None,
        with_units: bool = True,
    ) -> Dict[str, Union[float, pint.Quantity]]:
        """Compute the flux through all polygons (films, holes, and flux regions)
        by integrating the calculated fields.

        Args:
            polygons: Name(s) of the polygon(s) for which to compute the flux.
                Default: All polygons.
            with_units: Whether to a dict of pint.Quantities with units attached.

        Returns:
            dict of flux for each polygon in units of ``[self.field_units * device.length_units**2]``.
        """
        films = list(self.device.films)
        holes = list(self.device.holes)
        abstract_regions = list(self.device.abstract_regions)
        all_polygons = films + holes + abstract_regions

        if isinstance(polygons, str):
            polygons = [polygons]
        if polygons is None:
            polygons = all_polygons
        else:
            for poly in polygons:
                if poly not in all_polygons:
                    raise ValueError(f"Unknown polygon, {poly}.")

        ureg = self.device.ureg
        new_units = units or f"{self.field_units} * {self.device.length_units}**2"
        if isinstance(new_units, str):
            new_units = ureg(new_units)

        points = self.device.points
        triangles = self.device.triangles

        tri_areas = areas(points, triangles) * ureg(f"{self.device.length_units}") ** 2
        xt, yt = centroids(points, triangles).T
        mesh = Triangulation(*points.T, triangles=triangles)
        fluxes = {}
        for name in polygons:
            if name in films:
                poly = self.device.films[name]
            elif name in holes:
                poly = self.device.holes[name]
            else:
                poly = self.device.abstract_regions[name]
            ix = poly.contains_points(xt, yt, index=True)
            h_interp = LinearTriInterpolator(mesh, self.fields[poly.layer])
            field = h_interp(xt, yt)
            # LinearTriInterpolator returns a masked array
            field = np.asarray(field[ix]) * ureg(self.field_units)
            area = tri_areas[ix]
            # Convert field to B = mu0 * H
            field = convert_field(field, "mT", ureg=ureg)
            flux = np.sum(field * area).to(new_units)
            if with_units:
                fluxes[name] = flux
            else:
                fluxes[name] = flux.magnitude
        return fluxes

    def field_at_position(
        self,
        positions: np.ndarray,
        zs: Optional[Union[float, np.ndarray]] = None,
        vector: bool = False,
        units: Optional[str] = None,
        with_units: Optional[bool] = True,
        return_sum: bool = True,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Calculate the field due to currents in the device at any point(s) in space.

        Args:
            positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array of (x, y, z)
                coordinates at which to calculate the magnetic field. A single list like [x, y]
                or [x, y, z] is also allowed.
            zs: z coordinates at which to calculate the field. If positions has shape (m, 3), then
                this argument is not allowed. If zs is a scalar, then the fields are calculated in
                a plane parallel to the x-y plane. If zs is any array, then it must be same length
                as positions.
            vector: Whether to return the full vector magnetic field or just the z component.
            units: Units to which to convert the fields (can be either magnetic field H or
                magnetic flux density B = mu0 * H). If not give, then the fields are returned in
                units of ``self.field_units``.
            with_units: Whether to return the fields as ``pint.Quantity`` with units attached.
            return_sum: Whether to return the sum of the fields from all layers in the device,
                or a dict of ``{layer_name: field_from_layer}``.

        Returns:
            An np.ndarray if return_sum is True, otherwise a dict of ``{layer_name: field_from_layer}``.
            If with_units is True, then the array(s) will contain pint.Quantities. ``field_from_layer``
            will have shape ``(m, )`` if vector is False, or shape ``(m, 3)`` if ``vector`` is True.
        """
        device = self.device
        ureg = device.ureg
        points = device.points
        triangles = device.triangles

        units = units or self.field_units
        old_units = f"{self.current_units} / {device.length_units}"

        # In case something like a list [x, y] or [x, y, z] is given
        positions = np.atleast_2d(positions)
        # If positions includes z coordinates, peel those off here
        if positions.shape[1] == 3:
            if zs is not None:
                raise ValueError(
                    "If positions has shape (m, 3) then zs cannot be specified."
                )
            zs = positions[:, 2]
            positions = positions[:, :2]
        elif isinstance(zs, (int, float)):
            # constant zs
            zs = zs * np.ones(points.shape[0], dtype=float)
        if not isinstance(zs, np.ndarray):
            raise ValueError(f"Expected zs to be an ndarray, but got {type(zs)}.")
        if zs.ndim == 1:
            # We need zs to be shape (m, 1)
            zs = zs[:, np.newaxis]

        tri_points = centroids(points, triangles)
        tri_areas = areas(points, triangles)
        # Compute ||(x, y) - (xt, yt)||^2
        rho2 = cdist(positions, tri_points, metric="sqeuclidean")
        mesh = Triangulation(*points.T, triangles=triangles)

        Hz_applied = self.applied_field(positions[:, 0], positions[:, 1], zs)
        if vector:
            H_applied = np.stack(
                [np.zeros_like(Hz_applied), np.zeros_like(Hz_applied), Hz_applied],
                axis=1,
            )
        else:
            H_applied = Hz_applied
        H_applied = convert_field(
            H_applied,
            units,
            old_units=old_units,
            ureg=ureg,
            magnitude=(not with_units),
        )

        fields = {}
        # Compute the fields at the specified positions from the currents in each layer
        for name, layer in device.layers.items():
            dz = zs - layer.z0
            if np.any(dz == 0):
                raise ValueError(
                    f"Cannot calculate fields in the same plane as layer {name}."
                )
            g_interp = LinearTriInterpolator(mesh, self.streams[name])
            # g has units of [current]
            g = g_interp(*tri_points.T)
            # Q is the dipole kernel for the z component, Hz
            # Q has units of [length]^(2*(1-5/2)) = [length]^(-3)
            Q = (2 * dz ** 2 - rho2) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
            # tri_areas has units of [length]^2
            # So here Hz is in units of [current] * [length]^(-1)
            Hz = np.asarray(np.sum(tri_areas * Q * g, axis=1))
            if vector:
                # See Eq. 15 of Kirtley RSI 2016, arXiv:1605.09483
                # Pairwise difference between all x positions
                d = np.subtract.outer(positions[:, 0], tri_points[:, 0])
                # Kernel for x component, Hx
                Q = (3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                Hx = np.asarray(np.sum(tri_areas * Q * g, axis=1))

                # Pairwise difference between all y positions
                d = np.subtract.outer(positions[:, 1], tri_points[:, 1])
                # Kernel for y component, Hy
                Q = (3 * dz * d) / (4 * np.pi * (dz ** 2 + rho2) ** (5 / 2))
                Hy = np.asarray(np.sum(tri_areas * Q * g, axis=1))

                H = np.stack([Hx, Hy, Hz], axis=1)
            else:
                H = Hz
            fields[name] = convert_field(
                H,
                units,
                old_units=old_units,
                ureg=ureg,
                magnitude=(not with_units),
            )
        if return_sum:
            fields = sum(fields.values()) + H_applied
        else:
            fields["applied_field"] = H_applied
        return fields
