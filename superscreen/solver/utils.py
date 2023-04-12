import itertools
import logging
from typing import Optional, Union

import numpy as np
import pint
import scipy.sparse as sp

from ..fem import cdist_batched

logger = logging.getLogger("solve")


class LambdaInfo:
    lambda_str = "\u03bb"
    Lambda_str = "\u039b"

    def __init__(
        self,
        *,
        layer: str,
        Lambda: np.ndarray,
        london_lambda: Optional[np.ndarray] = None,
        thickness: Optional[float] = None,
    ):
        self.layer = layer
        self.Lambda = Lambda
        self.london_lambda = london_lambda
        self.thickness = thickness
        self.inhomogeneous = (
            np.ptp(self.Lambda) / max(np.min(np.abs(self.Lambda)), np.finfo(float).eps)
            > 1e-6
        )
        if self.inhomogeneous:
            logger.warning(
                f"Inhomogeneous {LambdaInfo.Lambda_str} in layer '{self.layer}', "
                f"which violates the assumptions of the London model. "
                f"Results may not be reliable."
            )
        if self.london_lambda is not None:
            assert self.thickness is not None
            assert np.allclose(self.Lambda, self.london_lambda**2 / self.thickness)
        if np.any(self.Lambda < 0):
            raise ValueError(f"Negative Lambda in layer '{layer}'.")


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
    distances = cdist_batched(points, points, batch_size=batch_size, metric="euclidean")
    if dtype is not None:
        distances = distances.astype(dtype, copy=False)
    with np.errstate(divide="ignore"):
        q = 1 / (4 * np.pi * distances**3)
    np.fill_diagonal(q, np.inf)
    return q.astype(dtype, copy=False)


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


def convert_field(
    value: Union[np.ndarray, float, str, pint.Quantity],
    new_units: Union[str, pint.Unit],
    old_units: Optional[Union[str, pint.Unit]] = None,
    ureg: Optional[pint.UnitRegistry] = None,
    with_units: bool = True,
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
        with_units: Whether to return a ``pint.Quantity`` with units attached.

    Returns:
        The converted value, either a pint.Quantity (scalar or array with units),
        or an array or float without units, depending on the ``with_units`` argument.
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
    if not with_units:
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
