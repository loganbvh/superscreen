import itertools
from typing import Union, Tuple

import numpy as np
from scipy.constants import mu_0

from ..parameter import Parameter


def dipole_field(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    r0: Union[np.ndarray, Tuple[float, float, float]] = (0, 0, 0),
    moment: Union[np.ndarray, Tuple[float, float, float]] = (0, 0, 0),
) -> np.ndarray:
    """Returns the 3D field from a single dipole with the given moment
    (in units of the Bohr magneton) located at the position ``r0``, evaluated
    at coordinates ``(x, y, z)``.

    Given :math:`\\vec{r}=(x, y, z) - \\vec{r}_0`, the magnetic field is:

    .. math::

        \\vec{B}(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}
            \\frac{
                3\\hat{r}(\\hat{r}\\cdot\\vec{m}) - \\vec{m}
            }{
                |\\vec{r}|^3
            }

    Args:
        x: x-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        y: y-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        z: z-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        r0: Coordinates ``(x0, y0, z0)`` of the dipole position,
            shape ``(3,)`` or ``(1, 3)``.
        moment: Dipole moment ``(mx, my, mz)`` in units of the Bohr magneton,
            shape ``(3,)`` or ``(1, 3)``.

    Returns:
        Magnetic field ``(Bx, By, Bz)`` in Tesla evaluated at ``(x, y, z)``:
        An array with shape ``(3, )`` if ``x, y, z`` are scalars, or shape ``(n, 3)``
        if ``x, y, z`` are vectors with shape ``(n, )``.
    """
    moment, r0 = np.atleast_1d(moment, r0)
    x, y, z = np.atleast_1d(x, y, z)
    r = np.stack([x, y, z], axis=1)
    r = r - r0
    # \sqrt{\vec{r}\cdot\vec{r}}
    norm_r = np.sqrt(np.einsum("ij,ij -> i", r, r))[:, np.newaxis]
    # \vec{m}\cdot\vec{r}
    m_dot_r = np.einsum("j,ij -> i", moment, r)[:, np.newaxis]
    # \frac{3\hat{r}(\hat{r}\cdot\vec{m}) - \vec{m}}{|\vec{r}|^3} =
    # \frac{3\vec{r}(\vec{r}\cdot\vec{m})}{|\vec{r}|^5} - \frac{\vec{m}}{|\vec{r}|^3}
    B = 3 * r * m_dot_r / norm_r ** 5 - moment / norm_r ** 3
    return mu_0 / (4 * np.pi) * B.squeeze()


def dipole_distribution(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    dipole_positions: np.ndarray,
    dipole_moments: Union[np.ndarray, Tuple[float, float, float]],
) -> np.ndarray:
    """Returns the 3D field :math:`\\vec{B}=\\mu_0\\vec{H}` from a
    distribution of dipoles with given moments (in units of the Bohr magneton)
    located at the given positions, evaluated at coordinates ``(x, y, z)``.

    Args:
        x: x-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        y: y-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        z: z-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        dipole_positions: Coordinates ``(x0_i, y0_i, z0_i)`` of the position of
            each dipole ``i``, shape ``(m, 3)``.
        dipole_moments: Dipole moments ``(mx_i, my_i, mz_i)`` in units of the
            Bohr magneton. If dipole_moments has shape ``(3, )`` or ``(1, 3)``,
            then all dipoles are assigned the same moment. Otherwise, dipole_moments
            must have shape ``(m, 3)``, i.e. the moment is specified for each dipole.

    Returns:
        Magnetic field ``(Bx, By, Bz)`` in Tesla evaluated at ``(x, y, z)``:
        An array with shape ``(3, )`` if ``x, y, z`` are scalars, or shape ``(n, 3)``
        if ``x, y, z`` are vectors with shape ``(n, )``.
    """
    dipole_positions, dipole_moments = np.atleast_2d(dipole_positions, dipole_moments)
    if dipole_moments.shape[0] == 1:
        # Assign each dipole the same moment
        dipole_moments = itertools.repeat(dipole_moments[0], dipole_positions.shape[0])
    elif dipole_moments.shape[0] != dipole_positions.shape[0]:
        raise ValueError(
            f"The number of dipole moments ({dipole_moments.shape[0]}) must be either"
            f"1 or equal to the the number of dipole positions "
            f"({dipole_positions.shape[0]})."
        )
    return sum(
        dipole_field(x, y, z, moment=moment, r0=r0)
        for moment, r0 in zip(dipole_moments, dipole_positions)
    )


def dipole_distribution_comp(x, y, z, *, dipole_positions, dipole_moments, component):
    index = "xyz".index(component)
    B = dipole_distribution(
        x,
        y,
        z,
        dipole_positions=dipole_positions,
        dipole_moments=dipole_moments,
    )
    return np.atleast_2d(B)[:, index]


def DipoleField(
    *,
    dipole_positions: Union[np.ndarray, Tuple[float, float, float]],
    dipole_moments: Union[np.ndarray, Tuple[float, float, float]],
    component: str = "z",
) -> Parameter:
    """Returns a Parameter that computes a given component of the field from
    a distribution of dipoles with given moments (in units of the Bohr magneton)
    located at the given positions.

    Given dipole positions :math:`\\vec{r}_{0, i}` and moments :math:`\\vec{m}_i`,
    the magnetic field is:

    .. math::

        \\mu_0\\vec{H}(\\vec{r}) = \\sum_i\\frac{\\mu_0}{4\\pi}
            \\frac{
                3\\hat{r}_i(\\hat{r}_i\\cdot\\vec{m}_i) - \\vec{m}_i
            }{
                |\\vec{r}_i|^3
            },

    where :math:`\\vec{r}_i=(x, y, z) - \\vec{r}_{0, i}`.

    Args:
        dipole_positions: Coordinates ``(x0_i, y0_i, z0_i)`` of the position of
            each dipole ``i``. Shape ``(3, )`` or ``(1, 3)`` for a single dipole, or
            shape ``(m, 3)`` for m dipoles.
        dipole_moments: Dipole moments ``(mx_i, my_i, mz_i)`` in units of the
            Bohr magneton. If dipole_moments has shape ``(3, )`` or ``(1, 3)``, then
            all dipoles are assigned the same moment. Otherwise, dipole_moments
            must have shape ``(m, 3)``, i.e. the moment is specified for each dipole.
        component: The component of the field to calculate: "x", "y", or "z".

    Returns:
        A Parameter that computes a given component of the field
        :math:`\\mu_0\\vec{H}(x, y, z)` in Tesla for a given distribution of dipoles.
    """
    component = component.lower()
    if component not in "xyz":
        raise ValueError(f"Component must be 'x', 'y', or 'z' (got '{component}').")
    dipole_positions, dipole_moments = np.atleast_2d(dipole_positions, dipole_moments)
    return Parameter(
        dipole_distribution_comp,
        dipole_positions=dipole_positions,
        dipole_moments=dipole_moments,
        component=component,
    )
