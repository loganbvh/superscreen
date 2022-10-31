import itertools
from typing import Optional, Tuple, Union

import numpy as np
from scipy.constants import mu_0

from ..parameter import Parameter
from ..units import ureg


def dipole_field(
    eval_coords: np.ndarray,
    r0: Union[np.ndarray, Tuple[float, float, float]] = (0, 0, 0),
    moment: Union[np.ndarray, Tuple[float, float, float]] = (0, 0, 0),
) -> np.ndarray:
    """Returns the 3D field from a single dipole with the given moment
    (in units of amps * meters ** 2) located at the position ``r0``, evaluated
    at coordinates ``eval_coords = [x, y, z]``.

    Given :math:`\\vec{r}=(x, y, z) - \\vec{r}_0`, the magnetic field is:

    .. math::

        \\vec{B}(\\vec{r}) = \\frac{\\mu_0}{4\\pi}
            \\frac{
                3\\hat{r}(\\hat{r}\\cdot\\vec{m}) - \\vec{m}
            }{
                |\\vec{r}|^3
            }

    Args:
        eval_coords: (x, y, z) coordinates (in meters) at which to
            evaluate the field. Either a sequence of length 3
            (for a single position) or an array of shape  ``(n, 3)``
            (for ``n`` positions.).
        r0: Coordinates ``(x0, y0, z0)`` (in meters) of the dipole position,
            shape ``(3,)`` or ``(1, 3)``.
        moment: Dipole moment ``(mx, my, mz)`` in units of amps * meters ** 2,
            shape ``(3,)`` or ``(1, 3)``.

    Returns:
        Magnetic field ``(Bx, By, Bz)`` in Tesla evaluated at ``(x, y, z)``:
        An array with shape ``(3, )`` if ``x, y, z`` are scalars, or shape ``(n, 3)``
        if ``x, y, z`` are vectors with shape ``(n, )``.
    """
    moment, r0 = np.atleast_1d(moment, r0)
    r = np.atleast_2d(eval_coords).reshape((-1, 3))
    r = r - r0
    # \sqrt{\vec{r}\cdot\vec{r}}
    norm_r = np.sqrt(np.einsum("ij, ij -> i", r, r))[:, np.newaxis]
    # \vec{m}\cdot\vec{r}
    m_dot_r = np.einsum("j, ij -> i", moment, r)[:, np.newaxis]
    # \frac{3\hat{r}(\hat{r}\cdot\vec{m}) - \vec{m}}{|\vec{r}|^3} =
    # \frac{3\vec{r}(\vec{r}\cdot\vec{m})}{|\vec{r}|^5} - \frac{\vec{m}}{|\vec{r}|^3}
    B = 3 * r * m_dot_r / norm_r**5 - moment / norm_r**3
    return mu_0 / (4 * np.pi) * B.squeeze()


def dipole_distribution(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    dipole_positions: np.ndarray,
    dipole_moments: Union[np.ndarray, Tuple[float, float, float]],
    component: Optional[str] = None,
    length_units: str = "um",
    moment_units: str = "mu_B",
) -> np.ndarray:
    """Returns the 3D field :math:`\\vec{B}=\\mu_0\\vec{H}`, or one of its components,
    from a distribution of dipoles with given moments (in units of the Bohr magneton)
    located at the given positions, evaluated at coordinates ``(x, y, z)``.

    Args:
        x: x-coordinate(s) (in meters) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        y: y-coordinate(s) (in meters) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        z: z-coordinate(s) (in meters) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        dipole_positions: Coordinates ``(x0_i, y0_i, z0_i)`` of the position of
            each dipole ``i``, shape ``(m, 3)`` (in meters) .
        dipole_moments: Dipole moments ``(mx_i, my_i, mz_i)`` in units of
            amps * meters ** 2. If dipole_moments has shape ``(3, )`` or ``(1, 3)``,
            then all dipoles are assigned the same moment. Otherwise, dipole_moments
            must have shape ``(m, 3)``, i.e. the moment is specified for each dipole.
        component: The component of the magnetic field to return: "x", "y", "z",
            or None. If None, the vector magnetic field (shape ``(n, 3)``) is returned.
        length_units: The units for the positions coordinates ``x``, ``y``, ``z``,
            and ``dipole_positions``.
        moment_units: The units for ``dipole_moments``, for example the Bohr magneton
            "mu_B" or SI base units "A * m ** 2".

    Returns:
        Magnetic field ``(Bx, By, Bz)`` (or one of its components) in Tesla evaluated
        at ``(x, y, z)``: An array with shape ``(3, )`` if ``x, y, z`` are scalars,
        or shape ``(n, 3)`` if ``x, y, z`` are vectors with shape ``(n, )``.
    """
    index = Ellipsis if component is None else list("xyz").index(component)
    length_units = ureg(length_units)
    dipole_moments = (dipole_moments * ureg(moment_units)).to("A * m ** 2").magnitude
    dipole_positions = (dipole_positions * length_units).to("m").magnitude
    x = (x * length_units).to("m").magnitude
    y = (y * length_units).to("m").magnitude
    z = (z * length_units).to("m").magnitude
    if len(z) == 1:
        z = z * np.ones_like(x)
    eval_coords = np.stack(np.atleast_1d(x, y, z), axis=1)
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
    B = sum(
        dipole_field(eval_coords, moment=moment, r0=r0)
        for moment, r0 in zip(dipole_moments, dipole_positions)
    )
    return np.atleast_2d(B)[:, index]


def DipoleField(
    *,
    dipole_positions: Union[np.ndarray, Tuple[float, float, float]],
    dipole_moments: Union[np.ndarray, Tuple[float, float, float]],
    component: Optional[str] = None,
    length_units: str = "um",
    moment_units: str = "mu_B",
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
        component: The component of the field to calculate: "x", "y", "z", or None.
            If None, then the vector field (shape ``(m, 3)``) is returned.
        length_units: The units for the positions coordinates ``x``, ``y``, ``z``,
            and ``dipole_positions``.
        moment_units: The units for ``dipole_moments``, for example the Bohr magneton
            "mu_B" or SI base units "A * m ** 2".

    Returns:
        A Parameter that computes a given component of the field
        :math:`\\mu_0\\vec{H}(x, y, z)` in Tesla for a given distribution of dipoles.
    """
    if component not in (None, "x", "y", "z"):
        raise ValueError(
            f"Component must be 'x', 'y', 'z', or None (got {component!r})."
        )
    return Parameter(
        dipole_distribution,
        dipole_positions=dipole_positions,
        dipole_moments=dipole_moments,
        component=component,
        length_units=length_units,
        moment_units=moment_units,
    )
