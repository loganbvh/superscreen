from typing import Union

import numpy as np

from ..parameter import Parameter


def vortex(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    x0: float = 0,
    y0: float = 0,
    z0: float = 0,
    nPhi0: Union[int, float] = 1,
) -> Union[float, np.ndarray]:
    """Field :math:`\\mu_0H_z` from an isolated vortex (monopole)
    in units of ``Phi_0 / (length_units)**2``.

    .. math::

        \\mu_0H_z(\\vec{r}-\\vec{r}_0) = \\frac{n\\Phi_0}{2\\pi}
            \\frac{(\\vec{r}-\\vec{r}_0)\\cdot\\hat{z}}{|(\\vec{r}-\\vec{r}_0)|^3}

    Args:
        x, y, z: Position coordinates.
        x0, y0, z0: Vortex position
        nPhi0: Number of flux quanta contained in the vortex.

    Returns:
        The field at the given coordinates in units of ``Phi_0 / (length_units)**2``.
    """
    xp = x - x0
    yp = y - y0
    zp = z - z0
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


def VortexField(
    x0: float = 0, y0: float = 0, z0: float = 0, nPhi0: Union[int, float] = 1
) -> Parameter:
    """Returns a Parameter that computes the z-component of the field from a vortex
    (monopole) located at position ``(x0, y0, z0)`` containing a total of
    ``nPhi0`` flux quanta.

    .. math::

        \\mu_0H_z(\\vec{r}-\\vec{r}_0) = \\frac{n\\Phi_0}{2\\pi}
            \\frac{(\\vec{r}-\\vec{r}_0)\\cdot\\hat{z}}{|(\\vec{r}-\\vec{r}_0)|^3}

    Args:
        x0, y0, z0: Coordinates of the vortex position.
        nPhi0: Number of flux quanta contained in the vortex.

    Returns:
        A Parameter that returns the out-of-plane field in units of
        ``Phi_0 / (length_units)**2``.
    """
    return Parameter(vortex, x0=x0, y0=y0, z0=z0, nPhi0=nPhi0)
