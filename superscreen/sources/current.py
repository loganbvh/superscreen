from typing import Union

import numpy as np
from scipy.constants import mu_0
from scipy.spatial import Delaunay

from ..units import ureg
from ..fem import mass_matrix
from ..parameter import Parameter


def biot_savart_2d(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    positions: np.ndarray,
    current_densities: np.ndarray,
    z0: float = 0,
    length_units: str = "um",
    current_units: str = "uA",
) -> np.ndarray:
    """Returns the z-component of the magnetic field (in tesla) from a sheet of current
    located at vertical positon ``z0`` (in units of ``lengt_units``). The current is
    parameterized by a set of ``current_densities`` (in units of
    ``current_units / length_units``) and x-y ``positions`` (in units of
    ``length_units``), and the field is evaluated at coordinates ``(x, y, z)``.

    Args:
        x: x-coordinate(s) (in meters) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        y: y-coordinate(s) (in meters) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        z: z-coordinate(s) (in meters) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        positions: Coordinates ``(x0, y0)`` (in meters) of the current sheet,
            shape ``(m, 2)``.
        current_densities: 2D current density ``(Jx, Jy)`` in units of amps / meter,
            shape``(m, 2)``.
        z0: Vertical (z) position of the current sheet.
        length_units: The units for all coordinates.
        current_units: The units for current values. The ``current_densities`` are
            assumed to be in units of ``current_units / length_units``.

    Returns:
        Magnetic field ``Bz`` in tesla evaluated at ``(x, y, z)``, shape ``(n, )``
    """
    # Convert everything to base units: meters and amps / meter.
    to_meter = ureg(length_units).to("m").magnitude
    to_amp_per_meter = ureg(f"{current_units} / {length_units}").to("A / m").magnitude
    x, y, z = np.atleast_1d(x, y, z)
    if z.shape[0] == 1:
        z = z * np.ones_like(x)
    x = x * to_meter
    y = y * to_meter
    z = z * to_meter
    positions, current_densities = np.atleast_2d(positions, current_densities)
    positions = positions * to_meter
    z0 = z0 * to_meter
    current_densities = current_densities * to_amp_per_meter
    # Calculate the pairwise distance between the current sheet and evaluation
    # points for each axis.
    x0, y0 = positions[:, 0], positions[:, 1]
    Jx, Jy = current_densities[:, 0], current_densities[:, 1]
    dx = np.subtract.outer(x, x0)
    dy = np.subtract.outer(y, y0)
    dz = np.subtract.outer(z, z0 * np.ones_like(x0))
    # Triangulate the current sheet to assign an effective area to each vertex.
    triangles = Delaunay(positions).simplices
    areas = mass_matrix(positions, triangles)
    # Evaluate the Biot-Savart integral.
    return (mu_0 / (4 * np.pi)) * (
        areas * (Jx * dy - Jy * dx) / (dx**2 + dy**2 + dz**2) ** (3 / 2)
    ).sum(axis=1)


def SheetCurrentField(
    *,
    sheet_positions: np.ndarray,
    current_densities: np.ndarray,
    z0: float,
    length_units: str = "um",
    current_units: str = "uA",
) -> Parameter:
    """Returns a Parameter that computes the z-component of the field from
    a 2D sheet of current parameterized by the given positions and current densities.

    The :math:`z`-component of the field from a 2D sheet of current :math:`S` lying in
    the plane :math:`z=z_0`  with spatially varying current density
    :math:`\\vec{J}=(J_x, J_y)` is given by:

    .. math::

        \\mu_0H_z(\\vec{r})=\\frac{\\mu_0}{2\\pi}\\int_S
        \\frac{J_x(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{y}
        - J_y(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{x}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r',

    where :math:`\\vec{r}=(x, y, z)` and :math:`\\vec{r}'=(x', y', z_0)`.

    Args:
        sheet_positions: Coordinates ``(x0, y0)`` (in meters) of the current sheet,
            shape ``(m, 2)``.
        current_densities: 2D current density ``(Jx, Jy)`` in units of amps / meter,
            shape ``(m, 2)``.
        z0: Vertical (z) position of the current sheet.
        length_units: The units for all coordinates.
        current_units: The units for current values. The ``current_densities`` are
            assumed to be in units of ``current_units / length_units``.

    Returns:
        A Parameter that computes :math:`\\mu_0\\vec{H}_z(x, y, z)` in Tesla for a
        given sheet current.
    """
    return Parameter(
        biot_savart_2d,
        positions=sheet_positions,
        current_densities=current_densities,
        z0=z0,
        length_units=length_units,
        current_units=current_units,
    )
