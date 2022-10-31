from typing import Optional, Union

import numpy as np
from scipy.constants import mu_0
from scipy.spatial import Delaunay

from ..fem import mass_matrix
from ..parameter import Parameter
from ..units import ureg


def biot_savart_2d(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    positions: np.ndarray,
    current_densities: np.ndarray,
    z0: float = 0,
    areas: Optional[np.ndarray] = None,
    length_units: str = "um",
    current_units: str = "uA",
    vector: bool = True,
) -> np.ndarray:
    """Returns the magnetic field (in tesla) from a sheet of current located at
    vertical positon ``z0`` (in units of ``length_units``). The current is
    parameterized by a set of ``current_densities`` (in units of
    ``current_units / length_units``) and x-y ``positions`` (in units of
    ``length_units``), and the field is evaluated at coordinates ``(x, y, z)``.

    .. math::

        \\mu_0H_x(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        \\frac{J_y(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{z}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'\\\\
        \\mu_0H_y(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        -\\frac{J_x(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{z}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'\\\\
        \\mu_0H_z(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        \\frac{J_x(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{y}
        - J_y(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{x}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'


    where :math:`\\vec{r}=(x, y, z)` and :math:`\\vec{r}'=(x', y', z_0)`.

    Args:
        x: x-coordinate(s) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        y: y-coordinate(s) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        z: z-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        positions: Coordinates ``(x0, y0)`` of the current sheet,
            shape ``(m, 2)``.
        current_densities: 2D current density ``(Jx, Jy)``, shape``(m, 2)``.
        z0: Vertical (z) position of the current sheet.
        areas: Vertex areas for ``positions`` in units of ``length_units**2``. If None,
            the ``positions`` are triangulated to calculate vertex areas.
        length_units: The units for all coordinates.
        current_units: The units for current values. The ``current_densities`` are
            assumed to be in units of ``current_units / length_units``.
        vector: Return the full vector magnetic field (shape ``(n, 3)``) rather
            than just the z-component (shape ``(n, )``).

    Returns:
        Magnetic field in tesla evaluated at ``(x, y, z)``. If ``vector`` is True,
        returns the vector magnetic field :math:`\\mu_0\\vec{H}` (shape ``(n, 3)``).
        Otherwise, returns the the :math:`z`-component, :math:`\\mu_0H_z`
        (shape ``(n,)``).
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
    if areas is None:
        # Triangulate the current sheet to assign an effective area to each vertex.
        triangles = Delaunay(positions).simplices
        areas = mass_matrix(positions, triangles)
    else:
        areas = areas * to_meter**2
    # Evaluate the Biot-Savart integral.
    pref = (mu_0 / (4 * np.pi)) * areas * (dx**2 + dy**2 + dz**2) ** (-3 / 2)
    Jx_dy = np.einsum("ij, ij, j -> i", pref, dy, Jx)
    Jy_dx = np.einsum("ij, ij, j -> i", pref, dx, Jy)
    Bz = Jx_dy - Jy_dx
    if not vector:
        return Bz
    Jy_dz = np.einsum("ij, ij, j -> i", pref, dz, Jy)
    Jx_dz = np.einsum("ij, ij, j -> i", pref, dz, Jx)
    Bx = Jy_dz
    By = -Jx_dz
    return np.stack([Bx, By, Bz], axis=1)


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

        \\mu_0H_z(\\vec{r})=\\frac{\\mu_0}{4\\pi}\\int_S
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
        vector=False,
    )
