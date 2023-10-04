from typing import Optional, Union

import numba
import numpy as np
from scipy.constants import mu_0
from scipy.spatial import Delaunay

from ..device.utils import vertex_areas
from ..parameter import Parameter
from ..units import ureg


@numba.njit(fastmath=True, parallel=True)
def _biot_savart_2d_z(
    eval_positions: np.ndarray,
    positions: np.ndarray,
    current_densities: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Returns the z-component of the magnetic field (in tesla)
    from a given sheet current density distribution.

    Args:
        eval_positions: The ``(x, y, z)`` coordinates at which to evaluate the field in meters
        positions: The ``(x, y, z)`` coordinates of the sheet current in meters
        current_densities: The sheet current density in amps / meter
        areas: The effective areas of the sheet current mesh in meters**2

    Returns:
        The z-component of the magnetic field in tesla
    """
    assert eval_positions.ndim == 2
    assert eval_positions.shape[1] == 3
    assert positions.ndim == 2
    assert positions.shape[0] == areas.shape[0] == current_densities.shape[0]
    assert positions.shape[1] == 3

    Jx = current_densities[:, 0]
    Jy = current_densities[:, 1]
    Bz_out = np.empty(len(eval_positions), dtype=float)

    for i in numba.prange(eval_positions.shape[0]):
        Jx_dy = 0.0
        Jy_dx = 0.0
        for k in range(positions.shape[0]):
            dx = eval_positions[i, 0] - positions[k, 0]
            dy = eval_positions[i, 1] - positions[k, 1]
            dz = eval_positions[i, 2] - positions[k, 2]
            pref = (
                (mu_0 / (4 * np.pi))
                * areas[k]
                * (dx * dx + dy * dy + dz * dz) ** (-3 / 2)
            )
            Jx_dy += pref * Jx[k] * dy
            Jy_dx += pref * Jy[k] * dx
        Bz_out[i] = Jx_dy - Jy_dx
    return Bz_out


@numba.njit(fastmath=True, parallel=True)
def _biot_savart_2d_vector(
    eval_positions: np.ndarray,
    positions: np.ndarray,
    current_densities: np.ndarray,
    areas: np.ndarray,
) -> np.ndarray:
    """Returns the vector magnetic field (in tesla)
    from a given sheet current density distribution.

    Args:
        eval_positions: The ``(x, y, z)`` coordinates at which to evaluate the field in meters
        positions: The ``(x, y, z)`` coordinates of the sheet current in meters
        current_densities: The sheet current density in amps / meter
        areas: The effective areas of the sheet current mesh in meters**2

    Returns:
        The vector magnetic field in tesla
    """
    assert eval_positions.ndim == 2
    assert eval_positions.shape[1] == 3
    assert positions.ndim == 2
    assert positions.shape[0] == areas.shape[0] == current_densities.shape[0]
    assert positions.shape[1] == 3

    Jx = current_densities[:, 0]
    Jy = current_densities[:, 1]
    B_out = np.empty((len(eval_positions), 3), dtype=float)

    for i in numba.prange(eval_positions.shape[0]):
        Jx_dy = 0.0
        Jy_dx = 0.0
        Jx_dz = 0.0
        Jy_dz = 0.0
        for k in range(positions.shape[0]):
            dx = eval_positions[i, 0] - positions[k, 0]
            dy = eval_positions[i, 1] - positions[k, 1]
            dz = eval_positions[i, 2] - positions[k, 2]
            pref = (
                (mu_0 / (4 * np.pi))
                * areas[k]
                * (dx * dx + dy * dy + dz * dz) ** (-3 / 2)
            )
            Jx_dy += pref * Jx[k] * dy
            Jy_dx += pref * Jy[k] * dx
            Jx_dz += pref * Jx[k] * dz
            Jy_dz += pref * Jy[k] * dz
        B_out[i, 0] = Jy_dz
        B_out[i, 1] = -Jx_dz
        B_out[i, 2] = Jx_dy - Jy_dx
    return B_out


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
    eval_positions = np.array([x, y, z]).T * to_meter
    positions, current_densities = np.atleast_2d(positions, current_densities)
    current_densities = current_densities * to_amp_per_meter
    positions = positions * to_meter
    z0 = z0 * np.ones(len(positions)) * to_meter
    if areas is None:
        # Triangulate the current sheet to assign an effective area to each vertex.
        triangles = Delaunay(positions).simplices
        areas = vertex_areas(positions, triangles)
    else:
        areas = areas * to_meter**2
    positions = np.concatenate([positions, z0[:, np.newaxis]], axis=1)
    # Evaluate the Biot-Savart integral.
    if vector:
        B = _biot_savart_2d_vector(eval_positions, positions, current_densities, areas)
    else:
        B = _biot_savart_2d_z(eval_positions, positions, current_densities, areas)
    return B


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
