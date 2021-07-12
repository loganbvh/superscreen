from typing import Union, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .parameter import Parameter

Numeric = Union[int, float, np.ndarray]


def constant(
    x: Numeric, y: Numeric, z: Numeric, value: Optional[Union[int, float]] = 0
) -> Numeric:
    """Constant field.

    Args:
        x, y, z: Position coordinates.
        value: Value of the field.
    """
    return value * np.ones_like(x)


def vortex(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    vortex_position: Optional[Tuple[float, float, float]] = (0, 0, 0),
    nPhi0: Optional[int] = 1,
) -> Numeric:
    """Field from an isolated vortex.

    Args:
        x, y, z: Position coordinates.
        vortex_position: (x, y, z) position of the vortex.
        nPhi0: Number of flux quanta contained in the vortex.
    """
    xv, yv, zv = vortex_position
    xp = x - xv
    yp = y - yv
    zp = z - zv
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


def tilt(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    *,
    axis: str,
    angle: Optional[float] = 0,
    offset: Optional[float] = 0,
) -> Tuple[Numeric, Numeric, Numeric]:
    if axis not in "xy":
        raise ValueError(f"Axis must be 'x' or 'y', got {axis}.")

    if axis == "x":
        i = 1
    else:
        i = 0

    if angle == 0:
        return x, y, z

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.asarray(z)
    if z.ndim == 0:
        z = z * np.ones_like(x)
    points = np.array([x, y, z]).T

    points[:, i] -= offset
    r = Rotation.from_euler(axis, angle, degrees=True)
    points = r.apply(points)
    points[:, i] += offset
    x, y, z = [np.squeeze(a) for a in points.T]
    if x.ndim == 0:
        x = x.item()
    if y.ndim == 0:
        y = y.item()
    if z.ndim == 0:
        z = z.item()
    return x, y, z


def tilted_vortex(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    vortex_position: Optional[Tuple[float, float, float]] = (0, 0, 0),
    nPhi0: Optional[int] = 1,
    x_axis_tilt: Optional[float] = 0,
    x_axis_offset: Optional[float] = 0,
    y_axis_tilt: Optional[float] = 0,
    y_axis_offset: Optional[float] = 0,
    tilt_x_first: Optional[bool] = True,
):
    if tilt_x_first:
        x, y, z = tilt(x, y, z, axis="x", angle=x_axis_tilt, offset=x_axis_offset)
        x, y, z = tilt(x, y, z, axis="y", angle=y_axis_tilt, offset=y_axis_offset)
    else:
        x, y, z = tilt(x, y, z, axis="y", angle=y_axis_tilt, offset=y_axis_offset)
        x, y, z = tilt(x, y, z, axis="x", angle=x_axis_tilt, offset=x_axis_offset)

    return vortex(x, y, z, vortex_position=vortex_position, nPhi0=nPhi0)


def ConstantField(value: float) -> Parameter:
    return Parameter(constant, value=value)


def VortexField(
    position: Tuple[float, float, float], nPhi0: Optional[int] = 1
) -> Parameter:
    return Parameter(vortex, vortex_position=position, nPhi0=nPhi0)


def TiltedVortexField(
    vortex_position: Tuple[float, float, float],
    nPhi0: Optional[int] = 1,
    x_axis_tilt: Optional[float] = 0,
    x_axis_offset: Optional[float] = 0,
    y_axis_tilt: Optional[float] = 0,
    y_axis_offset: Optional[float] = 0,
    tilt_x_first: Optional[bool] = True,
) -> Parameter:
    return Parameter(
        tilted_vortex,
        vortex_position=vortex_position,
        nPhi0=nPhi0,
        x_axis_tilt=x_axis_tilt,
        x_axis_offset=x_axis_offset,
        y_axis_tilt=y_axis_tilt,
        y_axis_offset=y_axis_offset,
        tilt_x_first=tilt_x_first,
    )
