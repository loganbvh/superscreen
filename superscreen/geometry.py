from typing import Tuple

import numpy as np


def rotation_matrix(angle_radians: float) -> np.ndarray:
    """Returns a 2D rotation matrix."""
    c = np.cos(angle_radians)
    s = np.sin(angle_radians)
    return np.array([[c, -s], [s, c]])


def rotate(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotates an array of (x, y) coordinates counterclockwise by
    the specified angle.

    Args:
        coords: Shape (n, 2) array of (x, y) coordinates.
        angle_degrees: The angle by which to rotate the coordinates.

    Returns:
        Shape (n, 2) array of rotated coordinates (x', y')
    """
    coords = np.asarray(coords)
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    R = rotation_matrix(np.radians(angle_degrees))
    return (R @ coords.T).T


def ellipse(
    a: float,
    b: float,
    points: int = 100,
    center: Tuple[float, float] = (0, 0),
    angle: float = 0,
):
    """Returns the coordinates for an ellipse with major axis a and semimajor axis b,
    rotated by the specified angle about (0, 0), then translated to the specified center.

    Args:
        a: Major axis length
        b: Semi-major axis length
        points: Number of points in the circle
        center: Coordinates of the center of the circle
        angle: Angle (in degrees) by which to rotate counterclockwise about (0, 0)
            **before** translating to the specified center.

    Returns:
        A shape ``(points, 2)`` array of (x, y) coordinates
    """
    if a < b:
        raise ValueError("Expected a >= b for an ellipse.")
    x0, y0 = center
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    xs = a * np.cos(theta)
    ys = b * np.sin(theta)
    coords = np.stack([xs, ys], axis=1)
    if angle:
        coords = rotate(coords, angle)
    coords[:, 0] += x0
    coords[:, 1] += y0
    return coords


def circle(
    radius: float, points: int = 100, center: Tuple[float, float] = (0, 0)
) -> np.ndarray:
    """Returns the coordinates for a circle with a given radius, centered at the
    specified center.

    Args:
        radius: Radius of the circle
        points: Number of points in the circle
        center: Coordinates of the center of the circle

    Returns:
        A shape ``(points, 2)`` array of (x, y) coordinates
    """
    return ellipse(
        radius,
        radius,
        points=points,
        center=center,
        angle=0,
    )


def rectangle(
    width: float,
    height: float,
    x_points: int = 25,
    y_points: int = 25,
    center: Tuple[float, float] = (0, 0),
    angle: float = 0,
) -> np.ndarray:
    """Returns the coordinates for a rectangle with a given width and height,
    centered at the specified center.

    Args:
        width: Width of the rectangle (in the x direction).
        height: Height of the rectangle (in the y direction).
        x_points: Number of points in the top and bottom of the rectangle.
        y_points: Number of points in the sides of the rectangle.
        center: Coordinates of the center of the rectangle.
        angle: Angle (in degrees) by which to rotate counterclockwise about (0, 0)
            **before** translating to the specified center.

    Returns:
        A shape ``(2 * (x_points + y_points), 2)`` array of (x, y) coordinates
    """
    width = abs(width)
    height = abs(height)
    x0, y0 = center
    xs = np.concatenate(
        [
            width / 2 * np.ones(y_points),
            np.linspace(width / 2, -width / 2, x_points),
            -width / 2 * np.ones(y_points),
            np.linspace(-width / 2, width / 2, x_points),
        ]
    )
    ys = np.concatenate(
        [
            np.linspace(-height / 2, height / 2, y_points),
            height / 2 * np.ones(x_points),
            np.linspace(height / 2, -height / 2, y_points),
            -height / 2 * np.ones(x_points),
        ]
    )
    coords = np.stack([xs, ys], axis=1)
    if angle:
        coords = rotate(coords, angle)
    coords[:, 0] += x0
    coords[:, 1] += y0
    return coords


def square(
    side_length: float,
    points_per_side: int = 25,
    center: Tuple[float, float] = (0, 0),
    angle: float = 0,
) -> np.ndarray:
    """Returns the coordinates for a square with the given side length, centered at the
    specified center.

    Args:
        side_length: The width and height of the square
        points_per_side: Number of points in each side of the square.
        center: Coordinates of the center of the square
        angle: Angle by which to rotate counterclockwise about (0, 0)
            **before** translating to the specified center.

    Returns:
        A shape ``(4 * points_per_side, 2)`` array of (x, y) coordinates
    """
    return rectangle(
        side_length,
        side_length,
        x_points=points_per_side,
        y_points=points_per_side,
        center=center,
        angle=angle,
    )


def close_curve(points: np.ndarray) -> np.ndarray:
    """Close a curve (making the start point equal to the end point),
    if it is not already closed.

    Args:
        points: Shape ``(m, n)`` array of ``m`` coordinates in ``n`` dimensions.

    Returns:
        ``points`` with the first point appended to the end if the start point
        was not already equal to the end point. 
    """
    if not np.array_equal(points[0], points[-1]):
        points = np.concatenate([points, points[:1]], axis=0)
    return points
