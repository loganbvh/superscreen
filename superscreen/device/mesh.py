import logging
from typing import Optional, Tuple

import numpy as np
from matplotlib.tri import Triangulation
from meshpy import triangle
from scipy import spatial
from shapely.geometry import MultiLineString
from shapely.geometry.polygon import orient
from shapely.ops import polygonize

logger = logging.getLogger(__name__)


def generate_mesh(
    coords: np.ndarray,
    min_points: Optional[int] = None,
    convex_hull: bool = False,
    boundary: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a Delaunay mesh for a given set of polygon vertex coordinates.

    Additional keyword arguments are passed to ``triangle.build()``.

    Args:
        coords: Shape ``(n, 2)`` array of polygon ``(x, y)`` coordinates.
        min_points: The minimimum number of vertices in the resulting mesh.
        convex_hull: If True, then the entire convex hull of the polygon will be
            meshed. Otherwise, only the polygon interior is meshed.

    Returns:
        Mesh vertex coordinates and triangle indices.
    """
    # Coords is a shape (n, 2) array of vertex coordinates.
    coords = np.asarray(coords)
    # Remove duplicate coordinates, otherwise triangle.build() will segfault.
    # By default, np.unique() does not preserve order, so we have to remove
    # duplicates this way:
    _, ix = np.unique(coords, return_index=True, axis=0)
    coords = coords[np.sort(ix)]
    xmin = coords[:, 0].min()
    dx = np.ptp(coords[:, 0])
    ymin = coords[:, 1].min()
    dy = np.ptp(coords[:, 1])
    r0 = np.array([[xmin, ymin]]) + np.array([[dx, dy]]) / 2
    # Center the coordinates at (0, 0) to avoid floating point issues.
    coords = coords - r0
    # Facets is a shape (m, 2) array of edge indices.
    # coords[facets] is a shape (m, 2, 2) array of edge coordinates:
    # [(x0, y0), (x1, y1)]
    if convex_hull:
        facets = spatial.ConvexHull(coords).simplices
        if boundary is not None:
            raise ValueError(
                "Cannot have both boundary is not None and convex_hull = True."
            )
    else:
        indices = np.arange(coords.shape[0], dtype=int)
        if boundary is not None:
            boundary = np.asarray(boundary) - r0
            boundary = list(map(tuple, boundary))
            indices = [i for i in indices if tuple(coords[i]) in boundary]
        facets = np.stack([indices, np.roll(indices, -1)], axis=1)

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(coords)
    mesh_info.set_facets(facets)
    kwargs = kwargs.copy()

    max_vol = dx * dy / 100
    kwargs["max_volume"] = max_vol

    mesh = triangle.build(mesh_info=mesh_info, **kwargs)
    points = np.array(mesh.points) + r0
    triangles = np.array(mesh.elements)
    if min_points is None:
        return points, triangles

    num_points = 0
    i = 1
    while num_points < min_points:
        kwargs["max_volume"] = max_vol
        mesh = triangle.build(
            mesh_info=mesh_info,
            **kwargs,
        )
        points = np.array(mesh.points) + r0
        triangles = np.array(mesh.elements)
        num_points = points.shape[0]
        logger.debug(
            f"Iteration {i}: Made mesh with {points.shape[0]} points and "
            f"{triangles.shape[0]} triangles using max_volume={max_vol:.2e}. "
            f"Target number of points: {min_points}."
        )
        max_vol *= 0.9
        i += 1
    return points, triangles


def get_edges(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the edges from a list of triangle indices.

    Args:
        triangles: The triangle indices, shape ``(n, 3)``.

    Returns:
        A tuple containing an integer array of edges and a boolean array
        indicating whether each edge on in the boundary.
    """
    edges = np.concatenate([triangles[:, e] for e in [(0, 1), (1, 2), (2, 0)]])
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    return edges, counts == 1


def smooth_mesh(
    points: np.ndarray, triangles: np.ndarray, iterations: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Laplacian smoothing of the mesh, i.e., moving each interior vertex
    to the arithmetic average of its neighboring points.

    Args:
        points: Shape ``(n, 2)`` array of vertex coordinates.
        triangles: Shape ``(m, 3)`` array of triangle indices.
        iterations: The number of smoothing iterations to perform.

    Returns:
        Smoothed mesh vertex coordinates and triangle indices.
    """
    edges, _ = get_edges(triangles)
    n = points.shape[0]
    shape = (n, 2)
    boundary = boundary_vertices(points, triangles)
    for _ in range(iterations):
        num_neighbors = np.bincount(edges.ravel(), minlength=shape[0])
        new_points = np.zeros(shape)
        vals = points[edges[:, 1]].T
        new_points += np.array(
            [np.bincount(edges[:, 0], val, minlength=n) for val in vals]
        ).T
        vals = points[edges[:, 0]].T
        new_points += np.array(
            [np.bincount(edges[:, 1], val, minlength=n) for val in vals]
        ).T
        new_points /= num_neighbors[:, np.newaxis]
        # reset boundary points
        new_points[boundary] = points[boundary]
        points = new_points
    return points, triangles


def boundary_vertices(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns an array of boundary vertex indices, ordered counterclockwise.

    Args:
        points: Shape ``(n, 2)`` array of vertex coordinates.
        triangles: Shape ``(m, 3)`` array of triangle indices.

    Returns:
        An array of boundary vertex indices, ordered counterclockwise.
    """
    tri = Triangulation(points[:, 0], points[:, 1], triangles)
    boundary_edges = set()
    for i, neighbors in enumerate(tri.neighbors):
        for k in range(3):
            if neighbors[k] == -1:
                boundary_edges.add((triangles[i, k], triangles[i, (k + 1) % 3]))
    edges = MultiLineString([points[edge, :] for edge in boundary_edges])
    polygons = list(polygonize(edges))
    assert len(polygons) == 1, polygons
    polygon = orient(polygons[0])
    points_list = [tuple(xy) for xy in points]
    indices = np.array([points_list.index(xy) for xy in polygon.exterior.coords])
    return indices[:-1]
