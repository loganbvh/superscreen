from typing import Optional, Tuple
import logging

from matplotlib.tri import Triangulation
import numpy as np
from scipy import spatial
from meshpy import triangle
import optimesh
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
            boundary = list(map(tuple, boundary))
            indices = [i for i in indices if tuple(coords[i]) in boundary]
        facets = np.stack([indices, np.roll(indices, -1)], axis=1)

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(coords)
    mesh_info.set_facets(facets)
    kwargs = kwargs.copy()

    if min_points is None:
        mesh = triangle.build(mesh_info=mesh_info, **kwargs)
        points = np.array(mesh.points)
        triangles = np.array(mesh.elements)
    else:
        max_vol = 1
        num_points = 0
        i = 1
        while num_points < min_points:
            kwargs["max_volume"] = max_vol
            mesh = triangle.build(
                mesh_info=mesh_info,
                **kwargs,
            )
            points = np.array(mesh.points)
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


def optimize_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    steps: int,
    method: str = "cvt-block-diagonal",
    tolerance: float = 1e-3,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimizes an existing mesh using ``optimesh``.

    See ``optimesh`` documentation for additional options.

    Args:
        points: Mesh vertex coordinates.
        triangles: Mesh triangle indices.
        steps: Number of optimesh steps to perform.

    Returns:
        Optimized mesh vertex coordinates and triangle indices.
    """
    points, triangles = optimesh.optimize_points_cells(
        points,
        triangles,
        method,
        tolerance,
        steps,
        verbose=verbose,
        **kwargs,
    )
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
