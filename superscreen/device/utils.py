import logging
from typing import List, Optional, Tuple

import numpy as np
from matplotlib.tri import Triangulation
from meshpy import triangle
from scipy import spatial
from shapely.geometry import MultiLineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize

from ..geometry import ensure_unique

logger = logging.getLogger(__name__)


def generate_mesh(
    poly_coords: np.ndarray,
    hole_coords: Optional[List[np.ndarray]] = None,
    min_points: Optional[int] = None,
    max_edge_length: Optional[float] = None,
    convex_hull: bool = False,
    boundary: Optional[np.ndarray] = None,
    preserve_boundary: bool = False,
    min_angle: float = 32.5,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a Delaunay mesh for a given set of polygon vertex coordinates.

    Additional keyword arguments are passed to ``triangle.build()``.

    Args:
        poly_coords: Shape ``(n, 2)`` array of polygon ``(x, y)`` coordinates.
        hole_coords: A list of arrays of hole boundary coordinates.
        min_points: The minimimum number of vertices in the resulting mesh.
        max_edge_length: The maximum distance between interior vertices in the
            resulting mesh.
        convex_hull: If True, then the entire convex hull of the polygon (minus holes)
            will be meshed. Otherwise, only the polygon interior is meshed.
        boundary: Shape ``(m, 2)`` (where ``m <= n``) array of ``(x, y)`` coordinates
            for points on the boundary of the polygon.
        preserve_boundary: Do not add any mesh sites to the boundary.
        min_angle: The minimum angle in the mesh's triangles. Setting a larger value
            will make the triangles closer to equilateral, but the mesh generation
            may fail if the value is too large.

    Returns:
        Mesh vertex coordinates and triangle indices.
    """
    poly_coords = ensure_unique(poly_coords)
    if hole_coords is None:
        hole_coords = []
    hole_coords = [ensure_unique(coords) for coords in hole_coords]
    # Facets is a shape (m, 2) array of edge indices.
    # coords[facets] is a shape (m, 2, 2) array of edge coordinates:
    # [(x0, y0), (x1, y1)]
    coords = np.concatenate([poly_coords] + hole_coords, axis=0)
    xmin = coords[:, 0].min()
    dx = np.ptp(coords[:, 0])
    ymin = coords[:, 1].min()
    dy = np.ptp(coords[:, 1])
    r0 = np.array([[xmin, ymin]]) + np.array([[dx, dy]]) / 2
    # Center the coordinates at (0, 0) to avoid floating point issues.
    coords = coords - r0
    indices = np.arange(len(poly_coords), dtype=int)
    if convex_hull:
        if boundary is not None:
            raise ValueError(
                "Cannot have both boundary is not None and convex_hull = True."
            )
        facets = spatial.ConvexHull(coords).simplices
    else:
        if boundary is not None:
            boundary = list(map(tuple, ensure_unique(boundary - r0)))
            indices = [i for i in indices if tuple(coords[i]) in boundary]
        facets = np.array([indices, np.roll(indices, -1)]).T
    # Create facets for the holes.
    for hole in hole_coords:
        hole_indices = np.arange(
            indices[-1] + 1, indices[-1] + 1 + len(hole), dtype=int
        )
        hole_facets = np.array([hole_indices, np.roll(hole_indices, -1)]).T
        indices = np.concatenate([indices, hole_indices], axis=0)
        facets = np.concatenate([facets, hole_facets], axis=0)

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(coords)
    mesh_info.set_facets(facets)
    if hole_coords:
        # Triangle allows you to set holes by specifying a single point
        # that lies in each hole. Here we use the centroid of the hole.
        holes = [
            np.array(Polygon(hole).centroid.coords[0]) - r0.squeeze()
            for hole in hole_coords
        ]
        mesh_info.set_holes(holes)

    kwargs = kwargs.copy()
    kwargs["allow_boundary_steiner"] = not preserve_boundary
    if "min_angle" not in kwargs:
        kwargs["min_angle"] = min_angle

    mesh = triangle.build(mesh_info=mesh_info, **kwargs)
    points = np.array(mesh.points) + r0
    triangles = np.array(mesh.elements)
    if min_points is None and (max_edge_length is None or max_edge_length <= 0):
        return points, triangles

    kwargs["max_volume"] = dx * dy / 100
    i = 1
    if min_points is None:
        min_points = 0
    if max_edge_length is None or max_edge_length <= 0:
        max_edge_length = np.inf
    max_length = get_edge_lengths(points, triangles).max()
    while (len(points) < min_points) or (max_length > max_edge_length):
        mesh = triangle.build(mesh_info=mesh_info, **kwargs)
        points = np.array(mesh.points) + r0
        triangles = np.array(mesh.elements)
        edges, is_boundary = get_edges(triangles)
        if preserve_boundary:
            # Only constrain the length of interior edges, i.e. edges not on the boundary.
            edges = edges[~is_boundary]
        edge_lengths = np.linalg.norm(np.diff(points[edges], axis=1), axis=2)
        max_length = edge_lengths.max()
        logger.debug(
            f"Iteration {i}: Made mesh with {len(points)} points and "
            f"{len(triangles)} triangles with maximum interior edge length: "
            f"{max_length:.2e}. Target maximum edge length: {max_edge_length:.2e}."
        )
        if np.isfinite(max_edge_length):
            kwargs["max_volume"] *= min(0.98, np.sqrt(max_edge_length / max_length))
        else:
            kwargs["max_volume"] *= 0.98
        i += 1
    return points, triangles


def get_edges(triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the edges from a list of triangle indices.

    Args:
        triangles: The triangle indices, shape ``(n, 3)``.

    Returns:
        A tuple containing an integer array of edges and a boolean array
        indicating whether each edge is on the boundary.
    """
    edges = np.concatenate([triangles[:, e] for e in [(0, 1), (1, 2), (2, 0)]])
    edges = np.sort(edges, axis=1)
    edges, counts = np.unique(edges, return_counts=True, axis=0)
    return edges, counts == 1


def get_edge_lengths(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Returns the lengths of all edges in a triangulation.

    Args:
        points: Vertex coordinates.
        triangles: Triangle indices.

    Returns:
        An array of edge lengths.
    """
    edges, _ = get_edges(triangles)
    return np.linalg.norm(np.diff(points[edges], axis=1), axis=2).squeeze()


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


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Calculates the area of each triangle.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices

    Returns:
        Shape (m, ) array of triangle areas
    """
    xy = points[triangles]
    # s1 = xy[:, 2, :] - xy[:, 1, :]
    # s2 = xy[:, 0, :] - xy[:, 2, :]
    # s3 = xy[:, 1, :] - xy[:, 0, :]
    # which can be simplified to
    # s = xy[:, [2, 0, 1]] - xy[:, [1, 2, 0]]  # 3D
    s = xy[:, [2, 0]] - xy[:, [1, 2]]  # 2D
    a = np.linalg.det(s)
    return a * 0.5


def vertex_areas(
    points: np.ndarray,
    triangles: np.ndarray,
    tri_areas: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculates the vertex effective areas by averaging adjactent triangle areas.

    Args:
        points: Shape (n, 2) array of x, y coordinates of vertices
        triangles: Shape (m, 3) array of triangle indices
        tri_areas: Pre-computed array of triangle areas.

    Returns:
        Shape (n, ) array of vertex areas
    """
    if tri_areas is None:
        tri_areas = triangle_areas(points, triangles)
    v_areas = np.zeros(len(points), dtype=float)
    for a, t in zip(tri_areas / 3, triangles):
        v_areas[t[0]] += a
        v_areas[t[1]] += a
        v_areas[t[2]] += a
    return v_areas
