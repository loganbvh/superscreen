import numpy as np
import pytest

import superscreen as sc

from superscreen.device.mesh import Mesh, MeshOperators
from superscreen.device.utils import generate_mesh


def test_C_vector_uses_bounding_box_center():
    # A rectangle whose sites are crowded on one side: the result must not
    # depend on where the sites happen to be dense.
    x, y = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-1, 1, 11))
    grid = np.column_stack([x.ravel(), y.ravel()])
    rng = np.random.default_rng(0)
    crowd = np.column_stack([rng.uniform(1, 2, 500), rng.uniform(-1, 1, 500)])
    C = MeshOperators.C_vector(np.concatenate([grid, crowd]))[: len(grid)]
    C_mirror = MeshOperators.C_vector(np.concatenate([grid * [-1, 1], -crowd]))
    C_mirror = C_mirror[: len(grid)]
    finite = C < 1e29
    np.testing.assert_allclose(C[finite], C_mirror[finite], rtol=1e-12)


def _mesh(film, holes=(), max_edge_length=0.2):
    points, triangles = generate_mesh(
        np.asarray(film),
        hole_coords=[np.asarray(h) for h in holes],
        max_edge_length=max_edge_length,
    )
    return Mesh.from_triangulation(points, triangles)


def _interior(mesh):
    return np.setdiff1d(np.arange(len(mesh.sites)), mesh.boundary_indices)


def _C_region(point, x0, x1, y0, y1):
    """(1/4 pi) times the integral of 1/r^3 over a rectangle not containing point."""
    from scipy.integrate import dblquad

    def f(y, x):
        return ((x - point[0]) ** 2 + (y - point[1]) ** 2) ** -1.5

    return dblquad(f, x0, x1, y0, y1, epsabs=1e-13, epsrel=1e-11)[0] / (4 * np.pi)


def test_C_vector_exact_matches_box_formula_for_rectangle():
    mesh = _mesh(sc.geometry.box(4, 2))
    ix = _interior(mesh)
    exact = MeshOperators.C_vector(mesh.sites, elements=mesh.elements)
    box = MeshOperators.C_vector(mesh.sites)
    np.testing.assert_allclose(exact[ix], box[ix], rtol=1e-10)
    assert np.all(exact[mesh.boundary_indices] > 1e27)


@pytest.mark.parametrize("shape", ["notch", "hole"])
def test_C_vector_exact_nonconvex(shape):
    # Film = 4 x 2 box minus a 1 x 1 square, either cut from a corner (notch)
    # or from the middle (hole). The missing square adds its own 1/r^3 integral
    # to the outer box's C.
    outer = sc.geometry.box(4, 2)
    if shape == "notch":
        x0, x1, y0, y1 = 1.0, 2.0, 0.0, 1.0
        cut = sc.Polygon(points=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        mesh = _mesh(sc.Polygon(points=outer).difference(cut).points)
    else:
        x0, x1, y0, y1 = -0.5, 0.5, -0.5, 0.5
        mesh = _mesh(outer, holes=[sc.geometry.box(1, 1)])
    exact = MeshOperators.C_vector(mesh.sites, elements=mesh.elements)
    box = MeshOperators.C_vector(mesh.sites)
    ix = _interior(mesh)
    rng = np.random.default_rng(1)
    for i in rng.choice(ix, 8, replace=False):
        expected = box[i] + _C_region(mesh.sites[i], x0, x1, y0, y1)
        np.testing.assert_allclose(exact[i], expected, rtol=1e-8)
    if shape == "hole":
        assert np.all(exact[mesh.boundary_indices] > 1e27)


def test_from_mesh_uses_exact_C():
    mesh = _mesh(
        sc.geometry.box(4, 2), holes=[sc.geometry.box(1, 1)], max_edge_length=0.3
    )
    ops = MeshOperators.from_mesh(mesh)
    ix = _interior(mesh)
    C = MeshOperators.C_vector(mesh.sites, elements=mesh.elements)
    np.testing.assert_allclose((ops.Q @ ops.weights)[ix], C[ix], rtol=1e-8)
