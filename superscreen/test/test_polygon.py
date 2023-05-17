import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely

import superscreen as sc
from superscreen.visualization import non_gui_backend


def test_set_polygon_points():
    box = shapely.geometry.box(0, 0, 1, 1).exterior.coords
    hole = shapely.geometry.box(0.25, 0.25, 0.75, 0.75, ccw=False)
    polygon = shapely.geometry.polygon.Polygon(box, holes=[hole.exterior.coords])

    with pytest.raises(ValueError):
        _ = sc.Polygon("bad", layer="None", points=polygon)

    invalid = shapely.geometry.polygon.LinearRing(
        [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]
    )

    with pytest.raises(ValueError):
        _ = sc.Polygon(points=invalid)

    x, y = sc.geometry.circle(1).T
    z = np.ones_like(x)
    points = np.stack([x, y, z], axis=1)
    with pytest.raises(ValueError):
        _ = sc.Polygon(points=points)


def test_polygon_on_boundary(radius=1):
    points = sc.geometry.circle(radius, points=501)
    polygon = sc.Polygon(points=points)
    Delta_x, Delta_y = polygon.extents
    assert np.isclose(Delta_x, 2 * radius)
    assert np.isclose(Delta_y, 2 * radius)

    smaller = sc.geometry.circle(radius - 0.01)
    bigger = sc.geometry.circle(radius + 0.01)
    assert polygon.on_boundary(smaller, radius=0.1).all()
    assert polygon.on_boundary(bigger, radius=0.1).all()
    assert not polygon.on_boundary(smaller, radius=0.001).any()
    assert not polygon.on_boundary(bigger, radius=0.001).any()
    assert issubclass(polygon.on_boundary(smaller, index=True).dtype.type, np.integer)


def test_polygon_join():
    square1 = sc.Polygon(points=sc.geometry.box(1))
    square2 = sc.Polygon(points=sc.geometry.translate(sc.geometry.box(1), 0.5, 0.5))
    square3 = sc.geometry.box(1, center=(-0.25, 0.25))
    name = "name"
    layer = "layer"
    for items in (
        [square1, square2, square3],
        [square1.points, square2.points, square3],
        [square1.polygon, square2.polygon, square3],
    ):
        _ = sc.Polygon.from_union(items, name=name, layer=layer)
        _ = sc.Polygon.from_difference(items, name=name, layer=layer)
        _ = sc.Polygon.from_intersection(items, name=name, layer=layer)

    assert (
        square1.union(square2, square3).polygon
        == sc.Polygon.from_union(items, name=name, layer=layer).polygon
    )

    assert (
        square1.intersection(square2, square3).polygon
        == sc.Polygon.from_intersection(items, name=name, layer=layer).polygon
    )

    assert (
        square1.difference(square2, square3).polygon
        == sc.Polygon.from_difference(items, name=name, layer=layer).polygon
    )

    square1.layer = square2.layer = None
    with pytest.raises(ValueError):
        _ = sc.Polygon.from_difference([square1, square1], name=name, layer=layer)

    with pytest.raises(ValueError):
        _ = square1._join_via(square2, "invalid")

    with pytest.raises(ValueError):
        _ = sc.Polygon.from_difference(
            [square1, square2],
            name=name,
            layer=layer,
            symmetric=True,
        )

    assert square1.resample(False) == square1
    assert square1.resample(None).points.shape == square1.points.shape
    assert square1.resample(71).points.shape != square1.points.shape

    with pytest.raises(ValueError):
        bowtie = [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]
        _ = sc.Polygon(name="bowtie", layer="layer", points=bowtie)


@pytest.mark.parametrize("min_points", [None, 1000])
@pytest.mark.parametrize("max_edge_length", [None, 0.25])
@pytest.mark.parametrize("convex_hull", [False, True])
@pytest.mark.parametrize("smooth", [0, 100])
@pytest.mark.parametrize("build_operators", [False, True])
def test_polygon_make_mesh(
    min_points, max_edge_length, convex_hull, smooth, build_operators
):
    poly = sc.Polygon(points=sc.geometry.box(2))
    poly = (
        poly.difference(poly.translate(dx=-1, dy=-1))
        .set_name("name")
        .set_layer("layer")
    )
    mesh = poly.make_mesh(
        min_points=min_points,
        max_edge_length=max_edge_length,
        convex_hull=convex_hull,
        smooth=smooth,
        build_operators=build_operators,
    )
    assert isinstance(mesh, sc.Mesh)


def test_plot_polygon():
    with non_gui_backend():
        ax = sc.Polygon(points=sc.geometry.box(1)).plot()
        assert isinstance(ax, plt.Axes)
