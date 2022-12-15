import contextlib
import copy
import pickle
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp
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
    assert polygon.on_boundary(smaller, index=True).dtype is np.dtype(int)


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

    with pytest.raises(ValueError):
        square1.name = None
        sc.Device._validate_polygons([square1], "label")

    for min_points, smooth in [(0, 0), (500, 0), (500, 10)]:
        points, triangles = square1.make_mesh(min_points=min_points, smooth=smooth)
        if min_points:
            assert points.shape[0] > min_points


def test_plot_polygon():
    with non_gui_backend():
        ax = sc.Polygon("square1", layer="layer", points=sc.geometry.box(1)).plot()
        assert isinstance(ax, plt.Axes)


@pytest.fixture(scope="module")
def device():

    layers = [
        sc.Layer("layer0", london_lambda=1, thickness=0.1, z0=0),
        sc.Layer("layer1", london_lambda=2, thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring", layer="layer1", points=sc.geometry.ellipse(3, 2, angle=5)),
    ]

    holes = [
        sc.Polygon("hole", layer="layer1", points=sc.geometry.circle(1)),
    ]

    offset_film = films[0].buffer(1, join_style="mitre", as_polygon=False)
    assert isinstance(offset_film, np.ndarray)
    assert offset_film.shape[0] >= films[0].points.shape[0]

    offset_poly = films[0].buffer(1)
    assert isinstance(offset_poly, sc.Polygon)
    assert films[0].name in offset_poly.name

    assert films[0].contains_points([0, 0])
    assert np.array_equal(
        films[0].contains_points([[0, 0], [2, 1]], index=True), np.array([0, 1])
    )
    assert np.isclose(films[0].area, np.pi * 5**2, rtol=1e-3)
    assert np.isclose(films[1].area, np.pi * 3 * 2, rtol=1e-3)

    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="layer0",
            points=sc.geometry.box(12, angle=90),
        ),
    ]

    with pytest.raises(ValueError):
        device = sc.Device(
            "device",
            layers=layers[:1],
            films=films,
            holes=holes,
            abstract_regions=abstract_regions,
        )

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )

    with pytest.raises(AttributeError):
        device.layers["layer0"].Lambda = 0
    with pytest.raises(ValueError):
        device.compute_matrices()
    with pytest.raises(ValueError):
        device.solve_dtype = "int64"

    assert device.get_arrays() is None

    with pytest.raises(TypeError):
        device.scale(xfact=-1, origin="center")
    with pytest.raises(TypeError):
        device.rotate(90, origin="centroid")

    assert isinstance(device.scale(xfact=-1), sc.Device)
    assert isinstance(device.scale(yfact=-1), sc.Device)
    assert isinstance(device.rotate(90), sc.Device)
    assert isinstance(device.mirror_layers(about_z=0), sc.Device)
    dx = 1
    dy = -1
    dz = 1
    assert isinstance(device.translate(dx, dy, dz=dz), sc.Device)
    d = device.copy()
    assert d.translate(dx, dy, dz=dz, inplace=True) is d

    return device


def test_layer_bad_init():
    with pytest.raises(ValueError):
        _ = sc.Layer("layer0", Lambda=10, london_lambda=1, thickness=0.1, z0=0)

    with pytest.raises(ValueError):
        _ = sc.Layer("layer0", Lambda=None, london_lambda=1, thickness=None, z0=0)


@pytest.fixture(scope="module")
def device_with_mesh():

    with pytest.raises(ValueError):
        _ = sc.Polygon("poly", layer="", points=sc.geometry.circle(1).T)

    layers = [
        sc.Layer("layer0", london_lambda=1, thickness=0.1, z0=0),
        sc.Layer("layer1", Lambda=sc.Constant(2), z0=0.5),
    ]

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon(
            "ring",
            layer="layer1",
            points=sc.geometry.close_curve(sc.geometry.circle(4)),
        ),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="layer0",
            points=sc.geometry.box(12),
        )
    ]

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
    assert device.vertex_distances is None
    assert device.triangle_areas is None
    device.make_mesh(min_points=3000)
    assert isinstance(device.vertex_distances, np.ndarray)
    assert isinstance(device.triangle_areas, np.ndarray)
    centroids = sc.fem.centroids(device.points, device.triangles)
    assert centroids.shape[0] == device.triangles.shape[0]

    print(device)
    assert device == device
    assert all(film == film for film in films)
    assert all(layer == layer for layer in layers)
    assert films[0] != layers[0]
    assert layers[0] != films[0]
    assert layers[0] == layers[0].copy()
    assert layers[0] is not layers[0].copy()
    assert films[0] == films[0].copy()
    assert films[0] is not films[0].copy()
    assert device != layers[0]
    with pytest.raises(TypeError):
        device.layers = []
    with pytest.raises(TypeError):
        device.films = []
    with pytest.raises(TypeError):
        device.holes = []
    with pytest.raises(TypeError):
        device.abstract_regions = []
    device.layers["layer1"].Lambda = sc.Constant(3)

    assert (
        copy.deepcopy(device)
        == copy.copy(device)
        == device.copy(with_arrays=True)
        == device
    )

    d = device.scale(xfact=-1)
    assert isinstance(d, sc.Device)
    assert d.points is None
    d = device.scale(yfact=-1)
    assert isinstance(d, sc.Device)
    assert d.points is None
    d = device.rotate(90)
    assert isinstance(d, sc.Device)
    assert d.points is None
    d = device.mirror_layers(about_z=-1)
    assert isinstance(d, sc.Device)
    assert np.array_equal(d.points, device.points)
    dx = 1
    dy = -1
    dz = 1
    assert isinstance(device.translate(dx, dy, dz=dz), sc.Device)
    d = device.copy(with_arrays=True, copy_arrays=True)
    assert d.translate(dx, dy, dz=dz, inplace=True) is d

    for points in ("poly_points", "points"):
        x0, y0 = getattr(device, points).mean(axis=0)
        z0s = {layer.name: layer.z0 for layer in layers}
        with device.translation(dx, dy, dz=dz):
            x, y = getattr(device, points).mean(axis=0)
            assert np.isclose(x, x0 + dx)
            assert np.isclose(y, y0 + dy)
            for layer in device.layers.values():
                assert np.isclose(layer.z0, z0s[layer.name] + dz)
        x, y = getattr(device, points).mean(axis=0)
        assert np.isclose(x, x0)
        assert np.isclose(y, y0)
        for layer in device.layers.values():
            assert np.isclose(layer.z0, z0s[layer.name])

    return device


@pytest.mark.parametrize("subplots", [False, True])
@pytest.mark.parametrize("legend", [False, True])
def test_plot_device(device, device_with_mesh, legend, subplots, mesh=True):
    with non_gui_backend():
        fig, axes = device.plot(legend=legend, subplots=subplots)
        if subplots:
            assert isinstance(axes, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
            with pytest.raises(ValueError):
                fig, ax = plt.subplots()
                _ = device.plot(ax=ax, subplots=subplots)
        with pytest.raises(RuntimeError):
            _ = device.plot(legend=legend, subplots=subplots, mesh=mesh)
        fig, axes = device_with_mesh.plot(legend=legend, subplots=subplots, mesh=mesh)
        plt.close("all")


@pytest.mark.parametrize("subplots", [False, True])
@pytest.mark.parametrize("legend", [False, True])
def test_draw_device(device, legend, subplots):
    with non_gui_backend():
        fig, axes = device.draw(exclude="disk", legend=legend, subplots=subplots)
        fig, axes = device.draw(
            legend=legend,
            subplots=subplots,
            layer_order="decreasing",
        )
        if subplots:
            assert isinstance(axes, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
            with pytest.raises(ValueError):
                fig, ax = plt.subplots()
                _ = device.draw(ax=ax, subplots=subplots)

    with pytest.raises(ValueError):
        _ = device.draw(layer_order="invalid")

    with non_gui_backend():
        fig, ax = plt.subplots()
        _ = sc.Device(
            "device",
            layers=[sc.Layer("layer", Lambda=0, z0=0)],
            films=[sc.Polygon("disk", layer="layer", points=sc.geometry.circle(1))],
        ).draw(ax=ax)
        plt.close("all")


@pytest.mark.parametrize(
    ", ".join(["min_points", "smooth"]),
    [(None, None), (None, 20), (1200, None), (1200, 20)],
)
@pytest.mark.parametrize(
    "weight_method", ["uniform", "half_cotangent", "inv_euclidean", "invalid"]
)
def test_make_mesh(device, min_points, smooth, weight_method):
    if weight_method == "invalid":
        context = pytest.raises(ValueError)
    else:
        context = contextlib.nullcontext()

    with context:
        device.make_mesh(
            min_points=min_points,
            smooth=smooth,
            weight_method=weight_method,
        )

    if weight_method == "invalid":
        return

    assert device.points is not None
    assert device.triangles is not None
    if min_points:
        assert device.points.shape[0] >= min_points

    assert isinstance(device.weights, np.ndarray)
    assert device.weights.shape == (device.points.shape[0],)


@pytest.mark.parametrize("save_mesh", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_device_to_file(device, device_with_mesh, save_mesh, compressed):

    with tempfile.TemporaryDirectory() as directory:
        device.to_file(directory, save_mesh=save_mesh, compressed=compressed)
        loaded_device = sc.Device.from_file(directory)
    assert device == loaded_device

    with tempfile.TemporaryDirectory() as directory:
        device_with_mesh.to_file(directory, save_mesh=save_mesh, compressed=compressed)
        loaded_device = sc.Device.from_file(directory)
    assert device_with_mesh == loaded_device


def test_pickle_device(device, device_with_mesh):

    loaded_device = pickle.loads(pickle.dumps(device))
    loaded_device_with_mesh = pickle.loads(pickle.dumps(device_with_mesh))

    assert loaded_device == device
    assert loaded_device_with_mesh == device_with_mesh

    assert loaded_device.ureg("1 m") == loaded_device.ureg("1000 mm")


@pytest.mark.parametrize("dense", [False, True])
def test_copy_arrays(device_with_mesh, dense):
    arrays = device_with_mesh.get_arrays(copy_arrays=False, dense=dense)
    copied_arrays = device_with_mesh.get_arrays(copy_arrays=True, dense=dense)

    for key, val in arrays.items():
        assert val is not copied_arrays[key]
        if sp.issparse(val):
            assert not dense
            assert np.array_equal(val.toarray(), copied_arrays[key].toarray())
        else:
            assert np.array_equal(val, copied_arrays[key])


def poly_derivative(coeffs):
    """Given polynomial coefficients (c0, c1, c2, ...) specifying a polynomial
    y = c0 * x**0 + c1 * x**1 + c2 * x**2 + ..., returns the coefficients of the first
    derivative of y: (c1, 2 * c2, ...)
    """
    return tuple(c * (n + 1) for n, c in enumerate(coeffs[1:])) + (0,)


def assert_consistent_polynomial(xs, ys, coeffs, rtol=1e-3, atol=5e-3):
    deg = len(coeffs) - 1
    fit_coeffs = np.polyfit(xs, ys, deg)[::-1]
    poly_coeffs = np.array(poly_derivative(coeffs))
    assert np.allclose(fit_coeffs, poly_coeffs, rtol=rtol, atol=atol)


@pytest.mark.parametrize("poly_degree", [0, 1, 2])
def test_gradient_triangles(device_with_mesh, poly_degree):
    points = device_with_mesh.points
    triangles = device_with_mesh.triangles

    x, y = points[:, 0], points[:, 1]
    centroids = sc.fem.centroids(points, triangles)
    xt, yt = centroids[:, 0], centroids[:, 1]

    Gx, Gy = sc.fem.gradient_triangles(points, triangles)

    rng = np.random.default_rng(seed=poly_degree)
    poly_coeffs = 2 * (rng.random(size=poly_degree + 1) - 0.5)

    fx = sum(c * x**n for n, c in enumerate(poly_coeffs))
    fy = sum(c * y**n for n, c in enumerate(poly_coeffs))

    dfx_dx = Gx @ fx
    dfx_dy = Gy @ fx
    dfy_dx = Gx @ fy
    dfy_dy = Gy @ fy

    for array in [dfx_dx, dfx_dy, dfy_dx, dfy_dy]:
        assert array.shape == (triangles.shape[0],)

    assert_consistent_polynomial(xt, dfx_dx, poly_coeffs)
    assert_consistent_polynomial(yt, dfx_dy, 0 * poly_coeffs)
    assert_consistent_polynomial(xt, dfy_dx, 0 * poly_coeffs)
    assert_consistent_polynomial(yt, dfy_dy, poly_coeffs)


@pytest.mark.parametrize("poly_degree", [0, 1, 2])
def test_gradient_vertices(device_with_mesh, poly_degree):
    points = device_with_mesh.points
    triangles = device_with_mesh.triangles

    x, y = points[:, 0], points[:, 1]

    Gx, Gy = sc.fem.gradient_vertices(points, triangles)

    rng = np.random.default_rng(seed=poly_degree)
    poly_coeffs = 2 * (rng.random(size=poly_degree + 1) - 0.5)

    fx = sum(c * x**n for n, c in enumerate(poly_coeffs))
    fy = sum(c * y**n for n, c in enumerate(poly_coeffs))

    dfx_dx = Gx @ fx
    dfx_dy = Gy @ fx
    dfy_dx = Gx @ fy
    dfy_dy = Gy @ fy

    for array in [dfx_dx, dfx_dy, dfy_dx, dfy_dy]:
        assert array.shape == (points.shape[0],)

    assert_consistent_polynomial(x, dfx_dx, poly_coeffs)
    assert_consistent_polynomial(y, dfx_dy, 0 * poly_coeffs)
    assert_consistent_polynomial(x, dfy_dx, 0 * poly_coeffs)
    assert_consistent_polynomial(y, dfy_dy, poly_coeffs)

    grad = np.stack([Gx.toarray(), Gy.toarray()], axis=0)

    dfx = grad @ fx
    dfx_dx = dfx[0]
    dfx_dy = dfx[1]

    dfy = grad @ fy
    dfy_dx = dfy[0]
    dfy_dy = dfy[1]

    for array in [dfx_dx, dfx_dy, dfy_dx, dfy_dy]:
        assert array.shape == (points.shape[0],)

    assert_consistent_polynomial(x, dfx_dx, poly_coeffs)
    assert_consistent_polynomial(y, dfx_dy, 0 * poly_coeffs)
    assert_consistent_polynomial(x, dfy_dx, 0 * poly_coeffs)
    assert_consistent_polynomial(y, dfy_dy, poly_coeffs)

    poly_coeffs2 = 2 * (rng.random(size=poly_degree + 2) - 0.5)
    gx = sum(c * x**n for n, c in enumerate(poly_coeffs2))
    gy = sum(c * y**n for n, c in enumerate(poly_coeffs2))

    ix1d = np.arange(1000, dtype=int)
    ix3d = np.ix_([True, True], ix1d, ix1d)

    for f, g in [(fx, gx), (fx, gy), (fy, gx), (fy, gy)]:
        assert grad.shape == (2, points.shape[0], points.shape[0])
        df = grad[ix3d] @ f[ix1d]
        assert df.shape == (2, ix1d.shape[0])
        dg = grad[ix3d] @ g[ix1d]
        assert dg.shape == (2, ix1d.shape[0])
        df_dot_dg1 = np.einsum("ij, ij -> j", df, dg)
        assert df_dot_dg1.shape == (ix1d.shape[0],)
        df_dot_grad = np.einsum("ij, ijk -> jk", df, grad[ix3d])
        assert df_dot_grad.shape == (ix1d.shape[0], ix1d.shape[0])
        df_dot_dg2 = df_dot_grad @ g[ix1d]
        assert df_dot_dg2.shape == (ix1d.shape[0],)
        assert np.allclose(df_dot_dg1, df_dot_dg2)
