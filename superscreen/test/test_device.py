import copy
import os
import pickle
import tempfile
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.display import HTML

import superscreen as sc
from superscreen.visualization import non_gui_backend


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
            "abstract",
            layer="layer0",
            points=sc.geometry.box(5, angle=90),
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
        device.solve_dtype = "int64"

    assert device.meshes is None

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
    assert d == device
    assert d.translate(dx, dy, dz=dz, inplace=True) is d

    with d.translation(-dx, -dy, dz=-dz):
        assert d == device

    assert device.mesh_stats_dict() is None
    assert device.mesh_stats() is None

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
    assert device.meshes is None
    device.make_mesh(min_points=3000)
    assert isinstance(device.meshes, dict)
    assert set(device.meshes) == set(device.films)
    assert all(isinstance(mesh, sc.Mesh) for mesh in device.meshes.values())
    for mesh in device.meshes.values():
        centroids = sc.fem.centroids(mesh.sites, mesh.elements)
        assert centroids.shape[0] == mesh.elements.shape[0]

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
    device.layers["layer1"].Lambda = sc.Constant(3)

    assert (
        copy.deepcopy(device)
        == copy.copy(device)
        == device.copy(with_mesh=True)
        == device
    )

    d = device.scale(xfact=-1)
    assert isinstance(d, sc.Device)
    assert d.meshes is None
    d = device.scale(yfact=-1)
    assert isinstance(d, sc.Device)
    assert d.meshes is None
    d = device.rotate(90)
    assert isinstance(d, sc.Device)
    assert d.meshes is None
    d = device.mirror_layers(about_z=-1)
    assert isinstance(d, sc.Device)
    assert d.meshes is None
    dx = 1
    dy = -1
    dz = 1
    assert isinstance(device.translate(dx, dy, dz=dz), sc.Device)
    d = device.copy(with_mesh=True, copy_mesh=True)
    assert isinstance(device.mesh_stats_dict(), dict)
    assert isinstance(device.mesh_stats(), HTML)
    return device


@pytest.mark.parametrize("subplots", [False, True])
@pytest.mark.parametrize("legend", [False, True])
@pytest.mark.parametrize("show_sites", [False, True])
@pytest.mark.parametrize("show_edges", [False, True])
def test_plot_device(
    device: sc.Device,
    device_with_mesh: sc.Device,
    legend: bool,
    subplots: bool,
    show_sites: bool,
    show_edges: bool,
):
    with non_gui_backend():
        fig, axes = device.plot_polygons(legend=legend, subplots=subplots)
        if subplots:
            assert isinstance(axes, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
            with pytest.raises(ValueError):
                fig, ax = plt.subplots()
                _ = device.plot_polygons(ax=ax, subplots=subplots)
        fig, axes = device_with_mesh.plot_mesh(
            subplots=subplots, show_sites=show_sites, show_edges=show_edges
        )
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
    ", ".join(["min_points", "max_edge_length", "smooth"]),
    [(None, None, None), (None, 0.4, 20), (1200, 0.4, None), (1200, 0.4, 20)],
)
@pytest.mark.parametrize(
    "buffer_factor, buffer", [(None, None), (None, 0.5), (0.05, None)]
)
@pytest.mark.parametrize("preserve_boundary", [False, True])
def test_make_mesh(
    device: sc.Device,
    min_points: Optional[int],
    max_edge_length: Optional[float],
    smooth: Optional[int],
    buffer_factor: Optional[float],
    buffer: Optional[float],
    preserve_boundary: bool,
):
    device.make_mesh(
        min_points=min_points,
        max_edge_length=max_edge_length,
        smooth=smooth,
        buffer_factor=buffer_factor,
        buffer=buffer,
        preserve_boundary=preserve_boundary,
    )

    assert isinstance(device.meshes, dict)
    assert set(device.meshes) == set(device.films)
    if min_points:
        for mesh in device.meshes.values():
            assert len(mesh.sites) >= min_points

    mesh = device.meshes["disk"]
    for weight_method in ("uniform", "inv_euclidean", "half_cotangent"):
        _ = sc.fem.laplace_operator(
            mesh.sites, mesh.elements, weight_method=weight_method
        )


@pytest.mark.parametrize("save_mesh", [False, True])
@pytest.mark.parametrize("compress", [False, True])
def test_device_to_hdf5(
    device: sc.Device,
    device_with_mesh: sc.Device,
    save_mesh: bool,
    compress: bool,
):
    for sc_device in (device, device_with_mesh):
        with tempfile.TemporaryDirectory() as directory:
            fname = os.path.join(directory, "test-0.h5")
            sc_device.to_hdf5(fname, save_mesh=save_mesh, compress=compress)
            loaded_device = sc.Device.from_hdf5(fname)
            assert loaded_device == sc_device

            with h5py.File(os.path.join(directory, "test-1.h5"), "x") as h5file:
                sc_device.to_hdf5(h5file, save_mesh=save_mesh, compress=compress)
                loaded_device = sc.Device.from_hdf5(h5file)
                assert loaded_device == sc_device


def test_pickle_device(device, device_with_mesh):
    loaded_device = pickle.loads(pickle.dumps(device))
    loaded_device_with_mesh = pickle.loads(pickle.dumps(device_with_mesh))

    assert loaded_device == device
    assert loaded_device_with_mesh == device_with_mesh

    assert loaded_device.ureg("1 m") == loaded_device.ureg("1000 mm")


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
def test_gradient_triangles(device_with_mesh: sc.Device, poly_degree: int):
    mesh = device_with_mesh.meshes["disk"]
    points = mesh.sites
    triangles = mesh.elements

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
def test_gradient_vertices(device_with_mesh: sc.Device, poly_degree: int):
    mesh = device_with_mesh.meshes["disk"]
    points = mesh.sites
    triangles = mesh.elements

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
