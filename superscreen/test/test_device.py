import copy
import pickle
import tempfile

import pytest
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

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

    offset_film = films[0].offset_points(1)
    assert isinstance(offset_film, np.ndarray)
    assert offset_film.shape[0] >= films[0].points.shape[0]

    offset_poly = films[0].offset_points(1, as_polygon=True)
    assert isinstance(offset_poly, sc.Polygon)
    assert films[0].name in offset_poly.name

    assert films[0].contains_points([0, 0])
    assert np.isclose(films[0].area, np.pi * 5 ** 2, rtol=1e-3)
    assert np.isclose(films[1].area, np.pi * 3 * 2, rtol=1e-3)

    abstract_regions = [
        sc.Polygon(
            "bounding_box",
            layer="layer0",
            points=sc.geometry.square(12, angle=90),
        ),
    ]

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=None,
        abstract_regions=abstract_regions,
    )

    with pytest.raises(AttributeError):
        device.layers["layer0"].Lambda = 0
    with pytest.raises(ValueError):
        device.compute_matrices()

    assert device.get_arrays() is None

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

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    assert device.abstract_regions == {}
    device.make_mesh(min_triangles=2500)

    print(device)
    assert all(polygon.counter_clockwise for polygon in films + holes)
    assert sc.Polygon(
        "cw_circle", layer="layer0", points=sc.geometry.circle(2)[::-1]
    ).clockwise
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
    return device


@pytest.mark.parametrize("subplots", [False, True])
@pytest.mark.parametrize("legend", [False, True])
def test_plot_polygons(device, device_with_mesh, legend, subplots, plot_mesh=True):
    with non_gui_backend():
        fig, axes = device.plot_polygons(legend=legend, subplots=subplots)
        if subplots:
            assert isinstance(axes, np.ndarray)
            assert all(isinstance(ax, plt.Axes) for ax in axes.flat)
        plt.close(fig)

        with pytest.raises(RuntimeError):
            _ = device.plot_polygons(
                legend=legend, subplots=subplots, plot_mesh=plot_mesh
            )

        fig, axes = device_with_mesh.plot_polygons(
            legend=legend, subplots=subplots, plot_mesh=plot_mesh
        )
        plt.close(fig)


@pytest.mark.parametrize(
    ", ".join(["min_triangles", "optimesh_steps"]),
    [(None, None), (None, 20), (2500, None), (2500, 20)],
)
@pytest.mark.parametrize(
    "weight_method", ["uniform", "half_cotangent", "inv_euclidean", "invalid"]
)
def test_make_mesh(device, min_triangles, optimesh_steps, weight_method):
    if weight_method == "invalid":
        context = pytest.raises(ValueError)
    else:
        context = sc.io.NullContextManager()

    with context:
        device.make_mesh(
            min_triangles=min_triangles,
            optimesh_steps=optimesh_steps,
            weight_method=weight_method,
        )

    if weight_method == "invalid":
        return

    assert device.points is not None
    assert device.triangles is not None
    if min_triangles:
        assert device.triangles.shape[0] >= min_triangles

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
