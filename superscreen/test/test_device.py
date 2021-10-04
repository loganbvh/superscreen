import copy
import pickle
import tempfile

import pytest
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import superscreen as sc
from superscreen.visualization import non_gui_backend


@pytest.fixture
def device():

    layers = [
        sc.Layer("layer0", london_lambda=1, thickness=0.1, z0=0),
        sc.Layer("layer1", london_lambda=2, thickness=0.05, z0=0.5),
    ]

    with pytest.raises(ValueError):
        sc.Polygon("ring", layer="layer1", points=sc.geometry.ellipse(2, 3))

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring", layer="layer1", points=sc.geometry.ellipse(3, 2, angle=5)),
    ]

    assert films[0].contains_points(0, 0)

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
        sc.Polygon("ring", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    assert device.abstract_regions == {}
    device.make_mesh(min_triangles=2500)

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
    return device


@pytest.mark.parametrize("legend", [False, True])
def test_plot_polygons(device, device_with_mesh, legend):
    with non_gui_backend():
        ax = device.plot_polygons(legend=legend)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = device_with_mesh.plot_polygons(legend=legend)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)


@pytest.mark.parametrize("edges", [False, True])
@pytest.mark.parametrize("vertices", [False, True])
def test_plot_mesh(device, device_with_mesh, edges, vertices):
    with non_gui_backend():
        with pytest.raises(RuntimeError):
            ax = device.plot_mesh(edges=edges, vertices=vertices)

        ax = device_with_mesh.plot_mesh(edges=edges, vertices=vertices)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)


@pytest.mark.parametrize("min_triangles", [None, 2500])
@pytest.mark.parametrize("optimesh_steps", [None, 20])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "weight_method", ["uniform", "half_cotangent", "inv_euclidean", "invalid"]
)
def test_make_mesh(device, min_triangles, optimesh_steps, sparse, weight_method):
    if weight_method == "invalid":
        context = pytest.raises(ValueError)
    else:
        context = sc.io.NullContextManager()

    with context:
        device.make_mesh(
            min_triangles=min_triangles,
            optimesh_steps=optimesh_steps,
            sparse=sparse,
            weight_method=weight_method,
        )

    if weight_method == "invalid":
        return

    assert device.points is not None
    assert device.triangles is not None
    if min_triangles:
        assert device.triangles.shape[0] >= min_triangles

    if sparse:
        assert sp.issparse(device.weights)
    else:
        assert isinstance(device.weights, np.ndarray)
    assert device.weights.shape == (device.points.shape[0],) * 2


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
        if key == "C_vectors":
            for name, arr in val.items():
                assert arr is not copied_arrays[key][name]
                if sp.issparse(arr):
                    assert not dense
                    assert np.array_equal(
                        arr.toarray(), copied_arrays[key][name].toarray()
                    )
                else:
                    assert np.array_equal(arr, copied_arrays[key][name])
        else:
            assert val is not copied_arrays[key]
            if sp.issparse(val):
                assert not dense
                assert np.array_equal(val.toarray(), copied_arrays[key].toarray())
            else:
                assert np.array_equal(val, copied_arrays[key])
