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

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)

    return device


@pytest.fixture(scope="module")
def device_with_mesh():

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
    device.make_mesh(min_triangles=2500)

    return device


def test_plot_polygons(device, device_with_mesh):
    with non_gui_backend():
        ax = device.plot_polygons()
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = device_with_mesh.plot_polygons()
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
@pytest.mark.parametrize("optimesh_steps", [None, 100])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    "weight_method", ["uniform", "half_cotangent", "inv_euclidean"]
)
def test_make_mesh(device, min_triangles, optimesh_steps, sparse, weight_method):

    device.make_mesh(
        min_triangles=min_triangles,
        optimesh_steps=optimesh_steps,
        sparse=sparse,
        weight_method=weight_method,
    )

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
def test_device_to_file(device, device_with_mesh, save_mesh):

    with tempfile.TemporaryDirectory() as directory:
        device.to_file(directory, save_mesh=save_mesh)
        loaded_device = sc.Device.from_file(directory)
    assert device == loaded_device

    with tempfile.TemporaryDirectory() as directory:
        device_with_mesh.to_file(directory, save_mesh=save_mesh)
        loaded_device = sc.Device.from_file(directory)
    assert device_with_mesh == loaded_device
