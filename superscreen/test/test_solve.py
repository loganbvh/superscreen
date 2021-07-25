import numpy as np
import pytest

import superscreen as sc


@pytest.fixture(scope="module")
def device():

    layers = [
        sc.Layer("layer1", london_lambda=2, thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("ring", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(3)),
    ]

    abstract_regions = [
        sc.Polygon("bounding_box", layer="layer1", points=sc.geometry.square(10))
    ]

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
    device.make_mesh(min_triangles=10000, optimesh_steps=200)

    return device


def test_current_value(device):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = {
        "ring_hole": "1 mA",
    }

    solution = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        field_units="mT",
        iterations=1,
    )[-1]

    grid_shape = 500

    xgrid, ygrid, J = solution.current_density(
        grid_shape=grid_shape, units="uA / um", with_units=False
    )
    jx, jy = J["layer1"]
    x, y = sc.grids_to_vecs(xgrid, ygrid)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    N = grid_shape // 2

    assert np.isclose(np.sum(-jy[N, :N]) * dx, 1000)
    assert np.isclose(np.sum(+jy[N, N:]) * dx, 1000)
    assert np.isclose(np.sum(-jx[N:, N]) * dy, 1000)
    assert np.isclose(np.sum(+jx[:N, N]) * dy, 1000)
