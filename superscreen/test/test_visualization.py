import numpy as np
import pytest
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
        sc.Polygon("ring", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    device.make_mesh(min_triangles=2500)

    return device


@pytest.fixture(scope="module")
def solution(device):

    applied_field = sc.sources.ConstantField(1)

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=None,
        field_units="mT",
        iterations=5,
    )

    return solutions[-1]


@pytest.mark.parametrize("layers", [None, "layer0"])
@pytest.mark.parametrize("units", [None, "mA"])
def test_plot_streams(solution, layers, units):
    with non_gui_backend():
        fig, ax = sc.plot_streams(
            solution,
            layers=layers,
            units=units,
        )
        plt.close(fig)


@pytest.mark.parametrize("layers", [None, "layer0"])
@pytest.mark.parametrize("units", [None, "mA/um"])
@pytest.mark.parametrize("streamplot", [False, True])
@pytest.mark.parametrize("cross_section_xs, cross_section_ys", [(None, 0), (0, None)])
# @pytest.mark.parametrize("auto_range_cutoff", [None, 1])
# @pytest.mark.parametrize("share_color_scale", [False, True])
# @pytest.mark.parametrize("symmetric_color_scale", [False, True])
def test_plot_currents_and_fields(
    solution,
    layers,
    units,
    streamplot,
    cross_section_xs,
    cross_section_ys,
    share_color_scale=True,
    symmetric_color_scale=True,
    auto_range_cutoff=1,
):
    with non_gui_backend():
        fig, ax = sc.plot_currents(
            solution,
            grid_shape=(50, 50),
            layers=layers,
            units=units,
            cross_section_xs=cross_section_xs,
            cross_section_ys=cross_section_ys,
            cross_section_angle=45,
            streamplot=streamplot,
            auto_range_cutoff=auto_range_cutoff,
            share_color_scale=share_color_scale,
            symmetric_color_scale=symmetric_color_scale,
        )
        plt.close(fig)

        fig, ax = sc.plot_fields(
            solution,
            grid_shape=(50, 50),
            layers=layers,
            units=units,
            cross_section_xs=cross_section_xs,
            cross_section_ys=cross_section_ys,
            cross_section_angle=45,
            auto_range_cutoff=auto_range_cutoff,
            share_color_scale=share_color_scale,
            symmetric_color_scale=symmetric_color_scale,
        )
        plt.close(fig)


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize(
    "positions, zs",
    [
        (np.random.rand(200).reshape((-1, 2)), np.random.rand(100)),
        (np.random.rand(200).reshape((-1, 2)), 1),
        (np.random.rand(300).reshape((-1, 3)), None),
    ],
)
@pytest.mark.parametrize("units", [None, "mT"])
@pytest.mark.parametrize("cross_section_xs, cross_section_ys", [(None, 0), (0, None)])
@pytest.mark.parametrize("auto_range_cutoff", [None, 1])
def test_plot_field_at_positions(
    solution,
    positions,
    zs,
    units,
    cross_section_xs,
    cross_section_ys,
    auto_range_cutoff,
    vector,
):
    with non_gui_backend():
        fig, ax = sc.plot_field_at_positions(
            solution,
            positions,
            zs=zs,
            vector=vector,
            grid_shape=(50, 50),
            units=units,
            cross_section_xs=cross_section_xs,
            cross_section_ys=cross_section_ys,
            cross_section_angle=45,
            auto_range_cutoff=auto_range_cutoff,
        )
        plt.close(fig)
