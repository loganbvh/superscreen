import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    device.make_mesh(min_points=1200)

    return device


@pytest.fixture(scope="module")
def solutions(device):
    applied_field = sc.sources.ConstantField(1)

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=None,
        field_units="mT",
        iterations=5,
    )

    return solutions


@pytest.fixture(scope="module")
def solution(solutions):
    return solutions[-1]


@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("logy", [False, True])
@pytest.mark.parametrize("units", [None, "Phi_0"])
def test_plot_polygon_flux(solutions, diff, absolute, logy, units):
    with non_gui_backend():
        if diff:
            fig, ax = plt.subplots()
        else:
            ax = None
        fig, ax = sc.plot_polygon_flux(
            solutions,
            ax=ax,
            diff=diff,
            absolute=absolute,
            units=units,
            logy=logy,
            grid=True,
        )
    plt.close(fig)


@pytest.mark.parametrize("films", [None, "disk", ["disk", "ring"]])
@pytest.mark.parametrize("units", [None, "mA"])
def test_plot_streams(solution, films, units):
    with non_gui_backend():
        fig, ax = sc.plot_streams(
            solution,
            films=films,
            units=units,
        )
        plt.close(fig)


cross_section_coord_params = [
    None,
    np.array([np.linspace(-1, 1, 101), 1 * np.ones(101)]).T,
    [
        np.array([np.linspace(-1, 1, 101), 1 * np.ones(101)]).T,
        np.array([1 * np.ones(101), np.linspace(-1, 1, 101)]).T,
    ],
]
thetas = np.linspace(0, 2 * np.pi, endpoint=False)
cross_section_coord_params.append(
    [r * np.array([np.cos(thetas), np.sin(thetas)]).T for r in (0.5, 1.0, 1.5)]
)


@pytest.mark.parametrize("interp_method", ["linear", "cubic"])
@pytest.mark.parametrize("cross_section_coords", cross_section_coord_params)
def test_cross_section(solution, cross_section_coords, interp_method):
    if cross_section_coords is None:
        return

    dataset_coords = solution.device.meshes["disk"].sites
    dataset_values = solution.film_solutions["disk"].total_field

    with pytest.raises(ValueError):
        _ = sc.visualization.cross_section(
            dataset_coords,
            dataset_values,
            cross_section_coords,
            interp_method="invalid",
        )

    coords, paths, cross_sections = sc.visualization.cross_section(
        dataset_coords,
        dataset_values,
        cross_section_coords,
        interp_method=interp_method,
    )
    for item in (coords, paths, cross_sections):
        assert isinstance(item, list)
    for c, p, cs in zip(coords, paths, cross_sections):
        assert p.shape == cs.shape
        assert c.shape == (p.shape[0], 2)
    if isinstance(cross_section_coords, list):
        assert (
            len(coords)
            == len(paths)
            == len(cross_sections)
            == len(cross_section_coords)
        )
    else:
        assert len(coords) == len(paths) == len(cross_sections) == 1
        assert np.array_equal(coords[0], cross_section_coords)


def test_cross_section_bad_shape(solution: sc.Solution):
    dataset_coords = solution.device.meshes["disk"].sites
    dataset_values = solution.film_solutions["disk"].total_field
    cross_section_coords = np.array([np.ones(101)] * 3).T

    with pytest.raises(ValueError):
        _ = sc.visualization.cross_section(
            dataset_coords,
            dataset_values,
            cross_section_coords,
            interp_method="invalid",
        )


@pytest.mark.parametrize("films", [None, "disk", ["disk", "ring"]])
@pytest.mark.parametrize("units", [None, "mA/um"])
@pytest.mark.parametrize("streamplot", [False, True])
@pytest.mark.parametrize("cross_section_coords", cross_section_coord_params)
@pytest.mark.parametrize(
    "dataset", ["field", "self_field", "applied_field", "field_from_other_films"]
)
# @pytest.mark.parametrize("auto_range_cutoff", [None, 1])
# @pytest.mark.parametrize("share_color_scale", [False, True])
# @pytest.mark.parametrize("symmetric_color_scale", [False, True])
def test_plot_currents_and_fields(
    solution,
    films,
    units,
    streamplot,
    dataset,
    cross_section_coords,
    share_color_scale=True,
    symmetric_color_scale=True,
    auto_range_cutoff=1,
):
    with non_gui_backend():
        fig, ax = sc.plot_currents(
            solution,
            grid_shape=200,
            films=films,
            units=units,
            cross_section_coords=cross_section_coords,
            streamplot=streamplot,
            auto_range_cutoff=auto_range_cutoff,
            share_color_scale=share_color_scale,
            symmetric_color_scale=symmetric_color_scale,
        )
        plt.close(fig)

        fig, ax = sc.plot_fields(
            solution,
            dataset=dataset,
            films=films,
            units=units,
            cross_section_coords=cross_section_coords,
            auto_range_cutoff=auto_range_cutoff,
            share_color_scale=share_color_scale,
            symmetric_color_scale=symmetric_color_scale,
        )
        plt.close(fig)


@pytest.mark.parametrize(
    "positions, zs",
    [
        (np.random.rand(200).reshape((-1, 2)), np.random.rand(100)),
        (np.random.rand(200).reshape((-1, 2)), 1),
        (np.random.rand(300).reshape((-1, 3)), None),
        (None, 1),
    ],
)
@pytest.mark.parametrize("units", [None, "mT"])
@pytest.mark.parametrize("auto_range_cutoff", [None, 1])
@pytest.mark.parametrize("cross_section_coords", cross_section_coord_params)
def test_plot_field_at_positions(
    solution: sc.Solution,
    positions,
    zs,
    units,
    cross_section_coords,
    auto_range_cutoff,
):
    if positions is None:
        positions = list(solution.device.meshes.values())[0]
    with non_gui_backend():
        fig, ax = sc.plot_field_at_positions(
            solution,
            positions,
            zs=zs,
            units=units,
            cross_section_coords=cross_section_coords,
            auto_range_cutoff=auto_range_cutoff,
        )
        plt.close(fig)
