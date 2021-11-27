import os

import pint
import pytest
import numpy as np
import matplotlib.pyplot as plt


import superscreen as sc
import superscreen.geometry as geo


@pytest.fixture(scope="module")
def device():

    layers = [
        sc.Layer("layer1", london_lambda=2, thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("ring", layer="layer1", points=geo.circle(4)),
    ]

    holes = [
        sc.Polygon(
            "ring_hole",
            layer="layer1",
            points=geo.circle(3, center=(0.5, 0.5)),
        ),
    ]

    abstract_regions = [sc.Polygon("bounding_box", layer="layer1", points=geo.box(10))]

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
    device.make_mesh(min_points=5000, optimesh_steps=200)

    return device


@pytest.fixture(scope="module")
def two_rings():
    length_units = "um"
    inner_radius = 2.5
    outer_radius = 5

    layers = [
        sc.Layer("layer0", Lambda=0.5, z0=0),
        sc.Layer("layer1", Lambda=0.5, z0=1),
    ]

    films = [
        sc.Polygon(
            "big_ring",
            layer="layer0",
            points=geo.circle(1.5 * outer_radius, points=200),
        ),
        sc.Polygon(
            "little_ring",
            layer="layer1",
            points=geo.circle(outer_radius, points=100),
        ),
    ]

    holes = [
        sc.Polygon(
            "big_hole",
            layer="layer0",
            points=geo.circle(1.5 * inner_radius, points=100),
        ),
        sc.Polygon(
            "little_hole",
            layer="layer1",
            points=geo.circle(inner_radius, points=100),
        ),
    ]

    abstract_regions = [
        sc.Polygon(
            "bbox",
            layer="layer0",
            points=geo.circle(2 * outer_radius, points=51),
        )
    ]

    device = sc.Device(
        "two_rings",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
        length_units=length_units,
    )
    device.make_mesh(min_points=4000)
    return device


@pytest.mark.parametrize("return_solutions", [False, True])
@pytest.mark.parametrize("save", [False, True])
def test_current_value(device, return_solutions, save, tmp_path):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = {
        "ring_hole": "1 mA",
    }

    # https://docs.pytest.org/en/stable/tmpdir.html
    directory = str(tmp_path) if save else None

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        field_units="mT",
        iterations=1,
        return_solutions=return_solutions,
        directory=directory,
    )
    if directory is not None:
        assert os.path.isdir(directory)
        assert len(os.listdir(directory)) == 1

    if return_solutions:
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        solution = solutions[0]
        grid_shape = 500

        xgrid, ygrid, J = solution.grid_current_density(
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
    else:
        assert solutions is None


def test_invalid_vortex_args(device):

    with pytest.raises(TypeError):
        _ = sc.solve(device=device, vortices=[0, 1, 2])

    # Vortex in unknown layer
    with pytest.raises(ValueError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=3.5, y=0, layer="invalid")],
        )

    # Vortex in hole
    with pytest.raises(ValueError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=0, y=0, layer="layer1")],
        )

    # Vortex outside film
    with pytest.raises(ValueError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=4.5, y=0, layer="layer1")],
        )


def test_mutual_inductance_errors(two_rings):
    with pytest.raises(ValueError):
        _ = two_rings.mutual_inductance_matrix(
            upper_only=True,
            lower_only=True,
        )

    with pytest.raises(ValueError):
        _ = two_rings.mutual_inductance_matrix(
            hole_polygon_mapping={"invalid": None},
        )

    with pytest.raises(ValueError):
        _ = two_rings.mutual_inductance_matrix(
            hole_polygon_mapping={"round_hole": geo.circle(1)},
        )


@pytest.fixture(scope="module")
def mutual_inductance_matrix(two_rings, iterations=3):
    M = two_rings.mutual_inductance_matrix(
        iterations=iterations,
        all_iterations=True,
    )
    # Check that M is symmetric
    assert np.isclose(M[-1][0, 1], M[-1][1, 0], rtol=5e-2)
    return M


def test_mutual_inductance_matrix(
    two_rings,
    mutual_inductance_matrix,
    iterations=3,
):
    hole_polygon_mapping = {
        "big_hole": geo.circle(6),
        "little_hole": geo.circle(4),
    }

    M = two_rings.mutual_inductance_matrix(
        hole_polygon_mapping=hole_polygon_mapping,
        iterations=iterations,
        all_iterations=True,
    )
    assert isinstance(M, list)
    assert len(M) == iterations + 1
    assert all(isinstance(m, pint.Quantity) for m in M)
    assert all(isinstance(m.magnitude, np.ndarray) for m in M)
    M = M[-1]
    assert isinstance(M, pint.Quantity)
    assert isinstance(M.magnitude, np.ndarray)
    M2 = mutual_inductance_matrix[-1]
    assert np.allclose(M, M2, rtol=5e-2)
    # Check that M is symmetric
    assert np.isclose(M[0, 1], M[1, 0], rtol=5e-2)


def test_mutual_inductance_upper_lower(two_rings, iterations=1):

    M = two_rings.mutual_inductance_matrix(
        iterations=iterations,
        upper_only=True,
    ).magnitude
    assert np.array_equal(M, np.triu(M))

    M = two_rings.mutual_inductance_matrix(
        iterations=iterations,
        lower_only=True,
    ).magnitude
    assert np.array_equal(M, np.tril(M))


def test_plot_mutual_inductance(mutual_inductance_matrix):
    M = mutual_inductance_matrix
    with sc.visualization.non_gui_backend():
        fig, ax = sc.plot_mutual_inductance(
            M,
            diff=True,
            logy=True,
        )
        fig, ax = sc.plot_mutual_inductance(
            [m for m in M],
            diff=True,
            absolute=True,
            logy=False,
        )
        fig, ax = sc.plot_mutual_inductance(
            [m.magnitude for m in M],
            diff=True,
            logy=True,
        )
        fig, ax = plt.subplots()
        f, a = sc.plot_mutual_inductance(
            M,
            ax=ax,
            diff=False,
            logy=True,
        )
        assert f is fig
        assert a is ax
        plt.close(fig)

    with pytest.raises(ValueError):
        _ = sc.plot_mutual_inductance(M[0])

    with pytest.raises(ValueError):
        _ = sc.plot_mutual_inductance([M[0], 0])


def test_fluxoid_single(device):

    from scipy.optimize import RootResults

    with pytest.raises(ValueError):
        solution, result = sc.find_fluxoid_solution(device, fluxoids=dict())

    _ = sc.make_fluxoid_polygons(device, interp_points=None)
    _ = sc.make_fluxoid_polygons(device, interp_points=101)

    fluxoids = {hole: 0 for hole in device.holes}
    solution, result = sc.find_fluxoid_solution(device, fluxoids=fluxoids)
    assert isinstance(solution, sc.Solution)
    assert isinstance(result, RootResults)
    fluxoid = solution.hole_fluxoid(list(device.holes)[0])
    assert np.isclose(sum(fluxoid).to("Phi_0").m, 0)


def test_fluxoid_multi(two_rings):

    from scipy.optimize import OptimizeResult

    fluxoids = {hole: 0 for hole in two_rings.holes}
    solution, result = sc.find_fluxoid_solution(
        two_rings,
        fluxoids=fluxoids,
        applied_field=sc.sources.ConstantField(0.1),
        field_units="mT",
        current_units="mA",
    )
    assert isinstance(solution, sc.Solution)
    assert isinstance(result, OptimizeResult)
    for hole_name in two_rings.holes:
        fluxoid = solution.hole_fluxoid(hole_name)
        assert np.isclose(
            sum(fluxoid).to("Phi_0").m,
            fluxoids[hole_name],
            atol=1e-6,
        )
