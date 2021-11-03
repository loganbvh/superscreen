import os

import pint
import pytest
import numpy as np


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

    abstract_regions = [
        sc.Polygon("bounding_box", layer="layer1", points=geo.square(10))
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


@pytest.fixture(scope="module")
def two_rings():
    length_units = "um"
    inner_radius = 2.5
    outer_radius = 5

    layers = [
        sc.Layer("layer0", Lambda=1, z0=0),
        sc.Layer("layer1", Lambda=1, z0=1),
    ]

    films = [
        sc.Polygon(
            "square_ring",
            layer="layer0",
            points=geo.square(1.5 * outer_radius, points_per_side=50),
        ),
        sc.Polygon(
            "round_ring", layer="layer1", points=geo.circle(outer_radius, points=200)
        ),
    ]

    holes = [
        sc.Polygon(
            "square_hole",
            layer="layer0",
            points=geo.square(1 * inner_radius, points_per_side=50),
        ),
        sc.Polygon(
            "round_hole", layer="layer1", points=geo.circle(inner_radius, points=100)
        ),
    ]

    abstract_regions = [
        sc.Polygon(
            "bbox",
            layer="layer0",
            points=geo.square(1.25 * 2 * outer_radius, points_per_side=10),
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
    device.make_mesh(min_triangles=8_000)
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


@pytest.mark.parametrize("all_iterations", [False, True])
def test_mutual_inductance_matrix(two_rings, all_iterations, iterations=3):
    hole_polygon_mapping = {
        "square_hole": geo.square(6, points_per_side=101),
        "round_hole": geo.circle(4, points=301),
    }

    with pytest.raises(ValueError):
        _ = two_rings.mutual_inductance_matrix(
            hole_polygon_mapping={"invalid": None},
            iterations=iterations,
            all_iterations=all_iterations,
        )

    with pytest.raises(ValueError):
        _ = two_rings.mutual_inductance_matrix(
            hole_polygon_mapping={"round_hole": geo.circle(1)},
            iterations=iterations,
            all_iterations=all_iterations,
        )

    M = two_rings.mutual_inductance_matrix(
        hole_polygon_mapping=hole_polygon_mapping,
        iterations=iterations,
        all_iterations=all_iterations,
    )
    M2 = two_rings.mutual_inductance_matrix(
        iterations=iterations,
        all_iterations=all_iterations,
    )
    if all_iterations:
        assert isinstance(M, list)
        assert len(M) == iterations + 1
        assert all(isinstance(m, pint.Quantity) for m in M)
        assert all(isinstance(m.magnitude, np.ndarray) for m in M)
        M = M[-1]
        M2 = M2[-1]
    else:
        assert isinstance(M, pint.Quantity)
        assert isinstance(M.magnitude, np.ndarray)
    assert np.allclose(M, M2, rtol=1e-2)
    # Check that M is symmetric
    assert np.isclose(M[0, 1], M[1, 0], rtol=1e-2)


def test_fluxoid_single(device):

    from scipy.optimize import RootResults

    solution, result = sc.find_fluxoid_solution(device)
    assert isinstance(solution, sc.Solution)
    assert isinstance(result, RootResults)
    fluxoid = solution.hole_fluxoid(device.holes_list[0].name)
    assert np.isclose(sum(fluxoid).to("Phi_0").m, 0)


@pytest.mark.parametrize(
    "hole_polygon_fluxoid_mapping",
    [
        None,
        {"round_hole": (None, 0), "square_hole": (None, 0)},
        {"round_hole": (None, None)},
    ],
)
def test_fluxoid_multi(two_rings, hole_polygon_fluxoid_mapping):

    from scipy.optimize import OptimizeResult

    solution, result = sc.find_fluxoid_solution(
        two_rings,
        hole_polygon_fluxoid_mapping=hole_polygon_fluxoid_mapping,
        applied_field=sc.sources.ConstantField(0.1),
        field_units="mT",
        current_units="mA",
    )
    assert isinstance(solution, sc.Solution)
    assert isinstance(result, OptimizeResult)
    for hole_name in two_rings.holes:
        fluxoid = solution.hole_fluxoid(hole_name)
        assert np.isclose(sum(fluxoid).to("Phi_0").m, 0, atol=1e-6)
