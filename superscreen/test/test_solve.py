import h5py
import matplotlib.pyplot as plt
import numpy as np
import pint
import pytest

import superscreen as sc
import superscreen.geometry as geo


@pytest.fixture(scope="module")
def device():
    layers = [
        sc.Layer("layer1", london_lambda=0.5, thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("ring", layer="layer1", points=geo.circle(4)),
    ]

    holes = [
        sc.Polygon(
            "ring_hole",
            layer="layer1",
            points=geo.circle(2),
        ),
    ]
    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
    )
    device.make_mesh(max_edge_length=0.15)

    return device


@pytest.fixture(scope="module")
def two_rings():
    length_units = "um"
    inner_radius = 2.5
    outer_radius = 5

    layers = [
        sc.Layer("layer0", Lambda=0, z0=0),
        sc.Layer("layer1", Lambda=0, z0=1),
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


@pytest.mark.parametrize("pre_factorize", [False, True])
@pytest.mark.parametrize("return_solutions", [False, True])
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("inhomogeneous", [False, True])
def test_current_value(
    device, pre_factorize, return_solutions, save, tmp_path, inhomogeneous
):
    applied_field = sc.sources.ConstantField(0)

    circulating_currents = {
        "ring_hole": "1 mA",
    }

    # https://docs.pytest.org/en/stable/tmpdir.html
    save_path = tmp_path / "solution.h5" if save else None

    old_lambda = device.layers["layer1"].london_lambda
    try:
        if inhomogeneous:

            def linear(x, y, offset=0):
                return offset + 0.1 * ((y - y.min()) + (x - x.min()))

            device.layers["layer1"].london_lambda = sc.Parameter(
                linear, offset=old_lambda
            )
        if pre_factorize:
            model = sc.factorize_model(
                device=device,
                circulating_currents=circulating_currents,
                current_units="uA",
            )
            model_save_path = tmp_path / "model.h5"
            with h5py.File(model_save_path, "x") as h5file:
                model.to_hdf5(h5file)

            with h5py.File(model_save_path, "r") as h5file:
                model = sc.FactorizedModel.from_hdf5(h5file)

            solutions = sc.solve(
                model=model,
                applied_field=applied_field,
                field_units="mT",
                iterations=1,
                return_solutions=return_solutions,
                save_path=save_path,
            )
        else:
            solutions = sc.solve(
                device=device,
                applied_field=applied_field,
                circulating_currents=circulating_currents,
                field_units="mT",
                current_units="uA",
                iterations=1,
                return_solutions=return_solutions,
                save_path=save_path,
            )
    finally:
        device.layers["layer1"].london_lambda = old_lambda

    if return_solutions:
        assert isinstance(solutions, list)
        assert len(solutions) == 1
        solution = solutions[0]
        xs = np.linspace(1.9, 4.1, 1001)
        ys = np.zeros_like(xs)
        positions = np.array([xs, ys]).T
        rtol = 5e-2
        for angle, axis in [(0, 1), (90, 0), (180, 1), (270, 0)]:
            coords = sc.geometry.rotate(positions, angle)
            current = solution.current_through_path(
                coords,
                film="ring",
                units="uA",
                with_units=False,
            )
            assert np.isclose(abs(current), 1000, rtol=rtol)

            j = solution.interp_current_density(
                coords,
                film="ring",
                units="uA / um",
                with_units=False,
            )
            dr = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            current = np.sum(j[1:, axis] * dr)
            assert np.isclose(abs(current), 1000, rtol=rtol)


def test_invalid_vortex_args(device):
    with pytest.raises(TypeError):
        _ = sc.solve(device=device, vortices=[0, 1, 2])

    # Vortex in unknown film
    with pytest.raises(KeyError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=3.5, y=0, film="invalid")],
        )

    # Vortex in hole
    with pytest.raises(ValueError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=0, y=0, film="ring")],
        )

    # Vortex outside film
    with pytest.raises(ValueError):
        _ = sc.solve(
            device=device,
            vortices=[sc.Vortex(x=4.5, y=0, film="ring")],
        )


def test_mutual_inductance_errors(two_rings):
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
    _ = sc.make_fluxoid_polygons(device, interp_points=None)
    _ = sc.make_fluxoid_polygons(device, interp_points=101)

    fluxoids = {hole: 0 for hole in device.holes}
    solution = sc.find_fluxoid_solution(device, fluxoids=fluxoids)
    assert isinstance(solution, sc.Solution)
    fluxoid = solution.hole_fluxoid(list(device.holes)[0])
    assert np.isclose(sum(fluxoid).to("Phi_0").m, 0)


def test_fluxoid_multi(two_rings):
    fluxoids = {hole: 0 for hole in two_rings.holes}
    solution = sc.find_fluxoid_solution(
        two_rings,
        fluxoids=fluxoids,
        applied_field=sc.sources.ConstantField(0.1),
        field_units="mT",
        current_units="mA",
    )
    assert isinstance(solution, sc.Solution)
    for hole_name in two_rings.holes:
        fluxoid = solution.hole_fluxoid(hole_name)
        assert np.isclose(
            sum(fluxoid).to("Phi_0").m,
            fluxoids[hole_name],
            atol=1e-5,
        )


def test_applied_field_shape(device: sc.Device):
    dipole_field = sc.sources.DipoleField(
        dipole_positions=[(0, 0, 1000)],
        dipole_moments=[(0, 0, 100000)],
        moment_units="uA * um ** 2",
        length_units="um",
    )
    with pytest.raises(ValueError):
        _ = sc.solve(device=device, applied_field=dipole_field)
