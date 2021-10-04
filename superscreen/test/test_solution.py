import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pint
import pytest

import superscreen as sc


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

    abstract_regions = [
        sc.Polygon("bounding_box", layer="layer0", points=sc.geometry.square(12)),
    ]

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
        abstract_regions=abstract_regions,
    )
    device.make_mesh(min_triangles=5000)

    return device


@pytest.fixture(scope="module")
def solution1(device):

    applied_field = sc.sources.ConstantField(1)

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=None,
        field_units="mT",
        iterations=1,
    )

    return solutions[-1]


@pytest.fixture(scope="module")
def solution2(device):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = {
        "ring_hole": "1 mA",
    }

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        field_units="mT",
        iterations=1,
    )

    return solutions[-1]


def test_solution_equals(solution1, solution2):

    assert solution1 == solution1
    assert solution2 == solution2
    assert solution1 != solution2
    assert solution1 != 0


@pytest.mark.parametrize("dataset", ["streams", "fields", "screening_fields"])
@pytest.mark.parametrize("with_units", [False, True])
def test_grid_data(solution1, dataset, with_units):

    with pytest.raises(ValueError):
        _ = solution1.grid_data("invalid_dataset")

    with pytest.raises(ValueError):
        _ = solution1.grid_data("streams", layers="invalid_layer")

    with pytest.raises(TypeError):
        _ = solution1.grid_data("streams", grid_shape=(100, 100, 2))

    xgrid, ygrid, zgrids = solution1.grid_data(
        dataset,
        grid_shape=(20, 20),
        with_units=with_units,
    )

    if with_units:
        assert isinstance(xgrid.magnitude, np.ndarray)
        assert isinstance(ygrid.magnitude, np.ndarray)
    else:
        assert isinstance(xgrid, np.ndarray)
        assert isinstance(ygrid, np.ndarray)
    assert isinstance(zgrids, dict)

    for layer, array in zgrids.items():
        assert isinstance(layer, str)
        if with_units:
            assert isinstance(array, pint.Quantity)
            assert isinstance(array.magnitude, np.ndarray)
            if dataset == "streams":
                assert array.units == solution1.device.ureg(solution1.current_units)
            else:
                assert array.units == solution1.device.ureg(solution1.field_units)
        else:
            assert isinstance(array, np.ndarray)
        assert xgrid.shape == ygrid.shape == array.shape


@pytest.mark.parametrize("units", [None, "mA / um"])
@pytest.mark.parametrize("with_units", [False, True])
def test_current_density(solution1, units, with_units):

    xgrid, ygrid, current_density = solution1.current_density(
        grid_shape=(20, 20),
        units=units,
        with_units=with_units,
    )

    if with_units:
        assert isinstance(xgrid.magnitude, np.ndarray)
        assert isinstance(ygrid.magnitude, np.ndarray)
    else:
        assert isinstance(xgrid, np.ndarray)
        assert isinstance(ygrid, np.ndarray)
    assert isinstance(current_density, dict)

    for layer, array in current_density.items():
        assert isinstance(layer, str)
        if with_units:
            assert isinstance(array, pint.Quantity)
            assert isinstance(array.magnitude, np.ndarray)
            if units is None:
                units = f"{solution1.current_units} / {solution1.device.length_units}"
                assert array.units == solution1.device.ureg(units)
            else:
                assert array.units == solution1.device.ureg(units)
        else:
            assert isinstance(array, np.ndarray)
        assert array.shape[0] == 2
        assert xgrid.shape == ygrid.shape == array.shape[1:]


@pytest.mark.parametrize("units", [None, "Phi_0", "mT * um**2"])
@pytest.mark.parametrize("with_units", [False, True])
def test_polygon_flux(solution2, units, with_units):

    with pytest.raises(ValueError):
        _ = solution2.polygon_flux(polygons="invalid_polygon")

    flux_dict = solution2.polygon_flux(units=units, with_units=with_units)

    assert isinstance(flux_dict, dict)
    assert len(flux_dict) == (
        len(solution2.device.films)
        + len(solution2.device.holes)
        + len(solution2.device.abstract_regions)
    )

    units = units or f"{solution2.field_units} * {solution2.device.length_units}**2"
    ureg = solution2.device.ureg

    for name, flux in flux_dict.items():
        assert isinstance(name, str)
        if with_units:
            assert isinstance(flux, pint.Quantity)
            assert flux.units == ureg(units)
        else:
            assert isinstance(flux, float)


@pytest.mark.parametrize(
    "positions, zs",
    [
        (np.array([[-1, 1], [1, -1]]), "layer1"),
        (np.array([[-1, 1, -2], [1, -1, 2]]), None),
        (np.array([[-1, 1], [1, -1]]), 2),
        (np.array([[-1, 1], [1, -1]]), np.array([2, 2])),
        (np.array([[-1, 1], [1, -1]]), np.array([[2], [2]])),
    ],
)
@pytest.mark.parametrize("units", [None, "mT", "mA/um"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("return_sum", [False, True])
def test_field_at_positions(
    solution2, positions, zs, vector, units, with_units, return_sum
):

    if isinstance(zs, str):
        zs = solution2.device.layers["layer1"].z0
        if vector:
            with pytest.raises(ValueError):
                H = solution2.field_at_position(
                    positions,
                    zs=zs,
                    vector=vector,
                    units=units,
                    with_units=with_units,
                    return_sum=return_sum,
                )
            return

    H = solution2.field_at_position(
        positions,
        zs=zs,
        vector=vector,
        units=units,
        with_units=with_units,
        return_sum=return_sum,
    )

    units = units or solution2.field_units
    units = solution2.device.ureg(units)

    if return_sum:
        if with_units:
            assert isinstance(H, pint.Quantity)
            assert isinstance(H.magnitude, np.ndarray)
            assert H.units == units
        if vector:
            assert H.shape == positions.shape[:1] + (3,)
        else:
            assert H.shape == positions.shape[:1]

    else:
        assert isinstance(H, dict)
        assert "applied_field" in H
        for item in H.values():
            if with_units:
                assert isinstance(item, pint.Quantity)
                assert isinstance(item.magnitude, np.ndarray)
                assert item.units == units
            else:
                assert isinstance(item, np.ndarray)
            if vector:
                assert item.shape == positions.shape[:1] + (3,)
            else:
                assert item.shape == positions.shape[:1]


@pytest.mark.parametrize("save_mesh", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_save_solution(solution1, solution2, save_mesh, compressed):

    with tempfile.TemporaryDirectory() as directory:
        solution1.to_file(directory, save_mesh=save_mesh, compressed=compressed)
        loaded_solution1 = sc.Solution.from_file(directory)
    assert solution1 == loaded_solution1

    with tempfile.TemporaryDirectory() as other_directory:
        solution2.to_file(other_directory, save_mesh=save_mesh, compressed=compressed)
        loaded_solution2 = sc.Solution.from_file(other_directory)
    assert solution2 == loaded_solution2


@pytest.mark.parametrize("center", [(-4, 0), (-2, 2), (0, 0), (1, -2)])
@pytest.mark.parametrize("layers", ["layer0", ["layer0"], None])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("flux_units", ["Phi_0", None])
def test_fluxoid_simply_connected(solution1, flux_units, with_units, layers, center):

    if layers is None:
        with pytest.raises(ValueError):
            _ = solution1.rectangle_fluxoid(
                width=3,
                height=3,
                x_points=200,
                y_points=200,
                center=center,
                layers=layers,
                flux_units=flux_units,
                with_units=with_units,
            )
        return

    if center == (-4, 0):
        # The rectangle goes outside of the film -> raise ValueError
        context = pytest.raises(ValueError)
    else:
        context = sc.io.NullContextManager()

    with context:
        fluxoid_dict = solution1.rectangle_fluxoid(
            width=3,
            height=3,
            x_points=200,
            y_points=200,
            center=center,
            layers=layers,
            flux_units=flux_units,
            with_units=with_units,
        )

    if center == (-4, 0):
        return

    for name, fluxoid in fluxoid_dict.items():
        assert name in solution1.device.layers
        flux_part, supercurrent_part = fluxoid
        desired_type = pint.Quantity if with_units else float
        assert isinstance(flux_part, desired_type)
        assert isinstance(supercurrent_part, desired_type)
        total_fluxoid = sum(fluxoid)
        if with_units:
            flux_part = flux_part.m
            total_fluxoid = total_fluxoid.m
        # Total fluxoid should vanish for a simply connected region,
        # so we assert that it's small relative to the flux part.
        assert abs(total_fluxoid) / abs(flux_part) < 2e-2


@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("units", ["uA / um", None])
@pytest.mark.parametrize("layers", ["layer0", ["layer0"], None])
@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("positions", [[0, 0], np.array([[1, 0], [0, 1]]), None])
def test_interp_current_density(
    solution1, positions, method, layers, units, with_units
):
    if positions is None:
        positions = solution1.device.points

    with pytest.raises(ValueError):
        _ = solution1.interp_current_density(
            positions,
            layers=layers,
            method="invalid_method",
            units=units,
            with_units=with_units,
        )

    current_densities = solution1.interp_current_density(
        positions,
        layers=layers,
        method=method,
        units=units,
        with_units=with_units,
    )
    if layers is None:
        assert set(current_densities) == set(solution1.device.layers)
    else:
        assert set(current_densities) == set(["layer0"])
    for array in current_densities.values():
        assert array.shape == np.atleast_2d(positions).shape
        if with_units:
            assert isinstance(array, pint.Quantity)


def test_visualization(solution1):
    with sc.visualization.non_gui_backend():
        fig, _ = solution1.plot_streams()
        plt.close(fig)

        fig, _ = solution1.plot_fields()
        plt.close(fig)

        fig, _ = solution1.plot_currents()
        plt.close(fig)

        fig, _ = solution1.plot_field_at_positions(solution1.device.points, zs=0.3333)
        plt.close(fig)
