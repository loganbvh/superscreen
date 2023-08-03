import contextlib
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pint
import pytest

import superscreen as sc


@pytest.fixture(scope="module")
def device() -> sc.Device:
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

    device = sc.Device(
        "device",
        layers=layers,
        films=films,
        holes=holes,
    )
    device.make_mesh(max_edge_length=0.4, min_points=3000)
    return device


@pytest.fixture(scope="module")
def solution1(device: sc.Device) -> sc.Solution:
    applied_field = sc.sources.ConstantField(1)

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        vortices=[sc.Vortex(x=0, y=0, film="disk")],
        circulating_currents=None,
        field_units="mT",
        iterations=1,
    )

    return solutions[-1]


@pytest.fixture(scope="module")
def solution2(device: sc.Device) -> sc.Solution:
    applied_field = sc.sources.ConstantField(0)

    circulating_currents = {"ring_hole": "1 mA"}

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        field_units="mT",
        iterations=5,
    )

    return solutions[-1]


def test_save_and_load_solutions(solution1, tmp_path):
    h5path = tmp_path / "solution1.h5"
    sc.Solution.save_solutions([solution1] * 5, h5path)

    solutions = sc.Solution.load_solutions(h5path)
    assert all(solution == solution1 for solution in solutions)


def test_solution_equals(solution1: sc.Solution, solution2: sc.Solution):
    assert solution1 == solution1
    assert solution2 == solution2
    assert solution1 != solution2
    assert solution1 != 0


@pytest.mark.parametrize("units", [None, "Phi_0", "mT * um**2"])
@pytest.mark.parametrize("with_units", [False, True])
def test_polygon_flux(solution2: sc.Solution, units, with_units):
    with pytest.raises(ValueError):
        _ = solution2.polygon_flux(name="invalid_polygon")

    flux_dict = {}
    for polygon in solution2.device.get_polygons(include_terminals=False):
        flux_dict[polygon.name] = solution2.polygon_flux(
            polygon.name, units=units, with_units=with_units
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
        (np.array([[-1, 1, -2], [1, -1, 2]]), None),
        (np.array([[-1, 1], [1, -1]]), 0.0),
        (np.array([[-1, 1], [1, -1]]), 0.5),
        (np.array([[-1, 1], [1, -1]]), 2),
        (np.array([[-1, 1], [1, -1]]), np.array([2, 2])),
        (np.array([[-1, 1], [1, -1]]), np.array([[2], [2]])),
    ],
)
@pytest.mark.parametrize("units", [None, "mT", "mA/um"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("return_sum", [False, True])
def test_field_at_positions(
    solution2: sc.Solution, positions, zs, units, with_units, return_sum
):
    H = solution2.field_at_position(
        positions,
        zs=zs,
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
            assert item.shape == positions.shape[:1]


@pytest.mark.parametrize("compress", [False, True])
def test_save_solution(
    solution1: sc.Solution,
    solution2: sc.Solution,
    compress,
    tmp_path,
):
    solution1.to_hdf5(tmp_path / "solution1.h5", compress=compress)
    loaded_solution1 = sc.Solution.from_hdf5(tmp_path / "solution1.h5")
    assert solution1 == loaded_solution1

    solution2.to_hdf5(tmp_path / "solution2.h5", compress=compress)
    loaded_solution2 = sc.Solution.from_hdf5(tmp_path / "solution2.h5")
    assert solution2 == loaded_solution2

    loaded_solution2._time_created = dt.datetime.now()
    assert solution2 != loaded_solution2


@pytest.mark.parametrize("polygon_shape", ["circle", "rectangle"])
@pytest.mark.parametrize("center", [(-4, 0), (-2, 2), (0, 0), (1, -2)])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("units", ["Phi_0", None])
def test_fluxoid_simply_connected(
    solution1: sc.Solution,
    units,
    with_units,
    center,
    polygon_shape,
):
    ureg = solution1.device.ureg
    if polygon_shape == "circle":
        coords = sc.geometry.circle(1.5, points=201, center=center)
    else:
        coords = sc.geometry.box(3, 2, points=401, center=center)[::-1]

    if center == (-4, 0):
        # The polygon goes outside of the film -> raise ValueError
        context = pytest.raises(ValueError)
    else:
        context = contextlib.nullcontext()

    with context:
        fluxoid = solution1.polygon_fluxoid(
            polygon_coords=coords,
            film="disk",
            units=units,
            with_units=with_units,
        )
    if units is None:
        units = f"{solution1.field_units} * {solution1.device.length_units} ** 2"

    if center == (-4, 0):
        return

    flux_part, supercurrent_part = fluxoid
    desired_type = pint.Quantity if with_units else float
    assert isinstance(flux_part, desired_type)
    assert isinstance(supercurrent_part, desired_type)
    total_fluxoid = sum(fluxoid)
    if with_units:
        flux_part = flux_part.m
        total_fluxoid = total_fluxoid.m
    # For a simply connected region, the total fluxoid should be equal to
    # Phi0 times the number of vortices in the region.
    total_vortex_flux = 0
    for vortex in solution1.vortices:
        if sc.fem.in_polygon(coords, (vortex.x, vortex.y)):
            total_vortex_flux += (vortex.nPhi0 * ureg("Phi_0")).to(units).magnitude
    if total_vortex_flux:
        # There are vortices in the region.
        assert (abs(total_fluxoid - total_vortex_flux) / abs(total_vortex_flux)) < 5e-2
    else:
        # No vortices - fluxoid should be zero.
        assert abs(total_fluxoid) / abs(flux_part) < 5e-2


@pytest.mark.parametrize("units, with_units", [("uA / um", False), (None, True)])
@pytest.mark.parametrize(
    "film, method, positions",
    [
        ("disk", "linear", [0, 0]),
        ("ring", "linear", np.array([[1, 0], [0, 1]])),
        ("disk", "cubic", None),
    ],
)
def test_interp_current_density(
    solution1: sc.Solution, positions, method, film, units, with_units
):
    if positions is None:
        positions = solution1.device.meshes["disk"].sites

    with pytest.raises(KeyError):
        _ = solution1.interp_current_density(
            positions,
            film=film,
            method="invalid_method",
            units=units,
            with_units=with_units,
        )

    current_densities = solution1.interp_current_density(
        positions,
        film=film,
        method=method,
        units=units,
        with_units=with_units,
    )
    assert current_densities.shape == np.atleast_2d(positions).shape
    if with_units:
        assert isinstance(current_densities, pint.Quantity)


def test_visualization(solution1: sc.Solution):
    with sc.visualization.non_gui_backend():
        fig, _ = solution1.plot_streams()
        plt.close(fig)

        fig, _ = solution1.plot_fields()
        plt.close(fig)

        fig, _ = solution1.plot_currents()
        plt.close(fig)

        fig, _ = solution1.plot_field_at_positions(
            solution1.device.meshes["disk"].sites, zs=0.3333
        )
        plt.close(fig)


@pytest.mark.parametrize("use_zs", [False, True, "z0"])
@pytest.mark.parametrize("units", [None, "uT * um"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("return_sum", [False, True])
def test_bz_from_vector_potential(
    solution2: sc.Solution, use_zs, units, with_units, return_sum
):
    solution = solution2
    applied_field = solution.applied_field_func
    device = solution.device
    ureg = device.ureg
    bz_units = None if units is None else "uT"
    gradx = device.meshes["disk"].operators.gradient_x.toarray()
    grady = device.meshes["disk"].operators.gradient_y.toarray()
    if with_units:
        gradx = gradx / ureg(device.length_units)
        grady = grady / ureg(device.length_units)
    positions = device.meshes["disk"].sites
    z0 = 1.5
    zs = z0 * np.ones_like(positions[:, :1])
    applied_field = applied_field(positions[:, 0], positions[:, 1], zs[0])
    applied_field = applied_field * ureg(solution.field_units)
    if bz_units:
        applied_field.ito(bz_units)
    if not use_zs:
        positions = np.concatenate([positions, zs], axis=1)
        zs = None
    elif use_zs == "z0":
        zs = z0
    Bz = solution.field_at_position(
        positions, zs=zs, units=bz_units, with_units=with_units
    )
    A = solution.vector_potential_at_position(
        positions,
        zs=zs,
        units=units,
        with_units=with_units,
        return_sum=return_sum,
    )
    if not return_sum:
        assert isinstance(A, dict)
        assert len(A) == len(device.films)
        A = sum(A.values())
    Ax = A[:, 0]
    Ay = A[:, 1]
    if not with_units:
        applied_field = applied_field.magnitude
    Bz_from_A = applied_field + (gradx @ Ay - grady @ Ax)
    if with_units:
        Bz_from_A.ito(Bz.units)
    assert np.all(np.abs(Bz_from_A - Bz) < 5e-2 * np.max(np.abs(Bz)))


@pytest.mark.parametrize("units", [None, "mT", "mA/um"])
@pytest.mark.parametrize("with_units", [False, True])
@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_interp_field(solution2: sc.Solution, units, with_units, method):
    solution = solution2
    positions = np.random.random(size=(100, 2))
    field = solution.interp_field(
        positions,
        film="disk",
        units=units,
        with_units=with_units,
        method=method,
    )
    if with_units:
        assert isinstance(field, pint.Quantity)
        assert isinstance(field.magnitude, np.ndarray)
    assert field.shape == positions.shape[:1]
