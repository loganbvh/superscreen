import os
import tempfile

import h5py
import numpy as np
import pytest

import superscreen as sc


@pytest.fixture()
def plus_device():
    layer = sc.Layer("base", Lambda=1)
    width, height = 10, 2
    points = sc.geometry.box(width, height)
    bar = sc.Polygon("plus", points=points)
    plus = bar.union(bar.rotate(90)).resample(501)
    plus.name = "plus"
    plus.layer = layer.name
    terminal = sc.Polygon(
        points=sc.geometry.box(height, width / 100, center=(0, -width / 2))
    )
    terminals = []
    for i, name in enumerate(["drain", "source1", "source2", "source3"]):
        term = terminal.rotate(i * 90)
        term.name = name
        terminals.append(term)

    device = sc.Device(
        "plus",
        films=[plus],
        layers=[layer],
        terminals={"plus": terminals},
        length_units="um",
    )
    device.make_mesh(max_edge_length=0.2)
    return device


@pytest.fixture()
def plus_device_no_terminals():
    layer = sc.Layer("base", Lambda=1)
    width, height = 10, 2
    points = sc.geometry.box(width, height)
    bar = sc.Polygon("plus", points=points)
    plus = bar.union(bar.rotate(90)).resample(501)
    plus.name = "plus"
    plus.layer = layer.name
    device = sc.Device(
        "plus",
        films=[plus],
        layers=[layer],
        terminals=None,
        length_units="um",
    )
    device.make_mesh(max_edge_length=0.2)
    return device


@pytest.fixture()
def holey_device():
    width = 1
    height = width * 2
    slot_height = height / 5
    slot_width = width / 4
    dx, dy = (0, 0)
    length_units = "um"

    film = (
        sc.Polygon("film", layer="base", points=sc.geometry.box(width, height))
        .difference(
            sc.geometry.box(
                slot_width, slot_height, center=(-(width - slot_width) / 2, 0)
            )
        )
        .difference(
            sc.geometry.box(
                slot_width, slot_height, center=(+(width - slot_width) / 2, 0)
            )
        )
        .resample(251)
    )

    source_terminal = sc.Polygon(
        "source", points=sc.geometry.box(width, height / 100, center=(0, height / 2))
    )
    drain_terminal = sc.Polygon(
        "drain", points=sc.geometry.box(width, height / 100, center=(0, -height / 2))
    )

    device = sc.Device(
        "constriction",
        layers=[sc.Layer("base", Lambda=2)],
        films=[film],
        holes=[
            sc.Polygon(
                "hole1",
                layer="base",
                points=sc.geometry.circle(width / 4, center=(0, +height / 4)),
            ),
            sc.Polygon(
                "hole2",
                layer="base",
                points=sc.geometry.circle(width / 4, center=(0, -height / 4)),
            ),
        ],
        terminals={"film": [source_terminal, drain_terminal]},
        length_units=length_units,
    ).translate(dx=dx, dy=dy)
    device.make_mesh(max_edge_length=0.2)
    print(device)
    return device


def test_save_and_load_device_with_terminals(holey_device: sc.Device, tmp_path):
    h5path = tmp_path / "holey_device.h5"
    holey_device.to_hdf5(h5path)
    loaded_device = sc.Device.from_hdf5(h5path)
    assert loaded_device == holey_device


@pytest.fixture()
def factorized_model(holey_device) -> sc.FactorizedModel:
    model = sc.factorize_model(
        device=holey_device,
        current_units="uA",
        terminal_currents={"film": {"source": "10 uA", "drain": "-10 uA"}},
        circulating_currents={"hole1": "5 uA"},
        vortices=[sc.Vortex(x=0, y=0, film="film")],
    )
    return model


def test_save_and_load_factorized_model(factorized_model: sc.FactorizedModel, tmp_path):
    h5path = tmp_path / "factorized-model.h5"
    with h5py.File(h5path, "x") as f:
        factorized_model.to_hdf5(f)

    with h5py.File(h5path, "r") as f:
        loaded_model = sc.FactorizedModel.from_hdf5(f)

    assert isinstance(loaded_model, sc.FactorizedModel)


@pytest.mark.parametrize("applied_field", [0, -1, 2])
def test_multi_terminal_currents(plus_device, applied_field):
    xs = np.linspace(-2, 2, 201)
    ys = -3 * np.ones_like(xs)
    rs = np.stack([xs, ys], axis=1)
    sections = [sc.geometry.rotate(rs, i * 90) for i in range(4)]

    with pytest.raises(ValueError):
        # Current not conserved
        terminal_currents = {
            "plus": {
                "drain": -5,
                "source1": "1 uA",
                "source2": sc.ureg("2 uA"),
                "source3": 3,
            }
        }
        solution = sc.solve(
            plus_device,
            terminal_currents=terminal_currents,
            applied_field=sc.sources.ConstantField(applied_field),
            current_units="uA",
            field_units="uT",
        )[-1]

    terminal_currents = {
        "plus": {
            "drain": -6,
            "source1": "1 uA",
            "source2": sc.ureg("2 uA"),
            "source3": 3,
        }
    }
    solution = sc.solve(
        plus_device,
        terminal_currents=terminal_currents,
        applied_field=sc.sources.ConstantField(applied_field),
        current_units="uA",
        field_units="uT",
    )[-1]

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "solution.h5")
        solution.to_hdf5(path)
        loaded_solution = sc.Solution.from_hdf5(path)
        assert loaded_solution == solution

    currents = []
    for coords in sections:
        J = solution.interp_current_density(
            coords, film="plus", units="uA/um", with_units=False
        )
        _, unit_normals = sc.geometry.path_vectors(coords)
        dr = np.linalg.norm(np.diff(coords, axis=0), axis=1)[0]
        currents.append(np.sum(J * dr * unit_normals))
    target_currents = solution.terminal_currents["plus"].values()
    assert np.abs(np.sum(currents) / terminal_currents["plus"]["drain"]) < 5e-2
    for actual, target in zip(currents, target_currents):
        assert np.isclose(-actual, target, rtol=5e-2)


def test_holey_device(holey_device):
    device = holey_device
    terminal_currents = {"film": {"source": "2 uA", "drain": "-2 uA"}}
    circulating_currents = {
        "hole1": "1 uA",
        "hole2": "-1 uA",
    }

    solution = sc.solve(
        device,
        terminal_currents=terminal_currents,
        circulating_currents=circulating_currents,
        applied_field=sc.sources.ConstantField(0),
        field_units="uT",
        current_units="uA",
    )[-1]

    xs_left = np.linspace(-0.5, 0, 201)
    ys_left = np.ones_like(xs_left)
    xs_right = -xs_left[::-1]
    ys_right = ys_left
    xs = np.linspace(-0.5, 0.5, 401)
    ys = np.ones_like(xs)
    sections = [
        np.stack([xs, 0 * ys], axis=1),
        np.stack([xs_right, -0.5 * ys_right], axis=1),
        np.stack([xs_left, +0.5 * ys_left], axis=1),
        np.stack([xs_right, +0.5 * ys_right], axis=1),
        np.stack([xs_left, -0.5 * ys_left], axis=1),
    ]
    target_currents = [2, 2, 2, 0, 0]
    currents = []
    for coords in sections:
        J = solution.interp_current_density(
            coords,
            film="film",
            units="uA/um",
            with_units=False,
        )
        _, unit_normals = sc.geometry.path_vectors(coords)
        dr = np.linalg.norm(np.diff(coords, axis=0), axis=1)[0]
        currents.append(np.sum(J * dr * unit_normals))
    for actual, target in zip(currents, target_currents):
        assert np.isclose(actual, target, rtol=5e-2, atol=1e-2)


def test_no_terminals(plus_device_no_terminals):
    device = plus_device_no_terminals
    solutions = sc.solve(
        device=device,
        applied_field=sc.sources.ConstantField(1),
        field_units="mT",
    )
    assert len(solutions) == 1
