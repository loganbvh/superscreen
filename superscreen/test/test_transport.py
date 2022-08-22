import numpy as np
import pytest

import superscreen as sc


@pytest.fixture()
def plus_device():
    layer = sc.Layer("base", Lambda=1)
    width, height = 10, 2
    points = sc.geometry.box(width, height)
    bar = sc.Polygon("plus", points=points)
    plus = bar.union(bar.rotate(90))
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
    drain, *sources = terminals
    device = sc.TransportDevice(
        "plus",
        film=plus,
        layer=layer,
        source_terminals=sources,
        drain_terminal=drain,
        length_units="um",
    )
    device.make_mesh(min_points=2000, optimesh_steps=20)
    return device


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("applied_field", [0, -1, 2])
def test_multi_terminal_currents(plus_device, gpu, applied_field):
    xs = np.linspace(-2, 2, 201)
    ys = -3 * np.ones_like(xs)
    rs = np.stack([xs, ys], axis=1)
    sections = [sc.geometry.rotate(rs, i * 90) for i in range(4)]

    terminal_currents = {
        "source1": "1 uA",
        "source2": sc.ureg("2 uA"),
        "source3": 3,
    }
    solution = sc.solve(
        plus_device,
        terminal_currents=terminal_currents,
        applied_field=sc.sources.ConstantField(applied_field),
        current_units="uA",
        field_units="uT",
        gpu=gpu,
    )[-1]

    currents = []
    for coords in sections:
        J = solution.interp_current_density(coords, units="uA/um", with_units=False)[
            "base"
        ]
        _, unit_normals = sc.geometry.path_vectors(coords)
        dr = np.linalg.norm(np.diff(coords, axis=0), axis=1)[0]
        currents.append(np.sum(J * dr * unit_normals))
    drain_current, *source_currents = currents
    target_currents = solution.terminal_currents.values()
    total_current = sum(target_currents)
    assert np.isclose(drain_current, total_current, rtol=1e-3)
    for actual, target in zip(source_currents, target_currents):
        assert np.isclose(-actual, target, rtol=1e-3)
