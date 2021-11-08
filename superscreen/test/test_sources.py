import pytest
import numpy as np

from superscreen import Parameter
from superscreen.sources import (
    ConstantField,
    VortexField,
    PearlVortexField,
    DipoleField,
)


@pytest.mark.parametrize("shape", [(), (10,), (100,)])
def test_constant_field(shape):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    value = 10.5
    param = ConstantField(value)
    field = param(x, y, z)
    assert isinstance(param, Parameter)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.all(field == value)


@pytest.mark.parametrize("shape", [(), (10,), (100,)])
@pytest.mark.parametrize("vortex_position", [(0, 0, 0), (1, 0, 1), (5, -5, 4)])
def test_vortex_field(shape, vortex_position):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    x0, y0, z0 = vortex_position
    param = VortexField(x0=x0, y0=y0, z0=z0)
    field = param(x, y, z)
    assert isinstance(param, Parameter)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.all(field != 0)


@pytest.mark.parametrize(
    "dipole_positions", [(0, 0, 0), np.array([[0, 0, 0]]), np.array([1, 5, 2])]
)
@pytest.mark.parametrize("shape", [(), (10,), (100,)])
def test_dipole_field_single(shape, dipole_positions):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    moments = [
        (0, 0, 1),
        np.array([0, 0, 1]),
    ]

    for moment in moments:
        param = DipoleField(dipole_positions=dipole_positions, dipole_moments=moment)
        field = param(x, y, z)
        assert isinstance(param, Parameter)
        if shape == ():
            assert isinstance(field, float)
        else:
            assert field.shape == shape
        assert np.isfinite(field).all()

    moments = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    param = DipoleField(dipole_positions=dipole_positions, dipole_moments=moments)
    with pytest.raises(ValueError):
        field = param(x, y, z)

    for comp in "xyz":
        assert isinstance(
            DipoleField(
                dipole_positions=dipole_positions,
                dipole_moments=moments,
                component=comp,
            ),
            Parameter,
        )

    with pytest.raises(ValueError):
        _ = DipoleField(
            dipole_positions=dipole_positions,
            dipole_moments=moments,
            component="invalid",
        )


@pytest.mark.parametrize("num_dipoles", [1, 5, 200])
@pytest.mark.parametrize("shape", [(), (10,), (100,)])
def test_dipole_field(shape, num_dipoles):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    dipole_positions = np.random.random(3 * num_dipoles).reshape((num_dipoles, 3))

    moments = [
        (0, 0, 1),
        np.array([0, 0, 1]),
    ]

    for moment in moments:
        param = DipoleField(dipole_positions=dipole_positions, dipole_moments=moment)
        field = param(x, y, z)
        assert isinstance(param, Parameter)
        if shape == ():
            assert isinstance(field, float)
        else:
            assert field.shape == shape
        assert np.isfinite(field).all()

    moments = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    param = DipoleField(dipole_positions=dipole_positions, dipole_moments=moments)
    with pytest.raises(ValueError):
        field = param(x, y, z)

    moments = np.random.random(3 * num_dipoles).reshape((num_dipoles, 3))
    param = DipoleField(dipole_positions=dipole_positions, dipole_moments=moments)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.isfinite(field).all()


@pytest.mark.parametrize("shape", [(), (10,), (100,)])
@pytest.mark.parametrize("vortex_position", [(0, 0, 0), (1, 0, 1), (5, -5, 4)])
def test_pearl_vortex_field(shape, vortex_position):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)
    x0, y0, z0 = vortex_position

    xs = np.linspace(-2, 2, 101)
    ys = np.linspace(-2, 2, 101)

    if (
        (x - x0).min() < xs.min()
        or (x - x0).max() > xs.max()
        or (y - y0).min() < xs.min()
        or (y - y0).max() > ys.max()
    ):
        with pytest.raises(ValueError):
            param = PearlVortexField(x0=x0, y0=y0, z0=z0, xs=xs, ys=ys)
            field = param(x, y, z)
        return

    if shape != ():
        with pytest.raises(ValueError):
            param = PearlVortexField(x0=x0, y0=y0, z0=z0, xs=xs, ys=ys)
            field = param(x, y, z)
        z = np.atleast_1d(z)[0] * np.ones(shape)
    param = PearlVortexField(x0=x0, y0=y0, z0=z0, xs=xs, ys=xs)
    field = param(x, y, z)
    assert isinstance(param, Parameter)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.all(field >= 0)
