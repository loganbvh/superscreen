from superscreen.sources.dipole import DipoleField
import pytest
import numpy as np

from superscreen import Parameter
from superscreen.sources import ConstantField, VortexField, DipoleField

@pytest.mark.parametrize("shape", [(1, ), (10, ), (100, )])
def test_constant_field(shape):
    size = np.prod(shape)
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    value = 10.5
    param = ConstantField(value)
    field = param(x, y, z)
    assert isinstance(param, Parameter)
    assert field.shape == shape
    assert np.all(field == value)


@pytest.mark.parametrize("shape", [(1, ), (10, ), (100, )])
@pytest.mark.parametrize("vortex_position", [(0, 0, 0), (1, 0, 1), (5, -5, 4)])
def test_vortex_field(shape, vortex_position):
    size = np.prod(shape)
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    x0, y0, z0 = vortex_position
    param = VortexField(x0=x0, y0=y0, z0=z0)
    field = param(x, y, z)
    assert isinstance(param, Parameter)
    assert field.shape == shape
    assert np.all(field != 0)

@pytest.mark.parametrize(
    "dipole_positions",
    [(0, 0, 0), np.array([[0, 0, 0]]), np.array([1, 5, 2])]
)
@pytest.mark.parametrize("shape", [(1, ), (10, ), (100, )])
def test_dipole_field_single(shape, dipole_positions):
    size = np.prod(shape)
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


@pytest.mark.parametrize("num_dipoles", [1, 5, 200])
@pytest.mark.parametrize("shape", [(1, ), (10, ), (100, )])
def test_dipole_field_single(shape, num_dipoles):
    size = np.prod(shape)
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
    assert field.shape == shape
    assert np.isfinite(field).all()
