import numpy as np
import pytest

import superscreen as sc


@pytest.mark.parametrize("shape", [(), (10,), (100,)])
def test_constant_field(shape):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    value = 10.5
    param = sc.sources.ConstantField(value)
    field = param(x, y, z)
    assert isinstance(param, sc.Parameter)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.all(field == value)


@pytest.mark.parametrize("shape", [(), (10,), (100,)])
@pytest.mark.parametrize("vortex_position", [(0, 0, 0), (1, 0, 1), (5, -5, 4)])
@pytest.mark.parametrize("vector", [False, True])
def test_vortex_field(shape, vortex_position, vector):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    param = sc.sources.VortexField(r0=vortex_position, vector=vector)
    field = param(x, y, z)
    assert isinstance(param, sc.Parameter)
    if vector:
        if shape == ():
            assert field.shape == (3,)
        else:
            assert field.shape == shape + (3,)
        assert np.all(np.linalg.norm(np.atleast_2d(field), axis=1) != 0)
    else:
        if shape == ():
            assert isinstance(field, float)
        else:
            assert field.shape == shape
        assert np.all(field != 0)


@pytest.mark.parametrize(
    "dipole_positions", [(0, 0, 0), np.array([[0, 0, 0]]), np.array([1, 5, 2])]
)
@pytest.mark.parametrize("shape", [(), (10,), (100,)])
@pytest.mark.parametrize("component", [None, "x", "y", "z"])
def test_dipole_field_single(shape, dipole_positions, component):
    size = int(np.prod(shape))
    x = np.random.random(size).reshape(shape)
    y = np.random.random(size).reshape(shape)
    z = np.random.random(size).reshape(shape)

    moments = [
        (0, 0, 1),
        np.array([0, 0, 1]),
    ]

    for moment in moments:
        param = sc.sources.DipoleField(
            dipole_positions=dipole_positions,
            dipole_moments=moment,
            component=component,
        )
        field = param(x, y, z)
        assert isinstance(param, sc.Parameter)
        if shape == ():
            if component is not None:
                assert isinstance(field, float)
            else:
                assert field.shape == (3,)
        else:
            if component is not None:
                assert field.shape == shape
            else:
                assert field.shape == shape + (3,)
        assert np.isfinite(field).all()

    moments = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    param = sc.sources.DipoleField(
        dipole_positions=dipole_positions,
        dipole_moments=moments,
        component=component,
    )
    with pytest.raises(ValueError):
        field = param(x, y, z)

    with pytest.raises(ValueError):
        _ = sc.sources.DipoleField(
            dipole_positions=dipole_positions,
            dipole_moments=moments,
            component="invalid",
        )


@pytest.mark.parametrize("num_dipoles", [1, 5, 200])
@pytest.mark.parametrize("shape", [(), (10,), (100,)])
@pytest.mark.parametrize("component", [None, "x", "y", "z"])
def test_dipole_field(shape, num_dipoles, component):
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
        param = sc.sources.DipoleField(
            dipole_positions=dipole_positions,
            dipole_moments=moment,
            component=component,
        )
        field = param(x, y, z)
        assert isinstance(param, sc.Parameter)
        if shape == ():
            if component is not None:
                assert isinstance(field, float)
            else:
                assert field.shape == (3,)
        else:
            if component is not None:
                assert field.shape == shape
            else:
                assert field.shape == shape + (3,)
        assert np.isfinite(field).all()

    moments = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    param = sc.sources.DipoleField(
        dipole_positions=dipole_positions,
        dipole_moments=moments,
        component=component,
    )
    with pytest.raises(ValueError):
        field = param(x, y, z)

    moments = np.random.random(3 * num_dipoles).reshape((num_dipoles, 3))
    param = sc.sources.DipoleField(
        dipole_positions=dipole_positions,
        dipole_moments=moments,
        component=component,
    )
    if shape == ():
        if component is not None:
            assert isinstance(field, float)
        else:
            assert field.shape == (3,)
    else:
        if component is not None:
            assert field.shape == shape
        else:
            assert field.shape == shape + (3,)
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
            param = sc.sources.PearlVortexField(r0=vortex_position, xs=xs, ys=ys)
            field = param(x, y, z)
        return

    if shape != ():
        with pytest.raises(ValueError):
            param = sc.sources.PearlVortexField(r0=vortex_position, xs=xs, ys=ys)
            field = param(x, y, z)
        z = np.atleast_1d(z)[0] * np.ones(shape)
    param = sc.sources.PearlVortexField(r0=vortex_position, xs=xs, ys=xs)
    field = param(x, y, z)
    assert isinstance(param, sc.Parameter)
    if shape == ():
        assert isinstance(field, float)
    else:
        assert field.shape == shape
    assert np.all(field >= 0)


def test_sheet_current():
    wire = sc.Polygon(points=sc.geometry.box(12, 1))
    points, _ = wire.make_mesh(min_points=2000)

    current_densities = np.array([1, 0]) * np.ones((points.shape[0], 1))

    field = sc.sources.SheetCurrentField(
        sheet_positions=points,
        current_densities=current_densities,
        z0=0,
        length_units="um",
        current_units="mA",
    )

    # Coordinates at which to evaluate the magnetic field (in microns)
    N = 101
    eval_xs = eval_ys = np.linspace(-5, 5, N)
    eval_z = 0.5
    xgrid, ygrid, zgrid = np.meshgrid(eval_xs, eval_ys, eval_z)
    xgrid = np.squeeze(xgrid)
    ygrid = np.squeeze(ygrid)
    zgrid = np.squeeze(zgrid)

    # field returns shape (N * N, ) and the units are tesla
    Hz = field(xgrid.ravel(), ygrid.ravel(), zgrid.ravel()) * sc.ureg("tesla")
    # Reshape to (N, N) and convert to mT
    Hz = Hz.reshape((N, N)).to("mT").magnitude

    assert np.isclose(Hz.max(), -Hz.min(), rtol=1e-3)
