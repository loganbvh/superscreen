import json
import inspect
import datetime
import tempfile

import numpy as np
import pytest

import superscreen as sc

from backports.datetime_fromisoformat import MonkeyPatch

MonkeyPatch.patch_fromisoformat()


@pytest.fixture(scope="module")
def device():

    layers = [
        sc.Layer("layer0", london_lambda=1, thickness=0.1, z0=0),
        sc.Layer("layer1", london_lambda=sc.Constant(2), thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    device.make_mesh(min_points=2000)

    return device


@pytest.fixture(scope="module")
def solutions(device):

    applied_field = sc.sources.ConstantField(1)

    solutions = sc.solve(
        device=device,
        applied_field=applied_field,
        circulating_currents=None,
        field_units="mT",
        iterations=5,
    )

    return solutions


@pytest.mark.parametrize("to_zip", [False, True])
@pytest.mark.parametrize("return_paths", [False, True])
def test_save_and_load_solutions(solutions, return_paths, to_zip):

    with tempfile.TemporaryDirectory() as directory:
        paths = sc.save_solutions(
            solutions,
            directory,
            return_paths=return_paths,
            to_zip=to_zip,
        )
        if return_paths:
            assert isinstance(paths, list)
            assert all(isinstance(path, str) for path in paths)
        else:
            assert paths is None
        loaded_solutions = sc.load_solutions(directory)
        assert isinstance(loaded_solutions, list)
        assert loaded_solutions == solutions

        assert inspect.isgeneratorfunction(sc.iload_solutions)
        loaded_solutions = sc.iload_solutions(directory)
        assert inspect.isgenerator(loaded_solutions)
        assert list(loaded_solutions) == solutions


def test_json_serialization():
    json_data = {
        "datetime": datetime.datetime.now(),
        "array": np.random.rand(100).reshape((10, 10)),
        "complex_array": np.random.rand(100) + 1j * np.random.rand(100),
        "float": np.pi,
    }

    loaded_data = json.loads(
        json.dumps(json_data, cls=sc.io.NumpyJSONEncoder),
        object_hook=sc.io.json_numpy_obj_hook,
    )

    assert loaded_data["datetime"] == json_data["datetime"]
    assert loaded_data["float"] == json_data["float"]
    assert np.array_equal(loaded_data["array"], json_data["array"])
    assert np.array_equal(loaded_data["complex_array"], json_data["complex_array"])
