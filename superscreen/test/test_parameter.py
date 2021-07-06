import numpy as np
import pytest

import superscreen as sc


def test_parameter_against_func():
    def func(x, y, sigma=1):
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    param = sc.Parameter(func, sigma=2)

    x = np.random.rand(100)
    y = np.random.rand(100)

    assert np.array_equal(func(x, y, sigma=2), param(x, y))


def func_2d(x, y, sigma=1):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def func_3d(x, y, z, sigma=1):
    return np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2))


@pytest.mark.parametrize(
    "func, args",
    [
        (func_2d, (np.random.rand(100), np.random.rand(100))),
        (func_3d, (np.random.rand(100), np.random.rand(100), np.random.rand(100))),
    ],
)
def test_parameter_math(func, args):

    param1 = sc.Parameter(func, sigma=10)
    param2 = sc.Parameter(func, sigma=0.1)

    assert np.array_equal((param1 ** 2)(*args), func(*args, sigma=10) ** 2)
    assert np.array_equal((2 * param1)(*args), 2 * func(*args, sigma=10))

    assert np.array_equal(
        (param1 + param2)(*args),
        func(*args, sigma=10) + func(*args, sigma=0.1),
    )

    assert np.array_equal(
        (param1 - param2)(*args),
        func(*args, sigma=10) - func(*args, sigma=0.1),
    )

    assert np.array_equal(
        (param1 * param2)(*args),
        func(*args, sigma=10) * func(*args, sigma=0.1),
    )

    assert np.array_equal(
        (param1 / param2)(*args),
        func(*args, sigma=10) / func(*args, sigma=0.1),
    )

    assert np.array_equal(
        (param1 ** param2)(*args),
        func(*args, sigma=10) ** func(*args, sigma=0.1),
    )

    assert param1 == param1
    assert param2 == param2
    assert param1 != param2
