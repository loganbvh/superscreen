import pickle

import numpy as np
import pytest

import superscreen as sc


def test_function_repr():
    def func(x, *args, **kwargs):
        pass

    print(sc.parameter.function_repr(func))

    def func(x, a: bool = True, b=None):
        pass

    print(sc.parameter.function_repr(func))

    def func(x, *, a: int, b: float, c=None):
        pass

    print(sc.parameter.function_repr(func))


def func_2d(x, y, sigma=1):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


def func_3d(x, y, z, sigma=1):
    return np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))


def test_parameter_against_func():
    def func(x, y, sigma=1):
        return np.exp(-(x**2 + y**2) / (2 * sigma**2))

    param = sc.Parameter(func, sigma=2)
    print(param)

    x = np.random.rand(100)
    y = np.random.rand(100)

    assert np.array_equal(func(x, y, sigma=2), param(x, y))

    assert param != sc.Parameter(func, sigma=1)

    def func2(x, y, sigma=1):
        return np.exp(-np.abs(x + y) / (2 * sigma**2))

    assert param != sc.Parameter(func2, sigma=1)


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
    print(param1, param2)

    assert np.array_equal((param1**2)(*args), func(*args, sigma=10) ** 2)
    assert np.array_equal((2 * param1)(*args), 2 * func(*args, sigma=10))
    assert np.array_equal((1 + param1)(*args), 1 + func(*args, sigma=10))
    assert np.array_equal((1 - param1)(*args), 1 - func(*args, sigma=10))
    assert np.array_equal((2**param1)(*args), 2 ** func(*args, sigma=10))
    assert np.array_equal((1 / (10 + param1))(*args), 1 / (10 + func(*args, sigma=10)))

    with pytest.raises(ValueError):
        _ = sc.Constant(2, dimensions=1)

    assert np.array_equal(
        2 + param1(*args), (sc.Constant(2, dimensions=len(args)) + param1)(*args)
    )

    assert np.array_equal(
        (param1 + param2)(*args), func(*args, sigma=10) + func(*args, sigma=0.1)
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
        (param1**param2)(*args),
        func(*args, sigma=10) ** func(*args, sigma=0.1),
    )

    assert param1 == param1
    assert param2 == param2
    assert param1 != param2

    p3 = param1 * param2
    assert p3 == p3

    assert (param1 * param2) == (param1 * param2)
    assert (param1**2) != param1
    assert (param1**2) != param1 * param2
    print(param1**2)
    print(param1 * param2 + param1)
    print(param1 * (param2 + param1))
    print(2 * param2, param1 / 5)

    with pytest.raises(TypeError):
        _ = complex(1, 1) + param1

    with pytest.raises(TypeError):
        _ = param2 + complex(1, 1)


def test_bad_composite_parameter():
    with pytest.raises(TypeError):
        _ = sc.parameter.CompositeParameter(1, 2, "+")

    p1 = sc.Parameter(func_2d, sigma=10)
    p2 = sc.Parameter(func_2d, sigma=5)

    with pytest.raises(ValueError):
        _ = sc.parameter.CompositeParameter(p1, p2, "<<")


@pytest.mark.parametrize("func", [func_2d, func_3d])
def test_pickle_parameter(func):

    param1 = sc.Parameter(func, sigma=10)
    param2 = sc.Parameter(func, sigma=0.1)

    assert pickle.loads(pickle.dumps(param1)) == param1
    assert pickle.loads(pickle.dumps(param1)) != param2

    assert pickle.loads(pickle.dumps(param1 * param2)) == (param1 * param2)
    assert pickle.loads(pickle.dumps(param1 - param2)) == (param1 - param2)
    assert pickle.loads(pickle.dumps(param1 * param2)) != (param1 / param2)


def test_bad_parameters():
    def func(a, x, y, b=0):
        pass

    with pytest.raises(ValueError):
        _ = sc.Parameter(func, b=0)

    def func(x, y, a, z):
        pass

    with pytest.raises(ValueError):
        _ = sc.Parameter(func)

    def func(x, y, z, a):
        pass

    with pytest.raises(ValueError):
        _ = sc.Parameter(func)

    def func(x, y, a=0, b=0):
        pass

    with pytest.raises(ValueError):
        _ = sc.Parameter(func, a=0, c=None)
