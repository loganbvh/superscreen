# This file is part of superscreen.
#
#     Copyright (c) 2021 Logan Bishop-Van Horn
#
#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import inspect
import operator
from typing import Callable, Union, Optional

import numpy as np

Numeric = Union[int, float, np.ndarray]


class _FakeArgSpec(object):
    def __init__(
        self,
        args=None,
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=None,
        kwonlydefaults=None,
        annotations=None,
    ):
        self.args = args
        self.varargs = varargs
        self.varkw = varkw
        self.defaults = defaults
        self.kwonlyargs = kwonlyargs
        self.kwonlydefaults = kwonlydefaults
        self.annotations = annotations


def function_repr(
    func: Callable,
    argspec: Optional[Union[_FakeArgSpec, inspect.FullArgSpec]] = None,
) -> str:
    """Returns a human-readable string representation for a function."""
    if argspec is None:
        argspec = inspect.getfullargspec(func)
    args = [str(arg) for arg in argspec.args]

    if argspec.defaults:
        for i, val in enumerate(argspec.defaults[::-1]):
            args[-(i + 1)] = args[-(i + 1)] + f"={val}"

    if argspec.varargs:
        args.append("*" + argspec.varargs)

    if argspec.kwonlyargs:
        if not argspec.varargs:
            args.append("*")
        args.extend(argspec.kwonlyargs)
    if argspec.kwonlydefaults:
        for i, name in enumerate(args):
            if name in argspec.kwonlydefaults:
                args[i] = args[i] + f"={argspec.kwonlydefaults[name]}"
    if argspec.varkw:
        args.append("**" + argspec.varkw)

    if argspec.annotations:
        for i, name in enumerate(args):
            if name in argspec.annotations:
                args[i] = args[i] + f": {argspec.annotations[name].__name__}"

    return func.__name__ + "(" + ", ".join(args) + ")"


class Parameter(object):
    """A callable object that computes a scalar parameter
    as a function of position coordinates x, y (and optionally z).

    Addition, subtraction, multiplication, and division
    between multiple Parameters and/or real numbers (ints and floats)
    is supported. The result of any of these operations is a
    ``CompositeParameter`` object.

    Args:
        func: A callable/function that actually calculates the parameter's value.
            The function must take x, y (and optionally z) as the first and only
            positional arguments, and all other arguments must be keyword arguments.
            Therefore func should have a signature like ``func(x, y, z, a=1, b=2, c=True)``,
            ``func(x, y, *, a, b, c)``, ``func(x, y, z, *, a, b, c)``,
            or ``func(x, y, z, *, a, b=None, c=3)``.
        kwargs: Keyword arguments for func.
    """

    def __init__(self, func: Callable, **kwargs):
        argspec = inspect.getfullargspec(func)
        args = argspec.args
        num_args = 2
        if args[:num_args] != ["x", "y"]:
            raise ValueError(
                "The first function arguments must be x and y, "
                f"not {', '.join(args[:num_args])}."
            )
        if "z" in args:
            if args.index("z") != num_args:
                raise ValueError(
                    "If the function takes an argument z, "
                    "it must be the third argument (x, y, z)."
                )
            else:
                num_args = 3
        defaults = argspec.defaults or []
        if len(defaults) != len(args) - num_args:
            raise ValueError(
                "All arguments other than x, y, z must be keyword arguments."
            )
        kwonlyargs = set(kwargs) - set(argspec.args[num_args:])
        if kwonlyargs != set(argspec.kwonlyargs or []):
            raise ValueError(
                f"Provided keyword-only arguments ({kwonlyargs}) "
                f"do not match the function signature: {function_repr(func)}."
            )

        self.func = func
        self.kwargs = kwargs

    def __call__(
        self,
        x: Numeric,
        y: Numeric,
        z: Optional[Numeric] = None,
    ) -> Numeric:
        kwargs = self.kwargs.copy()
        if z is not None:
            kwargs["z"] = z
        return self.func(x, y, **kwargs)

    def _get_argspec(self) -> _FakeArgSpec:
        kwargs, kwarg_values = list(zip(*self.kwargs.items()))
        return _FakeArgSpec(
            args=list(kwargs),
            defaults=kwarg_values,
        )

    def __repr__(self) -> str:
        func_repr = function_repr(self.func, argspec=self._get_argspec())
        return f"Parameter<{func_repr}>"

    def __add__(self, other) -> "CompositeParameter":
        """self + other"""
        return CompositeParameter(self, other, operator.add)

    def __radd__(self, other) -> "CompositeParameter":
        """other + self"""
        return CompositeParameter(other, self, operator.add)

    def __sub__(self, other) -> "CompositeParameter":
        """self - other"""
        return CompositeParameter(self, other, operator.sub)

    def __rsub__(self, other) -> "CompositeParameter":
        """other - self"""
        return CompositeParameter(other, self, operator.sub)

    def __mul__(self, other) -> "CompositeParameter":
        """self * other"""
        return CompositeParameter(self, other, operator.mul)

    def __rmul__(self, other) -> "CompositeParameter":
        """other * self"""
        return CompositeParameter(other, self, operator.mul)

    def __truediv__(self, other) -> "CompositeParameter":
        """self / other"""
        return CompositeParameter(self, other, operator.truediv)

    def __rtruediv__(self, other) -> "CompositeParameter":
        """other / self"""
        return CompositeParameter(other, self, operator.truediv)

    def __pow__(self, other) -> "CompositeParameter":
        """self ** other"""
        return CompositeParameter(self, other, operator.pow)

    def __rpow__(self, other) -> "CompositeParameter":
        """other ** self"""
        return CompositeParameter(other, self, operator.pow)

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Parameter):
            return False

        # Check if function bytecode is the same
        if self.__call__.__code__ != other.__call__.__code__:
            return False

        # Checks function name and kwargs
        return repr(self) == repr(other)


class CompositeParameter(Parameter):

    """A callable object that behaves like a Parameter
    (i.e. it computes a scalar value as a function of
    position coordinates x, y, z). A CompositeParameter object is created as
    a result of mathematical operations between Parameters, CompositeParameters,
    and/or real numbers.

    Addition, subtraction, multiplication, division, and exponentiation
    between Parameters, CompositeParameters and real numbers (ints and floats)
    is supported. The result of any of these operations is a new
    CompositeParameter object.

    Args:
        left: The object on the left-hand side of the operator.
        right: The object on the right-hand side of the operator.
        operator: The operator acting on left and right.
    """

    VALID_OPERATORS = {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
        operator.pow: "**",
    }

    def __init__(
        self,
        left: Union[int, float, Parameter, "CompositeParameter"],
        right: Union[int, float, Parameter, "CompositeParameter"],
        operator: Callable,
    ):
        valid_types = (int, float, Parameter, CompositeParameter)
        if not isinstance(left, valid_types):
            raise TypeError(
                f"Left must be a number, Parameter, or CompositeParameter, "
                f"not {type(left)}."
            )
        if not isinstance(right, valid_types):
            raise TypeError(
                f"Right must be a number, Parameter, or CompositeParameter, "
                f"not {type(right)}."
            )
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            raise TypeError(
                "Either left or right must be a Parameter or CompositeParameter."
            )
        if operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Unknown operator, {operator}. "
                f"Valid operators are {list(self.VALID_OPERATORS)}."
            )
        self.left = left
        self.right = right
        self.operator = operator

    def __call__(
        self,
        x: Numeric,
        y: Numeric,
        z: Optional[Numeric] = None,
    ) -> Numeric:
        if isinstance(self.left, (int, float)):
            left_val = self.left
        else:
            left_val = self.left(x, y, z)
        if isinstance(self.right, (int, float)):
            right_val = self.right
        else:
            right_val = self.right(x, y, z)
        return self.operator(left_val, right_val)

    def _bare_repr(self) -> str:
        op_str = self.VALID_OPERATORS[self.operator]
        if isinstance(self.left, CompositeParameter):
            left_repr = self.left._bare_repr()
        elif isinstance(self.left, Parameter):
            left_argspec = self.left._get_argspec()
            left_repr = function_repr(self.left.func, left_argspec)
        else:
            left_repr = str(self.left)

        if isinstance(self.right, CompositeParameter):
            right_repr = self.right._bare_repr()
        elif isinstance(self.right, Parameter):
            right_argspec = self.right._get_argspec()
            right_repr = function_repr(self.right.func, right_argspec)
        else:
            right_repr = str(self.right)

        return f"({left_repr} {op_str} {right_repr})"

    def __repr__(self) -> str:
        return f"CompositeParameter<{self._bare_repr()}>"


class Constant(Parameter):
    """A Parameter whose value doesn't depend on position."""

    def __init__(self, value, dimensions=2):
        if dimensions not in (2, 3):
            raise ValueError("Dimensions must be 2 or 3.")
        if dimensions == 2:

            def constant(x, y, value=0):
                return value * np.ones_like(x)

        else:

            def constant(x, y, z, value=0):
                return value * np.ones_like(x)

        super().__init__(constant, value=value)
