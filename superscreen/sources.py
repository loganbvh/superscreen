import inspect
import operator
from typing import Callable, Union, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

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


class FieldSource(object):
    """A callable object that computes a scalar field
    as a function of position coordinates x, y, z.

    Addition, subtraction, multiplication, and division
    between multiple FieldSources and/or real numbers (ints and floats)
    is supported. The result of any of these operations is a
    CompositeFieldSource object.

    Args:
        func: A callable/function that actually calculcates the field.
            The function must take x, y, z as its first three (positional)
            arguments, and all other arguments must be keyword arguments.
            Therefore func should have a signature like func(x, y, z, a=1, b=2, c=True),
            func(x, y, z, *, a, b, c), or func(x, y, z, *, a, b=None, c=3).
        **kwargs: Keyword arguments for func.
    """

    def __init__(self, func: Callable, **kwargs):
        argspec = inspect.getfullargspec(func)
        args = argspec.args
        if args[:3] != ["x", "y", "z"]:
            raise ValueError(
                "The first three function arguments must be x, y, z, "
                f"not {', '.join(args[:3])}."
            )
        defaults = argspec.defaults or []
        if len(defaults) != len(args) - len("xyz"):
            raise ValueError(
                "All arguments other than x, y, z must be keyword arguments."
            )
        kwonlyargs = set(kwargs) - set(argspec.args[3:])
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
        z: Numeric,
    ) -> Numeric:
        return self.func(x, y, z, **self.kwargs)

    def _get_argspec(self) -> _FakeArgSpec:
        kwargs, kwarg_values = list(zip(*self.kwargs.items()))
        return _FakeArgSpec(
            args=list(kwargs),
            defaults=kwarg_values,
        )

    def __repr__(self) -> str:
        func_repr = function_repr(self.func, argspec=self._get_argspec())
        return f"FieldSource<{func_repr}>"

    def __add__(self, other) -> "CompositeFieldSource":
        """self + other"""
        return CompositeFieldSource(self, other, operator.add)

    def __radd__(self, other) -> "CompositeFieldSource":
        """other + self"""
        return CompositeFieldSource(other, self, operator.add)

    def __sub__(self, other) -> "CompositeFieldSource":
        """self - other"""
        return CompositeFieldSource(self, other, operator.sub)

    def __rsub__(self, other) -> "CompositeFieldSource":
        """other - self"""
        return CompositeFieldSource(other, self, operator.sub)

    def __mul__(self, other) -> "CompositeFieldSource":
        """self * other"""
        return CompositeFieldSource(self, other, operator.mul)

    def __rmul__(self, other) -> "CompositeFieldSource":
        """other * self"""
        return CompositeFieldSource(other, self, operator.mul)

    def __truediv__(self, other) -> "CompositeFieldSource":
        """self / other"""
        return CompositeFieldSource(self, other, operator.truediv)

    def __rtruediv__(self, other) -> "CompositeFieldSource":
        """other / self"""
        return CompositeFieldSource(other, self, operator.truediv)


class CompositeFieldSource(object):

    """A callable object that behaves like a FieldSource
    (i.e. it computes a scalar field as a function of
    position coordinates x, y, z). A CompositeFieldSource object is created as
    a result of mathematical operations between FieldSources, CompositeFieldSources,
    and/or real numbers.

    Addition, subtraction, multiplication, and division
    between FieldSources, CompositeFieldSources and real numbers (ints and floats)
    is supported. The result of any of these operations is a new
    CompositeFieldSource object.

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
    }

    def __init__(
        self,
        left: Union[int, float, FieldSource, "CompositeFieldSource"],
        right: Union[int, float, FieldSource, "CompositeFieldSource"],
        operator: Callable,
    ):
        valid_types = (int, float, FieldSource, CompositeFieldSource)
        if not isinstance(left, valid_types):
            raise TypeError(
                f"Left must be a number, FieldSource, or CompositeFieldSource, "
                f"not {type(left)}."
            )
        if not isinstance(right, valid_types):
            raise TypeError(
                f"Right must be a number, FieldSource, or CompositeFieldSource, "
                f"not {type(right)}."
            )
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            raise TypeError(
                "Either left or right must be a FieldSource or CompositeFieldSource."
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
        z: Numeric,
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
        if isinstance(self.left, FieldSource):
            left_argspec = self.left._get_argspec()
            left_repr = function_repr(self.left.func, left_argspec)
        elif isinstance(self.left, CompositeFieldSource):
            left_repr = self.left._bare_repr()
        else:
            left_repr = str(self.left)

        if isinstance(self.right, FieldSource):
            right_argspec = self.right._get_argspec()
            right_repr = function_repr(self.right.func, right_argspec)
        elif isinstance(self.right, CompositeFieldSource):
            right_repr = self.right._bare_repr()
        else:
            right_repr = str(self.right)

        return f"({left_repr} {op_str} {right_repr})"

    def __repr__(self) -> str:
        return f"CompositeFieldSource<{self._bare_repr()}>"


def constant(
    x: Numeric, y: Numeric, z: Numeric, value: Optional[Union[int, float]] = 0
) -> Numeric:
    """Constant field.

    Args:
        x, y, z: Position coordinates.
        value: Value of the field. Default: 0.
    """
    return value * np.ones_like(x)


def vortex(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    vortex_position: Optional[Tuple[float, float, float]] = (0, 0, 0),
    nPhi0: Optional[int] = 1,
) -> Numeric:
    """Field from an isolated vortex.

    Args:
        x, y, z: Position coordinates.
        vortex_position: (x, y, z) position of the vortex.
        nPhi0: Number of flux quanta contained in the vortex.
    """
    xv, yv, zv = vortex_position
    xp = x - xv
    yp = y - yv
    zp = z - zv
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


def tilt(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    *,
    axis: str,
    angle: Optional[float] = 0,
    offset: Optional[float] = 0,
) -> Tuple[Numeric, Numeric, Numeric]:
    if axis not in "xy":
        raise ValueError(f"Axis must be 'x' or 'y', got {axis}.")

    if axis == "x":
        i = 1
    else:
        i = 0

    if angle == 0:
        return x, y, z

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.asarray(z)
    if z.ndim == 0:
        z = z * np.ones_like(x)
    points = np.array([x, y, z]).T

    points[:, i] -= offset
    r = Rotation.from_euler(axis, angle, degrees=True)
    points = r.apply(points)
    points[:, i] += offset
    x, y, z = [np.squeeze(a) for a in points.T]
    if x.ndim == 0:
        x = x.item()
    if y.ndim == 0:
        y = y.item()
    if z.ndim == 0:
        z = z.item()
    return x, y, z


def tilted_vortex(
    x: Numeric,
    y: Numeric,
    z: Numeric,
    vortex_position: Optional[Tuple[float, float, float]] = (0, 0, 0),
    nPhi0: Optional[int] = 1,
    x_axis_tilt: Optional[float] = 0,
    x_axis_offset: Optional[float] = 0,
    y_axis_tilt: Optional[float] = 0,
    y_axis_offset: Optional[float] = 0,
    tilt_x_first: Optional[bool] = True,
):
    if tilt_x_first:
        x, y, z = tilt(x, y, z, axis="x", angle=x_axis_tilt, offset=x_axis_offset)
        x, y, z = tilt(x, y, z, axis="y", angle=y_axis_tilt, offset=y_axis_offset)
    else:
        x, y, z = tilt(x, y, z, axis="y", angle=y_axis_tilt, offset=y_axis_offset)
        x, y, z = tilt(x, y, z, axis="x", angle=x_axis_tilt, offset=x_axis_offset)

    return vortex(x, y, z, vortex_position=vortex_position, nPhi0=nPhi0)


def ConstantField(value: float) -> FieldSource:
    return FieldSource(constant, value=value)


def VortexField(
    position: Tuple[float, float, float], nPhi0: Optional[int] = 1
) -> FieldSource:
    return FieldSource(vortex, vortex_position=position, nPhi0=nPhi0)


def TiltedVortexField(
    vortex_position: Tuple[float, float, float],
    nPhi0: Optional[int] = 1,
    x_axis_tilt: Optional[float] = 0,
    x_axis_offset: Optional[float] = 0,
    y_axis_tilt: Optional[float] = 0,
    y_axis_offset: Optional[float] = 0,
    tilt_x_first: Optional[bool] = True,
) -> FieldSource:
    return FieldSource(
        tilted_vortex,
        vortex_position=vortex_position,
        nPhi0=nPhi0,
        x_axis_tilt=x_axis_tilt,
        x_axis_offset=x_axis_offset,
        y_axis_tilt=y_axis_tilt,
        y_axis_offset=y_axis_offset,
        tilt_x_first=tilt_x_first,
    )
