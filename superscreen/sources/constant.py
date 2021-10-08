from typing import Union

import numpy as np

from ..parameter import Parameter


def constant(
    x: Union[int, float, np.ndarray],
    y: Union[int, float, np.ndarray],
    z: Union[int, float, np.ndarray],
    value: Union[int, float] = 0,
) -> Union[int, float, np.ndarray]:
    """Constant field.

    Args:
        x, y, z: Position coordinates.
        value: Value of the field.
    """
    return value * np.ones_like(x)


def ConstantField(value: float = 0) -> Parameter:
    """Returns a Parameter that computes a constant as a function of ``x, y, z``.

    Args:
        value: The constant value of the field.

    Returns:
        A Parameter that returns ``value`` at all ``x, y, z``.
    """
    return Parameter(constant, value=float(value))
