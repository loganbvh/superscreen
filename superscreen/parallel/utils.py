import itertools
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import pint

from ..device import Device
from ..parameter import Parameter
from ..solution import Solution, Vortex

logger = logging.getLogger("parallel")


def cpu_count(logical: bool = False):
    return joblib.cpu_count(only_physical_cores=(not logical))


def create_models(
    device: Device,
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
    vortices: Optional[Union[List[Vortex], List[List[Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    product: bool = False,
) -> List[Tuple[Device, Parameter, Dict[str, Union[float, str, pint.Quantity]]]]:
    """Generate a list of (device, applied_field, circulating_currents).

    Args:
        device: The Device to simulate.
        applied_fields: A callable or list of callables that compute(s) the applied
            magnetic field as a function of x, y, z coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}`` or list
            of such dicts. If circulating_current is a float, then it is assumed to
            be in units of current_units. If circulating_current is a string, then
            it is converted to a pint.Quantity.
        vortices: A list (or list of lists) of ``Vortex`` objects.
        layer_updater: A callable with signature
            ``layer_updater(layer: Layer, **kwargs) -> Layer`` that updates
            parameter(s) of each layer in a device.
        layer_update_kwargs: A list of dicts of keyword arguments passed to
            ``layer_updater``.
        product: If True, then all combinations of applied_fields,
            circulating_currrents, and layer_update_kwargs are simulated (the
            behavior is given by itertools.product()). Otherwise, the behavior is
            similar to zip().

    Returns:
        A list of "models", (device, applied_field, circulating_currents, vortices).
    """

    if not isinstance(applied_fields, (list, tuple)):
        applied_fields = [applied_fields]
    for param in applied_fields:
        if param is not None and (not isinstance(param, Parameter)):
            raise TypeError(
                f"All applied fields must be Parameters (got {type(param)})."
            )

    layer_update_kwargs = layer_update_kwargs or []
    circulating_currents = circulating_currents or {}
    vortices = vortices or [[]]

    if not isinstance(circulating_currents, (list, tuple)):
        circulating_currents = [circulating_currents]

    if isinstance(vortices, Vortex):
        vortices = [vortices]
    for i, item in enumerate(vortices):
        if not isinstance(item, list):
            vortices[i] = [vortices[i]]

    if layer_updater and layer_update_kwargs:
        devices = []
        for kwargs in layer_update_kwargs:
            # This does a deepcopy of Layers and Polygons
            d = device.copy(with_mesh=False)
            updated_layers = [layer_updater(layer, **kwargs) for layer in d.layers_list]
            d.layers_list = updated_layers
            devices.append(d)
    else:
        devices = [device.copy(with_mesh=False)]

    if product:
        models = list(
            itertools.product(devices, applied_fields, circulating_currents, vortices)
        )
    else:
        max_len = max(
            len(devices),
            len(applied_fields),
            len(circulating_currents),
            len(vortices),
        )
        for lst in [devices, applied_fields, circulating_currents, vortices]:
            if len(lst) == 1 and max_len > 1:
                item = lst[0]
                lst.extend([item] * (max_len - 1))
        if not (
            len(devices)
            == len(applied_fields)
            == len(circulating_currents)
            == len(vortices)
        ):
            raise ValueError(
                "Devices, applied_fields, circulating_currents, and vortices "
                "must be lists of the same length."
            )
        models = list(zip(devices, applied_fields, circulating_currents, vortices))

    return models


def load_solutions(
    *,
    directory: str,
    num_models: int,
    iterations: int,
    device: Device,
    keep_only_final_solution: bool,
) -> Union[List[Solution], List[List[Solution]]]:
    solutions = []
    for i in range(num_models):
        if keep_only_final_solution:
            solution = Solution.from_file(os.path.join(directory, str(i)))
            solution.device = device
            solutions.append(solution)
        else:
            solutions.append([])
            for j in range(iterations):
                solution = Solution.from_file(os.path.join(directory, str(i), str(j)))
                solution.device = device
                solutions[-1].append(solution)
    return solutions


def cleanup(directory: str, iterations: int) -> None:
    final = os.path.join(directory, str(iterations))
    for f in os.listdir(final):
        shutil.move(os.path.join(final, f), directory)
    for j in range(iterations + 1):
        shutil.rmtree(os.path.join(directory, str(j)))


# Set docstrings for functions in parallel.py based on solve_many.
def patch_docstring(func):
    from ..solver import solve_many

    func.__doc__ = (
        func.__doc__
        + "\n"
        + "\n".join(
            [
                line
                for line in solve_many.__doc__.splitlines()
                if "parallel_method:" not in line
            ][2:]
        )
    )
    annotations = solve_many.__annotations__.copy()
    _ = annotations.pop("parallel_method", None)
    func.__annotations__.update(annotations)
