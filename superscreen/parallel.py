# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import os
import logging
import itertools
import warnings
import multiprocessing as mp
from typing import Union, Callable, Optional, Dict, Tuple, List, Any

import numpy as np

try:
    import ray
except ImportError:
    ray = None

from . import brandt
from .brandt import CirculatingCurrentsType
from .io import save_solutions
from .device import Device
from .parameter import Parameter


logger = logging.getLogger(__name__)


def create_setups(
    device: Device,
    applied_fields: Union[Parameter, List[Parameter]],
    circulating_currents: Optional[
        Union[CirculatingCurrentsType, List[CirculatingCurrentsType]]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    product: bool = False,
) -> List[Tuple[Device, Parameter, CirculatingCurrentsType]]:
    """Generate a list of (device, applied_field, circulating_currents).

    Args:
        device: The Device to simulate.
        applied_fields: A callable or list of callables that compute(s) the applied
            magnetic field as a function of x, y, z coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}`` or list
            of such dicts. If circulating_current is a float, then it is assumed to
            be in units of current_units. If circulating_current is a string, then
            it is converted to a pint.Quantity.
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
        A list of "setups", (device, applied_field, circulating_currents).
    """

    if not isinstance(applied_fields, (list, tuple)):
        applied_fields = [applied_fields]
    for param in applied_fields:
        if not isinstance(param, Parameter):
            raise TypeError(
                f"All applied fields must be Parameters (got {type(param)})."
            )

    layer_update_kwargs = layer_update_kwargs or []
    circulating_currents = circulating_currents or {}

    if not isinstance(circulating_currents, (list, tuple)):
        circulating_currents = [circulating_currents]

    if layer_updater and layer_update_kwargs:
        devices = []
        for kwargs in layer_update_kwargs:
            d = device.copy(with_arrays=False)
            updated_layers = [layer_updater(layer, **kwargs) for layer in d.layers_list]
            d.layers_list = updated_layers
            devices.append(d)
    else:
        devices = [device.copy(with_arrays=False)]

    if product:
        setups = list(itertools.product(devices, applied_fields, circulating_currents))
    else:
        max_len = max(len(devices), len(applied_fields), len(circulating_currents))
        for lst in [devices, applied_fields, circulating_currents]:
            if len(lst) == 1 and max_len > 1:
                item = lst[0]
                lst.extend([item] * (max_len - 1))
        if not (len(devices) == len(applied_fields) == len(circulating_currents)):
            raise ValueError(
                "Devices, applied_fields, and circulating_current must be lists "
                "of the same length."
            )
        setups = list(zip(devices, applied_fields, circulating_currents))

    return setups


#######################################################################################
# Synchronous (serial) execution
#######################################################################################


def solve_many_serial(
    *,
    device: Device,
    applied_fields: Union[Parameter, List[Parameter]],
    circulating_currents: Optional[
        Union[CirculatingCurrentsType, List[CirculatingCurrentsType]]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
    coupled: Optional[bool] = True,
    iterations: Optional[int] = 1,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
):
    """Solve many models in a single process."""
    setups = create_setups(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    arrays = device.get_arrays(copy_arrays=False, dense=True)

    if directory is not None:
        device.to_file(os.path.join(directory, device.name), save_mesh=True)

    if return_solutions:
        all_solutions = []
    if directory is not None:
        paths = []

    logger.info(f"Solving {len(setups)} models serially with 1 process.")

    for i, (device, applied_field, circulating_currents) in enumerate(setups):

        device.set_arrays(arrays)

        solutions = brandt.solve(
            device=device,
            applied_field=applied_field,
            circulating_currents=circulating_currents,
            field_units=field_units,
            current_units=current_units,
            coupled=coupled,
            iterations=iterations,
            check_inversion=check_inversion,
            log_level=log_level,
        )
        if directory is not None:
            path = os.path.abspath(os.path.join(directory, str(i)))
            if keep_only_final_solution:
                solutions[-1].to_file(path, save_mesh=False)
            else:
                save_solutions(solutions, path, save_mesh=False)
            paths.append(path)
        if return_solutions:
            if keep_only_final_solution:
                all_solutions.append(solutions[-1])
            else:
                all_solutions.append(solutions)

    if directory is None:
        paths = None
    if not return_solutions:
        all_solutions = None

    return all_solutions, paths


#######################################################################################
# Concurrency using multiprocessing
#######################################################################################

# See: http://thousandfold.net/cz/2014/05/01/sharing-numpy-arrays-between-processes-using-multiprocessing-and-ctypes/
# See: https://stackoverflow.com/questions/37705974/why-are-multiprocessing-sharedctypes-assignments-so-slow


def shared_array_to_numpy(
    shared_array: mp.RawArray, shape: Tuple[int, ...]
) -> np.ndarray:
    """Convert a shared RawArray to a numpy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        array = np.ctypeslib.as_array(shared_array).reshape(shape)
    return array


def numpy_to_shared_array(array: np.ndarray) -> mp.RawArray:
    """Convert a numpy array to a shared RayArray."""
    dtype = np.ctypeslib.as_ctypes_type(array.dtype)
    shared_array = mp.RawArray(dtype, array.size)
    sh_np = np.ctypeslib.as_array(shared_array).reshape(array.shape)
    np.copyto(sh_np, array)
    return shared_array


def share_arrays(device: Device) -> Dict[str, Tuple[mp.RawArray, Tuple[int, ...]]]:
    """Convert all arrays in the device to shared RawArrays."""
    arrays = device.get_arrays(copy_arrays=False, dense=True)
    C_vectors = {
        name: (numpy_to_shared_array(array), array.shape)
        for name, array in arrays["C_vectors"].items()
    }
    shared_arrays = {
        name: (numpy_to_shared_array(array), array.shape)
        for name, array in arrays.items()
        if name != "C_vectors"
    }
    shared_arrays["C_vectors"] = C_vectors
    return shared_arrays


def init(shared_arrays):
    global init_arrays
    init_arrays = shared_arrays


def solve_single_mp(kwargs: Dict[str, Any]) -> str:
    """Solve a single setup (multiprocessing)."""
    directory = kwargs.pop("directory", None)
    index = kwargs.pop("index")
    return_solutions = kwargs.pop("return_solutions")
    keep_only_final_solution = kwargs.pop("keep_only_final_solution")

    device = kwargs["device"]
    numpy_arrays = {}
    numpy_arrays["C_vectors"] = {}
    # init_arrays is the dict of arrays stored in shared memory
    for name, data in init_arrays.items():
        if name == "C_vectors":
            for layer, (vec, shape) in data.items():
                numpy_arrays["C_vectors"][layer] = shared_array_to_numpy(vec, shape)
        else:
            shared_array, shape = data
            numpy_arrays[name] = shared_array_to_numpy(shared_array, shape)

    device.set_arrays(numpy_arrays)
    kwargs["device"] = device

    solutions = brandt.solve(**kwargs)

    if directory is None:
        path = None
    else:
        path = os.path.abspath(os.path.join(directory, str(index)))
        if keep_only_final_solution:
            solutions[-1].to_file(path, save_mesh=False)
        else:
            save_solutions(solutions, path, save_mesh=False)
    s = None
    if return_solutions:
        if keep_only_final_solution:
            s = solutions[-1]
        else:
            s = solutions
    return s, path


def solve_many_mp(
    *,
    device: Device,
    applied_fields: Union[Parameter, List[Parameter]],
    circulating_currents: Optional[
        Union[CirculatingCurrentsType, List[CirculatingCurrentsType]]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
    coupled: Optional[bool] = True,
    iterations: Optional[int] = 1,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
) -> List[str]:
    """Solve many models in parallel using multiprocessing."""
    setups = create_setups(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    # Put the device's big arrays in shared memory
    shared_arrays = share_arrays(device)

    if directory is not None:
        device.to_file(os.path.join(directory, device.name), save_mesh=True)

    kwargs = []
    for i, (device, applied_field, circulating_currents) in enumerate(setups):
        kwargs.append(
            dict(
                directory=directory,
                index=i,
                return_solutions=return_solutions,
                keep_only_final_solution=keep_only_final_solution,
                device=device,
                applied_field=applied_field,
                circulating_currents=circulating_currents,
                field_units=field_units,
                current_units=current_units,
                coupled=coupled,
                iterations=iterations,
                check_inversion=check_inversion,
                log_level=log_level,
            )
        )

    nproc = min(len(setups), mp.cpu_count())
    logger.info(
        f"Solving {len(setups)} models in parallel with {nproc} process(es) "
        f"using multiprocessing."
    )
    with mp.Pool(processes=nproc, initializer=init, initargs=(shared_arrays,)) as pool:
        results = pool.map(solve_single_mp, kwargs)
        pool.close()
        pool.join()

    all_solutions, paths = zip(*results)
    if all(s is None for s in all_solutions):
        all_solutions = None
    if all(p is None for p in paths):
        paths = None

    return all_solutions, paths


#######################################################################################
# Concurrency using ray
#######################################################################################


@ray.remote
def solve_single_ray(
    *, directory, index, arrays, return_solutions, keep_only_final_solution, **kwargs
):
    """Solve a single setup (ray)."""
    device = kwargs["device"]
    device.set_arrays(arrays)

    solutions = brandt.solve(**kwargs)

    log_level = kwargs.pop("log_level", None)
    if log_level is not None:
        logging.basicConfig(level=log_level)

    if directory is None:
        path = None
    else:
        path = os.path.abspath(os.path.join(directory, str(index)))
        if keep_only_final_solution:
            solutions[-1].to_file(path, save_mesh=False)
        else:
            save_solutions(solutions, path, save_mesh=False)
    s = None
    if return_solutions:
        if keep_only_final_solution:
            s = solutions[-1]
        else:
            s = solutions
    return s, path


def solve_many_ray(
    *,
    device: Device,
    applied_fields: Union[Parameter, List[Parameter]],
    circulating_currents: Optional[
        Union[CirculatingCurrentsType, List[CirculatingCurrentsType]]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
    coupled: Optional[bool] = True,
    iterations: Optional[int] = 1,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
):
    """Solve many models in parallel using ray."""
    if ray is None:
        raise EnvironmentError(
            "ray is not installed. Please run 'pip install ray[default]' to install it."
        )

    setups = create_setups(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    ncpus = mp.cpu_count()
    if not ray.is_initialized():
        ncpus = min(len(setups), ncpus)
        ray.init(num_cpus=ncpus)

    # Put the device's big arrays in shared memory
    arrays = device.get_arrays(copy_arrays=False, dense=True)
    arrays_ref = ray.put(arrays)

    if directory is not None:
        device.to_file(os.path.join(directory, device.name), save_mesh=True)

    logger.info(
        f"Solving {len(setups)} models in parallel with {ncpus} process(es) using ray."
    )

    result_ids = []
    for i, (device, applied_field, circulating_currents) in enumerate(setups):
        result_ids.append(
            solve_single_ray.remote(
                directory=directory,
                index=i,
                arrays=arrays_ref,
                return_solutions=return_solutions,
                keep_only_final_solution=keep_only_final_solution,
                device=device,
                applied_field=applied_field,
                circulating_currents=circulating_currents,
                field_units=field_units,
                current_units=current_units,
                coupled=coupled,
                iterations=iterations,
                check_inversion=check_inversion,
                log_level=log_level,
            )
        )

    results = ray.get(result_ids)

    all_solutions, paths = zip(*results)
    if all(s is None for s in all_solutions):
        all_solutions = None
    if all(p is None for p in paths):
        paths = None

    return all_solutions, paths
