import gc
import os
import time
import psutil
import logging
import itertools
import tempfile
import warnings
import multiprocessing as mp
from typing import Union, Callable, Optional, Dict, Tuple, List, Any

import ray
import pint
import numpy as np

from . import brandt
from .io import load_solutions, save_solutions
from .device import Device
from .parameter import Parameter
from .solution import Solution


logger = logging.getLogger(__name__)


class NullContextManager(object):
    """Does nothing."""

    def __init__(self, resource=None):
        self.resource = resource

    def __enter__(self):
        return self.resource

    def __exit__(self, *args):
        pass


def create_models(
    device: Device,
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
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
        A list of "models", (device, applied_field, circulating_currents).
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
        models = list(itertools.product(devices, applied_fields, circulating_currents))
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
        models = list(zip(devices, applied_fields, circulating_currents))

    return models


#######################################################################################
# Synchronous (serial) execution
#######################################################################################


def solve_many_serial(
    *,
    device: Device,
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = True,
    iterations: int = 1,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
):
    """Solve many models in a single process."""

    solver = "superscreen.solve_many:serial:1"

    models = create_models(
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

    t0 = time.time()

    logger.info(f"Solving {len(models)} models serially with 1 process.")

    for i, (device, applied_field, circulating_currents) in enumerate(models):

        device.set_arrays(arrays)

        solutions = brandt.solve(
            device=device,
            applied_field=applied_field,
            circulating_currents=circulating_currents,
            field_units=field_units,
            current_units=current_units,
            iterations=iterations,
            check_inversion=check_inversion,
            log_level=log_level,
        )
        for solution in solutions:
            solution._solver = solver
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

    t1 = time.time()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models serially with 1 process in "
        f"{elapsed_seconds:.3f} seconds ({seconds_per_model:.3f} seconds per model)."
    )

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
    """Convert a numpy array to a shared RawArray."""
    dtype = np.ctypeslib.as_ctypes_type(array.dtype)
    shared_array = mp.RawArray(dtype, array.size)
    sh_np = np.ctypeslib.as_array(shared_array).reshape(array.shape)
    np.copyto(sh_np, array)
    return shared_array


def share_arrays(
    arrays: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
) -> Dict[str, Tuple[mp.RawArray, Tuple[int, ...]]]:
    """Convert all arrays in the device to shared RawArrays."""
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
    solver = kwargs.pop("solver")
    use_shared_memory = kwargs.pop("use_shared_memory")

    device = kwargs["device"]

    if use_shared_memory:
        # init_arrays is the dict of arrays stored in shared memory
        numpy_arrays = {}
        numpy_arrays["C_vectors"] = {}
        for name, data in init_arrays.items():
            if name == "C_vectors":
                for layer, (vec, shape) in data.items():
                    numpy_arrays["C_vectors"][layer] = shared_array_to_numpy(vec, shape)
            else:
                shared_array, shape = data
                numpy_arrays[name] = shared_array_to_numpy(shared_array, shape)
    else:
        numpy_arrays = init_arrays

    device.set_arrays(numpy_arrays)
    kwargs["device"] = device

    solutions = brandt.solve(**kwargs)

    for solution in solutions:
        solution._solver = solver

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
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = True,
    iterations: int = 0,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
) -> List[str]:
    """Solve many models in parallel using multiprocessing."""
    models = create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    if directory is not None:
        device.to_file(os.path.join(directory, device.name), save_mesh=True)

    t0 = time.time()

    arrays = device.get_arrays(copy_arrays=False, dense=True)

    if use_shared_memory:
        # Put the device's big arrays in shared memory
        shared_arrays = share_arrays(arrays)
    else:
        shared_arrays = arrays

    nproc = min(len(models), psutil.cpu_count(logical=False))
    solver = f"superscreen.solve_many:multiprocessing:{nproc}"

    kwargs = []
    for i, (device, applied_field, circulating_currents) in enumerate(models):
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
                iterations=iterations,
                check_inversion=check_inversion,
                log_level=log_level,
                solver=solver,
                use_shared_memory=use_shared_memory,
            )
        )
    if use_shared_memory:
        shared_mem_str = "with shared memory"
    else:
        shared_mem_str = "without shared memory"
    logger.info(
        f"Solving {len(models)} models in parallel using multiprocessing with "
        f"{nproc} process(es) {shared_mem_str}."
    )
    with mp.Pool(processes=nproc, initializer=init, initargs=(shared_arrays,)) as pool:
        results = pool.map(solve_single_mp, kwargs)
        pool.close()
        pool.join()

    t1 = time.time()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models in parallel using multiprocessing with {nproc} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    all_solutions, paths = zip(*results)
    all_solutions = list(all_solutions)
    paths = list(paths)
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
    *,
    directory,
    index,
    arrays,
    keep_only_final_solution,
    solver,
    **kwargs,
):
    """Solve a single setup (ray)."""
    kwargs["device"].set_arrays(arrays)

    log_level = kwargs.pop("log_level", None)
    if log_level is not None:
        logging.basicConfig(level=log_level)

    solutions = brandt.solve(**kwargs)

    for solution in solutions:
        solution._solver = solver

    if directory is None:
        path = None
    else:
        path = os.path.abspath(os.path.join(directory, str(index)))
        if keep_only_final_solution:
            solutions[-1].to_file(path, save_mesh=False)
        else:
            save_solutions(solutions, path, save_mesh=False)
    del kwargs
    del solution
    del solutions
    return path


def solve_many_ray(
    *,
    device: Device,
    applied_fields: Optional[Union[Parameter, List[Parameter]]] = None,
    circulating_currents: Optional[
        Union[
            Dict[str, Union[float, str, pint.Quantity]],
            List[Dict[str, Union[float, str, pint.Quantity]]],
        ]
    ] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: Optional[bool] = True,
    iterations: int = 0,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
):
    """Solve many models in parallel using ray."""

    models = create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    initialized_ray = False
    nproc = psutil.cpu_count(logical=False)
    if not ray.is_initialized():
        nproc = min(len(models), nproc)
        logger.info(f"Initializing ray with {nproc} process(es).")
        ray.init(num_cpus=nproc)
        initialized_ray = True

    ray_resources = ray.available_resources()
    nproc = int(ray_resources["CPU"])
    logger.info(f"ray resources: {ray_resources}")

    solver = f"superscreen.solve_many:ray:{nproc}"

    if directory is not None:
        device.to_file(os.path.join(directory, device.name), save_mesh=True)

    t0 = time.time()

    # Put the device's big arrays in shared memory.
    # The copy is necessary here so that the arrays do not get pinned in shared memory.
    arrays = device.get_arrays(copy_arrays=True, dense=True)
    if use_shared_memory:
        arrays_ref = ray.put(arrays)
    else:
        arrays_ref = arrays

    if use_shared_memory:
        shared_mem_str = "with shared memory"
    else:
        shared_mem_str = "without shared memory"
    logger.info(
        f"Solving {len(models)} models in parallel using ray with "
        f"{nproc} process(es) {shared_mem_str}."
    )

    # To prevent filling up the ray object store, we save solutions to disk
    # even if return_solutions is True.
    if directory is None:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = NullContextManager(directory)

    with save_context as save_directory:
        result_ids = []
        for i, (device_copy, applied_field, circulating_currents) in enumerate(models):
            result_ids.append(
                solve_single_ray.remote(
                    directory=save_directory,
                    index=i,
                    arrays=arrays_ref,
                    keep_only_final_solution=keep_only_final_solution,
                    solver=solver,
                    device=device_copy,
                    applied_field=applied_field,
                    circulating_currents=circulating_currents,
                    field_units=field_units,
                    current_units=current_units,
                    iterations=iterations,
                    check_inversion=check_inversion,
                    log_level=log_level,
                )
            )

        paths = ray.get(result_ids)

        solutions = None
        if return_solutions:
            # Load solutions from disk.
            # Set arrays with the original device arrays,
            # not the ones in shared memory.
            arrays = device.get_arrays(copy_arrays=False, dense=False)
            solutions = []
            for path in paths:
                if keep_only_final_solution:
                    solution = Solution.from_file(path)
                    solution.device.set_arrays(arrays)
                    solutions.append(solution)
                else:
                    solutions.append(load_solutions(path))
                    for solution in solutions[-1]:
                        solution.device.set_arrays(arrays)

    t1 = time.time()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models in parallel using ray with {nproc} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    if initialized_ray:
        ray.shutdown()

    if directory is None:
        paths = None

    del arrays
    del arrays_ref
    del result_ids
    del models

    gc.collect()

    return solutions, paths
