import contextlib
import itertools
import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pint
import ray

from .device import Device
from .parameter import Parameter
from .solution import Solution, Vortex
from .solve import solve

logger = logging.getLogger(__name__)


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
            d = device.copy(with_arrays=False)
            updated_layers = [layer_updater(layer, **kwargs) for layer in d.layers_list]
            d.layers_list = updated_layers
            devices.append(d)
    else:
        devices = [device.copy(with_arrays=False)]

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


def _load_solutions(
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


def _cleanup(directory: str, iterations: int) -> None:
    final = os.path.join(directory, str(iterations))
    for f in os.listdir(final):
        shutil.move(os.path.join(final, f), directory)
    for j in range(iterations + 1):
        shutil.rmtree(os.path.join(directory, str(j)))


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
    vortices: Optional[Union[List[Vortex], List[List[Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 1,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_memory_cutoff: float = np.inf,
    log_level: Optional[int] = None,
    gpu: bool = False,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
):
    """Solve many models in a single process."""

    solver = "superscreen.solve_many:serial:1"

    models = create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    arrays = device.get_arrays(copy_arrays=False, dense=True)

    t0 = time.perf_counter()

    logger.info(f"Solving {len(models)} models serially with 1 process.")

    solutions = None
    paths = []

    if directory is None:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext(directory)

    with save_context as savedir:
        for i, model in enumerate(models):
            device_copy, applied_field, circulating_currents, vortices = model
            path = os.path.join(savedir, str(i))
            device_copy.set_arrays(arrays)
            _ = solve(
                device=device_copy,
                applied_field=applied_field,
                circulating_currents=circulating_currents,
                vortices=vortices,
                field_units=field_units,
                current_units=current_units,
                iterations=iterations,
                check_inversion=check_inversion,
                log_level=log_level,
                return_solutions=False,
                cache_memory_cutoff=cache_memory_cutoff,
                directory=path,
                gpu=gpu,
                _solver=solver,
            )
            paths.append(path)
            if keep_only_final_solution:
                _cleanup(path, iterations)
        if return_solutions:
            solutions = _load_solutions(
                directory=savedir,
                num_models=len(models),
                iterations=iterations,
                device=device,
                keep_only_final_solution=keep_only_final_solution,
            )

    t1 = time.perf_counter()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models serially with 1 process in "
        f"{elapsed_seconds:.3f} seconds ({seconds_per_model:.3f} seconds per model)."
    )

    if directory is None:
        paths = None
        save_context.cleanup()

    return solutions, paths


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
    shared_arrays = {
        name: (numpy_to_shared_array(array), array.shape)
        for name, array in arrays.items()
    }
    return shared_arrays


def init(shared_arrays):
    global init_arrays
    init_arrays = shared_arrays


def solve_single_mp(kwargs: Dict[str, Any]) -> str:
    """Solve a single setup (multiprocessing)."""
    use_shared_memory = kwargs.pop("use_shared_memory")
    keep_only_final_solution = kwargs.pop("keep_only_final_solution")
    device = kwargs["device"]

    if use_shared_memory:
        # init_arrays is the dict of arrays stored in shared memory
        numpy_arrays = {}
        for name, data in init_arrays.items():
            shared_array, shape = data
            numpy_arrays[name] = shared_array_to_numpy(shared_array, shape)
    else:
        numpy_arrays = init_arrays

    device.set_arrays(numpy_arrays)
    kwargs["device"] = device

    _ = solve(**kwargs)

    if keep_only_final_solution:
        _cleanup(kwargs["directory"], kwargs["iterations"])

    return kwargs["directory"]


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
    vortices: Optional[Union[List[Vortex], List[List[Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_memory_cutoff: float = np.inf,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
) -> List[str]:
    """Solve many models in parallel using multiprocessing."""
    models = create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    t0 = time.perf_counter()

    arrays = device.get_arrays(copy_arrays=False, dense=True)

    if use_shared_memory:
        # Put the device's big arrays in shared memory
        shared_arrays = share_arrays(arrays)
    else:
        shared_arrays = arrays
    if num_cpus is None:
        num_cpus = cpu_count(logical=False)
    num_cpus = min(len(models), num_cpus)
    solver = f"superscreen.solve_many:multiprocessing:{num_cpus}"

    solutions = None
    paths = None

    if directory is None:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext(directory)

    with save_context as savedir:
        kwargs = []
        for i, model in enumerate(models):
            device_copy, applied_field, circulating_currents, vortices = model
            path = os.path.join(savedir, str(i))
            kwargs.append(
                dict(
                    device=device_copy,
                    directory=os.path.join(savedir, path),
                    return_solutions=return_solutions,
                    keep_only_final_solution=keep_only_final_solution,
                    applied_field=applied_field,
                    circulating_currents=circulating_currents,
                    vortices=vortices,
                    field_units=field_units,
                    current_units=current_units,
                    iterations=iterations,
                    check_inversion=check_inversion,
                    log_level=log_level,
                    cache_memory_cutoff=cache_memory_cutoff,
                    _solver=solver,
                    use_shared_memory=use_shared_memory,
                )
            )
        if use_shared_memory:
            shared_mem_str = "with shared memory"
        else:
            shared_mem_str = "without shared memory"
        logger.info(
            f"Solving {len(models)} models in parallel using multiprocessing with "
            f"{num_cpus} process(es) {shared_mem_str}."
        )
        with mp.Pool(
            processes=num_cpus, initializer=init, initargs=(shared_arrays,)
        ) as pool:
            paths = pool.map(solve_single_mp, kwargs)
            pool.close()
            pool.join()
        if return_solutions:
            solutions = _load_solutions(
                directory=savedir,
                num_models=len(models),
                iterations=iterations,
                device=device,
                keep_only_final_solution=keep_only_final_solution,
            )

    t1 = time.perf_counter()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models in parallel using multiprocessing with {num_cpus} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    if directory is None:
        paths = None
        save_context.cleanup()

    return solutions, paths


#######################################################################################
# Concurrency using ray
#######################################################################################


@ray.remote
def solve_single_ray(*, arrays, **kwargs):
    """Solve a single setup (ray)."""
    keep_only_final_solution = kwargs.pop("keep_only_final_solution")
    kwargs["device"].set_arrays(arrays)

    log_level = kwargs.pop("log_level", None)
    if log_level is not None:
        logging.basicConfig(level=log_level)

    _ = solve(**kwargs)

    if keep_only_final_solution:
        _cleanup(kwargs["directory"], kwargs["iterations"])

    return kwargs["directory"]


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
    vortices: Optional[Union[List[Vortex], List[List[Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    product: bool = False,
    directory: Optional[str] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_memory_cutoff: float = np.inf,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
):
    """Solve many models in parallel using ray."""

    models = create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    initialized_ray = False
    if num_cpus is None:
        num_cpus = cpu_count(logical=False)
    elif ray.is_initialized():
        logger.warning("Ignoring num_cpus because ray is already initialized.")
        num_cpus = int(ray.available_resources()["CPU"])
    if not ray.is_initialized():
        num_cpus = min(len(models), num_cpus)
        logger.info(f"Initializing ray with {num_cpus} process(es).")
        ray.init(num_cpus=num_cpus)
        initialized_ray = True

    ray_resources = ray.available_resources()
    logger.info(f"ray resources: {ray_resources}")

    solver = f"superscreen.solve_many:ray:{num_cpus}"

    t0 = time.perf_counter()

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
        f"{num_cpus} process(es) {shared_mem_str}."
    )

    solutions = None
    paths = None

    if directory is None:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext(directory)
        paths = []

    result_ids = []
    with save_context as savedir:
        for i, model in enumerate(models):
            device_copy, applied_field, circulating_currents, vortices = model
            path = os.path.join(savedir, str(i))
            result_ids.append(
                solve_single_ray.remote(
                    arrays=arrays_ref,
                    device=device_copy,
                    applied_field=applied_field,
                    circulating_currents=circulating_currents,
                    vortices=vortices,
                    field_units=field_units,
                    current_units=current_units,
                    iterations=iterations,
                    check_inversion=check_inversion,
                    log_level=log_level,
                    return_solutions=False,
                    keep_only_final_solution=keep_only_final_solution,
                    cache_memory_cutoff=cache_memory_cutoff,
                    directory=path,
                    _solver=solver,
                )
            )
        paths = ray.get(result_ids)
        if return_solutions:
            solutions = _load_solutions(
                directory=savedir,
                num_models=len(models),
                iterations=iterations,
                device=device,
                keep_only_final_solution=keep_only_final_solution,
            )

    t1 = time.perf_counter()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models in parallel using ray with {num_cpus} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    if initialized_ray:
        ray.shutdown()

    if directory is None:
        paths = None
        save_context.cleanup()

    return solutions, paths


# Set docstrings for functions in parallel.py based on solve_many.
def _patch_docstring(func):
    from .solve import solve_many

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


for func in (solve_many_serial, solve_many_mp, solve_many_ray):
    _patch_docstring(func)
