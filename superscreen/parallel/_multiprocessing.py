import multiprocessing as mp
import os
import tempfile
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pint
import scipy.sparse as sp

from ..device import Device
from ..device.mesh import Mesh
from ..parameter import Parameter
from ..solution import Solution
from ..solver import solve
from . import utils

logger = utils.logger


#######################################################################################
# Concurrency using multiprocessing
#######################################################################################

# See: http://thousandfold.net/cz/2014/05/01/sharing-numpy-arrays-between-processes-using-multiprocessing-and-ctypes/
# See: https://stackoverflow.com/questions/37705974/why-are-multiprocessing-sharedctypes-assignments-so-slow


def shared_array_to_numpy(
    shared_array: mp.RawArray, shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    """Convert a shared RawArray to a numpy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        array = np.ctypeslib.as_array(shared_array)
        if shape is not None:
            array = array.reshape(shape)
    return array


def numpy_to_shared_array(array: np.ndarray) -> mp.RawArray:
    """Convert a numpy array to a shared RawArray."""
    dtype = np.ctypeslib.as_ctypes_type(array.dtype)
    shared_array = mp.RawArray(dtype, array.size)
    sh_np = np.ctypeslib.as_array(shared_array).reshape(array.shape)
    np.copyto(sh_np, array)
    return shared_array


def shared_arrays_to_sparse(
    shared_arrays: Tuple[mp.RawArray, mp.RawArray, mp.RawArray],
    shape: Tuple[int, ...],
    fmt: str,
) -> sp.spmatrix:
    """Convert shared RawArrays to a sparse matrix."""
    numpy_arrays = tuple(shared_array_to_numpy(a) for a in shared_arrays)
    csr = sp.csr_matrix(tuple(numpy_arrays), shape=shape, copy=False)
    return csr.asformat(fmt, copy=False)


def sparse_to_shared_arrays(
    mat: sp.spmatrix,
) -> Tuple[Tuple[mp.RawArray, mp.RawArray, mp.RawArray], Tuple[int, ...]]:
    """Convert a sparse matrix to shared RawArrays."""
    csr = mat.asformat("csr", copy=False)
    data = numpy_to_shared_array(csr.data)
    indices = numpy_to_shared_array(csr.indices)
    indptr = numpy_to_shared_array(csr.indptr)
    return (data, indices, indptr), csr.shape


def container_to_shared(item):
    """Recursively convert a container of arrays and sparse matrices to shared arrays."""
    if isinstance(item, np.ndarray):
        return numpy_to_shared_array(item)
    if isinstance(item, sp.spmatrix):
        return sparse_to_shared_arrays(item)
    if isinstance(item, dict):
        return {key: container_to_shared(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(container_to_shared(value) for value in item)
    return item


def shared_to_container(item):
    """Recursively convert a container of shared arrays into numpy arrays
    and sparse matrices.
    """
    if isinstance(item, mp.RawArray):
        return shared_array_to_numpy(item)
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and len(item[0]) == 3
        and all(isinstance(i, mp.RawArray) for i in item[0])
        and isinstance(item[1], tuple)
    ):
        # It's a shared sparse matrix.
        return shared_arrays_to_sparse(item)
    if isinstance(item, dict):
        return {key: shared_to_container(value) for key, value in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(shared_to_container(value) for value in item)
    return item


def meshes_to_shared_arrays(
    meshes: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
) -> Dict[str, Tuple[mp.RawArray, Tuple[int, ...]]]:
    """Convert all arrays in the device to shared RawArrays."""
    meshes_as_dicts = {name: mesh.to_dict() for name, mesh in meshes.items()}
    return container_to_shared(meshes_as_dicts)


def meshes_from_shared_arrays(shared_meshes) -> Dict[str, Mesh]:
    """Reconstruct meshes from shared arrays."""
    return {name: Mesh.from_dict(d) for name, d in shared_meshes.items()}


def init(shared_meshes):
    global init_meshes
    init_meshes = shared_meshes


def solve_single_mp(kwargs: Dict[str, Any]) -> str:
    """Solve a single setup (multiprocessing)."""
    use_shared_memory = kwargs.pop("use_shared_memory")
    device: Device = kwargs["device"]
    if use_shared_memory:
        meshes = meshes_to_shared_arrays(init_meshes)
    else:
        meshes = init_meshes
    device.meshes = meshes
    kwargs["device"] = device
    _ = solve(**kwargs)


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
    vortices: Optional[Union[List[utils.Vortex], List[List[utils.Vortex]]]] = None,
    layer_updater: Optional[Callable] = None,
    layer_update_kwargs: Optional[List[Dict[str, Any]]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    product: bool = False,
    save_path: Optional[os.PathLike] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_kernels: bool = True,
    log_level: Optional[int] = None,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
) -> Tuple[Optional[List[Solution]], Optional[str]]:
    """Solve many models in parallel using multiprocessing."""
    models = utils.create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    t0 = time.perf_counter()

    meshes = {name: mesh.to_dict() for name, mesh in device.meshes.items()}

    if use_shared_memory:
        # Put the device's big arrays in shared memory
        shared_meshes = meshes_to_shared_arrays(meshes)
    else:
        shared_meshes = meshes
    if num_cpus is None:
        num_cpus = utils.cpu_count(logical=False)
    num_cpus = min(len(models), num_cpus)
    solver = f"superscreen.solve_many:multiprocessing:{num_cpus}"

    with tempfile.TemporaryDirectory() as savedir:
        kwargs = []
        for i, model in enumerate(models):
            device_copy, applied_field, circulating_currents, vortices = model
            kwargs.append(
                dict(
                    device=device_copy,
                    save_path=os.path.join(savedir, f"solutions-{i}.h5"),
                    return_solutions=return_solutions,
                    applied_field=applied_field,
                    circulating_currents=circulating_currents,
                    vortices=vortices,
                    field_units=field_units,
                    current_units=current_units,
                    iterations=iterations,
                    check_inversion=check_inversion,
                    log_level=log_level,
                    cache_kernels=cache_kernels,
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
            processes=num_cpus, initializer=init, initargs=(shared_meshes,)
        ) as pool:
            pool.map(solve_single_mp, kwargs)
            pool.close()
            pool.join()

        solutions = utils.load_solutions(
            directory=savedir,
            num_models=len(models),
            device=device,
            keep_only_final_solution=keep_only_final_solution,
        )
        if save_path is not None:
            Solution.save_solutions(solutions, save_path)
        if not return_solutions:
            solutions = None

    t1 = time.perf_counter()
    elapsed_seconds = t1 - t0
    seconds_per_model = elapsed_seconds / len(models)
    logger.info(
        f"Solved {len(models)} models in parallel using multiprocessing with {num_cpus} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    return solutions, save_path


utils.patch_docstring(solve_many_mp)
