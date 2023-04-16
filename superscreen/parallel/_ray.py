import copy
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pint
import ray

from ..device import Device
from ..device.mesh import Mesh
from ..parameter import Parameter
from ..solution import Solution
from ..solver import solve
from . import utils

logger = utils.logger

#######################################################################################
# Concurrency using ray
#######################################################################################


@ray.remote
def solve_single_ray(*, meshes, **kwargs):
    """Solve a single setup (ray)."""
    kwargs["device"].meshes = {name: Mesh.from_dict(d) for name, d in meshes.items()}
    log_level = kwargs.pop("log_level", None)
    if log_level is not None:
        logging.basicConfig(level=log_level)
    _ = solve(**kwargs)


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
    """Solve many models in parallel using ray."""

    models = utils.create_models(
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
        num_cpus = utils.cpu_count(logical=False)
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

    if use_shared_memory:
        # Put the device's big arrays in shared memory.
        # The copy is necessary here so that the arrays do not get pinned in shared memory.
        meshes = {
            name: copy.deepcopy(mesh.to_dict()) for name, mesh in device.meshes.items()
        }
        meshes_ref = ray.put(meshes)
    else:
        meshes_ref = {name: mesh.to_dict() for name, mesh in device.meshes.items()}

    if use_shared_memory:
        shared_mem_str = "with shared memory"
    else:
        shared_mem_str = "without shared memory"
    logger.info(
        f"Solving {len(models)} models in parallel using ray with "
        f"{num_cpus} process(es) {shared_mem_str}."
    )

    result_ids = []
    with tempfile.TemporaryDirectory() as savedir:
        for i, model in enumerate(models):
            device_copy, applied_field, circulating_currents, vortices = model
            result_ids.append(
                solve_single_ray.remote(
                    meshes=meshes_ref,
                    device=device_copy,
                    save_path=os.path.join(savedir, f"solutions-{i}.h5"),
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
                    cache_kernels=cache_kernels,
                    _solver=solver,
                )
            )
        _ = ray.get(result_ids)
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
        f"Solved {len(models)} models in parallel using ray with {num_cpus} "
        f"process(es) in {elapsed_seconds:.3f} seconds "
        f"({seconds_per_model:.3f} seconds per model)."
    )

    if initialized_ray:
        ray.shutdown()

    return solutions, save_path


utils.patch_docstring(solve_many_ray)
