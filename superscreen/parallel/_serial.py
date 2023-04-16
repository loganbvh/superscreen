import contextlib
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pint

from ..device import Device
from ..device.mesh import Mesh
from ..parameter import Parameter
from ..solver import solve
from . import utils

logger = utils.logger

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
    vortices: Optional[Union[List[utils.Vortex], List[List[utils.Vortex]]]] = None,
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
    # Unused arguments, kept to preserve the function signature.
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
):
    """Solve many models in a single process."""

    solver = "superscreen.solve_many:serial:1"

    models = utils.create_models(
        device,
        applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        product=product,
    )

    meshes = {name: mesh.to_dict() for name, mesh in device.meshes.items()}

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
            device_copy.meshes = {name: Mesh.from_dict(d) for name, d in meshes.items()}
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
                utils.cleanup(path, iterations)
        if return_solutions:
            solutions = utils.load_solutions(
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


utils.patch_docstring(solve_many_serial)
