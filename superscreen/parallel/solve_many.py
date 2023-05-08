import logging
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numba
import pint

from ..device import Device
from ..parameter import Parameter
from ..solution import Solution, Vortex
from .utils import cpu_count

logger = logging.getLogger("solve")


def solve_many(
    device: Device,
    *,
    parallel_method: Optional[
        Literal[False, "serial", "multiprocessing", "mp", "ray"]
    ] = None,
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
    save_path: Optional[os.PathLike] = None,
    return_solutions: bool = False,
    keep_only_final_solution: bool = False,
    cache_kernels: bool = True,
    log_level: int = logging.INFO,
    use_shared_memory: bool = True,
    num_cpus: Optional[int] = None,
) -> Tuple[Optional[Union[List[Solution], List[List[Solution]]]], Optional[List[str]]]:
    """Solves many models involving the same device, optionally in parallel using
    multiple processes.

    Args:
        device: The Device to simulate.
        parallel_method: The method to use for multiprocessing (None, "mp", or "ray").
        applied_fields: A callable or list of callables that compute(s) the applied
            magnetic field as a function of x, y, z coordinates.
        circulating_currents: A dict of ``{hole_name: circulating_current}`` or list
            of such dicts. If circulating_current is a float, then it is assumed to
            be in units of current_units. If circulating_current is a string, then
            it is converted to a pint.Quantity.
        vortices: A list (list of lists) of ``Vortex`` objects.
        layer_updater: A callable with signature
            ``layer_updater(layer: Layer, **kwargs) -> Layer`` that updates
            parameter(s) of each layer in a device.
        layer_update_kwargs: A list of dicts of keyword arguments passed to
            ``layer_updater``.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        iterations: Number of times to compute the interactions between layers
        product: If True, then all combinations of applied_fields,
            circulating_currrents, and layer_update_kwargs are simulated (the
            behavior is given by itertools.product(), i.e. a nested for loop).
            Otherwise, the behavior is similar to zip().
            See superscreen.parallel.create_models for more details.
        directory: The directory in which to save the results. If None is given, then
            the results are not automatically saved to disk.
        return_solutions: Whether to return the Solution objects.
        keep_only_final_solution: Whether to keep/save only the Solution from
            the final iteration of superscreen.solve.solve for each setup.
        cache_memory_cutoff: If the memory needed for layer-to-layer kernel
            matrices exceeds ``cache_memory_cutoff`` times the current available
            system memory, then the kernel matrices will be cached to disk rather than
            in memory. Setting this value to ``inf`` disables caching to disk. In this
            case, the arrays will remain in memory unless they are swapped to disk
            by the operating system.
        log_level: Logging level to use, if any.
        use_shared_memory: Whether to use shared memory if parallel_method is not None.
        num_cpus: The number of processes to utilize.

    Returns:
        solutions, paths. If return_solutions is True, solutions is either a list of
        lists of Solutions (if keep_only_final_solution is False), or a list
        of Solutions (the final iteration for each setup). If directory is True,
        paths is a list of paths to the saved solutions, otherwise paths is None.
    """
    from ._multiprocessing import solve_many_mp
    from ._ray import solve_many_ray

    parallel_methods = {
        None: solve_many_mp,
        False: solve_many_mp,
        "serial": solve_many_mp,
        "multiprocessing": solve_many_mp,
        "mp": solve_many_mp,
        "ray": solve_many_ray,
    }

    if parallel_method not in parallel_methods:
        raise ValueError(
            f"Unknown parallel method, {parallel_method}. "
            f"Valid methods are {list(parallel_methods)}."
        )
    serial_methods = {None, False, "serial"}
    if num_cpus is not None and parallel_method in serial_methods:
        logger.warning(
            f"Ignoring num_cpus because parallel_method = {parallel_method!r}."
        )

    kwargs = dict(
        device=device,
        applied_fields=applied_fields,
        circulating_currents=circulating_currents,
        vortices=vortices,
        layer_updater=layer_updater,
        layer_update_kwargs=layer_update_kwargs,
        field_units=field_units,
        current_units=current_units,
        check_inversion=check_inversion,
        iterations=iterations,
        product=product,
        save_path=save_path,
        return_solutions=return_solutions,
        keep_only_final_solution=keep_only_final_solution,
        cache_kernels=cache_kernels,
        log_level=log_level,
        use_shared_memory=use_shared_memory,
        num_cpus=num_cpus,
    )

    solver = parallel_methods[parallel_method]
    # "serial" just uses multiprocessing with 1 CPU and no shared memory.
    if parallel_method in (None, False, "serial"):
        kwargs["num_cpus"] = 1
        kwargs["use_shared_memory"] = False

    # Limit the number of threads used by numba to
    # num_cpus_available // num_cpus_requested.
    numba_threads = numba.get_num_threads()
    cpus = cpu_count(logical=False)
    if solver in (solve_many_mp, solve_many_ray):
        if kwargs["num_cpus"] is None:
            numba.set_num_threads(1)
        elif kwargs["num_cpus"] > 1:
            numba.set_num_threads(max(1, cpus // num_cpus))

    solutions, path = solver(**kwargs)

    numba.set_num_threads(numba_threads)
    return solutions, path
