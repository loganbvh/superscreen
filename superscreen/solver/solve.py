import contextlib
import itertools
import logging
import os
import tempfile
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pint
import psutil
import scipy.sparse as sp
from scipy.spatial import distance

try:
    import jax
    import jax.numpy as jnp
except (ModuleNotFoundError, ImportError, RuntimeError):
    jax = None

from ..device import Device
from ..parameter import Constant
from ..solution import Solution, Vortex
from ..sources import ConstantField
from .solve_layer import solve_layer
from .utils import LambdaInfo, field_conversion_factor

logger = logging.getLogger("solve")


def solve(
    device: Device,
    *,
    applied_field: Optional[Callable] = None,
    terminal_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[List[Vortex]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    return_solutions: bool = True,
    directory: Optional[str] = None,
    cache_memory_cutoff: float = np.inf,
    log_level: int = logging.INFO,
    gpu: bool = False,
    _solver: str = "superscreen.solve",
) -> List[Solution]:
    """Computes the stream functions and magnetic fields for all layers in a ``Device``.

    The simulation strategy is:

    1. Compute the stream functions and fields for each layer given
    only the applied field.

    2. If iterations > 1 and there are multiple layers, then for each layer,
    calculate the screening field from all other layers and recompute the
    stream function and fields based on the sum of the applied field
    and the screening fields from all other layers.

    3. Repeat step 2 (iterations - 1) times.

    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        terminal_currents: A dict of ``{source_name: source_current}`` for
            each source terminal. This argument is only allowed if ``device``
            as an instance of ``TransportDevice``.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        vortices: A list of Vortex objects located in the Device.
        field_units: Units of the applied field. Can either be magnetic field H
            or magnetic flux density B = mu0 * H.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        check_inversion: Whether to verify the accuracy of the matrix inversion.
        iterations: Number of times to compute the interactions between layers.
        return_solutions: Whether to return a list of Solution objects.
        directory: If not None, resulting Solutions will be saved in this directory.
        cache_memory_cutoff: If the memory needed for layer-to-layer kernel
            matrices exceeds ``cache_memory_cutoff`` times the current available
            system memory, then the kernel matrices will be cached to disk rather than
            in memory. Setting this value to ``inf`` disables caching to disk. In this
            case, the arrays will remain in memory unless they are swapped to disk
            by the operating system.
        log_level: Logging level to use, if any.
        gpu: Solve on a GPU if available (requires JAX and CUDA).
        _solver: Name of the solver method used.

    Returns:
        If ``return_solutions`` is True, returns a list of Solutions of
        length ``iterations + 1``.
    """

    if log_level is not None:
        logging.basicConfig(level=log_level)

    if directory is not None:
        os.makedirs(directory, exist_ok=True)

    if device.points is None:
        raise ValueError(
            "The device does not have a mesh. Call device.make_mesh() to generate it."
        )
    if device.weights is None or device.Del2 is None:
        raise ValueError(
            "The device does not have a Laplace operator. "
            "Call device.compute_matrices() to calculate it."
        )
    if gpu:
        if jax is None:
            raise ValueError("Running solve(..., gpu=True) requires JAX.")
        jax_device = jax.devices()[0]
        logger.info(f"Using JAX with device {jax_device}.")
        if "cpu" in jax_device.device_kind:
            logger.warning("No GPU found. Using JAX on the CPU.")
            _solver = _solver + ":jax:cpu"
        else:
            _solver = _solver + ":jax:gpu"
        dtype = np.float32
    else:
        dtype = device.solve_dtype

    # Convert all circulating and terminal currents to floats.
    def current_to_float(value):
        if isinstance(value, str):
            value = device.ureg(value)
        if isinstance(value, pint.Quantity):
            value = value.to(current_units).magnitude
        return value

    circulating_currents = circulating_currents or {}
    _circ_currents = circulating_currents.copy()
    circulating_currents = {}
    for name, current in _circ_currents.items():
        circulating_currents[name] = current_to_float(current)
    terminal_currents = terminal_currents or {}
    _term_currents = terminal_currents.copy()
    terminal_currents = {}
    for name, current in _term_currents.items():
        terminal_currents[name] = current_to_float(current)

    points = device.points.astype(dtype, copy=False)
    weights = device.weights.astype(dtype, copy=False)
    Q = device.Q.astype(dtype, copy=False)
    if weights.ndim == 1:
        # Shape (n, ) --> (n, 1)
        weights = weights[:, np.newaxis]
    Del2 = device.Del2
    gradx = device.gradx
    grady = device.grady
    if sp.issparse(Del2):
        Del2 = Del2.toarray().astype(dtype, copy=False)
    if sp.issparse(gradx):
        gradx = gradx.toarray().astype(dtype, copy=False)
    if sp.issparse(grady):
        grady = grady.toarray().astype(dtype, copy=False)
    grad = np.stack([gradx, grady], axis=0)

    if gpu:
        weights = jax.device_put(weights)
        Del2 = jax.device_put(Del2)
        grad = jax.device_put(grad)
        Q = jax.device_put(Q)
        grad = jax.device_put(grad)

    solutions = []
    streams = {}
    currents = {}
    fields = {}
    screening_fields = {}
    applied_field = applied_field or ConstantField(0)
    vortices = vortices or []

    field_conversion = field_conversion_factor(
        field_units,
        current_units,
        length_units=device.length_units,
        ureg=device.ureg,
    )
    logger.debug(
        f"Conversion factor from {device.ureg(field_units).units:~P} to "
        f"{device.ureg(current_units) / device.ureg(device.length_units):~P}: "
        f"{field_conversion:~P}."
    )
    field_conversion_magnitude = field_conversion.magnitude
    # Only compute the applied field and Lambda once.
    layer_fields = {}
    layer_Lambdas = {}
    for name, layer in device.layers.items():
        # Units: current_units / device.length_units
        layer_field = (
            applied_field(device.points[:, 0], device.points[:, 1], layer.z0)
            * field_conversion_magnitude
        ).astype(dtype, copy=False)
        # Check and cache penetration depth
        london_lambda = layer.london_lambda
        d = layer.thickness
        Lambda = layer.Lambda
        if isinstance(london_lambda, (int, float)) and london_lambda <= d:
            length_units = device.ureg(device.length_units).units
            logger.warning(
                f"Layer '{name}': The film thickness, d = {d:.4f} {length_units:~P},"
                f" is greater than or equal to the London penetration depth, resulting"
                f" in an effective penetration depth {LambdaInfo.Lambda_str} = {Lambda:.4f}"
                f" {length_units:~P} <= {LambdaInfo.lambda_str} = {london_lambda:.4f}"
                f" {length_units:~P}. The assumption that the current density is nearly"
                f" constant over the thickness of the film may not be valid."
            )
        if isinstance(Lambda, (int, float)):
            Lambda = Constant(Lambda)
        Lambda = Lambda(device.points[:, 0], device.points[:, 1]).astype(
            dtype, copy=False
        )[:, np.newaxis]
        if london_lambda is not None:
            if isinstance(london_lambda, (int, float)):
                london_lambda = Constant(london_lambda)
            london_lambda = london_lambda(
                device.points[:, 0], device.points[:, 1]
            ).astype(dtype, copy=False)[:, np.newaxis]
        if gpu:
            layer_field = jax.device_put(layer_field)
            Lambda = jax.device_put(Lambda)
            if london_lambda is not None:
                london_lambda = jax.device_put(london_lambda)
        layer_fields[name] = layer_field
        layer_Lambdas[name] = LambdaInfo(
            layer=name,
            Lambda=Lambda,
            london_lambda=london_lambda,
            thickness=layer.thickness,
        )

    vortices_by_layer = defaultdict(list)
    for vortex in vortices:
        if not isinstance(vortex, Vortex):
            raise TypeError(f"Expected a Vortex, but got {type(vortex)}.")
        if vortex.layer not in device.layers:
            raise ValueError(f"Vortex located in unknown layer: {vortex}.")
        films_in_layer = [f for f in device.films.values() if f.layer == vortex.layer]
        holes_in_layer = [h for h in device.holes.values() if h.layer == vortex.layer]
        if not any(
            film.contains_points([vortex.x, vortex.y]) for film in films_in_layer
        ):
            raise ValueError(f"Vortex {vortex} is not located in a film.")
        if any(hole.contains_points([vortex.x, vortex.y]) for hole in holes_in_layer):
            raise ValueError(f"Vortex {vortex} is located in a hole.")
        vortices_by_layer[vortex.layer].append(vortex)

    # Compute the stream functions and fields for each layer
    # given only the applied field.
    for name, layer in device.layers.items():
        logger.info(f"Calculating {name} response to applied field.")
        g, J, total_field, screening_field = solve_layer(
            device=device,
            layer=name,
            applied_field=layer_fields[name],
            kernel=Q,
            terminal_currents=terminal_currents,
            circulating_currents=circulating_currents,
            vortices=vortices_by_layer[name],
            weights=weights,
            Del2=Del2,
            grad=grad,
            Lambda_info=layer_Lambdas[name],
            current_units=current_units,
            check_inversion=check_inversion,
            gpu=gpu,
        )
        # Units: current_units
        streams[name] = g.astype(dtype)
        # Units: currents_units / device.length_units
        currents[name] = J.astype(dtype, copy=False)
        # Units: current_units / device.length_units
        fields[name] = total_field.astype(dtype, copy=False)
        screening_fields[name] = screening_field.astype(dtype, copy=False)

    solution = Solution(
        device=device,
        streams={
            layer: np.asarray(stream, dtype=dtype) for layer, stream in streams.items()
        },
        current_densities=currents,
        fields={
            # Units: field_units
            layer: (field / field_conversion_magnitude).astype(dtype, copy=False)
            for layer, field in fields.items()
        },
        screening_fields={
            # Units: field_units
            layer: (field / field_conversion_magnitude).astype(dtype, copy=False)
            for layer, field in screening_fields.items()
        },
        applied_field=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
        vortices=vortices,
        solver=_solver,
    )
    if directory is not None:
        solution.to_file(os.path.join(directory, str(0)))
    if return_solutions:
        solutions.append(solution)
    else:
        del solution

    layer_names = list(device.layers)
    nlayers = len(layer_names)
    if nlayers < 2 or iterations < 1:
        if return_solutions:
            return solutions
        return
    rho2 = distance.cdist(points, points, metric="sqeuclidean").astype(
        dtype,
        copy=False,
    )
    # Cache kernel matrices.
    kernels = {}
    if cache_memory_cutoff is None:
        cache_memory_cutoff = np.inf
    if cache_memory_cutoff < 0:
        raise ValueError(
            f"Kernel cache memory cutoff must be greater than zero "
            f"(got {cache_memory_cutoff})."
        )
    nkernels = nlayers * (nlayers - 1) // 2
    total_bytes = nkernels * rho2.nbytes
    available_bytes = psutil.virtual_memory().available
    if total_bytes > cache_memory_cutoff * available_bytes:
        # Save kernel matrices to disk to avoid filling up memory.
        cache_kernels_to_disk = True
        context = tempfile.TemporaryDirectory()
    else:
        # Cache kernels in memory (much faster than saving to/loading from disk).
        cache_kernels_to_disk = False
        context = contextlib.nullcontext()
    if gpu:
        cache_kernels_to_disk = False
        rho2 = jax.device_put(rho2)
    with context as tempdir:
        for i in range(iterations):
            # Calculate the screening fields at each layer from every other layer
            zeros = jnp.zeros if gpu else np.zeros
            other_screening_fields = {
                name: zeros(points.shape[0], dtype=dtype) for name in layer_names
            }
            for layer, other_layer in itertools.product(device.layers_list, repeat=2):
                if layer.name == other_layer.name:
                    continue
                if (
                    i == 0
                    and layer.name == layer_names[0]
                    and other_layer.name == layer_names[1]
                ):
                    logger.info(
                        f"Caching {nkernels} layer-to-layer kernel(s) "
                        f"({total_bytes / 1024 ** 2:.0f} MB total) "
                        f"{'to disk' if cache_kernels_to_disk else 'in memory'}."
                    )
                logger.debug(
                    f"Calculating screening field at {layer.name} "
                    f"from {other_layer.name} ({i+1}/{iterations})."
                )
                g = streams[other_layer.name]
                dz = other_layer.z0 - layer.z0
                key = frozenset((layer.name, other_layer.name))
                # Get the cached kernel matrix,
                # or calculate it if it's not yet in the cache.
                q = kernels.get(key, None)
                if q is None:
                    q = (2 * dz**2 - rho2) / (4 * np.pi * (dz**2 + rho2) ** (5 / 2))
                    if cache_kernels_to_disk:
                        fname = os.path.join(tempdir, "_".join(key))
                        np.save(fname, q)
                        kernels[key] = f"{fname}.npy"
                    else:
                        kernels[key] = q
                elif isinstance(q, str):
                    # Kernel was cached to disk, so load it into memory.
                    q = np.load(q)
                # Calculate the dipole kernel and integrate
                # Eqs. 1-2 in [Brandt], Eqs. 5-6 in [Kirtley1], Eqs. 5-6 in [Kirtley2].
                screening_field = q @ (weights[:, 0] * g)
                other_screening_fields[layer.name] += screening_field
                del q, g
            # Solve again with the screening fields from all layers.
            # Calculate applied fields only once per iteration.
            new_layer_fields = {}
            for name, layer in device.layers.items():
                # Units: current_units / device.length_units
                new_layer_fields[name] = (
                    layer_fields[name] + other_screening_fields[name]
                )
            streams = {}
            fields = {}
            screening_fields = {}
            for name, layer in device.layers.items():
                logger.info(
                    f"Calculating {name} response to applied field and "
                    f"screening field from other layers ({i+1}/{iterations})."
                )
                g, J, total_field, screening_field = solve_layer(
                    device=device,
                    layer=name,
                    applied_field=new_layer_fields[name],
                    kernel=Q,
                    weights=weights,
                    Del2=Del2,
                    grad=grad,
                    Lambda_info=layer_Lambdas[name],
                    circulating_currents=circulating_currents,
                    vortices=vortices_by_layer[name],
                    current_units=current_units,
                    check_inversion=check_inversion,
                    gpu=gpu,
                )
                # Units: current_units
                streams[name] = g
                # Units: current_units / device.length_units
                currents[name] = J
                # Units: current_units / device.length_units
                fields[name] = total_field
                screening_fields[name] = screening_field

            solution = Solution(
                device=device,
                streams={
                    layer: np.asarray(g, dtype=dtype) for layer, g in streams.items()
                },
                current_densities=currents,
                fields={
                    # Units: field_units
                    layer: (np.asarray(field) / field_conversion_magnitude).astype(
                        dtype, copy=False
                    )
                    for layer, field in fields.items()
                },
                screening_fields={
                    # Units: field_units
                    layer: (np.asarray(field) / field_conversion_magnitude).astype(
                        dtype, copy=False
                    )
                    for layer, field in screening_fields.items()
                },
                applied_field=applied_field,
                field_units=field_units,
                current_units=current_units,
                circulating_currents=circulating_currents,
                terminal_currents=terminal_currents,
                vortices=vortices,
                solver=_solver,
            )
            if directory is not None:
                solution.to_file(os.path.join(directory, str(i + 1)))
            if return_solutions:
                solutions.append(solution)
            else:
                del solution
    if cache_kernels_to_disk:
        context.cleanup()
    if return_solutions:
        return solutions
