import itertools
import logging
import os
from typing import Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import pint

from ..device import Device
from ..distance import cdist
from ..solution import FilmSolution, Solution, Vortex
from ..sources import ConstantField
from . import utils
from .solve_film import build_linear_systems, solve_film

logger = logging.getLogger("solve")


def solve(
    device: Device,
    *,
    applied_field: Optional[Callable] = None,
    terminal_currents: Optional[
        Dict[str, Dict[str, Union[float, str, pint.Quantity]]]
    ] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[List[Vortex]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    return_solutions: bool = True,
    save_path: Optional[os.PathLike] = None,
    cache_kernels: bool = True,
    log_level: int = logging.INFO,
    _solver: str = "superscreen.solve",
) -> List[Solution]:
    """Computes the stream functions and magnetic fields for all layers in a ``Device``.

    The simulation strategy is:

    1. Compute the stream functions and fields for each film given
    only the applied field.

    2. If iterations > 1 and there are multiple films, then for each film,
    calculate the screening field from all other films and recompute the
    stream function and fields based on the sum of the applied field
    and the screening fields from all other films.

    3. Repeat step 2 (iterations - 1) times.

    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        terminal_currents: A dict like ``{film_name: {source_name: source_current}}`` for
            each film and terminal.
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
        save_path: Path to an HDF5 file in which to save the results.
        cache_kernels: Cache the film-to-film kernels in memory. This is much
            faster than not caching, but uses more memory.
        log_level: Logging level to use, if any.
        _solver: Name of the solver method used.

    Returns:
        If ``return_solutions`` is True, returns a list of Solutions of
        length ``iterations + 1``.
    """

    if log_level is not None:
        logging.basicConfig(level=log_level)

    if not device.meshes:
        raise ValueError(
            "The device does not have a mesh. Call device.make_mesh() to generate it."
        )

    dtype = device.solve_dtype
    circulating_currents = circulating_currents or {}
    circulating_currents = utils.currents_to_floats(
        circulating_currents, device.ureg, current_units
    )
    terminal_currents = terminal_currents or {}
    terminal_currents = {
        film_name: utils.currents_to_floats(currents, device.ureg, current_units)
        for film_name, currents in terminal_currents.items()
    }
    for film_name, currents in terminal_currents.items():
        if sum(currents.values()):
            raise ValueError(
                f"Terminal currents in film {film_name!r} are not conserved."
            )
    applied_field = applied_field or ConstantField(0)
    vortices = vortices or []

    field_conversion = utils.field_conversion_factor(
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

    film_info = utils.make_film_info(
        device=device,
        vortices=vortices,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
        dtype=dtype,
    )

    film_systems, hole_systems = build_linear_systems(device, film_info)

    applied_fields = {}
    for film, mesh in device.meshes.items():
        layer = device.layers[film_info[film].layer]
        z0 = layer.z0 * np.ones(len(mesh.sites), dtype=dtype)
        # Units: current_units / device.length_units
        applied_fields[film] = (
            applied_field(mesh.sites[:, 0], mesh.sites[:, 1], z0)
            * field_conversion_magnitude
        ).astype(dtype, copy=False)

    # Vortex flux in magnetization-like units,
    # i.e. H * area as opposed to B * area = mu_0 * H * area.
    # ([current] / [length]) * [length]^2 = [current] * [length]
    vortex_flux = (
        device.ureg("Phi_0 / mu_0")
        .to(f"{current_units} * {device.length_units}")
        .magnitude
    )

    solutions: List[Solution] = []
    film_solutions: Dict[str, FilmSolution] = {}

    # Compute the stream functions and fields for each film
    # given only the applied field.
    for film_name in device.films:
        logger.info(f"Calculating {film_name!r} response to applied field.")
        film_solutions[film_name] = solve_film(
            device=device,
            applied_field=applied_fields[film_name],
            field_from_other_films=None,
            film_system=film_systems[film_name],
            hole_systems=hole_systems[film_name],
            film_info=film_info[film_name],
            field_conversion=field_conversion_magnitude,
            vortex_flux=vortex_flux,
            check_inversion=check_inversion,
        )

    solution = Solution(
        device=device,
        film_solutions=film_solutions,
        applied_field_func=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
        vortices=vortices,
        solver=_solver,
    )
    if save_path is not None:
        with h5py.File(save_path, "x") as h5file:
            # Save device in the root group. Solutions from all iterations
            # will reference the device with an h5py.SoftLink.
            device.to_hdf5(h5file.create_group("device"))
            solution.to_hdf5(h5file.create_group(str(0)), device_path="/device")
    if return_solutions:
        solutions.append(solution)
    else:
        del solution

    if len(device.films) < 2 or iterations < 1:
        if return_solutions:
            return solutions
        return

    film_sites = {
        name: device.meshes[name].sites.astype(dtype, copy=False)
        for name in device.films
    }
    kernel_cache = {}

    for i in range(iterations):
        # Calculate the screening fields at each layer from every other layer
        other_screening_fields = {
            name: np.zeros(len(mesh.sites), dtype=dtype)
            for name, mesh in device.meshes.items()
        }
        for other_film, film in itertools.product(device.films, repeat=2):
            if film == other_film:
                continue
            layer = device.layers[film_info[film].layer]
            other_layer = device.layers[film_info[other_film].layer]
            logger.debug(
                f"Calculating screening field at {film!r} "
                f"from {other_film!r} ({i+1}/{iterations})."
            )
            weights = film_info[other_film].weights
            g = film_solutions[other_film].stream
            key = (film, other_film)
            if key in kernel_cache:
                q = kernel_cache[key]
            elif key[::-1] in kernel_cache:
                q = kernel_cache[key[::-1]].T
            else:
                dz = layer.z0 - other_layer.z0
                rho2 = cdist(
                    film_sites[film], film_sites[other_film], metric="sqeuclidean"
                ).astype(dtype, copy=False)
                q = (2 * dz**2 - rho2) / (4 * np.pi * (dz**2 + rho2) ** (5 / 2))
            if (
                cache_kernels
                and key not in kernel_cache
                and key[::-1] not in kernel_cache
            ):
                kernel_cache[key] = q
            # Calculate the dipole kernel and integrate
            # Eqs. 1-2 in [Brandt], Eqs. 5-6 in [Kirtley1], Eqs. 5-6 in [Kirtley2].
            other_screening_fields[film] += q @ (weights[:, 0] * g)
            del q, g

        # Solve again with the screening fields from all films.
        # Calculate applied fields only once per iteration.
        film_solutions = {}
        for film_name, film in device.films.items():
            logger.info(
                f"Calculating {film_name!r} response to applied field and "
                f"screening field from other films ({i+1}/{iterations})."
            )
            film_solutions[film_name] = solve_film(
                device=device,
                applied_field=applied_fields[film_name],
                field_from_other_films=other_screening_fields[film_name],
                film_system=film_systems[film_name],
                hole_systems=hole_systems[film_name],
                film_info=film_info[film_name],
                field_conversion=field_conversion_magnitude,
                vortex_flux=vortex_flux,
                check_inversion=check_inversion,
            )

        solution = Solution(
            device=device,
            film_solutions=film_solutions,
            applied_field_func=applied_field,
            field_units=field_units,
            current_units=current_units,
            circulating_currents=circulating_currents,
            terminal_currents=terminal_currents,
            vortices=vortices,
            solver=_solver,
        )
        if save_path is not None:
            with h5py.File(save_path, "r+") as h5file:
                solution.to_hdf5(h5file.create_group(str(i + 1)), device_path="/device")
        if return_solutions:
            solutions.append(solution)
        else:
            del solution
    if return_solutions:
        return solutions
