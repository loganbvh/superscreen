import itertools
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import h5py
import numba
import numpy as np
import pint
from tqdm import tqdm

from ..device import Device
from ..solution import FilmSolution, Solution, Vortex
from ..sources import ConstantField
from .solve_film import (
    LinearSystem,
    TerminalSystems,
    factorize_linear_systems,
    solve_film,
)
from .utils import FilmInfo, currents_to_floats, field_conversion_factor, make_film_info

logger = logging.getLogger("solve")


@numba.njit(fastmath=True, parallel=True)
def biot_savart_film_to_film(
    *,
    film1_sites: np.ndarray,
    film1_z0: float,
    film1_areas: np.ndarray,
    film1_J: np.ndarray,
    film2_sites: np.ndarray,
    film2_z0: np.ndarray,
) -> np.ndarray:
    """Computes the Biot-Savart field at ``film2_sites`` due to sheet current density
    ``film1_J`` flowing in ``film1_sites``.

    Args:
        film1_sites: Mesh sites for the source film, shape ``(n, 2)``
        film1_z0: z-position of the source film
        film1_areas: Mesh vertex areas for the source film, shape ``(n, )``
        film1_J: Sheet current density in the source film, shape ``(n, 2)``
        film2_sites: Mesh sites for the target film, shape ``(m, 2)``
        film2_z0: z-position of the target film

    Returns:
        The Biot-Savart field at the target film in magnetization-like units, shape ``(m, )``
    """
    assert film1_sites.shape[1] == 2
    assert film2_sites.shape[1] == 2
    assert film1_J.shape[0] == film1_sites.shape[0]
    assert film1_areas.shape[0] == film1_sites.shape[0]
    assert film1_J.shape[1] == 2
    one_over_4pi = 1 / (4 * np.pi)
    minus_three_halves = -3.0 / 2.0
    out = np.empty(film2_sites.shape[0], dtype=film1_J.dtype)
    dz2 = (film2_z0 - film1_z0) ** 2
    for i in numba.prange(film2_sites.shape[0]):
        tmp = 0.0
        for j in range(film1_sites.shape[0]):
            dx = film2_sites[i, 0] - film1_sites[j, 0]
            dy = film2_sites[i, 1] - film1_sites[j, 1]
            tmp += (
                one_over_4pi
                * film1_areas[j]
                * (film1_J[j, 0] * dy - film1_J[j, 1] * dx)
                * (dx**2 + dy**2 + dz2) ** minus_three_halves
            )
        out[i] = tmp
    return out


@dataclass
class FactorizedModel:
    """A pre-factorized SuperScreen model.

    Args:
        device: The :class:`superscreen.Device`
        film_info: A dict of ``{film_name: FilmInfo}``
        film_systems: A dict of ``{film_name: LinearSystem}``
        hole_systems: A dict of ``{film_name: {hole_name: LinearSystem}}``
        terminal_systems: A dict of ``{film_name: TerminalSystems}``
        terminal_currents: A dict of ``{film_name: {terminal_name: terminal_current}}``
        circulating_currents: A dict of ``{hole_name: circulating_current}``
        vortices: A dict of ``{film_name: vortices}``
        current_units: str
    """

    device: Device
    film_info: Dict[str, FilmInfo]
    film_systems: Dict[str, LinearSystem]
    hole_systems: Dict[str, Dict[str, LinearSystem]]
    terminal_systems: Dict[str, TerminalSystems]
    terminal_currents: Dict[str, Dict[str, float]]
    circulating_currents: Dict[str, float]
    vortices: Dict[str, Sequence[Vortex]]
    current_units: str

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save a :class:`superscreen.FactorizedModel` to an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` in which to save the model
        """
        h5group.attrs["current_units"] = self.current_units
        self.device.to_hdf5(h5group.create_group("device"))
        film_info_grp = h5group.create_group("film_info")
        for film, info in self.film_info.items():
            info.to_hdf5(film_info_grp.create_group(film))
        film_systems_grp = h5group.create_group("film_systems")
        for film, system in self.film_systems.items():
            system.to_hdf5(film_systems_grp.create_group(film))
        hole_systems_grp = h5group.create_group("hole_systems")
        for film, holes in self.hole_systems.items():
            film_grp = hole_systems_grp.create_group(film)
            for hole, system in holes.items():
                system.to_hdf5(film_grp.create_group(hole))
        terminal_systems_grp = h5group.create_group("terminal_systems")
        for film, systems in self.terminal_systems.items():
            systems.to_hdf5(terminal_systems_grp.create_group(film))
        term_grp = h5group.create_group("terminal_currents")
        for film, terminals in self.terminal_currents.items():
            film_grp = term_grp.create_group(film)
            film_grp.attrs.update(terminals)
        circ_grp = h5group.create_group("circulating_currents")
        circ_grp.attrs.update(self.circulating_currents)
        vortex_grp = h5group.create_group("vortices")
        for i, vortex in enumerate(self.vortices):
            vortex.to_hdf5(vortex_grp.create_group(str(i)))

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "FactorizedModel":
        """Load a :class:`superscreen.FactorizedModel` from an :class:`h5py.Group`.

        Args:
            h5group: The :class:`h5py.Group` from which to load the model

        Returns:
            The loaded :class:`superscreen.FactorizedModel`
        """
        current_units = h5group.attrs["current_units"]
        device = Device.from_hdf5(h5group["device"])
        film_info = {
            film: FilmInfo.from_hdf5(grp) for film, grp in h5group["film_info"].items()
        }
        film_systems = {
            film: LinearSystem.from_hdf5(grp)
            for film, grp in h5group["film_systems"].items()
        }
        hole_systems = {}
        for film, grp in h5group["hole_systems"].items():
            hole_systems[film] = {
                hole: LinearSystem.from_hdf5(subgrp) for hole, subgrp in grp.items()
            }
        terminal_systems = {
            film: TerminalSystems.from_hdf5(grp)
            for film, grp in h5group["terminal_systems"].items()
        }
        terminal_currents = {
            film: dict(grp.attrs) for film, grp in h5group["terminal_currents"].items()
        }
        circulating_currents = dict(h5group["circulating_currents"].attrs)
        vortex_grp = h5group["vortices"]
        vortices = [
            Vortex.from_hdf5(vortex_grp[i]) for i in sorted(vortex_grp, key=int)
        ]
        return FactorizedModel(
            device=device,
            film_info=film_info,
            film_systems=film_systems,
            hole_systems=hole_systems,
            terminal_systems=terminal_systems,
            terminal_currents=terminal_currents,
            circulating_currents=circulating_currents,
            vortices=vortices,
            current_units=current_units,
        )

    def set_circulating_currents(self, circulating_currents: Dict[str, float]) -> None:
        """Set the circulating currents for the model.

        Args:
            circulating_currents: A dict of ``{hole_name: current}``, where ``current``
                is a float in units of ``self.current_units``.
        """
        diff = set(circulating_currents) - set(self.device.holes)
        if diff:
            raise KeyError(
                "circulating_currents contains keys not in"
                f" self.device.holes: {list(diff)!r}"
            )
        self.circulating_currents = circulating_currents.copy()
        holes_by_film = self.device.holes_by_film()
        for film_name, film_info in self.film_info.items():
            holes = [hole.name for hole in holes_by_film[film_name]]
            film_info.circulating_currents = {}
            for hole, current in self.circulating_currents.items():
                if hole in holes:
                    film_info.circulating_currents[hole] = current


def factorize_model(
    *,
    device: Device,
    current_units: str,
    terminal_currents: Optional[
        Dict[str, Dict[str, Union[float, str, pint.Quantity]]]
    ] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[Sequence[Vortex]] = None,
) -> FactorizedModel:
    """Prepares the applied field-indepdendent portions of the model, including
    factorizing the linear system describing each film and hole.

    Args:
        device: The :class:`superscreen.Device` to simulate.
        current_units: Units to use for current quantities. The applied field will be
            converted to units of [current_units / device.length_units].
        terminal_currents: A dict like ``{film_name: {source_name: source_current}}`` for
            each film and terminal.
        circulating_currents: A dict of ``{hole_name: circulating_current}``.
            If circulating_current is a float, then it is assumed to be in units
            of current_units. If circulating_current is a string, then it is
            converted to a pint.Quantity.
        vortices: A list of :class:`superscreen.Vortex` objects
            located in the :class:`superscreen.Device`.

    Returns:
        A :class:`superscreen.FactorizedModel` instance that can be used to solve the model.
    """
    ureg = device.ureg
    circulating_currents = circulating_currents or {}
    circulating_currents = currents_to_floats(circulating_currents, ureg, current_units)
    terminal_currents = terminal_currents or {}
    terminal_currents = {
        film_name: currents_to_floats(currents, ureg, current_units)
        for film_name, currents in terminal_currents.items()
    }
    for film_name, currents in terminal_currents.items():
        if sum(currents.values()):
            raise ValueError(
                f"Terminal currents in film {film_name!r} are not conserved."
            )
    vortices = vortices or []

    film_info = make_film_info(
        device=device,
        vortices=vortices,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
    )
    # Factorize linear systems for all films and holes.
    film_systems, hole_systems, terminal_systems = factorize_linear_systems(
        device, film_info
    )
    return FactorizedModel(
        device,
        film_info,
        film_systems,
        hole_systems,
        terminal_systems,
        terminal_currents,
        circulating_currents,
        vortices,
        current_units,
    )


def solve(
    device: Optional[Device] = None,
    *,
    model: Optional[FactorizedModel] = None,
    applied_field: Optional[Callable] = None,
    terminal_currents: Optional[
        Dict[str, Dict[str, Union[float, str, pint.Quantity]]]
    ] = None,
    circulating_currents: Optional[Dict[str, Union[float, str, pint.Quantity]]] = None,
    vortices: Optional[Sequence[Vortex]] = None,
    field_units: str = "mT",
    current_units: str = "uA",
    check_inversion: bool = False,
    iterations: int = 0,
    return_solutions: bool = True,
    save_path: Optional[os.PathLike] = None,
    log_level: Optional[int] = None,
    progress_bar: bool = True,
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
        device: The Device to simulate. Required if ``model`` is not provided.
        model: A :class:`superscreen.FactorizedModel` instance.
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
        log_level: Logging level to use, if any.
        progress_bar: Show a progress bar for self-consistent iterations.
        _solver: Name of the solver method used.

    Returns:
        If ``return_solutions`` is True, returns a list of Solutions of
        length ``iterations + 1``.
    """

    if log_level is not None:
        logging.basicConfig(level=log_level)

    if model is None:
        if device is None:
            raise ValueError("Either a model or a device must be provided.")
        logger.info("Factorizing model.")
        model = factorize_model(
            device=device,
            current_units=current_units,
            terminal_currents=terminal_currents,
            circulating_currents=circulating_currents,
            vortices=vortices,
        )
    else:
        if (
            device is not None
            or terminal_currents is not None
            or circulating_currents is not None
            or vortices is not None
        ):
            raise ValueError(
                "If model argument is provided, device, terminal_currents,"
                " circulating_currents, and vortices must be None."
            )
        if current_units is not None and current_units != model.current_units:
            logger.warning(
                "Keyword argument 'current_units' is ignored when "
                "a factorized model is provided. "
                f"Using model.current_units = {model.current_units!r}"
            )

    if not isinstance(model, FactorizedModel):
        raise TypeError(
            f"model must be an instance of FactorizedModel (got {type(model)})."
        )

    device = model.device
    film_info = model.film_info
    film_systems = model.film_systems
    hole_systems = model.hole_systems
    terminal_systems = model.terminal_systems
    terminal_currents = model.terminal_currents
    circulating_currents = model.circulating_currents
    vortices = model.vortices
    current_units = model.current_units

    if not device.meshes:
        raise ValueError(
            "The device does not have a mesh. Call device.make_mesh() to generate it."
        )

    dtype = device.solve_dtype
    ureg = device.ureg
    length_units = device.length_units
    meshes = device.meshes
    applied_field = applied_field or ConstantField(0)
    field_conversion = field_conversion_factor(
        field_units,
        current_units,
        length_units=length_units,
        ureg=ureg,
    )
    logger.debug(
        f"Conversion factor from {ureg(field_units).units:~P} to "
        f"{ureg(current_units) / ureg(length_units):~P}: {field_conversion:~P}."
    )

    applied_fields = {}
    for film, mesh in meshes.items():
        layer = device.layers[film_info[film].layer]
        z0 = layer.z0 * np.ones(len(mesh.sites))
        # Units: current_units / device.length_units
        Hz_applied = np.squeeze(
            applied_field(mesh.sites[:, 0], mesh.sites[:, 1], z0)
            * field_conversion.magnitude
        ).astype(dtype, copy=False)
        if Hz_applied.ndim != 1:
            raise ValueError(
                "Expected applied_field to return a 1D vector,"
                f" got a {Hz_applied.shape[1]}D vector."
            )
        applied_fields[film] = Hz_applied

    # Vortex flux in magnetization-like units,
    # i.e. H * area as opposed to B * area = mu_0 * H * area.
    # ([current] / [length]) * [length]^2 = [current] * [length]
    vortex_flux = ureg("Phi_0 / mu_0").to(f"{current_units} * {length_units}")
    vortex_flux = vortex_flux.magnitude

    solution_kwargs = dict(
        applied_field_func=applied_field,
        field_units=field_units,
        current_units=current_units,
        circulating_currents=circulating_currents,
        terminal_currents=terminal_currents,
        vortices=vortices,
        solver=_solver,
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
            field_conversion=field_conversion.magnitude,
            vortex_flux=vortex_flux,
            terminal_systems=terminal_systems.get(film_name, None),
            check_inversion=check_inversion,
        )

    solution = Solution(device=device, film_solutions=film_solutions, **solution_kwargs)
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

    for i in tqdm(
        range(iterations), desc="Solver iterations", disable=(not progress_bar)
    ):
        # Calculate the screening fields at each layer from every other layer
        other_screening_fields = {
            name: np.zeros(len(mesh.sites), dtype=dtype)
            for name, mesh in meshes.items()
        }
        for source_film, film in itertools.product(device.films, repeat=2):
            if film == source_film:
                continue
            layer = device.layers[film_info[film].layer]
            other_layer = device.layers[film_info[source_film].layer]
            logger.debug(
                f"Calculating screening field at {film!r} "
                f"from {source_film!r} ({i+1}/{iterations})."
            )
            other_screening_fields[film] += biot_savart_film_to_film(
                film1_sites=meshes[source_film].sites,
                film1_z0=other_layer.z0,
                film1_areas=film_info[source_film].weights,
                film1_J=film_solutions[source_film].current_density,
                film2_sites=meshes[film].sites,
                film2_z0=layer.z0,
            )

        # Solve again with the screening fields from all films.
        # Calculate applied fields only once per iteration.
        film_solutions = {}
        for film_name, film in device.films.items():
            logger.debug(
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
                field_conversion=field_conversion.magnitude,
                vortex_flux=vortex_flux,
                terminal_systems=terminal_systems.get(film_name, None),
                check_inversion=check_inversion,
            )

        solution = Solution(
            device=device, film_solutions=film_solutions, **solution_kwargs
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
