from collections import defaultdict
import warnings
import itertools
from contextlib import contextmanager
from typing import Optional, Union, Tuple, List, Dict, Sequence

import pint
import numpy as np
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .solve import convert_field
from .solution import Solution


@contextmanager
def non_gui_backend():
    """A contextmanager that temporarily uses a non-GUI backend for matplotlib."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Matplotlib is currently using agg"
        )
        try:
            old_backend = mpl.get_backend()
            mpl.use("Agg")
            yield
        finally:
            mpl.use(old_backend)


def auto_range_iqr(
    data_array: np.ndarray,
    cutoff_percentile: Union[float, Tuple[float, float]] = 1,
) -> Tuple[float, float]:
    """Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75 and 25 percent of the distribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    Taken from `qcodes <https://github.com/QCoDeS/Qcodes/blob/
    6c8f7202f6b6fca4884bfc0f6e1e9a6564628d75/qcodes/utils/plotting.py#L28-L76>`_.

    Args:
        data_array: Array of arbitrary dimension containing the
            statistical data.
        cutoff_percentile: Percentile of data that may maximally be
            clipped on both sides of the distribution. If given a
            tuple (a, b) the percentile limits will be a and 100-b.

    Returns:
        vmin, vmax
    """
    if isinstance(cutoff_percentile, tuple):
        bottom, top = cutoff_percentile
    else:
        bottom = cutoff_percentile
        top = 100 - bottom
    z = data_array.flatten()
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    zrange = zmax - zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [bottom, 75, 25, top])
    iqr = q3 - q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    if zrange == 0.0 or iqr / zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5 * iqr, zmin)
        vmax = min(q3 + 1.5 * iqr, zmax)
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax


def auto_grid(
    num_plots: int,
    max_cols: int = 3,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Creates a grid of at least ``num_plots`` subplots
    with at most ``max_cols`` columns.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        num_plots: Total number of plots that will be populated.
        max_cols: Maximum number of columns in the grid.

    Returns:
        matplotlib figure and axes
    """
    ncols = min(max_cols, num_plots)
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    return fig, axes


def grids_to_vecs(
    xgrid: np.ndarray, ygrid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts coordinate vectors from 2D meshgrids.

    Args:
        xgrid: meshgrid of x coordinates
        ygrid: meshgrid of y coordinates

    Returns:
        vector of x coordinates, vector of y coordinates
    """
    return xgrid[0, :], ygrid[:, 0]


def setup_color_limits(
    dict_of_arrays: Dict[str, np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Set up color limits (vmin, vmax) for a dictionary of numpy arrays.

    Args:
        dict_of_arrays: Dict of ``{name: array}`` for which to compute color limits.
        vmin: If provided, this vmin will be used for all arrays. If vmin is not None,
            then vmax must also not be None.
        vmax: If provided, this vmax will be used for all arrays. If vmax is not None,
            then vmin must also not be None.
        share_color_scale: Whether to force all arrays to share the same color scale.
            This option is ignored if vmin and vmax are provided.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
            This option is ignored if vmin and vmax are provided.
        auto_range_cutoff: Cutoff percentile for ``auto_range_iqr``.

    Returns:
        A dict of ``{name: (vmin, vmax)}``
    """
    if (vmin is not None and vmax is None) or (vmax is not None and vmin is None):
        raise ValueError("If either vmin or max is provided, both must be provided.")
    if vmin is not None:
        return {name: (vmin, vmax) for name in dict_of_arrays}

    if auto_range_cutoff is None:
        clims = {
            name: (np.nanmin(array), np.nanmax(array))
            for name, array in dict_of_arrays.items()
        }
    else:
        clims = {
            name: auto_range_iqr(array, cutoff_percentile=auto_range_cutoff)
            for name, array in dict_of_arrays.items()
        }

    if share_color_scale:
        # All subplots share the same color scale
        global_vmin = np.inf
        global_vmax = -np.inf
        for vmin, vmax in clims.values():
            global_vmin = min(vmin, global_vmin)
            global_vmax = max(vmax, global_vmax)
        clims = {name: (global_vmin, global_vmax) for name in dict_of_arrays}

    if symmetric_color_scale:
        # Set vmin = -vmax
        new_clims = {}
        for name, (vmin, vmax) in clims.items():
            new_vmax = max(vmax, -vmin)
            new_clims[name] = (-new_vmax, new_vmax)
        clims = new_clims

    return clims


def cross_section(
    dataset_coords: np.ndarray,
    dataset_values: np.ndarray,
    cross_section_coords: Union[np.ndarray, Sequence[np.ndarray]],
    interp_method: str = "linear",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Takes a cross-section of the specified dataset values along
    a path given by the given dataset coordinates.

    Args:
        dataset_coords: A shape (n, 2) array of (x, y) coordinates for the dataset.
        dataset_values: A shape (n, ) array of dataset values of which
            to take a cross-section.
        cross_section_coords: A shape (m, 2) array of (x, y) coordinates specifying
            the cross-section path (or a list of such arrays for multiple
            cross sections).
        interp_method: The interpolation method to use: "nearest", "linear", "cubic".

    Returns:
        A list of coordinate arrays, a list of curvilinear coordinate (path) arrays,
        and a list of cross section values.
    """
    valid_methods = ("nearest", "linear", "cubic")
    if interp_method not in valid_methods:
        raise ValueError(
            f"Interpolation method must be one of {valid_methods} "
            f"(got {interp_method})."
        )
    if interp_method == "nearest":
        interpolator = interpolate.NearestNDInterpolator
    elif interp_method == "linear":
        interpolator = interpolate.LinearNDInterpolator
    else:  # "cubic"
        interpolator = interpolate.CloughTocher2DInterpolator

    if not (isinstance(cross_section_coords, Sequence)):
        cross_section_coords = [cross_section_coords]
    cross_section_coords = [np.asarray(c) for c in cross_section_coords]
    for i, arr in enumerate(cross_section_coords):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Invalid shape for coordinate array {i}: {arr.shape}. "
                f"Coordinate arrays must have shape (n, 2)."
            )
    # Calculcate curvilinear cross section coordinates
    paths = []
    for c in cross_section_coords:
        path = np.cumsum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
        paths.append(np.concatenate([[0], path], axis=0))
    # Calculate cross sections.
    cross_sections = []
    mask = np.isfinite(dataset_values)
    z_interp = interpolator(dataset_coords[mask], dataset_values[mask])
    for c in cross_section_coords:
        cross_sections.append(z_interp(c[:, 0], c[:, 1]))

    return cross_section_coords, paths, cross_sections


def plot_streams_layer(
    solution: Solution,
    layer: str,
    units: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma",
    levels: int = 101,
    colorbar: bool = True,
    **kwargs,
) -> Tuple[plt.Axes, Optional[Colorbar]]:
    """Plots the stream function for a single layer in a Device.

    Additional keyword arguments are passed to plt.subplots() if ax is None.

    Args:
        solution: The Solution from which to extract the stream function.
        layer: Name of the layer in solution.device.layers for which to plot
            the stream function.
        units: Units in which to plot the stream function. Defaults to
            solution.current_units.
        ax: matplotlib axis on which to plot the data. If None is provided,
            a new figure is created.
        cmap: Name of the matplotlib colormap to use.
        levels: Number of contour levels to used.
        colorbar: Whether to add a colorbar to the plot.

    Returns:
        matplotlib axis and Colorbar if one was created (None otherwise)
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.figure
    device = solution.device
    length_units = device.ureg(device.length_units).units
    x = device.points[:, 0]
    y = device.points[:, 1]
    triangles = device.triangles
    units = units or solution.current_units
    if isinstance(units, str):
        units = device.ureg(units).units
    stream = (solution.streams[layer] * device.ureg(solution.current_units)).to(units)

    im = ax.tricontourf(x, y, triangles, stream.magnitude, cmap=cmap, levels=levels)
    ax.set_xlabel(f"$x$ [${length_units:~L}$]")
    ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    ax.set_title(f"$g$ ({layer})")
    ax.set_aspect("equal")
    cbar = None
    if colorbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="8%", pad="4%")
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"$g$ [${units:~L}$]")
    return ax, cbar


def plot_streams(
    solution: Solution,
    layers: Optional[Union[List[str], str]] = None,
    units: Optional[str] = None,
    max_cols: int = 3,
    cmap: str = "magma",
    levels: int = 101,
    colorbar: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the stream function for multiple layers in a Device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The Solution from which to extract stream functions.
        layers: Name(s) of layer(s) for which to plot the stream function.
            By default, the stream function is plotted for all layers in the Device.
        units: Units in which to plot the stream function. Defaults to
            solution.current_units.
        max_cols: Maximum number of columns in the grid of subplots.
        cmap: Name of the matplotlib colormap to use.
        levels: Number of contour levels to used.
        colorbar: Whether to add a colorbar to each subplot.

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    if layers is None:
        layers = list(device.layers)
    if isinstance(layers, str):
        layers = [layers]
    fig, axes = auto_grid(len(layers), max_cols=max_cols, **kwargs)
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        ax, cbar = plot_streams_layer(
            solution,
            layer,
            units=units,
            ax=ax,
            cmap=cmap,
            levels=levels,
            colorbar=colorbar,
        )
        used_axes.append(ax)
        if cbar is not None:
            used_axes.append(cbar.ax)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    fig.tight_layout()
    return fig, axes


def plot_fields(
    solution: Solution,
    layers: Optional[Union[List[str], str]] = None,
    dataset: str = "fields",
    normalize: bool = False,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    max_cols: int = 3,
    cmap: str = "cividis",
    colorbar: bool = True,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cross_section_coords: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots either the total field or the screening field for
    multiple layers in a Device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The Solution from which to extract fields.
        layers: Name(s) of layer(s) for which to plot the fields.
            By default, the stream function is plotted for all layers in the Device.
        dataset: Which set of fields to plot, either "fields" or "screening_fields".
        normalize: Whether to normalize the fields by the applied field.
        units: Units in which to plot the fields. Defaults to solution.field_units.
            This argument is ignored if normalize is True.
        grid_shape: Shape of the desired rectangular grid. If a single integer n
            is given, then the grid will be square, shape = (n, n).
        grid_method: Interpolation method to use (see scipy.interpolate.griddata).
        max_cols: Maximum number of columns in the grid of subplots.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        auto_range_cutoff: Cutoff percentile for ``auto_range_iqr``.
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        cross_section_coords: Shape (m, 2) array of (x, y) coordinates for a
            cross-section (or a list of such arrays).

    Returns:
        matplotlib figure and axes
    """
    if dataset not in ("fields", "screening_fields"):
        raise ValueError("Dataset must be 'fields' or 'screening_fields'.")

    device = solution.device
    # Length units from the Device
    length_units = device.ureg(device.length_units).units
    # The units the fields are currently in
    old_units = device.ureg(solution.field_units).units
    # The units we want to convert to
    if units is None:
        units = old_units
    if isinstance(units, str):
        units = device.ureg(units).units

    if layers is None:
        layers = list(device.layers)
    if isinstance(layers, str):
        layers = [layers]

    fig, axes = auto_grid(len(layers), max_cols=max_cols, **kwargs)
    xgrid, ygrid, fields = solution.grid_data(
        dataset=dataset,
        grid_shape=grid_shape,
        method=grid_method,
        layers=layers,
    )
    if dataset == "fields":
        clabel = "$H_z$"
    else:
        clabel = "$H_{sc}$"
    if "[mass]" in units.dimensionality:
        # We want flux density, B = mu0 * H
        clabel = "$\\mu_0$" + clabel
    if normalize:
        for layer, field in fields.items():
            z0 = device.layers[layer].z0
            field /= solution.applied_field(xgrid, ygrid, z0)
        clabel = clabel + " / $H_{applied}$"
    else:
        for layer in layers:
            fields[layer] = convert_field(
                fields[layer],
                units,
                old_units=old_units,
                ureg=device.ureg,
                with_units=False,
            )
        clabel = clabel + f" [${units:~L}$]"
    clim_dict = setup_color_limits(
        fields,
        vmin=vmin,
        vmax=vmax,
        share_color_scale=share_color_scale,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        field = fields[layer]
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = mpl.colors.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{clabel.split('[')[0].strip()} ({layer})")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        used_axes.append(ax)
        if cross_section_coords is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, paths, cross_sections = cross_section(
                np.stack([xgrid.ravel(), ygrid.ravel()], axis=1),
                field.ravel(),
                cross_section_coords=cross_section_coords,
            )
            for i, (coord, path, cross) in enumerate(
                zip(coords, paths, cross_sections)
            ):
                color = f"C{i % 10}"
                ax.plot(*coord.T, "--", color=color, lw=2)
                ax.plot(*coord[0], "o", color=color)
                ax.plot(*coord[-1], "s", color=color)
                cax.plot(path, cross, color=color, lw=2)
                cax.plot(path[0], cross[0], "o", color=color)
                cax.plot(path[-1], cross[-1], "s", color=color)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(clabel)
            used_axes.append(cax)
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    return fig, axes


def plot_currents(
    solution: Solution,
    layers: Optional[Union[List[str], str]] = None,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    max_cols: int = 3,
    cmap: str = "inferno",
    colorbar: bool = True,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    streamplot: bool = True,
    min_stream_amp: float = 0.025,
    cross_section_coords: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the current density (sheet current) for each layer in a Device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The Solution from which to extract sheet current.
        layers: Name(s) of layer(s) for which to plot the sheet current.
            By default, the stream function is plotted for all layers in the Device.
        units: Units in which to plot the current density. Defaults to
            solution.current_units / solution.device.length_units.
        grid_shape: Shape of the desired rectangular grid. If a single integer n
            is given, then the grid will be square, shape = (n, n).
        grid_method: Interpolation method to use (see scipy.interpolate.griddata).
        max_cols: Maximum number of columns in the grid of subplots.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        auto_range_cutoff: Cutoff percentile for ``auto_range_iqr``.
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
            (ignored if share_color_scale is True).
        vmax: Color scale maximum to use for all layers
            (ignored if share_color_scale is True).
        streamplot: Whether to overlay current streamlines on the plot.
        min_stream_amp: Streamlines will not be drawn anywhere the
            current density is less than min_stream_amp * max(current_density).
            This avoids streamlines being drawn where there is no current flowing.
        cross_section_coords: Shape (m, 2) array of (x, y) coordinates for a
            cross-section (or a list of such arrays).

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    length_units = device.ureg(device.length_units).units
    old_units = device.ureg(f"{solution.current_units} / {device.length_units}").units
    units = units or old_units
    if isinstance(units, str):
        units = device.ureg(units).units
    if layers is None:
        layers = list(device.layers)
    if isinstance(layers, str):
        layers = [layers]
    fig, axes = auto_grid(len(layers), max_cols=max_cols, **kwargs)
    xgrid, ygrid, current_densities = solution.grid_current_density(
        grid_shape=grid_shape,
        method=grid_method,
        layers=layers,
    )
    jcs = {}
    Js = {}
    for layer, jc in current_densities.items():
        jc = jx, jy = (jc * old_units).to(units).magnitude
        jcs[layer] = jc
        Js[layer] = np.sqrt(jx ** 2 + jy ** 2)
    clabel = "$|\\,\\vec{J}\\,|$" + f" [${units:~L}$]"
    clim_dict = setup_color_limits(
        Js,
        vmin=vmin,
        vmax=vmax,
        share_color_scale=share_color_scale,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        Jx, Jy = jcs[layer]
        J = Js[layer]
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = mpl.colors.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, J, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{clabel.split('[')[0].strip()} ({layer})")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        used_axes.append(ax)
        if cross_section_coords is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, paths, cross_sections = cross_section(
                np.stack([xgrid.ravel(), ygrid.ravel()], axis=1),
                J.ravel(),
                cross_section_coords,
            )
            for i, (coord, path, cross) in enumerate(
                zip(coords, paths, cross_sections)
            ):
                color = f"C{i % 10}"
                ax.plot(*coord.T, "--", color=color, lw=2)
                ax.plot(*coord[0], "o", color=color)
                ax.plot(*coord[-1], "s", color=color)
                cax.plot(path, cross, color=color, lw=2)
                cax.plot(path[0], cross[0], "o", color=color)
                cax.plot(path[-1], cross[-1], "s", color=color)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(clabel)
            used_axes.append(cax)
        if streamplot:
            if min_stream_amp is not None:
                cutoff = np.nanmax(J) * min_stream_amp
                Jx[J < cutoff] = np.nan
                Jy[J < cutoff] = np.nan
            ax.streamplot(xgrid, ygrid, Jx, Jy, color="w", density=1, linewidth=0.75)
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    return fig, axes


def plot_field_at_positions(
    solution: Solution,
    positions: np.ndarray,
    zs: Optional[Union[float, np.ndarray]] = None,
    vector: bool = False,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    cmap: str = "cividis",
    colorbar: bool = True,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cross_section_coords: Optional[Union[float, List[float]]] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the total field (either all three components or just the
    z component) at a given set of positions (x, y, z) outside of the device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The Solution from which to extract fields.
        positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array of (x, y, z)
            coordinates at which to calculate the magnetic field. A single list like [x, y]
            or [x, y, z] is also allowed.
        zs: z coordinates at which to calculate the field. If positions has shape (m, 3), then
            this argument is not allowed. If zs is a scalar, then the fields are calculated in
            a plane parallel to the x-y plane. If zs is any array, then it must be same length
            as positions.
        vector: Whether to return the full vector magnetic field or just the z component.
        units: Units in which to plot the fields. Defaults to solution.field_units.
            This argument is ignored if normalize is True.
        grid_shape: Shape of the desired rectangular grid. If a single integer n
            is given, then the grid will be square, shape = (n, n).
        grid_method: Interpolation method to use (see scipy.interpolate.griddata).
        max_cols: Maximum number of columns in the grid of subplots.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        auto_range_cutoff: Cutoff percentile for ``auto_range_iqr``.
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        cross_section_coords: Shape (m, 2) array of (x, y) coordinates for a
            cross-section (or a list of such arrays).

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    # Length units from the Device
    length_units = device.ureg(device.length_units).units
    # The units the fields are currently in
    old_units = device.ureg(solution.field_units).units
    # The units we want to convert to
    if units is None:
        units = old_units
    if isinstance(units, str):
        units = device.ureg(units).units

    fields = solution.field_at_position(
        positions,
        zs=zs,
        vector=vector,
        units=units,
        with_units=False,
    )
    if fields.ndim == 1:
        fields = fields[:, np.newaxis]
    if vector:
        num_subplots = 3
    else:
        num_subplots = 1
    fig, axes = plt.subplots(1, num_subplots, **kwargs)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    x, y, *_ = positions.T
    xs = np.linspace(x.min(), x.max(), grid_shape[1])
    ys = np.linspace(y.min(), y.max(), grid_shape[0])
    xgrid, ygrid = np.meshgrid(xs, ys)
    # Shape grid_shape or (grid_shape + (3, ))
    fields = interpolate.griddata(
        positions[:, :2],
        fields,
        (xgrid, ygrid),
        method=grid_method,
    )
    clabels = [f"{label} [${units:~L}$]" for label in ["$H_x$ ", "$H_y$ ", "$H_z$ "]]
    if "[mass]" in units.dimensionality:
        # We want flux density, B = mu0 * H
        clabels = ["$\\mu_0$" + clabel for clabel in clabels]
    if not vector:
        clabels = clabels[-1:]
    fields_dict = {label: fields[:, :, i] for i, label in enumerate(clabels)}
    clim_dict = setup_color_limits(
        fields_dict,
        vmin=vmin,
        vmax=vmax,
        share_color_scale=share_color_scale,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )
    for ax, label in zip(fig.axes, clabels):
        field = fields_dict[label]
        layer_vmin, layer_vmax = clim_dict[label]
        norm = mpl.colors.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{label.split('[')[0].strip()}")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        if cross_section_coords is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, paths, cross_sections = cross_section(
                np.stack([xgrid.ravel(), ygrid.ravel()], axis=1),
                field.ravel(),
                cross_section_coords=cross_section_coords,
            )
            for i, (coord, path, cross) in enumerate(
                zip(coords, paths, cross_sections)
            ):
                color = f"C{i % 10}"
                ax.plot(*coord.T, "--", color=color, lw=2)
                ax.plot(*coord[0], "o", color=color)
                ax.plot(*coord[-1], "s", color=color)
                cax.plot(path, cross, color=color, lw=2)
                cax.plot(path[0], cross[0], "o", color=color)
                cax.plot(path[-1], cross[-1], "s", color=color)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(label)
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
            cbar.set_label(label)
    return fig, axes


def plot_mutual_inductance(
    M: Union[np.ndarray, List[np.ndarray]],
    diff: bool = False,
    iteration_offset: int = 0,
    absolute: bool = False,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    logy: bool = False,
    grid: bool = True,
    legend: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the convergence vs. iteration of a set of mutual inductance matrices,
    given by the output of :meth:`superscreen.Device.mutual_inductance_matrix`
    with ``all_iterations=True``.

    Args:
        M: A length ``m`` list of shape ``(n, n)`` mutual inductance matrices, or
            a shape ``(m, n, n)`` array representing the same.
        diff: If True, plots the difference in mutual inductance between subsequent
            iterations.
        iteration_offset: The first iteration (index in ``M``) to consider when
            calculating convergence.
        absolute: If True (and diff is True), plots the absolute change in mutual
            inductance vs. iteration, otherwise plots relative change.
        ax: Matplotlib Axes instance on which to plot.
        figsize: Matplotlib figure size to create if ``ax`` is None.
        logy: If True, sets the y axis scaling to logarithmic.
        grid: If True, turns on plot grid lines.
        legend: If True, adds a legend to the plot.
        kwargs: Passed to ``ax.plot()``.

    Returns:
        Matplotlib Figure and Axes.
    """
    if isinstance(M, list):
        for i, item in enumerate(M):
            is_quantity = isinstance(item, pint.Quantity) and isinstance(
                item.magnitude, np.ndarray
            )
            if not (
                is_quantity
                or isinstance(item, np.ndarray)
                and item.ndim == 2
                and item.shape[0] == item.shape[1]
            ):
                raise ValueError(
                    f"Element {i} of list M is not a square array: {item!r}."
                )
        M = np.stack(M, axis=0)
    if isinstance(M, pint.Quantity):
        units = f"${M.units:~L}$"
        M = M.magnitude
    else:
        units = "?"
    if not (isinstance(M, np.ndarray) and M.ndim == 3 and M.shape[1] == M.shape[2]):
        raise ValueError(f"Expected M to be a shape (m, n, n) array, but got {M!r}.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    n = M.shape[1]
    i0 = int(iteration_offset)
    iterations = np.arange(M.shape[0])
    plot_kwargs = kwargs.copy()
    for i, j in itertools.product(range(n), repeat=2):
        plot_kwargs["label"] = f"$M_{{{i}{j}}}$"
        if diff:
            xs = iterations[i0 + 1 :]
            ys = np.abs(np.diff(M[i0:, i, j]))
            if not absolute:
                ys = ys / np.abs(M[i0 + 1 :, i, j])
            ax.plot(xs, ys, **plot_kwargs)
        else:
            xs = iterations[i0:]
            ax.plot(xs, M[i0:, i, j], **plot_kwargs)
    if logy:
        ax.set_yscale("log")
    if grid:
        ax.grid(True)
    if legend:
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    if diff:
        ylabel = "$\\Delta M_{{ij, k}}$"
        if absolute:
            title = "$\\Delta M_{{ij, k}} = |M_{{ij, k}} - M_{{ij, k-1}}|$"
            ylabel = ylabel + f" [{units}]"
        else:
            title = (
                "$\\Delta M_{{ij, k}} = "
                "\\frac{{|M_{{ij, k}} - M_{{ij, k-1}}|}}{{|M_{{ij, k}}|}}$"
            )
        ax.set_title(title)
    else:
        ylabel = f"$M_{{ij, k}}$ [{units}]"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration, $k$")
    return fig, ax


def plot_polygon_flux(
    solutions: List[Solution],
    diff: bool = False,
    iteration_offset: int = 0,
    absolute: bool = False,
    units: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    logy: bool = False,
    grid: bool = True,
    legend: bool = True,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the convergence vs. iteration of the flux through all polygons in a Device,
    given by the output of :meth:`superscreen.solve()`

    Args:
        solutions: A list of ``Solutions``, one per solve iteration.
        diff: If True, plots the difference in flux between subsequent iterations.
        iteration_offset: The first iteration (index in ``solutions``) to consider when
            calculating convergence.
        absolute: If True (and diff is True), plots the absolute change in flux vs.
            iteration, otherwise plots relative change.
        units: The flux units to display if ``absolute`` is True.
        ax: Matplotlib Axes instance on which to plot.
        figsize: Matplotlib figure size to create if ``ax`` is None.
        logy: If True, sets the y axis scaling to logarithmic.
        grid: If True, turns on plot grid lines.
        legend: If True, adds a legend to the plot.
        kwargs: Passed to ``ax.plot()``.

    Returns:
        Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    i0 = int(iteration_offset)
    iterations = np.arange(len(solutions))
    plot_kwargs = kwargs.copy()
    device = solutions[0].device
    polygon_names = list(device.polygons)
    polygon_flux = defaultdict(list)
    for i, solution in enumerate(solutions):
        flux_dict = solution.polygon_flux(polygons=polygon_names, units=units)
        for name, flux in flux_dict.items():
            units = flux.units
            polygon_flux[name].append(flux.magnitude)

    for name, flux_vals in polygon_flux.items():
        plot_kwargs["label"] = name
        if diff:
            xs = iterations[i0 + 1 :]
            ys = np.abs(np.diff(flux_vals[i0:]))
            if not absolute:
                ys = ys / np.abs(flux_vals[i0 + 1 :])
            ax.plot(xs, ys, **plot_kwargs)
        else:
            xs = iterations[i0:]
            ax.plot(xs, flux_vals[i0:], **plot_kwargs)
    if logy:
        ax.set_yscale("log")
    if grid:
        ax.grid(True)
    if legend:
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    if diff:
        ylabel = "$\\Delta\\Phi_i$"
        if absolute:
            title = "$\\Delta\\Phi_i = |\\Phi_i - \\Phi_{i-1}|$"
            ylabel = ylabel + f" [${units:~L}$]"
        else:
            title = "$\\Delta\\Phi_i = " "\\frac{|\\Phi_i - \\Phi_{i-1}|}{|\\Phi_{i}|}$"
        ax.set_title(title)
    else:
        ylabel = f"$\\Phi_i$ [${units:~L}$]"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration, $i$")
    return fig, ax


def _patch_docstring(func):
    other_func = getattr(Solution, func.__name__)
    other_func.__doc__ = (
        other_func.__doc__
        + "\n\n"
        + "\n".join(
            [line for line in func.__doc__.split("\n    ") if "solution:" not in line]
        )
    )
    annotations = func.__annotations__.copy()
    _ = annotations.pop("solution", None)
    other_func.__annotations__.update(annotations)


for func in [plot_streams, plot_currents, plot_fields, plot_field_at_positions]:
    _patch_docstring(func)
