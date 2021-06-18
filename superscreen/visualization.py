from typing import Optional, Union, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


from .brandt import BrandtSolution


def auto_grid(
    num_plots: int, max_cols: int = 3, figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """Creates a grid of at least ``num_plots`` subplots
    with at most ``max_cols`` columns.

    Args:
        num_plots: Total number of plots that will be populated.
        max_cols: Maximum number of columns in the grid.

    Returns:
        matplotlib figure and axes
    """
    ncols = min(max_cols, num_plots)
    nrows = int(np.ceil(num_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    return fig, axes


def plot_streams_layer(
    solution: BrandtSolution,
    layer: str,
    units: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma",
    levels: int = 101,
    colorbar: bool = True,
) -> Tuple[plt.Axes, Optional[Colorbar]]:
    """Plots the stream function for a single layer in a Device.

    Args:
        solution: The BrandtSolution from which to extract the stream function.
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
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    device = solution.device
    device_units = device.ureg(device.units).units
    x, y = device.points.T
    triangles = device.triangles
    units = units or solution.current_units
    stream = (solution.streams[layer] * device.ureg(solution.current_units)).to(units)

    im = ax.tricontourf(x, y, triangles, stream.magnitude, cmap=cmap, levels=levels)
    ax.set_xlabel(f"$x$ [${device_units:~L}$]")
    ax.set_ylabel(f"$y$ [${device_units:~L}$]")
    ax.set_title(layer)
    ax.set_aspect("equal")
    cbar = None
    if colorbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="8%", pad="4%")
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"$g$ [${device.ureg(units).units:~L}$]")
    return ax, cbar


def plot_streams(
    solution: BrandtSolution,
    layers: Optional[Union[List[str], str]] = None,
    units: Optional[str] = None,
    max_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "magma",
    levels: int = 101,
    colorbar: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the stream function for multiple layers in a Device.

    Args:
        solution: The BrandtSolution from which to extract stream functions.
        layers: Name(s) of layer(s) for which to plot the stream function.
            By default, the stream function is plotted for all layers in the Device.
        units: Units in which to plot the stream function. Defaults to
            solution.current_units.
        max_cols: Maximum number of columns in the grid of subplots.
        figsize: matplotlib figure size.
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
    fig, axes = auto_grid(len(layers), max_cols=max_cols, figsize=figsize)
    used_axes = []
    for layer, ax in zip(layers, fig.axes):
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
    solution: BrandtSolution,
    layers: Optional[Union[List[str], str]] = None,
    dataset: str = "fields",
    normalize: bool = False,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    max_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "cividis",
    colorbar: bool = True,
    share_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plots either the total field or the screening field for
    multiple layers in a Device.

    Args:
        solution: The BrandtSolution from which to extract fields.
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
        figsize: matplotlib figure size.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        share_color_scale: Whether to force all layers to use the same color scale.
        vmin: Color scale minimum to use for all layers
            (ignored if share_color_scale is True).
        vmax: Color scale maximum to use for all layers
            (ignored if share_color_scale is True).

    Returns:
        matplotlib figure and axes
    """
    if dataset not in ("fields", "screening_fields"):
        raise ValueError("Dataset must be 'fields' or 'screening_fields'.")
    device = solution.device
    device_units = device.ureg(device.units).units
    if layers is None:
        layers = list(device.layers)
    if isinstance(layers, str):
        layers = [layers]
    fig, axes = auto_grid(len(layers), max_cols=max_cols, figsize=figsize)
    used_axes = []
    xgrid, ygrid, fields = solution.grid_data(
        dataset=dataset,
        grid_shape=grid_shape,
        method=grid_method,
    )
    if dataset == "fields":
        clabel = "$H_z$"
    else:
        clabel = "$H_{sc}$"
    if normalize:
        for layer, field in fields.items():
            z0 = device.layers[layer].z0
            field /= solution.applied_field(xgrid, ygrid, z0)
        clabel = clabel + " / $H_{applied}$"
    else:
        units = units or solution.field_units
        new_fields = {}
        for layer, field in fields.items():
            new_field = (field * device.ureg(solution.field_units)).to(units)
            new_fields[layer] = new_field.magnitude
        fields = new_fields
        clabel = clabel + f" [${device.ureg(units).units:~L}$]"
    if share_color_scale:
        vmin = np.inf
        vmax = -np.inf
        for array in fields.values():
            vmin = min(vmin, np.nanmin(array))
            vmax = max(vmax, np.nanmax(array))
    clim_dict = {layer: (vmin, vmax) for layer in layers}
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        field = fields[layer]
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = cm.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(layer)
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${device_units:~L}$]")
        ax.set_ylabel(f"$y$ [${device_units:~L}$]")
        used_axes.append(ax)
        if colorbar:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="8%", pad="4%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    fig.tight_layout()
    return fig, axes


def plot_currents(
    solution: BrandtSolution,
    layers: Optional[Union[List[str], str]] = None,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    max_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    share_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    streamplot: bool = True,
    min_stream_amp: float = 0.05,
):
    """Plots the current density (sheet current) for each layer in a Device.

    Args:
        solution: The BrandtSolution from which to extract sheet current.
        layers: Name(s) of layer(s) for which to plot the sheet current.
            By default, the stream function is plotted for all layers in the Device.
        units: Units in which to plot the current density. Defaults to
            solution.current_units / solution.device.units.
        grid_shape: Shape of the desired rectangular grid. If a single integer n
            is given, then the grid will be square, shape = (n, n).
        grid_method: Interpolation method to use (see scipy.interpolate.griddata).
        max_cols: Maximum number of columns in the grid of subplots.
        figsize: matplotlib figure size.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        share_color_scale: Whether to force all layers to use the same color scale.
        vmin: Color scale minimum to use for all layers
            (ignored if share_color_scale is True).
        vmax: Color scale maximum to use for all layers
            (ignored if share_color_scale is True).
        streamplot: Whether to overlay current streamlines on the plot.
        min_stream_amp: Streamlines will not be drawn anywhere the
            current density is less than min_stream_amp * max(current_density).
            This avoids streamlines being drawn where there is no current flowing.

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    device_units = device.ureg(device.units).units
    if layers is None:
        layers = list(device.layers)
    if isinstance(layers, str):
        layers = [layers]
    fig, axes = auto_grid(len(layers), max_cols=max_cols, figsize=figsize)
    used_axes = []
    xgrid, ygrid, current_densities = solution.current_density(
        grid_shape=grid_shape,
        method=grid_method,
    )
    units = units or f"{solution.current_units} / {device.units}"
    jcs = {}
    for layer, jc in current_densities.items():
        old_units = f"{solution.current_units} / {device.units}"
        jc = (jc * device.ureg(old_units)).to(units)
        jcs[layer] = jc.magnitude
    clabel = "$|\\,\\vec{J}\\,|$" + f" [${device.ureg(units).units:~L}$]"
    if share_color_scale:
        vmin = np.inf
        vmax = -np.inf
        for array in jcs.values():
            vmin = min(vmin, np.nanmin(array))
            vmax = max(vmax, np.nanmax(array))
    clim_dict = {layer: (vmin, vmax) for layer in layers}
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        Jx, Jy = jcs[layer]
        J = np.sqrt(Jx ** 2 + Jy ** 2)
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = cm.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, J, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(layer)
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${device_units:~L}$]")
        ax.set_ylabel(f"$y$ [${device_units:~L}$]")
        used_axes.append(ax)
        if colorbar:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="8%", pad="4%")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
        if streamplot:
            if min_stream_amp is not None:
                cutoff = J.max() * min_stream_amp
                Jx[J < cutoff] = np.nan
                Jy[J < cutoff] = np.nan
            ax.streamplot(xgrid, ygrid, Jx, Jy, color="w", density=1.5, linewidth=0.75)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    fig.tight_layout()
    return fig, axes
