# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Tuple, List, Dict

import numpy as np
import scipy.ndimage
from scipy.interpolate import griddata
import matplotlib.colors as cm
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


from .brandt import convert_field, BrandtSolution


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

    Returns:
        A dict of ``{name: (vmin, vmax)}``
    """
    if (vmin is not None and vmax is None) or (vmax is not None and vmin is None):
        raise ValueError("If either vmin or max is provided, both must be provided.")
    if vmin is not None:
        return {name: (vmin, vmax) for name in dict_of_arrays}

    clims = {
        name: (np.nanmin(array), np.nanmax(array))
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
    solution: BrandtSolution,
    dataset: Optional[str] = None,
    layers: Optional[Union[List[str], str]] = None,
    xs: Optional[Union[float, List[float]]] = None,
    ys: Optional[Union[float, List[float]]] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    angle: Optional[float] = None,
) -> Tuple[
    Union[np.ndarray, List[np.ndarray]],
    np.ndarray,
    Union[Dict[str, Union[np.ndarray, List[np.ndarray]]], np.ndarray],
]:
    """Takes a cross-section of the specified dataset for each layer along the line
    specified by ``x`` or ``y`` after rotating counterclockwise about the center of
    the grid by ``angle``.

    Args:
        solution: The BrandtSolution from which to extract the data
        dataset: The dataset of which to take a cross-section. Required if
            existing grids are not provided, in which case dataset must
            be "streams", "fields", or "screening_fields".
        layers: Name(s) of the layer(s) for which to extract cross-sections.
        xs: x value(s) for a vertical cross-section(s) (required if ys is None).
        ys: y value(s) for a horizontal cross-section(s) (required if x is None).
        grid_shape: Shape of the desired rectangular grid for triangle-to-
            rectangle interpolation. Note that the shape of the output
            (cross-section) arrays depends upon ``angle``.
        angle: The angle by which to rotate the z-grids prior to taking
            a vertical or horizontal cross-section.

    Returns:
        cross-section coordinates [shape (m, 3) array or list thereof],
        cross-section axis [shape (m, ) array],
        dict of z values of cross-section(s) for each layer [shape (m, ) array
        or list thereof]
    """
    xgrid, ygrid, zgrids = solution.grid_data(
        dataset, grid_shape=grid_shape, layers=layers
    )
    slice_coords, slice_axis, cross_sections = image_cross_section(
        xgrid,
        ygrid,
        zgrids,
        xs=xs,
        ys=ys,
        angle=angle,
    )

    if xs is None:
        if len(xs) == 1:
            cross_sections = {
                name: crosses[0] for name, crosses in cross_sections.item()
            }
            slice_coords = slice_coords[0]
    else:
        if len(ys) == 1:
            cross_sections = {
                name: crosses[0] for name, crosses in cross_sections.item()
            }
            slice_coords = slice_coords[0]
    return slice_coords, slice_axis, cross_sections


def image_cross_section(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    zgrids: np.ndarray,
    xs: Optional[Union[float, List[float]]] = None,
    ys: Optional[Union[float, List[float]]] = None,
    angle: Optional[float] = None,
) -> Tuple[
    List[np.ndarray],
    np.ndarray,
    Union[Dict[str, List[np.ndarray]], List[np.ndarray]],
]:
    """Takes cross-section(s) of the given data along the line(s) specified by ``xs``
    or ``ys``, rotated counterclockwise about the center of the grid by ``angle``.

    Args:
        xgrid: xgrid to use. If not given, xgrid, ygrid, and zgrids
            will be calculcated using ``solution.grid_data()``.
        ygrid: ygrid to use. If not given, xgrid, ygrid, and zgrids
            will be calculcated using ``solution.grid_data()``.
        zgrids: Either a dict of zgrids to slice, or a single zgrid.
        xs: x value(s) for a vertical cross-section(s) (required if ys is None).
        ys: y value(s) for a horizontal cross-section(s) (required if x is None).
        angle: The angle by which to rotate the vertical or horizontal line
            about the center of the grid

    Returns:
        Cross-section x coordinates, cross-section y coordinates, and either
        a dict of cross-section z values for each grid in zgrids or a single
        array of zvalues (if zgrids was given as an array).
    """
    if xs is not None and ys is not None:
        raise ValueError("Can only take one cross section at a time.")
    if xs is None and ys is None:
        raise ValueError("Cross section axis not specified.")
    if xs is not None and isinstance(xs, (int, float)):
        xs = [xs]
    if ys is not None and isinstance(ys, (int, float)):
        ys = [ys]

    xvec, yvec = grids_to_vecs(xgrid, ygrid)

    if isinstance(zgrids, np.ndarray):
        # "None" is just a filler
        zgrids = {"None": zgrids}

    names = list(zgrids)

    if xs is None:
        slice_coords = [np.stack([xvec, y * np.ones_like(xvec)], axis=1) for y in ys]
    else:
        slice_coords = [np.stack([x * np.ones_like(yvec), yvec], axis=1) for x in xs]

    if angle:
        rotated_zgrids = {
            name: scipy.ndimage.rotate(grid, angle, cval=np.nan)
            for name, grid in zgrids.items()
        }
        z = zgrids[names[0]]
        rotated_z = rotated_zgrids[names[0]]
        x0 = xvec.mean()
        y0 = yvec.mean()
        tr = mtrans.Affine2D().rotate_deg_around(x0, y0, angle)
        try:
            plt.ioff()
            fig, ax = plt.subplots()
            _ = ax.pcolormesh(
                xvec, yvec, z, shading="auto", transform=(tr + ax.transData)
            )
            xvec = np.linspace(*ax.get_xlim(), rotated_z.shape[1])
            yvec = np.linspace(*ax.get_ylim(), rotated_z.shape[0])
            plt.close(fig)
        finally:
            plt.ion()
        zgrids = rotated_zgrids

    cross_sections = {name: [] for name in names}
    if xs is None:
        slice_axis = xvec
        for y in ys:
            i = np.argmin(np.abs(yvec - y))
            for name in names:
                cross_sections[name].append(zgrids[name][i, :])
        if angle:
            slice_coords = [
                tr.transform(np.stack([xvec, y * np.ones_like(xvec)], axis=1))
                for y in ys
            ]
    else:
        slice_axis = yvec
        for x in xs:
            j = np.argmin(np.abs(xvec - x))
            for name in names:
                cross_sections[name].append(zgrids[name][:, j])
        if angle:
            slice_coords = [
                tr.transform(np.stack([x * np.ones_like(yvec), yvec], axis=1))
                for x in xs
            ]
    if len(names) == 1 and names[0] == "None":
        cross_sections = cross_sections["None"]
    return slice_coords, slice_axis, cross_sections


def plot_streams_layer(
    solution: BrandtSolution,
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
        matplotlib axis and Colorbar if one was created (None otherwise)
    """
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.figure
    device = solution.device
    length_units = device.ureg(device.length_units).units
    x, y = device.points.T
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
    solution: BrandtSolution,
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
        solution: The BrandtSolution from which to extract stream functions.
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
    cmap: str = "cividis",
    colorbar: bool = True,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cross_section_xs: Optional[Union[float, List[float]]] = None,
    cross_section_ys: Optional[Union[float, List[float]]] = None,
    cross_section_angle: Optional[float] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots either the total field or the screening field for
    multiple layers in a Device.

    Additional keyword arguments are passed to plt.subplots().

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
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        cross_section_xs: x coordinate(s) for vertical cross-section(s)
        cross_section_ys: y coordinate(s) for horizontal cross_sections(s)
        cross_section_angle: Angle in degrees by which to rotate the cross-section
            lines above the center of the grid.

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
    used_axes = []
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
                magnitude=True,
            )
        clabel = clabel + f" [${units:~L}$]"
    clim_dict = setup_color_limits(
        fields,
        vmin=vmin,
        vmax=vmax,
        share_color_scale=share_color_scale,
        symmetric_color_scale=symmetric_color_scale,
    )
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        field = fields[layer]
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = cm.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{clabel.split('[')[0].strip()} ({layer})")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        used_axes.append(ax)
        if colorbar:
            if cross_section_xs is not None or cross_section_ys is not None:
                cbar_size = "5%"
                cbar_pad = "4%"
            else:
                cbar_size = "8%"
                cbar_pad = "4%"
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size=cbar_size, pad=cbar_pad)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
        if cross_section_xs is not None or cross_section_ys is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, axis, cross_sections = image_cross_section(
                xgrid,
                ygrid,
                field,
                xs=cross_section_xs,
                ys=cross_section_ys,
                angle=cross_section_angle,
            )
            xs = axis - axis.min()
            for coord, cross in zip(coords, cross_sections):
                ax.plot(*coord.T, ls="--", lw=2)
                cax.plot(xs, cross, lw=2)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(clabel)
            used_axes.append(cax)
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
    cmap: str = "inferno",
    colorbar: bool = True,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    streamplot: bool = True,
    min_stream_amp: float = 0.025,
    cross_section_xs: Optional[Union[float, List[float]]] = None,
    cross_section_ys: Optional[Union[float, List[float]]] = None,
    cross_section_angle: Optional[float] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the current density (sheet current) for each layer in a Device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The BrandtSolution from which to extract sheet current.
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
        cross_section_xs: x coordinate(s) for vertical cross-section(s)
        cross_section_ys: y coordinate(s) for horizontal cross_sections(s)
        cross_section_angle: Angle in degrees by which to rotate the cross-section
            lines above the center of the grid.

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
    used_axes = []
    xgrid, ygrid, current_densities = solution.current_density(
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
    )
    # Keep track of which axes are actually used,
    # and delete unused axes later
    used_axes = []
    for ax, layer in zip(fig.axes, layers):
        Jx, Jy = jcs[layer]
        J = Js[layer]
        layer_vmin, layer_vmax = clim_dict[layer]
        norm = cm.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, J, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{clabel.split('[')[0].strip()} ({layer})")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        used_axes.append(ax)
        if colorbar:
            if cross_section_xs is not None or cross_section_ys is not None:
                cbar_size = "5%"
                cbar_pad = "4%"
            else:
                cbar_size = "8%"
                cbar_pad = "4%"
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size=cbar_size, pad=cbar_pad)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(clabel)
            used_axes.append(cbar.ax)
        if cross_section_xs is not None or cross_section_ys is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, axis, cross_sections = image_cross_section(
                xgrid,
                ygrid,
                J,
                xs=cross_section_xs,
                ys=cross_section_ys,
                angle=cross_section_angle,
            )
            xs = axis - axis.min()
            for coord, cross in zip(coords, cross_sections):
                ax.plot(*coord.T, ls="--", lw=2)
                cax.plot(xs, cross, lw=2)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(clabel)
            used_axes.append(cax)
        if streamplot:
            if min_stream_amp is not None:
                cutoff = J.max() * min_stream_amp
                Jx[J < cutoff] = np.nan
                Jy[J < cutoff] = np.nan
            ax.streamplot(xgrid, ygrid, Jx, Jy, color="w", density=1, linewidth=0.75)
    for ax in fig.axes:
        if ax not in used_axes:
            fig.delaxes(ax)
    fig.tight_layout()
    return fig, axes


def plot_field_at_positions(
    solution: BrandtSolution,
    positions: np.ndarray,
    zs: Optional[Union[float, np.ndarray]] = None,
    vector: bool = False,
    units: Optional[str] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    cmap: str = "cividis",
    colorbar: bool = True,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cross_section_xs: Optional[Union[float, List[float]]] = None,
    cross_section_ys: Optional[Union[float, List[float]]] = None,
    cross_section_angle: Optional[float] = None,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plots the total field (either all three components or just the
    z component) at a given set of positions (x, y, z) outside of the device.

    Additional keyword arguments are passed to plt.subplots().

    Args:
        solution: The BrandtSolution from which to extract fields.
        positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array of (x, y, z)
            coordinates at which to calculcate the magnetic field. A single list like [x, y]
            or [x, y, z] is also allowed.
        zs: z coordinates at which to calculcate the field. If positions has shape (m, 3), then
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
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        cross_section_xs: x coordinate(s) for vertical cross-section(s)
        cross_section_ys: y coordinate(s) for horizontal cross_sections(s)
        cross_section_angle: Angle in degrees by which to rotate the cross-section
            lines above the center of the grid.

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
    fields = griddata(positions[:, :2], fields, (xgrid, ygrid), method=grid_method)
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
    )
    for ax, label in zip(fig.axes, clabels):
        field = fields_dict[label]
        layer_vmin, layer_vmax = clim_dict[label]
        norm = cm.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{label.split('[')[0].strip()}")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        if colorbar:
            if cross_section_xs is not None or cross_section_ys is not None:
                cbar_size = "5%"
                cbar_pad = "4%"
            else:
                cbar_size = "8%"
                cbar_pad = "4%"
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size=cbar_size, pad=cbar_pad)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label(label)
        if cross_section_xs is not None or cross_section_ys is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, axis, cross_sections = image_cross_section(
                xgrid,
                ygrid,
                field,
                xs=cross_section_xs,
                ys=cross_section_ys,
                angle=cross_section_angle,
            )
            cross_xs = axis - axis.min()
            for coord, cross in zip(coords, cross_sections):
                ax.plot(*coord.T, ls="--", lw=2)
                cax.plot(cross_xs, cross, lw=2)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(label)
    fig.tight_layout()
    return fig, axes
