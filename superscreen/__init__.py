from . import distance, fem, geometry, parallel, sources
from .about import version_dict, version_table
from .device import Device, Layer, Mesh, Polygon, TerminalSet
from .fluxoid import find_fluxoid_solution, make_fluxoid_polygons
from .parallel import solve_many
from .parameter import Constant, Parameter
from .solution import Fluxoid, Solution, Vortex
from .solver import convert_field, solve
from .units import ureg
from .version import __version__, __version_info__
from .visualization import (
    auto_grid,
    cross_section,
    grids_to_vecs,
    plot_currents,
    plot_field_at_positions,
    plot_fields,
    plot_mutual_inductance,
    plot_polygon_flux,
    plot_streams,
)
