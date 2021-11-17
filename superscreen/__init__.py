from .about import version_dict, version_table
from .solve import convert_field, solve, solve_many
from .device import Layer, Polygon, Device
from .fluxoid import (
    make_fluxoid_polygons,
    find_fluxoid_solution,
    find_single_fluxoid_solution,
)
from .io import save_solutions, load_solutions, iload_solutions
from .parameter import Parameter, Constant
from .solution import Solution, Fluxoid, Vortex
from .version import __version__, __version_info__
from .visualization import (
    auto_grid,
    grids_to_vecs,
    cross_section,
    plot_streams_layer,
    plot_streams,
    plot_fields,
    plot_currents,
    plot_field_at_positions,
    plot_mutual_inductance,
    plot_polygon_flux,
)
from . import fem
from . import geometry
from . import parallel
from . import sources
