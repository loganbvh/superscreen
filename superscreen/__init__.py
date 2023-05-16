from . import distance, fem, geometry, sources
from .about import version_dict, version_table
from .device import Device, Layer, Mesh, Polygon
from .fluxoid import find_fluxoid_solution, make_fluxoid_polygons
from .parameter import Constant, Parameter
from .solution import FilmSolution, Fluxoid, Solution, Vortex
from .solver import FactorizedModel, convert_field, factorize_model, solve
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
