from typing import Callable, Optional, Dict, List

from .brandt import brandt_layers as _brandt_layers
from .brandt import BrandtSolution
from .device import Layer, Polygon, Device
from .parameter import Parameter, Constant

from . import sources
from . import visualization

from .version import __version__, __version_info__


def solve(
    *,
    device: Device,
    applied_field: Callable,
    circulating_currents: Optional[Dict[str, float]] = None,
    check_inversion: Optional[bool] = True,
    coupled: Optional[bool] = True,
    iterations: Optional[int] = 1,
) -> List[BrandtSolution]:
    """Computes the stream functions and magnetic fields for all layers in a Device.

    The simulation strategy is:

    1. Compute the stream functions and fields for each layer given
    only the applied field.

    2. If coupled is True and there are multiple layers, then for each layer,
    calculcate the response field from all other layers and recompute the
    stream function and fields based on the sum of the applied field
    and the responses from all other layers.

    3. If iterations > 1, then repeat step 2 (iterations - 1) times.
    The solution should converge in only a few iterations.


    Args:
        device: The Device to simulate.
        applied_field: A callable that computes the applied magnetic field
            as a function of x, y, z coordinates.
        circulating_currents: A dict of {hole_name: hole_current}. Default: {}.
        check_inversion: Whether to verify the accuracy of the matrix inversion.
            Default: True.
        coupled: Whether to account for the interactions between different layers
            (e.g. shielding). Default: True.
        iterations: Number of times to compute the interactions between layers
            (iterations is ignored if coupled is False). Default: 1.

    Returns:
        A list of BrandtSolutions of length 1 if coupled is False,
        or length (iterations + 1) if coupled is True.
    """
    return _brandt_layers(
        device=device,
        applied_field=applied_field,
        circulating_currents=circulating_currents,
        check_inversion=check_inversion,
        coupled=coupled,
        iterations=iterations,
    )
