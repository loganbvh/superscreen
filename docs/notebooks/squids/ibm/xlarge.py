"""IBM SQUID susceptometer, 3 um inner radius pickup loop."""

import numpy as np

import superscreen as sc
from superscreen.geometry import box

from .layers import ibm_squid_layers


def make_squid(
    with_terminals: bool = True,
    align_layers: str = "middle",
    d_I1: float = 0.4,
    d_I2: float = 0.4,
) -> sc.Device:
    interp_points = 301
    pl_length = 11.5
    ri_pl = 3.0
    ro_pl = 3.5
    ri_fc = 6.0
    ro_fc = 8.8

    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.circle(ri_pl),
    ).union(sc.geometry.box(0.314, pl_length, center=(0, -pl_length / 2 - 0.9 * ri_pl)))

    pl = sc.Polygon("pl", layer="W1", points=sc.geometry.circle(ro_pl)).union(
        np.array(
            [
                [+0.8, -2.7],
                [-0.8, -2.7],
                [-4.6, -15.0],
                [+4.6, -15.0],
            ]
        )
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+2.6, -6.3],
                [+1.3, -3.6],
                [-1.3, -3.6],
                [-2.6, -6.3],
                [-6.0, -16.0],
                [+6.0, -16.0],
            ]
        ),
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+4.5, -13.2],
                [-4.5, -13.2],
                [-5.3, -15.5],
                [+5.3, -15.5],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [8.5, -10.3],
                [4.15, -4.15],
                [3.55, -4.75],
                [7.75, -10.75],
            ]
        )
    )

    fc = sc.Polygon("fc", layer="BE", points=sc.geometry.circle(ro_fc)).union(
        np.array(
            [
                [12.0, -9.6],
                [7.5, -4.8],
                [4.2, -4.2],
                [3.2, -7.8],
                [6.0, -13.5],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [13.3, -10.2],
                [7.7, -4.8],
                [3.3, -8.1],
                [6.1, -15.0],
            ]
        ),
    )

    films = [fc_shield, pl_shield1, pl_shield2, pl]
    holes = [fc_center, pl_center]
    for polygon in films + holes:
        polygon.points = polygon.resample(interp_points)

    terminals = None
    if with_terminals:
        fc_mask = sc.Polygon(points=box(8, 2)).rotate(33).translate(dx=8.5, dy=-11)
        fc = fc.difference(fc_mask, fc_center).resample(1001)
        source = (
            sc.Polygon("source", layer="BE", points=box(3.5, 0.2))
            .rotate(33)
            .translate(dx=9.5, dy=-9.1)
        )
        drain = (
            sc.Polygon("drain", layer="BE", points=box(3.5, 0.2))
            .rotate(33)
            .translate(dx=6.25, dy=-11.25)
        )
        terminals = {"fc": [source, drain]}
        holes = [pl_center]

    films.insert(0, fc)

    return sc.Device(
        "ibm_3000nm",
        layers=ibm_squid_layers(
            align=align_layers,
            d_I1=d_I1,
            d_I2=d_I2,
        ),
        films=films,
        holes=holes,
        terminals=terminals,
        length_units="um",
    )
