"""IBM SQUID susceptometer, 1 um inner radius pickup loop."""

import numpy as np

import superscreen as sc
from superscreen.geometry import box

from .layers import ibm_squid_layers


def make_squid(with_terminals: bool = True, align_layers="middle"):
    interp_points = 301
    pl_length = 4
    ri_pl = 1.0
    ro_pl = 1.5
    ri_fc = 2.5
    ro_fc = 3.5
    pl_center = sc.Polygon(
        "pl_center",
        layer="W1",
        points=sc.geometry.circle(ri_pl),
    ).union(sc.geometry.box(0.2, pl_length, center=(0, -pl_length / 2 - 0.9 * ri_pl)))

    pl = sc.Polygon("pl", layer="W1", points=sc.geometry.circle(ro_pl)).union(
        np.array(
            [
                [1.5, -5.7],
                [0.41, -1],
                [-0.41, -1],
                [-1.5, -5.7],
            ]
        )
    )

    pl_shield1 = sc.Polygon(
        "pl_shield1",
        layer="W2",
        points=np.array(
            [
                [+1.0, -2.8],
                [+0.6, -(ri_pl + 0.4)],
                [-0.6, -(ri_pl + 0.4)],
                [-1.0, -2.8],
                [-2.6, -6.4],
                [-2.75, -6.9],
                [+2.75, -6.9],
                [+2.6, -6.4],
            ]
        ),
    )

    pl_shield2 = sc.Polygon(
        "pl_shield2",
        layer="BE",
        points=np.array(
            [
                [+1.25, -(2.55 + ro_pl)],
                [-1.25, -(2.55 + ro_pl)],
                [-2.0, -6.2],
                [+2.0, -6.2],
            ]
        ),
    )

    fc_center = sc.Polygon(
        "fc_center", layer="BE", points=sc.geometry.circle(ri_fc)
    ).union(
        np.array(
            [
                [4.3, -4.2],
                [2.1, -1.0],
                [1.8, -1.6],
                [3.85, -4.55],
            ]
        )
    )

    fc = sc.Polygon("fc", layer="BE", points=sc.geometry.circle(ro_fc)).union(
        np.array(
            [
                [5.8, -3.9],
                [2.8, -0.9],
                [1.5, -2.3],
                [3.2, -6.0],
            ]
        )
    )

    fc_shield = sc.Polygon(
        "fc_shield",
        layer="W1",
        points=np.array(
            [
                [6.4, -4.05],
                [3.45, -1.4],
                [1.65, -3.3],
                [3.1, -6.8],
            ]
        ),
    )

    films = [fc_shield, pl_shield1, pl_shield2, pl]
    holes = [pl_center, fc_center]
    for polygon in films + holes:
        polygon.points = polygon.resample(interp_points)

    terminals = None
    if with_terminals:
        fc_mask = sc.Polygon(points=box(4, 1)).rotate(40).translate(dx=4.25, dy=-4.75)
        fc = fc.difference(fc_mask, fc_center).resample(1001)
        source = (
            sc.Polygon("source", layer="BE", points=box(1.5, 0.1))
            .rotate(40)
            .translate(dx=4.7, dy=-3.7)
        )
        drain = (
            sc.Polygon("drain", layer="BE", points=box(1.5, 0.1))
            .rotate(40)
            .translate(dx=3.3, dy=-4.9)
        )
        terminals = {"fc": [source, drain]}
        holes = [pl_center]

    films.insert(0, fc)

    return sc.Device(
        "ibm_1000nm",
        layers=ibm_squid_layers(align=align_layers),
        films=films,
        holes=holes,
        terminals=terminals,
        length_units="um",
    )
