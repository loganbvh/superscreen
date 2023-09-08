import numpy as np

import superscreen as sc
from superscreen.geometry import box, close_curve

from .layers import hypres_squid_layers


def make_squid(with_terminals: bool = True, align_layers: str = "middle"):
    layers = hypres_squid_layers(align=align_layers)
    layer_mapping = {
        "fc": "BE",
        "fc_center": "BE",
        "fc_shield": "W1",
        "pl": "W1",
        "pl_center": "W1",
        "pl_shield": "W2",
        "pl_shield2": "BE",
    }

    with np.load("squids/hypres/hypres-250nm.npz") as f:
        polygons = dict(f)

    films = {
        name: polygons[name]
        for name in ["fc", "fc_shield", "pl", "pl_shield", "pl_shield2"]
    }
    holes = {
        "pl_center": np.array(
            [
                [0.2, -4.75],
                [0.2, 0.01],
                [-0.3, 0.01],
                [-0.3, -4.75],
                [0.2, -4.75],
            ]
        ),
    }
    films["pl"] = films["pl"][np.abs(films["pl"][:, 1]) > 0.05]
    fc = sc.Polygon(points=polygons["fc"])
    fc = fc.intersection(sc.Polygon(points=box(12)).rotate(30))
    fc_points = np.roll(fc.points, -1, axis=0)
    mask = np.ones(len(fc_points), dtype=bool)

    if not with_terminals:
        mask[4:27] = False
        holes["fc_center"] = close_curve(fc_points[~mask])
        holes["fc_center"][[0, -1], :] = (3.9, -3.92)
        holes["fc_center"][-2, :] = (4.55, -3.5)
    films["fc"] = close_curve(fc_points[mask])

    fc_shield = sc.Polygon(points=films["fc_shield"])
    fc_shield = fc_shield.intersection(sc.Polygon(points=box(15)).rotate(30))
    films["fc_shield"] = fc_shield.points

    films = {name: sc.Polygon(name, points=poly) for name, poly in films.items()}
    holes = {name: sc.Polygon(name, points=poly) for name, poly in holes.items()}

    terminals = None
    if with_terminals:
        terminals = {
            "fc": [
                sc.Polygon("source", points=box(2, 0.1))
                .rotate(30)
                .translate(dx=5.7, dy=-3.66),
                sc.Polygon("drain", points=box(2, 0.1))
                .rotate(30)
                .translate(dx=3.75, dy=-4.75),
            ],
        }

    for name, poly in films.items():
        poly.layer = layer_mapping[name]
        if with_terminals and name == "fc":
            poly.points = poly.resample(401)
        else:
            poly.points = poly.resample(201)
    for name, poly in holes.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(201)

    return sc.Device(
        "hypres_250nm",
        layers=layers,
        films=list(films.values()),
        holes=list(holes.values()),
        terminals=terminals,
        length_units="um",
    )
