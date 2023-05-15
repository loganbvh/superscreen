import os

import numpy as np

import superscreen as sc
from superscreen.geometry import box

from .layers import hypres_squid_layers


def make_polygons():
    coords = {}
    npz_path = os.path.join(os.path.dirname(__file__), "hypres-400nm.npz")
    with np.load(npz_path) as df:
        coords = dict(df)
    films = ["fc", "fc_shield", "pl", "pl_shield"]
    holes = ["pl_center", "fc_center"]
    sc_films = {name: sc.Polygon(name, points=coords[name]) for name in films}
    sc_holes = {name: sc.Polygon(name, points=coords[name]) for name in holes}
    return sc_films, sc_holes


def make_squid(with_terminals: bool = True, align_layers: str = "middle"):
    films, holes = make_polygons()
    layers = hypres_squid_layers(align=align_layers)
    layer_mapping = {
        "fc": "BE",
        "fc_center": "BE",
        "fc_shield": "W1",
        "pl": "W1",
        "pl_center": "W1",
        "pl_shield": "W2",
    }

    for name, poly in films.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(151)
    for name, poly in holes.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(151)

    terminals = None
    if with_terminals:
        fc = films.pop("fc")
        fc_center = holes.pop("fc_center")
        fc_mask = sc.Polygon(points=box(5)).rotate(45).translate(dx=6.5, dy=-5.5)
        fc = fc.difference(fc_mask, fc_center).resample(501).set_layer("BE")
        films["fc"] = fc
        source = (
            sc.Polygon("source", layer="BE", points=box(2, 0.1))
            .rotate(45)
            .translate(dx=5.5, dy=-2.95)
        )
        drain = (
            sc.Polygon("drain", layer="BE", points=box(2, 0.1))
            .rotate(45)
            .translate(dx=3.95, dy=-4.5)
        )
        terminals = {"fc": [source, drain]}

    return sc.Device(
        "hypres_400nm",
        layers=layers,
        films=list(films.values()),
        holes=list(holes.values()),
        terminals=terminals,
        length_units="um",
    )
