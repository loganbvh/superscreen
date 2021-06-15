import os
import numpy as np
import scipy.io
from superscreen import Device, Layer, Polygon


def make_layout_small_susc():
    mat_path = os.path.join(os.path.dirname(__file__), "small_susc.mat")
    layout = scipy.io.loadmat(mat_path)
    origin = layout["origin"]
    pl = layout["pl"]
    pl_centers = layout["pl_centers"]
    pl_shield = layout["pl_shield"]
    pl_shield_2 = layout["pl_shield_2"]
    # A = layout['A']
    fc_in = layout["fc_in"]
    fc_out = layout["fc_out"]
    fc_shield = layout["fc_shield"]
    two_micron_scale = layout["two_micron_scale"]

    z0 = 0.0  # microns
    london_lambda = 0.080  # 80 nm London penetration depth for Nb films

    scale_factor = 2 / (two_micron_scale[1, 0] - two_micron_scale[1, 1])

    components = {}

    fc_in[:, 0] = (fc_in[:, 0] - origin[0, 0]) * scale_factor
    fc_in[:, 1] = -(fc_in[:, 1] - origin[0, 1]) * scale_factor
    components["fc_in"] = fc_in
    fc_out[:, 0] = (fc_out[:, 0] - origin[0, 0]) * scale_factor
    fc_out[:, 1] = -(fc_out[:, 1] - origin[0, 1]) * scale_factor
    components["fc_out"] = fc_out
    fc_shield[:, 0] = (fc_shield[:, 0] - origin[0, 0]) * scale_factor
    fc_shield[:, 1] = -(fc_shield[:, 1] - origin[0, 1]) * scale_factor
    components["fc_shield"] = fc_shield
    pl[:, 0] = (pl[:, 0] - origin[0, 0]) * scale_factor
    pl[:, 1] = -(pl[:, 1] - origin[0, 1]) * scale_factor
    components["pl"] = pl
    pl_shield[:, 0] = (pl_shield[:, 0] - origin[0, 0]) * scale_factor
    pl_shield[:, 1] = -(pl_shield[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield"] = pl_shield
    pl_shield_2[:, 0] = (pl_shield_2[:, 0] - origin[0, 0]) * scale_factor
    pl_shield_2[:, 1] = -(pl_shield_2[:, 1] - origin[0, 1]) * scale_factor
    components["pl_shield2"] = pl_shield_2
    pl_centers[:, 0] = (pl_centers[:, 0] - origin[0, 0]) * scale_factor
    pl_centers[:, 1] = -(pl_centers[:, 1] - origin[0, 1]) * scale_factor

    thicknesses = {"W2": 0.2, "I2": 0.13, "W1": 0.1, "I1": 0.15, "BE": 0.16}  # um
    heights = {
        "W2": z0 + thicknesses["W2"] / 2,
        "W1": z0 + sum([thicknesses[k] for k in ["W2", "I2"]]) + thicknesses["W1"] / 2,
        "BE": (
            z0
            + sum([thicknesses[k] for k in ["W2", "I2", "W1", "I1"]])
            + thicknesses["BE"] / 2
        ),
    }
    polygons = {
        "fc": np.concatenate([fc_in[:-2, :], np.flipud(fc_out)]),
        "fc_shield": np.append(fc_shield, fc_shield[0].reshape([-1, 2]), axis=0),
        "pl_shield": np.append(pl_shield, pl_shield[0].reshape([-1, 2]), axis=0),
        "pl_shield2": np.append(pl_shield_2, pl_shield_2[0].reshape([-1, 2]), axis=0),
        "pl": np.append(pl, pl[0].reshape([-1, 2]), axis=0),
    }

    layers = {
        "W2": Layer(
            "W2",
            thickness=thicknesses["W2"],
            london_lambda=london_lambda,
            z0=heights["W2"],
        ),
        "W1": Layer(
            "W1",
            thickness=thicknesses["W1"],
            london_lambda=london_lambda,
            z0=heights["W1"],
        ),
        "BE": Layer(
            "BE",
            thickness=thicknesses["BE"],
            london_lambda=london_lambda,
            z0=heights["BE"],
        ),
    }
    films = {
        "fc": Polygon("fc", layer="BE", points=polygons["fc"]),
        "pl_shield2": Polygon("pl_shield2", layer="BE", points=polygons["pl_shield2"]),
        "fc_shield": Polygon("fc_shield", layer="W1", points=polygons["fc_shield"]),
        "pl": Polygon("pl", layer="W1", points=polygons["pl"]),
        "pl_shield": Polygon("pl_shield", layer="W2", points=polygons["pl_shield"]),
    }
    holes = {}
    flux_regions = {
        "pl_center": Polygon(
            "pl_center",
            layer="W1",
            points=np.append(pl_centers, pl_centers[0].reshape([-1, 2]), axis=0),
        ),
    }

    name = "susceptometer_100nm"
    return Device(
        name,
        layers=layers,
        films=films,
        holes=holes,
        flux_regions=flux_regions,
    )


def make_layout_small_susc_v2():
    def rotate(arr, degrees):
        c = np.cos(np.radians(degrees))
        s = np.sin(np.radians(degrees))
        R = np.array([[c, -s], [s, c]])
        return R @ arr

    d = make_layout_small_susc()

    fc0 = rotate(d.films["fc"].points.T, -45).T

    # fc = np.concatenate(
    #     [
    #         fc0[29:],
    #         np.array(
    #             [
    #                 [2.45, -1.0],
    #                 [2.9, -1.8],
    #                 [3.5, -2.8],
    #                 [2.8, -2.8],
    #             ]
    #         ),
    #         fc0[:29],
    #         np.array(
    #             [
    #                 [1.9, -2.0],
    #                 [2.3, -2.8],
    #                 [1.6, -2.8],
    #                 [1.3, -2.0],
    #             ]
    #         ),
    #         fc0[29:30],
    #     ],
    #     axis=0,
    # )

    fc = np.concatenate(
        [
            fc0[29:],
            np.array(
                [
                    # [2.45, -1.0],
                    # [2.9, -1.8],
                    # [3.5, -2.8],
                    # [2.8, -2.8],
                    [fc0[-1][0], -1.3]
                ]
            ),
            fc0[:29],
            np.array(
                [
                    [1.9, -2.0],
                    # [2.3, -2.8],
                    [fc0[-1][0], -2.8],
                    [1.6, -2.8],
                    [1.3, -2.0],
                ]
            ),
            fc0[29:30],
        ],
        axis=0,
    )

    # fc_shield = np.array(
    #     [
    #         [0.81450159, -1.62204163],
    #         # [ 0.59869348, -1.12777144],
    #         [0.65, -1.1],
    #         [0.75880918, -0.04873086],
    #         # [ 1.19738696,  0.36200071],
    #         [1.25, 0.45],
    #         # [ 1.98404234, -0.18796191],
    #         [2.0, -0.14],
    #         [2.15111959, -0.31326984],
    #         [3.8, -3.0],
    #         [1.3, -3.0],
    #         [0.81450159, -1.62204163],
    #     ]
    # )

    fc_shield = np.array(
        [
            [0.81450159, -1.62204163],
            # [ 0.59869348, -1.12777144],
            [0.65, -1.1],
            [0.75880918, -0.04873086],
            # [ 1.19738696,  0.36200071],
            [1.25, 0.45],
            # [ 1.98404234, -0.18796191],
            [2.0, -0.14],
            [2.15111959, -0.31326984],
            # [3.8, -3.0],
            [2.15111959, -3.0],
            [1.3, -3.0],
            [0.81450159, -1.62204163],
        ]
    )

    pl_shield = np.array(
        [
            [-0.31326984, -0.10442328],
            [0.30630829, -0.11138483],
            [0.71007831, -1.74038802],
            [1.1, -3],
            [-1.1, -3],
            [-0.70311676, -1.55242611],
            [-0.31326984, -0.10442328],
        ]
    )

    pl_shield2 = np.array(
        [
            [-0.45250089, -1.44104128],
            [0.42465468, -1.44104128],
            [0.57084727, -1.87961906],
            [0.92, -2.9],
            [-0.92, -2.9],
            [-0.54996261, -1.71950336],
            [-0.45250089, -1.44104128],
        ]
    )

    pl0 = rotate(d.films["pl"].points.T, -45).T
    pl = np.concatenate(
        [
            pl0[:8],
            np.array(
                [
                    [0.75, -2.8],
                    [0.15, -2.8],
                ]
            ),
            pl0[9:17],
            np.array(
                [
                    [-0.15, -2.8],
                    [-0.75, -2.8],
                ]
            ),
            pl0[-1:],
        ],
        axis=0,
    )

    pl_center0 = rotate(d.flux_regions["pl_center"].points.T, -45).T
    pl_center = np.concatenate(
        [
            pl_center0[:10],
            np.array(
                [
                    [0.5, -2.7],
                    [-0.5, -2.7],
                ]
            ),
            pl_center0[:1],
        ],
        axis=0,
    )

    films = {
        "fc": Polygon("fc", layer="BE", points=fc),
        "pl_shield2": Polygon("pl_shield2", layer="BE", points=pl_shield2),
        "fc_shield": Polygon("fc_shield", layer="W1", points=fc_shield),
        "pl": Polygon("pl", layer="W1", points=pl),
        "pl_shield": Polygon("pl_shield", layer="W2", points=pl_shield),
    }
    holes = {}
    flux_regions = {
        "pl_center": Polygon(
            "pl_center",
            layer="W1",
            points=pl_center,
        ),
    }

    return Device(
        d.name,
        layers=d.layers,
        films=films,
        holes=holes,
        flux_regions=flux_regions,
    )


# def make_layout_small_susc_json():
#     path = os.path.join(
#         os.path.dirname(os.path.abspath(__file__)), "susceptometer_100nm.json"
#     )
#     return DeviceLayout.from_json(path)


# def make_layout_small_susc():
#     mat_path = os.path.join(os.path.dirname(__file__), "small_susc.mat")
#     layout = scipy.io.loadmat(mat_path)
#     origin = layout["origin"]
#     pl = layout["pl"]
#     pl_centers = layout["pl_centers"]
#     pl_shield = layout["pl_shield"]
#     pl_shield_2 = layout["pl_shield_2"]
#     # A = layout['A']
#     fc_in = layout["fc_in"]
#     fc_out = layout["fc_out"]
#     fc_shield = layout["fc_shield"]
#     two_micron_scale = layout["two_micron_scale"]

#     z0 = 0.0  # microns
#     london_lambda = 0.080  # 80 nm London penetration depth for Nb films

#     scale_factor = 2 / (two_micron_scale[1, 0] - two_micron_scale[1, 1])

#     components = {}

#     fc_in[:, 0] = (fc_in[:, 0] - origin[0, 0]) * scale_factor
#     fc_in[:, 1] = -(fc_in[:, 1] - origin[0, 1]) * scale_factor
#     components["fc_in"] = fc_in
#     fc_out[:, 0] = (fc_out[:, 0] - origin[0, 0]) * scale_factor
#     fc_out[:, 1] = -(fc_out[:, 1] - origin[0, 1]) * scale_factor
#     components["fc_out"] = fc_out
#     fc_shield[:, 0] = (fc_shield[:, 0] - origin[0, 0]) * scale_factor
#     fc_shield[:, 1] = -(fc_shield[:, 1] - origin[0, 1]) * scale_factor
#     components["fc_shield"] = fc_shield
#     pl[:, 0] = (pl[:, 0] - origin[0, 0]) * scale_factor
#     pl[:, 1] = -(pl[:, 1] - origin[0, 1]) * scale_factor
#     components["pl"] = pl
#     pl_shield[:, 0] = (pl_shield[:, 0] - origin[0, 0]) * scale_factor
#     pl_shield[:, 1] = -(pl_shield[:, 1] - origin[0, 1]) * scale_factor
#     components["pl_shield"] = pl_shield
#     pl_shield_2[:, 0] = (pl_shield_2[:, 0] - origin[0, 0]) * scale_factor
#     pl_shield_2[:, 1] = -(pl_shield_2[:, 1] - origin[0, 1]) * scale_factor
#     components["pl_shield2"] = pl_shield_2
#     pl_centers[:, 0] = (pl_centers[:, 0] - origin[0, 0]) * scale_factor
#     pl_centers[:, 1] = -(pl_centers[:, 1] - origin[0, 1]) * scale_factor

#     # use the order in which you want to calculate response fields
#     polygons = {
#         "fc": np.concatenate([fc_in[:-2, :], np.flipud(fc_out)]),
#         "fc_shield": np.append(fc_shield, fc_shield[0].reshape([-1, 2]), axis=0),
#         "pl_shield": np.append(pl_shield, pl_shield[0].reshape([-1, 2]), axis=0),
#         "pl_shield2": np.append(pl_shield_2, pl_shield_2[0].reshape([-1, 2]), axis=0),
#         "pl": np.append(pl, pl[0].reshape([-1, 2]), axis=0),
#     }
#     layers = {
#         "fc": "BE",
#         "pl_shield2": "BE",
#         "fc_shield": "W1",
#         "pl": "W1",
#         "pl_shield": "W2",
#     }
#     # order: from closest to vortex to furthest from sample
#     thicknesses = {"W2": 0.2, "I2": 0.13, "W1": 0.1, "I1": 0.15, "BE": 0.16}  # um
#     heights = {
#         "W2": z0 + thicknesses["W2"] / 2,
#         "W1": z0 + sum([thicknesses[k] for k in ["W2", "I2"]]) + thicknesses["W1"] / 2,
#         "BE": z0
#         + sum([thicknesses[k] for k in ["W2", "I2", "W1", "I1"]])
#         + thicknesses["BE"] / 2,
#     }
#     lambdas = {name: london_lambda for name in ["W2", "W1", "BE"]}
#     flux_regions = {"pl_center": (np.append(pl_centers, pl_centers[0].reshape([-1, 2]), axis=0), "W1")}
#     name = "susceptometer_100nm"
#     return DeviceLayout(
#         name, polygons, layers, thicknesses, heights, lambdas, flux_regions=flux_regions
#     )
