import os
import json
import logging
from typing import Union, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import optimesh

from .utils import inpolygon, PathMaker, json_numpy_obj_hook, NumpyJSONEncoder
from . import core

logger = logging.getLogger("Device")


class Layer(object):
    def __init__(self, name: str, *, thickness: float, london_lambda: float, z0: float):
        self.name = name
        self.thickness = thickness
        self.london_lambda = london_lambda
        self.z0 = z0

    @property
    def Lambda(self) -> float:
        return self.london_lambda ** 2 / self.thickness

    def __repr__(self) -> str:
        return 'Layer("{}", thickness={:.3f}, london_lambda={:.3f}, z0={:.3f})'.format(
            self.name, self.thickness, self.london_lambda, self.z0
        )


class Polygon(object):
    def __init__(self, name: str, *, layer: str, points: np.ndarray):
        self.name = name
        self.layer = layer
        self.points = np.asarray(points)

        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError(
                "Expected shape (n, 2), but got {}.".format(self.points.shape)
            )

    def contains_points(
        self,
        xq: Union[float, np.ndarray],
        yq: Union[float, np.ndarray],
        index: bool = False,
    ) -> Union[bool, np.ndarray]:
        x, y = self.points.T
        bool_array = inpolygon(xq, yq, x, y)
        if index:
            return np.where(bool_array)[0]
        return bool_array

    def __repr__(self) -> str:
        return 'Polygon("{}", layer="{}", points=ndarray[shape={}])'.format(
            self.name, self.layer, self.points.shape
        )


class Device(object):
    """Object representing a device composed of multiple layers of thin film superconductor.
    All distances/lengths should be in the units specified by self.units.
    """

    args = [
        "name",
    ]

    kwargs = [
        "layers",
        "films",
        "holes",
        "flux_regions",
        "mesh_refinements",
        "units",
        "origin",
    ]

    def __init__(
        self,
        name: str,
        *,
        layers: Dict[str, Layer],
        films: Dict[str, Polygon],
        holes: Optional[Dict[str, Polygon]] = None,
        flux_regions: Optional[Dict[str, Polygon]] = None,
        mesh_refinements: int = 0,
        units: str = "um",
        origin: Tuple[float, float, float] = (0, 0, 0),
    ):
        self.name = name
        self.layers = layers
        self.films = films
        self.holes = holes or {}
        self.flux_regions = flux_regions or {}
        self.units = units
        self._origin = tuple(origin)

        self.points = np.unique(
            np.concatenate(
                [film.points for film in self.films.values()]
                + [hole.points for hole in self.holes.values()]
                + [flux_region.points for flux_region in self.flux_regions.values()]
            ),
            axis=0,
        )

        self.adj = None
        self.weights = None
        self.q = None
        self.C = None
        self.Q = None
        self.Del2 = None

        self.mesh = None
        self.make_mesh(n_refine=mesh_refinements)

    @property
    def mesh_points(self) -> np.ndarray:
        return np.stack([self.mesh.x, self.mesh.y], axis=1)

    @property
    def triangles(self) -> np.ndarray:
        return self.mesh.triangles

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self._origin

    @origin.setter
    def origin(self, new_origin: Tuple[float, float, float]):
        xp, yp, zp = new_origin
        self._update_origin(xp, yp, zp)

    def _update_origin(self, xp: float, yp: float, zp: float) -> None:
        x0, y0, z0 = self._origin
        dx = xp - x0
        dy = yp - y0
        self._origin = (xp, yp, zp)
        if dx or dy:
            points = self.mesh_points
            points[:, 0] += dx
            points[:, 1] += dy
            self.points = points
            for film in self.films.values():
                film.points[:, 0] += dx
                film.points[:, 1] += dy
            for hole in self.holes.values():
                hole.points[:, 0] += dx
                hole.points[:, 1] += dy
            for flux_region in self.flux_regions.values():
                flux_region.points[:, 0] += dx
                flux_region.points[:, 1] += dy
            self.make_mesh(n_refine=0, compute_arrays=False)

    def make_mesh(self, n_refine: int = 0, compute_arrays: bool = True):
        print("Making mesh with origin {}...".format(self.origin))
        x, y = self.points.T
        if self.mesh is None:
            triangles = None
        else:
            triangles = self.mesh.triangles
        mesh = mtri.Triangulation(x, y, triangles=triangles)
        points, cells = optimesh.optimize_points_cells(
            self.points, mesh.triangles, "cpt-fixed-point", 1e-4, 10000
        )
        self.mesh = mtri.Triangulation(points[:, 0], points[:, 1], cells)

        if n_refine:
            x, y = self.mesh_points.T
            triangles = self.mesh.triangles
            logger.info(
                "Initial mesh: {} nodes, {} triangles".format(len(x), len(triangles))
            )
            tri = mtri.Triangulation(x, y, triangles=triangles)
            refiner = mtri.UniformTriRefiner(tri)
            refined = refiner.refine_triangulation(subdiv=n_refine)
            points = np.stack([refined.x, refined.y], axis=1)
            points, cells = optimesh.optimize_points_cells(
                points, refined.triangles, "cpt-fixed-point", 1e-4, 10000
            )
            self.mesh = mtri.Triangulation(points[:, 0], points[:, 1], triangles=cells)
            msg = "Refined mesh ({} iterations): {} nodes, {} triangles"
            logger.info(msg.format(n_refine, len(points), len(cells)))

        if compute_arrays:
            points = self.mesh_points
            triangles = self.triangles
            self.adj = core.compute_adj(triangles)
            self.weights = core.uniform_weights(self.adj)
            self.q = core.q_matrix(points)
            self.C = core.C_vector(points)
            self.Q = core.Q_matrix(self.q, self.C, self.weights)
            self.Del2 = core.laplacian(self.weights)

    def plot_polygons(self, ax=None, grid=True, legend=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for name, film in self.films.items():
            ax.plot(*film.points.T, "-", label=name, **kwargs)
        for name, hole in self.holes.items():
            ax.plot(*hole.points.T, ".-.", label=name, **kwargs)
        for name, region in self.flux_regions.items():
            ax.plot(*region.points.T, "--.", label=name, **kwargs)
        ax.set_aspect("equal")
        ax.grid(grid)
        if legend:
            ax.legend(loc="best")

    def plot_mesh(self, ax=None, triangles=True, vertices=False, grid=True):
        x, y = self.mesh_points.T
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if triangles:
            ax.triplot(x, y, self.triangles, linewidth=1)
        if vertices:
            ax.plot(x, y, "k.")
        ax.set_aspect("equal")
        ax.grid(grid)

    @classmethod
    def from_json(cls, fname):
        """Create a Device from json metadata, i.e. output from Device.to_json()."""
        with open(fname, "r") as f:
            meta = json.load(f, object_hook=json_numpy_obj_hook)
        args = [meta[k] for k in cls.args]
        kwargs = {k: meta[k] for k in cls.kwargs}
        return cls(*args, **kwargs)

    @property
    def metadata(self):
        metadata = {attr: getattr(self, attr) for attr in self.args + self.kwargs}
        metadata["points"] = len(self.mesh_points)
        metadata["triangles"] = len(self.triangles)
        return metadata

    def to_json(self, fname=None):
        if fname is None:
            path = PathMaker(directory="layouts").dir
            fname = os.path.join(path, self.name)
        if not fname.endswith(".json"):
            fname += ".json"
        with open(fname, "w") as f:
            json.dump(self.metadata, f, indent=4, encoder=NumpyJSONEncoder)

    def save_mesh(self, fname=None):
        import meshio

        if fname is None:
            path = PathMaker(directory="mesh").dir
            fname = os.path.join(path, self.name) + ".msh"
        mesh = meshio.Mesh(self.mesh_points, {"triangle": self.triangles})
        meshio.write(fname, mesh)
