from copy import deepcopy
from typing import Optional, Union, Tuple

import numpy as np
from matplotlib import path
from scipy import interpolate
from shapely import geometry
import shapely

from ..geometry import close_curve
from ..parameter import Parameter


class Layer(object):
    """A single layer of a superconducting device.

    You can provide either an effective penetration depth Lambda,
    or both a London penetration depth (lambda_london) and a layer
    thickness. Lambda and london_lambda can either be real numers or
    Parameters which compute the penetration depth as a function of
    position.

    Args:
        name: Name of the layer.
        thickness: Thickness of the superconducting film(s) located in the layer.
        london_lambda: London penetration depth of the superconducting film(s)
            located in the layer.
        z0: Vertical location of the layer.
    """

    def __init__(
        self,
        name: str,
        Lambda: Optional[Union[float, Parameter]] = None,
        london_lambda: Optional[Union[float, Parameter]] = None,
        thickness: Optional[float] = None,
        z0: float = 0,
    ):
        self.name = name
        self.thickness = thickness
        self.london_lambda = london_lambda
        self.z0 = z0
        if Lambda is None:
            if london_lambda is None or thickness is None:
                raise ValueError(
                    "You must provide either an effective penetration depth Lambda "
                    "or both a london_lambda and a thickness."
                )
            self._Lambda = None
        else:
            if london_lambda is not None or thickness is not None:
                raise ValueError(
                    "You must provide either an effective penetration depth Lambda "
                    "or both a london_lambda and a thickness (but not all three)."
                )
            self._Lambda = Lambda

    @property
    def Lambda(self) -> Union[float, Parameter]:
        """Effective penetration depth of the superconductor."""
        if self._Lambda is not None:
            return self._Lambda
        return self.london_lambda ** 2 / self.thickness

    @Lambda.setter
    def Lambda(self, value: Union[float, Parameter]) -> None:
        """Effective penetration depth of the superconductor."""
        if self._Lambda is None:
            raise AttributeError(
                "Can't set Lambda directly. Set london_lambda and/or thickness instead."
            )
        self._Lambda = value

    def __repr__(self) -> str:
        Lambda = self.Lambda
        if isinstance(Lambda, (int, float)):
            Lambda = f"{Lambda:.3f}"
        d = self.thickness
        if isinstance(d, (int, float)):
            d = f"{d:.3f}"
        london = self.london_lambda
        if isinstance(london, (int, float)):
            london = f"{london:.3f}"
        return (
            f'{self.__class__.__name__}("{self.name}", Lambda={Lambda}, '
            f"thickness={d}, london_lambda={london}, z0={self.z0:.3f})"
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.name == other.name
            and self.thickness == other.thickness
            and self.london_lambda == other.london_lambda
            and self.Lambda == other.Lambda
            and self.z0 == other.z0
        )

    def copy(self):
        return deepcopy(self)


class Polygon(object):
    """A polygonal region located in a Layer.

    Args:
        name: Name of the polygon.
        layer: Name of the layer in which the polygon is located.
        points: An array of shape (n, 2) specifying the x, y coordinates of
            the polyon's vertices, or a shapely Polygon instance.
        mesh: Whether to include this polygon when computing a mesh.

    """

    def __init__(
        self,
        name: str,
        *,
        layer: str,
        points: Union[np.ndarray, geometry.polygon.Polygon],
        mesh: bool = True,
    ):
        self.name = name
        self.layer = layer
        if isinstance(points, shapely.geometry.polygon.Polygon):
            if points.interiors:
                raise ValueError("Expected a simply-connected polygon.")
            points = shapely.geometry.polygon.orient(points).exterior.coords
        points = close_curve(np.asarray(points))
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {points.shape}.")
        self._points = points
        self.mesh = mesh

    @property
    def points(self) -> np.ndarray:
        """A shape ``(n, 2)`` array of counter-clockwise-oriented polygon vertices."""
        return self._points

    @property
    def area(self) -> float:
        """The area of the polygon."""
        return self.polygon.area

    @property
    def extents(self) -> Tuple[float, float]:
        """Returns the total x, y extent of the polygon, (Dx, Dy)."""
        minx, miny, maxx, maxy = self.polygon.bounds
        return (maxx - minx), (maxy - miny)

    @property
    def path(self) -> path.Path:
        """A matplotlib.path.Path representing the polygon boundary."""
        return path.Path(self.points, closed=True)

    @property
    def polygon(self) -> geometry.polygon.Polygon:
        """A :class:`shapely.geometry.polygon.Polygon` representing the Polygon."""
        return geometry.polygon.Polygon(self._points)

    def contains_points(
        self,
        points: np.ndarray,
        index: bool = False,
        radius: float = 0,
    ) -> Union[bool, np.ndarray]:
        """Determines whether ``points`` lie within the polygon.

        Args:
            points: Shape ``(n, 2)`` array of x, y coordinates.
            index: If True, then return the indices of the points in ``points``
                that lie within the polygon. Otherwise, returns a shape ``(n, )``
                boolean array.
            radius: An additional margin on ``self.path``.
                See :meth:`matplotlib.path.Path.contains_points`.

        Returns:
            If index is True, returns the indices of the points in ``points``
            that lie within the polygon. Otherwise, returns a shape ``(n, )``
            boolean array indicating whether each point lies within the polygon.
        """
        bool_array = self.path.contains_points(np.atleast_2d(points), radius=radius)
        if index:
            return np.where(bool_array)[0]
        return bool_array

    def on_boundary(
        self, points: np.ndarray, radius: float = 1e-3, index: bool = False
    ):
        points = np.atleast_2d(points)
        p = self.path
        outer = p.contains_points(points, radius=radius)
        inner = p.contains_points(points, radius=-radius)
        boundary = np.logical_and(outer, ~inner)
        if index:
            return np.where(boundary)[0]
        return boundary

    # def offset_points(
    #     self,
    #     delta: float,
    #     join_type: str = "square",
    #     miter_limit: float = 2.0,
    #     as_polygon: bool = False,
    # ) -> Union[np.ndarray, "Polygon"]:
    #     """Returns polygon points or a Polygon object with vertices offset from
    #     ``self.points`` by a distance ``delta``. If ``delta > 0`` this "inflates"
    #     the polygon, and if ``delta < 0`` this shrinks the polygon.

    #     Args:
    #         delta: The amount by which to offset the polygon points, in units of
    #             ``self.length_units``.
    #         join_type: The join type to use in generating the offset points. See the
    #             `Clipper documentation <http://www.angusj.com/delphi/clipper/
    #             documentation/ Docs/Units/ClipperLib/Types/JoinType.htm>`_
    #             for more details.
    #         miter_limit: The MiterLimit to use if ``join_type == "miter"``. See the
    #             `Clipper documentation <http://www.angusj.com/delphi/clipper/
    #             documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/
    #             MiterLimit.htm>`_ for more details.
    #         as_polygon: Whether to return a new Polygon instance or just an array
    #             of vertices.

    #     Returns:
    #         A new :class:`superscreen.device.Polygon` instance with the
    #         offset polygon vertices if ``as_polygon`` is True. Otherwise returns
    #         a shape ``(m, 2)`` array of offset polygon vertices.
    #     """
    #     jt = {
    #         "square": pyclipper.JT_SQUARE,
    #         "miter": pyclipper.JT_MITER,
    #         "round": pyclipper.JT_ROUND,
    #     }[join_type.lower()]
    #     pco = pyclipper.PyclipperOffset()
    #     if jt == pyclipper.JT_MITER:
    #         pco.MiterLimit = miter_limit
    #     pco.AddPath(
    #         pyclipper.scale_to_clipper(self.points),
    #         jt,
    #         pyclipper.ET_CLOSEDPOLYGON,
    #     )
    #     solution = pco.Execute(pyclipper.scale_to_clipper(delta))
    #     points = np.array(pyclipper.scale_from_clipper(solution)).squeeze()
    #     points = close_curve(points)
    #     if points.shape[0] < self.points.shape[0]:
    #         tck, _ = interpolate.splprep(points.T, k=1, s=0)
    #         x, y = interpolate.splev(np.linspace(0, 1, self.points.shape[0]), tck)
    #         points = close_curve(np.stack([x, y], axis=1))
    #     if not as_polygon:
    #         return points
    #     return Polygon(
    #         f"{self.name} ({delta:+.3f})",
    #         layer=self.layer,
    #         points=points,
    #         mesh=self.mesh,
    #     )

    def _validate_join(
        self,
        other: Union["Polygon", geometry.polygon.Polygon, geometry.polygon.LinearRing],
        name: Optional[str],
        operator: str,
    ) -> Tuple[geometry.polygon.Polygon, Optional[str]]:
        if isinstance(other, Polygon):
            other_poly = other.polygon
            if self.layer != other.layer:
                raise ValueError("Cannot join with a polygon in other layer.")
            if name is None:
                name = f"{self.name} {operator} {other.name}"
        elif isinstance(
            other, (geometry.polygon.LinearRing, geometry.linestring.LineString)
        ):
            other_poly = geometry.polygon.Polygon(other)
        elif not isinstance(other, geometry.polygon.Polygon):
            raise TypeError(
                f"Expected other to be a superscreen Polygon, shapely Polygon, "
                f"or shapely LinearRing/LineString, but got {type(other)}."
            )
        return other_poly, name

    def union(
        self,
        other: Union["Polygon", geometry.polygon.Polygon, geometry.polygon.LinearRing],
        name: Optional[str] = None,
    ) -> "Polygon":
        other_poly, name = self._validate_join(other, name, "|")
        if name is None:
            name = self.name
        poly = self.polygon.union(other_poly)
        return Polygon(
            name,
            layer=self.layer,
            points=poly,
            mesh=self.mesh,
        )

    def intersection(
        self,
        other: Union["Polygon", geometry.polygon.Polygon, geometry.polygon.LinearRing],
        name: Optional[str] = None,
    ) -> "Polygon":
        other_poly, name = self._validate_join(other, "&")
        if name is None:
            name = self.name
        poly = self.polygon.intersection(other_poly)
        return Polygon(
            name,
            layer=self.layer,
            points=poly,
            mesh=self.mesh,
        )

    def difference(
        self,
        other: Union["Polygon", geometry.polygon.Polygon, geometry.polygon.LinearRing],
        name: Optional[str] = None,
    ) -> "Polygon":
        other_poly, name = self._validate_join(other, "\\")
        if name is None:
            name = self.name
        poly = self.polygon.difference(other_poly)
        return Polygon(
            name,
            layer=self.layer,
            points=poly,
            mesh=self.mesh,
        )

    def symmetric_difference(
        self,
        other: Union["Polygon", geometry.polygon.Polygon, geometry.polygon.LinearRing],
        name: Optional[str] = None,
    ) -> "Polygon":
        other_poly, name = self._validate_join(other, "\\")
        if name is None:
            name = self.name
        poly = self.polygon.symmetric_difference(other_poly)
        return Polygon(
            name,
            layer=self.layer,
            points=poly,
            mesh=self.mesh,
        )

    def buffer(
        self,
        distance: float,
        join_style: Union[str, int] = "mitre",
        mitre_limit: float = 5.0,
        single_sided: bool = True,
        as_polygon: bool = True,
    ) -> Union[np.ndarray, "Polygon"]:
        if isinstance(join_style, str):
            join_style = getattr(geometry.JOIN_STYLE, join_style)
        poly = self.polygon.buffer(
            distance,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
        )

        polygon = Polygon(
            f"{self.name} ({distance:+.3f})",
            layer=self.layer,
            points=poly,
            mesh=self.mesh,
        )
        npts = max(polygon.points.shape[0], self.points.shape[0])
        polygon = polygon.resample(npts)
        if as_polygon:
            return polygon
        return polygon.points

    def resample(
        self, num_points: Optional[int] = None, degree: int = 1, smooth: float = 0
    ) -> "Polygon":
        if num_points is None:
            return self
        tck, _ = interpolate.splprep(self.points.T, k=degree, s=smooth)
        x, y = interpolate.splev(np.linspace(0, 1, num_points), tck)
        points = close_curve(np.stack([x, y], axis=1))
        return Polygon(
            self.name,
            layer=self.layer,
            points=points,
            mesh=self.mesh,
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}("{self.name}", layer="{self.layer}", '
            f"points=ndarray[shape={self.points.shape}])"
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Polygon):
            return False

        return (
            self.name == other.name
            and self.layer == other.layer
            and np.allclose(self.points, other.points)
        )

    def copy(self) -> "Polygon":
        return deepcopy(self)
