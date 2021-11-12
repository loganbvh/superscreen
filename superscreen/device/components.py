from copy import deepcopy
from typing import Optional, Union, Tuple, Iterable

import numpy as np
from matplotlib import path
import matplotlib.pyplot as plt
from scipy import interpolate
from shapely import geometry as geo

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
        points: The polygon vertices. This can be a shape ``(n, 2)`` array of x, y
            coordinates or a shapely ``LineString``, ``LinearRing``, or ``Polygon``.
        mesh: Whether to include this polygon when computing a mesh.
    """

    def __init__(
        self,
        name: str,
        *,
        layer: str,
        points: Union[
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        ],
        mesh: bool = True,
    ):
        self.name = name
        self.layer = layer
        self.points = points
        self.mesh = mesh

    @property
    def points(self) -> np.ndarray:
        """A shape ``(n, 2)`` array of counter-clockwise-oriented polygon vertices."""
        return self._points

    @points.setter
    def points(self, points) -> None:
        geom_types = (
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        )
        if isinstance(points, Polygon):
            points = points.points
        if not isinstance(points, geom_types):
            points = np.asarray(points)
        points = geo.polygon.Polygon(points)
        points = geo.polygon.orient(points)
        if points.interiors:
            raise ValueError("Expected a simply-connected polygon.")
        if not points.is_valid:
            raise ValueError("The given points do not define a valid polygon.")
        points = close_curve(np.array(points.exterior.coords))
        if points.ndim != 2 or points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {points.shape}.")
        self._points = points

    @property
    def area(self) -> float:
        """The area of the polygon."""
        return self.polygon.area

    @property
    def extents(self) -> Tuple[float, float]:
        """Returns the total x, y extent of the polygon, (Delta_x, Delta_y)."""
        minx, miny, maxx, maxy = self.polygon.bounds
        return (maxx - minx), (maxy - miny)

    @property
    def path(self) -> path.Path:
        """A matplotlib.path.Path representing the polygon boundary."""
        return path.Path(self.points, closed=True)

    @property
    def polygon(self) -> geo.polygon.Polygon:
        """A shapely ``Polygon`` representing the Polygon."""
        return geo.polygon.Polygon(self.points)

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
        """Determines whether ``points`` lie within a given radius of the Polygon
        boundary.

        Args:
            points: Shape ``(n, 2)`` array of x, y coordinates.
            radius: Points within ``radius`` of the boundary are considered
                to lie on the boundary.
            index: If True, then return the indices of the points in ``points``
                that lie on the boundary. Otherwise, returns a shape ``(n, )``
                boolean array.

        Returns:
            If index is True, returns the indices of the points in ``points``
            that lie within the polygon. Otherwise, returns a shape ``(n, )``
            boolean array indicating whether each point lies within the polygon.
        """
        points = np.atleast_2d(points)
        p = self.path
        in_outer = p.contains_points(points, radius=radius)
        in_inner = p.contains_points(points, radius=-radius)
        boundary = np.logical_and(in_outer, ~in_inner)
        if index:
            return np.where(boundary)[0]
        return boundary

    def _join_via(
        self,
        other: Union[
            np.ndarray,
            "Polygon",
            geo.polygon.Polygon,
            geo.polygon.LinearRing,
        ],
        operation: str,
    ) -> geo.polygon.Polygon:
        """Joins ``self.polygon`` with another polygon-like object
        via a given operation.
        """
        valid_types = (
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        )
        valid_operations = (
            "union",
            "intersection",
            "difference",
            "symmetric_difference",
        )
        if operation not in valid_operations:
            raise ValueError(
                f"Unknown operation: {operation}. "
                f"Valid operations are {valid_operations}."
            )
        if isinstance(other, Polygon):
            other_poly = other.polygon
            if self.layer != other.layer:
                raise ValueError("Cannot join with a polygon in other layer.")
        elif isinstance(other, valid_types):
            other_poly = geo.polygon.Polygon(other)
        if not isinstance(other_poly, geo.polygon.Polygon):
            raise TypeError(
                f"Valid types are {(Polygon, ) + valid_types}, got {type(other)}."
            )
        joined = getattr(self.polygon, operation)(other_poly)
        if (
            not isinstance(joined, geo.polygon.Polygon)
            or joined.is_empty
            or not joined.is_valid
        ):
            raise ValueError(
                f"The {operation} of the two polygons is not a valid polygon."
            )
        return joined

    def union(
        self,
        other: Union[
            "Polygon",
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        ],
        name: Optional[str] = None,
    ) -> "Polygon":
        """Returns the union of the polygon and another polygon.

        Args:
            other: The object with which to join the polygon.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the union
            of ``self`` and ``other``.
        """
        return Polygon(
            name or self.name,
            layer=self.layer,
            points=self._join_via(other, "union"),
            mesh=self.mesh,
        )

    def intersection(
        self,
        other: Union[
            "Polygon",
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        ],
        name: Optional[str] = None,
    ) -> "Polygon":
        """Returns the intersection of the polygon and another polygon.

        Args:
            other: The object with which to join the polygon.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the intersection
            of ``self`` and ``other``.
        """
        return Polygon(
            name or self.name,
            layer=self.layer,
            points=self._join_via(other, "intersection"),
            mesh=self.mesh,
        )

    def difference(
        self,
        other: Union[
            "Polygon",
            np.ndarray,
            geo.linestring.LineString,
            geo.polygon.LinearRing,
            geo.polygon.Polygon,
        ],
        symmetric: bool = False,
        name: Optional[str] = None,
    ) -> "Polygon":
        """Returns the difference of the polygon and another polygon.

        Args:
            other: The object with which to join the polygon.
            symmetric: Whether to join via a symmetric difference operation.
                See the `shapely documentation`_.
            name: A name for the resulting joined Polygon (defaults to ``self.name``.)

        Returns:
            A new :class:`Polygon` instance representing the difference
            of ``self`` and ``other``.

        .. _shapely documentation: https://shapely.readthedocs.io/en/stable/manual.html
        """
        operation = "symmetric_difference" if symmetric else "difference"
        return Polygon(
            name or self.name,
            layer=self.layer,
            points=self._join_via(other, operation),
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
        """Returns polygon points or a new Polygon object with vertices offset from
        ``self.points`` by a given ``distance``. If ``distance > 0`` this "inflates"
        the polygon, and if ``distance < 0`` this shrinks the polygon.


        Args:
            join_style: One of "round" (1), "mitre" (2), or "bevel" (3).
                See the `shapely documentation`_.
            mitre_limit: See the `shapely documentation`_.
            single_sided: See the `shapely documentation`_.
            as_polygon: If True, returns a new ``Polygon`` instance, otherwise
                returns a shape ``(n, 2)`` array of polygon vertices.

        Returns:
            A new ``Polygon`` or an array of vertices offset by ``distance``.

        .. _shapely documentation: https://shapely.readthedocs.io/en/stable/manual.html
        """
        if isinstance(join_style, str):
            join_style = getattr(geo.JOIN_STYLE, join_style)
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
        """Resample vertices so that they are approximately uniformly distributed
        along the polygon boundary.

        Args:
            num_points: Number of points to interpolate to. If ``num_points`` is None,
                the polygon is resampled to ``len(self.points)`` points. If
                ``num_points`` is not None and has a boolean value of False,
                then an unaltered copy of the polygon is returned.
            degree: The degree of the spline with which to iterpolate.
                Defaults to 1 (linear spline).

        """
        if num_points is None:
            num_points = self.points.shape[0]
        if not num_points:
            return self.copy()
        points = self.points.copy()
        _, ix = np.unique(points, return_index=True, axis=0)
        points = points[np.sort(ix)]
        tck, _ = interpolate.splprep(points.T, k=degree, s=smooth)
        x, y = interpolate.splev(np.linspace(0, 1, num_points - 1), tck)
        points = close_curve(np.stack([x, y], axis=1))
        return Polygon(
            self.name,
            layer=self.layer,
            points=points,
            mesh=self.mesh,
        )

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Plots the Polygon's vertices.

        Args:
            ax: The matplotlib Axes on which to plot. If None is given, a new one
                is created.
            kwargs: Passed to ``ax.plot()``.

        Returns:
            The matplotlib Axes.
        """
        if ax is None:
            _, ax = plt.subplots()
        kwargs = kwargs.copy()
        kwargs["label"] = self.name
        ax.plot(*self.points.T, **kwargs)
        ax.set_aspect("equal")
        return ax

    @classmethod
    def from_union(
        cls,
        items: Iterable[
            Union[
                "Polygon",
                np.ndarray,
                geo.linestring.LineString,
                geo.polygon.LinearRing,
                geo.polygon.Polygon,
            ],
        ],
        *,
        name: str,
        layer: str,
        mesh: bool = True,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the union of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            layer: Name of the layer in which the polygon is located.
            mesh: Whether to include this polygon when computing a mesh.

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name, layer=layer, points=first, mesh=mesh)
        for item in rest:
            polygon = polygon.union(item)
        return polygon

    @classmethod
    def from_intersection(
        cls,
        items: Iterable[
            Union[
                "Polygon",
                np.ndarray,
                geo.linestring.LineString,
                geo.polygon.LinearRing,
                geo.polygon.Polygon,
            ],
        ],
        *,
        name: str,
        layer: str,
        mesh: bool = True,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the intersection
        of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            layer: Name of the layer in which the polygon is located.
            mesh: Whether to include this polygon when computing a mesh.

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name, layer=layer, points=first, mesh=mesh)
        for item in rest:
            polygon = polygon.intersection(item)
        return polygon

    @classmethod
    def from_difference(
        cls,
        items: Iterable[
            Union[
                "Polygon",
                np.ndarray,
                geo.linestring.LineString,
                geo.polygon.LinearRing,
                geo.polygon.Polygon,
            ],
        ],
        *,
        name: str,
        layer: str,
        mesh: bool = True,
        symmetric: bool = False,
    ) -> "Polygon":
        """Creates a new :class:`Polygon` from the difference
        of a sequence of polygons.

        Args:
            items: A sequence of polygon-like objects to join.
            name: Name of the polygon.
            layer: Name of the layer in which the polygon is located.
            mesh: Whether to include this polygon when computing a mesh.
            symmetric:

        Returns:
            A new :class:`Polygon`.
        """
        first, *rest = items
        polygon = cls(name, layer=layer, points=first, mesh=mesh)
        for item in rest:
            polygon = polygon.difference(item, symmetric=symmetric)
        return polygon

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
