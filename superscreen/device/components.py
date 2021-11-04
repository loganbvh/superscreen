from copy import deepcopy
from typing import Optional, Union, Tuple

import numpy as np
from matplotlib import path
from scipy import interpolate
import pyclipper

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
            the polyon's vertices.
        mesh: Whether to include this polygon when computing a mesh.

    """

    def __init__(self, name: str, *, layer: str, points: np.ndarray, mesh: bool = True):
        self.name = name
        self.layer = layer
        self.points = np.asarray(points)
        # Ensure that it is a closed polygon.
        self.points = close_curve(self.points)
        self.mesh = mesh

        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError(f"Expected shape (n, 2), but got {self.points.shape}.")

    @property
    def area(self) -> float:
        """The area of the polygon."""
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # https://stackoverflow.com/a/30408825/11655306
        x = self.points[:, 0]
        y = self.points[:, 1]
        return 0.5 * np.abs(np.dot(y, np.roll(x, 1)) - np.dot(x, np.roll(y, 1)))

    @property
    def extents(self) -> Tuple[float, float]:
        """Returns the total x, y extent of the polygon, (Dx, Dy)."""
        return tuple(np.ptp(self.points, axis=0))

    @property
    def clockwise(self) -> bool:
        """True if the polygon vertices are oriented clockwise."""
        # # https://stackoverflow.com/a/1165943
        # # https://www.element84.com/blog/
        # # determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
        x = self.points[:, 0]
        y = self.points[:, 1]
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0

    @property
    def counter_clockwise(self) -> bool:
        """True if the polygon vertices are oriented counter-clockwise."""
        return not self.clockwise

    @property
    def path(self) -> path.Path:
        """A matplotlib.path.Path representing the polygon boundary."""
        return path.Path(self.points, closed=True)

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

    def offset_points(
        self,
        delta: float,
        join_type: str = "square",
        miter_limit: float = 2.0,
        as_polygon: bool = False,
    ) -> Union[np.ndarray, "Polygon"]:
        """Returns polygon points or a Polygon object with vertices offset from
        ``self.points`` by a distance ``delta``. If ``delta > 0`` this "inflates"
        the polygon, and if ``delta < 0`` this shrinks the polygon.

        Args:
            delta: The amount by which to offset the polygon points, in units of
                ``self.length_units``.
            join_type: The join type to use in generating the offset points. See the
                `Clipper documentation <http://www.angusj.com/delphi/clipper/
                documentation/ Docs/Units/ClipperLib/Types/JoinType.htm>`_
                for more details.
            miter_limit: The MiterLimit to use if ``join_type == "miter"``. See the
                `Clipper documentation <http://www.angusj.com/delphi/clipper/
                documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/
                MiterLimit.htm>`_ for more details.
            as_polygon: Whether to return a new Polygon instance or just an array
                of vertices.

        Returns:
            A new :class:`superscreen.device.Polygon` instance with the
            offset polygon vertices if ``as_polygon`` is True. Otherwise returns
            a shape ``(m, 2)`` array of offset polygon vertices.
        """
        jt = {
            "square": pyclipper.JT_SQUARE,
            "miter": pyclipper.JT_MITER,
            "round": pyclipper.JT_ROUND,
        }[join_type.lower()]
        pco = pyclipper.PyclipperOffset()
        if jt == pyclipper.JT_MITER:
            pco.MiterLimit = miter_limit
        pco.AddPath(
            pyclipper.scale_to_clipper(self.points),
            jt,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        solution = pco.Execute(pyclipper.scale_to_clipper(delta))
        points = np.array(pyclipper.scale_from_clipper(solution)).squeeze()
        points = close_curve(points)
        if points.shape[0] < self.points.shape[0]:
            tck, _ = interpolate.splprep(points.T, k=1, s=0)
            x, y = interpolate.splev(np.linspace(0, 1, self.points.shape[0]), tck)
            points = close_curve(np.stack([x, y], axis=1))
        if not as_polygon:
            return points
        return Polygon(
            f"{self.name} ({delta:+.3f})",
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
