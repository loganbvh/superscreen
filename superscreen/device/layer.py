from copy import deepcopy
from typing import Optional, Union

import h5py

from ..io import deserialize_obj, serialize_obj
from ..parameter import Parameter


class Layer:
    """A single layer of a superconducting device.

    You can provide either an effective penetration depth ``Lambda``,
    or both a London penetration depth (``lambda_london``) and a layer
    ``thickness``. ``Lambda`` and ``london_lambda`` can either be real numers or
    instances of :class:`superscreen.Parameter` which compute the penetration
    depth as a function of position.

    Args:
        name: Name of the layer.
        Lambda: The effective magnetic penetration depth of the superconducting
            film(s) in the layer.
        thickness: Thickness of the superconducting film(s) located in the layer.
        london_lambda: London penetration depth of the superconducting film(s)
            located in the layer.
        z0: Vertical location of the layer.
    """

    __slots__ = ("name", "thickness", "london_lambda", "z0", "_Lambda")

    def __init__(
        self,
        name: str,
        Lambda: Union[float, Parameter, None] = None,
        london_lambda: Union[float, Parameter, None] = None,
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
        return self.london_lambda**2 / self.thickness

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
            f"{self.__class__.__name__}({self.name!r}, Lambda={Lambda}, "
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

    def to_hdf5(self, h5group: h5py.Group) -> None:
        h5group.attrs["name"] = self.name
        h5group.attrs["z0"] = self.z0
        if self.thickness is not None:
            h5group.attrs["thickness"] = self.thickness
        if self.london_lambda is not None:
            serialize_obj(h5group, self.london_lambda, "london_lambda", attr=True)
        else:
            serialize_obj(h5group, self.Lambda, "Lambda", attr=True)

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "Layer":
        name = h5group.attrs["name"]
        z0 = h5group.attrs["z0"]
        Lambda = london_lambda = None
        thickness = h5group.attrs.get("thickness", None)
        if "london_lambda" in h5group.attrs:
            london_lambda = h5group.attrs["london_lambda"]
        elif "london_lambda.pickle" in h5group.attrs:
            london_lambda = deserialize_obj(h5group, "london_lambda", attr=True)
        elif "Lambda" in h5group.attrs:
            Lambda = h5group.attrs["Lambda"]
        else:
            Lambda = deserialize_obj(h5group, "Lambda", attr=True)
        return Layer(
            name,
            Lambda=Lambda,
            london_lambda=london_lambda,
            thickness=thickness,
            z0=z0,
        )
