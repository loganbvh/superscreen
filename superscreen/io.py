from typing import Any

import dill
import h5py
import numpy as np


def serialize_obj(group: h5py.Group, obj: Any, name: str, attr: bool = False) -> None:
    """Serialize an arbitrary object to an :class:`h5py.Group`.

    Args:
        group: The :class:`h5py.Group` in which to save the object
        obj: The Python object to serialize
        name: The name to give the object in the :class:`h5py.Group`
        attr: Save the object as a group attribute if possbible.
    """
    if attr:
        try:
            group.attrs[name] = obj
        except TypeError:
            group.attrs[f"{name}.pickle"] = np.void(dill.dumps(obj))
    else:
        group[f"{name}.pickle"] = np.void(dill.dumps(obj))


def deserialize_obj(group: h5py.Group, name: str, attr: bool = False) -> Any:
    """Deserialize an object from an :class:`h5py.Group`.

    Args:
        group: The :class:`h5py.Group` from which to load the object
        name: The name of the object in the :class:`h5py.Group`
        attr: Whether the object was serialized as a group attribute

    Returns:
        The deserialized object
    """
    if attr:
        if name in group.attrs:
            return group.attrs[name]
        if f"{name}.pickle" in group.attrs:
            return dill.loads(np.void(group.attrs[f"{name}.pickle"]).tobytes())
    elif f"{name}.pickle" in group:
        return dill.loads(np.void(group[f"{name}.pickle"]).tobytes())
    raise IOError(f"Unable to load {name}.")
