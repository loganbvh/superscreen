from typing import Any

import dill
import h5py
import numpy as np


def serialize_obj(group: h5py.Group, obj: Any, name: str, attr: bool = False) -> None:
    if attr:
        try:
            group.attrs[name] = obj
        except TypeError:
            group.attrs[f"{name}.pickle"] = np.void(dill.dumps(obj))
    else:
        group[f"{name}.pickle"] = np.void(dill.dumps(obj))


def deserialize_obj(group: h5py.Group, name: str, attr: bool = False) -> Any:
    if attr:
        if name in group.attrs:
            return group.attrs[name]
        if f"{name}.pickle" in group:
            return dill.loads(np.void(group[f"{name}.pickle"]).tobytes())
        raise IOError(f"Unable to load {name}.")
    return dill.loads(np.void(group[f"{name}.pickle"]).tobytes())
