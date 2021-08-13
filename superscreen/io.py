import os
import json
import datetime
from typing import Union, List, Dict

import numpy as np

from .solution import Solution

from backports.datetime_fromisoformat import MonkeyPatch

MonkeyPatch.patch_fromisoformat()


class NullContextManager(object):
    """Does nothing."""

    def __init__(self, resource=None):
        self.resource = resource

    def __enter__(self):
        return self.resource

    def __exit__(self, *args):
        pass


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # scalar complex values only
        if isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, (np.void,)):
            return None

        # float, int, etc.
        if isinstance(obj, (np.generic,)):
            return obj.item()

        if isinstance(obj, datetime.datetime):
            return obj.isoformat()

        return super().default(self, obj)


def json_numpy_obj_hook(d: Dict) -> Dict:
    if set(d.keys()) == {"real", "imag"}:
        return complex(d["real"], d["imag"])

    for key, value in d.items():
        if isinstance(value, str) and len(value) >= 26:
            try:
                d[key] = datetime.datetime.fromisoformat(value)
            except ValueError:
                pass
    return d


def save_solutions(
    solutions: List[Solution],
    base_directory: str,
    save_mesh: bool = True,
    return_paths: bool = False,
) -> Union[None, List[str]]:
    """Saves a list of Solutions to disk.

    Args:
        base_directory: The name of the directory in which to save the solutions
            (must either be empty or not yet exist).

    Returns:
        If ``return_paths`` is True, returns a list of paths where each solution
        was saved.
    """
    if os.path.isdir(base_directory) and len(os.listdir(base_directory)):
        raise IOError(f"Directory '{base_directory}' already exists and is not empty.")
    os.makedirs(base_directory, exist_ok=True)

    paths = []
    for i, solution in enumerate(solutions):
        path = os.path.join(base_directory, str(i))
        solution.to_file(path, save_mesh=save_mesh)
        paths.append(os.path.abspath(path))

    if return_paths:
        return paths


def load_solutions(base_directory: str) -> List[Solution]:
    """Loads a sequence of Solutions from disk.

    Args:
        base_directory: The name of the directory from which to load the solutions.

    Returns:
        A list of Solutions
    """
    solutions = []
    for subdir in sorted(os.listdir(base_directory), key=int):
        path = os.path.join(base_directory, subdir)
        solutions.append(Solution.from_file(path))
    return solutions


def iload_solutions(base_directory: str) -> Solution:
    """A generator that loads a sequence of Solutions from disk.

    Args:
        base_directory: The name of the directory from which to load the solutions.

    Yields:
        Solution instances loaded from ``base_directory``
    """
    for subdir in sorted(os.listdir(base_directory), key=int):
        yield Solution.from_file(os.path.join(base_directory, subdir))
