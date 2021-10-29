import os
import shutil
import json
import datetime
import tempfile
from typing import Union, List, Dict, Iterator

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


def zip_solution(solution: Solution, directory: os.PathLike) -> str:
    """Save a Solution to a zip archive in the given directory.

    Args:
        Solution: The Solution to save.
        directory: The directory in which to save the Solution.

    Returns:
        The absolute path to the created zip file.
    """
    path = os.path.abspath(directory)
    solution.to_file(path)
    try:
        zip_name = shutil.make_archive(path, "zip", root_dir=path)
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path)
    return zip_name


def unzip_solution(path: os.PathLike) -> Solution:
    """Load a solution from a zip file.

    Args:
        path: The path to the zip file.

    Returns:
        The loaded Solution.
    """
    if not path.endswith(".zip"):
        path += ".zip"
    with tempfile.TemporaryDirectory() as extract_dir:
        shutil.unpack_archive(path, extract_dir=extract_dir)
        return Solution.from_file(extract_dir)


def save_solutions(
    solutions: List[Solution],
    base_directory: os.PathLike,
    save_mesh: bool = True,
    return_paths: bool = False,
    to_zip: bool = False,
) -> Union[None, List[os.PathLike]]:
    """Saves a list of Solutions to disk.

    Args:
        base_directory: The name of the directory in which to save the solutions
            (must either be empty or not yet exist).
        save_mesh: Whether to save the full mesh.
        return_paths: Whether to return a list of resulting paths.
        to_zip: Whether to save Solutions as zip archives.

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
        if to_zip:
            _ = zip_solution(solution, path)
        else:
            solution.to_file(path, save_mesh=save_mesh)
        paths.append(os.path.abspath(path))

    if return_paths:
        return paths


def iload_solutions(base_directory: os.PathLike) -> Iterator[Solution]:
    """An iterator that loads a sequence of Solutions from disk.

    Args:
        base_directory: The name of the directory from which to load the solutions.

    Yields:
        Solution instances loaded from ``base_directory``
    """
    paths = sorted(os.listdir(base_directory), key=lambda s: int(s.split(".")[0]))
    for path in paths:
        yield Solution.from_file(os.path.join(base_directory, path))


def load_solutions(base_directory: os.PathLike) -> List[Solution]:
    """Loads a sequence of Solutions from disk.

    Args:
        base_directory: The name of the directory from which to load the solutions.

    Returns:
        A list of Solutions
    """
    return list(iload_solutions(base_directory))
