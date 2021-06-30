# This file is part of superscreen.

#     Copyright (c) 2021 Logan Bishop-Van Horn

#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import os
import json
from typing import Union, List

import numpy as np

from .solution import BrandtSolution


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # scalar complex values only
        elif isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.void,)):
            return None

        # float, int, etc.
        elif isinstance(obj, (np.generic,)):
            return obj.item()

        return super().default(self, obj)


def json_numpy_obj_hook(item):
    if isinstance(item, dict) and set(item.keys()) == {"real", "imag"}:
        return complex(item["real"], item["imag"])
    return item


def save_solutions(
    solutions: List[BrandtSolution],
    base_directory: str,
    return_paths: bool = False,
) -> Union[None, List[str]]:
    """Saves a list of BrandtSolutions to disk.

    Args:
        base_directory: The name of the directory in which to save the solutions
            (must either be empty or not yet exist).

    Returns:
        If ``return_paths`` is True, returns a list of paths where each solution
        was saved.
    """
    if os.path.isdir(base_directory) and len(os.listdir(base_directory)):
        raise IOError(f"Directory '{base_directory}' already exists and is not empty.")
    else:
        os.makedirs(base_directory)

    paths = []
    for i, solution in enumerate(solutions):
        path = os.path.join(base_directory, str(i))
        solution.to_file(path)
        paths.append(os.path.abspath(path))

    if return_paths:
        return paths


def load_solutions(base_directory: str) -> List[BrandtSolution]:
    """Loads a sequence of BrandtSolutions from disk.

    Args:
        base_directory: The name of the directory from which to load the solutions.

    Returns:
        A list of BrandtSolutions
    """
    solutions = []
    for subdir in sorted(os.listdir(base_directory), key=int):
        path = os.path.join(base_directory, subdir)
        solutions.append(BrandtSolution.from_file(path))
    return solutions
