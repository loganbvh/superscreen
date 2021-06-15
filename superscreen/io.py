import json

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # scalar complex values only
        elif isinstance(obj, (complex, np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, np.void):
            return None

        # float, int, etc.
        elif isinstance(obj, np.generic):
            return obj.item()

        return super().default(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and set(dct.keys()) == {"real", "imag"}:
        return complex(dct["real"], dct["imag"])
    return dct
