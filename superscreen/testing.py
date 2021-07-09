# This file is part of superscreen.
#
#     Copyright (c) 2021 Logan Bishop-Van Horn
#
#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import os
import pytest
import warnings
import matplotlib


TESTDIR = os.path.join(
    os.path.pardir, os.path.dirname(os.path.abspath(__file__)), "test"
)


def run():
    # We want to temporarily use a non-GUI backend to avoid
    # spamming the user's screen with a bunch of plots.
    # Matplotlib may raise a UserWarning when using a non-GUI backend...
    with warnings.catch_warnings():
        old_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Matplotlib is currently using agg"
        )
        # Run the tests
        pytest.main(["-v", TESTDIR])
        matplotlib.pyplot.close("all")
        matplotlib.use(old_backend)


if __name__ == "__main__":
    run()
