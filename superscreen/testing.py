# This file is part of superscreen.
#
#     Copyright (c) 2021 Logan Bishop-Van Horn
#
#     This source code is licensed under the MIT license found in the
#     LICENSE file in the root directory of this source tree.

import os
import pytest

from superscreen.visualization import non_gui_backend


TESTDIR = os.path.join(
    os.path.pardir, os.path.dirname(os.path.abspath(__file__)), "test"
)


def run():
    with non_gui_backend():
        pytest.main(["-v", TESTDIR])


if __name__ == "__main__":
    run()
