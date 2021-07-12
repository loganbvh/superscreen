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
