import pytest

from IPython.display import HTML

from superscreen.about import version_dict, version_table


def test_version_dict():
    d = version_dict()
    assert isinstance(d, dict)


@pytest.mark.parametrize("verbose", [False, True])
def test_version_table(verbose):
    html = version_table(verbose)
    assert isinstance(html, HTML)