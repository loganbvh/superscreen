"""
# SuperScreen

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/loganbvh/superscreen/lint-and-test.yml?branch=main) [![Documentation Status](https://readthedocs.org/projects/superscreen/badge/?version=latest)](https://superscreen.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/loganbvh/superscreen/branch/main/graph/badge.svg?token=XW7LSY8WVD)](https://codecov.io/gh/loganbvh/superscreen) ![GitHub](https://img.shields.io/github/license/loganbvh/superscreen) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![DOI](https://zenodo.org/badge/376110557.svg)](https://zenodo.org/badge/latestdoi/376110557)

`SuperScreen` is a Python package for simulating the magnetic response of thin film superconducting devices. `SuperScreen` solves the coupled Maxwell's and London equations on a triangular mesh using a matrix inversion method described in the following paper:

>SuperScreen: An open-source package for simulating the magnetic response of two-dimensional superconducting devices, Computer Physics Communications, Volume 280, 2022, 108464 [https://doi.org/10.1016/j.cpc.2022.108464](https://doi.org/10.1016/j.cpc.2022.108464).

The accepted version of the paper can also be found on arXiv: [arXiv:2203.13388](https://doi.org/10.48550/arXiv.2203.13388). The GitHub repository accompanying the paper can be found [here](https://github.com/loganbvh/superscreen-paper).

## Learn `SuperScreen`

The documentation for `SuperScreen` can be found at [superscreen.readthedocs.io](https://superscreen.readthedocs.io/en/latest/).

## Try `SuperScreen`

Click the badge below to try `SuperScreen` interactively online via [Google Colab](https://colab.research.google.com/):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loganbvh/superscreen/blob/main/docs/notebooks/quickstart.ipynb)

"""

from setuptools import find_packages, setup

DESCRIPTION = "SuperScreen: simulate Meissner screening in 2D superconducting devices."
LONG_DESCRIPTION = __doc__

NAME = "superscreen"
AUTHOR = "Logan Bishop-Van Horn"
AUTHOR_EMAIL = "logan.bvh@gmail.com"
URL = "https://github.com/loganbvh/superscreen"
LICENSE = "MIT"
PYTHON_VERSION = ">=3.8, <3.12"

INSTALL_REQUIRES = [
    "dill",
    "h5py",
    "ipython",
    "joblib",
    "jupyter",
    "matplotlib",
    "meshpy",
    "numba",
    "numpy",
    "pint",
    "pytest",
    "scipy",
    "shapely",
    "tqdm",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest-cov",
        "pre-commit",
    ],
    "docs": [
        "sphinx<7",
        "sphinx_rtd_theme",
        "sphinx-autodoc-typehints",
        "nbsphinx",
        "pillow",  # required for image scaling in RTD
    ],
}

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Physics
"""

CLASSIFIERS = [line for line in CLASSIFIERS.splitlines() if line]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
KEYWORDS = "superconductor meissner screening"

exec(open("superscreen/version.py").read())

setup(
    name=NAME,
    version=__version__,  # noqa: F821
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    python_requires=PYTHON_VERSION,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
