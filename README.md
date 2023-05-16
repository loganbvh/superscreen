# SuperScreen

![SuperScreen Logo](docs/images/logo_currents_small.png)

`SuperScreen` is a Python package for simulating the magnetic response of thin film superconducting devices. `SuperScreen` solves the coupled Maxwell's and London equations on a triangular mesh using a matrix inversion method described in the following paper:

>SuperScreen: An open-source package for simulating the magnetic response of two-dimensional superconducting devices, Computer Physics Communications, Volume 280, 2022, 108464 [https://doi.org/10.1016/j.cpc.2022.108464](https://doi.org/10.1016/j.cpc.2022.108464).

The accepted version of the paper can also be found on arXiv: [arXiv:2203.13388](https://doi.org/10.48550/arXiv.2203.13388).

![PyPI](https://img.shields.io/pypi/v/superscreen) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/loganbvh/superscreen/lint-and-test.yml?branch=main) [![Documentation Status](https://readthedocs.org/projects/superscreen/badge/?version=latest)](https://superscreen.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/loganbvh/superscreen/branch/main/graph/badge.svg?token=XW7LSY8WVD)](https://codecov.io/gh/loganbvh/superscreen) ![GitHub](https://img.shields.io/github/license/loganbvh/superscreen) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![DOI](https://zenodo.org/badge/376110557.svg)](https://zenodo.org/badge/latestdoi/376110557)

## Learn `SuperScreen`

The documentation for `SuperScreen` can be found at [superscreen.readthedocs.io](https://superscreen.readthedocs.io/en/latest/).

## Try `SuperScreen`

Click the badge below to try `SuperScreen` interactively online via [Google Colab](https://colab.research.google.com/):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loganbvh/superscreen/blob/main/docs/notebooks/quickstart.ipynb)

## Install `SuperScreen`

`SuperScreen` requires `python >=3.8, <3.12`. We recommend installing `SuperScreen` in a fresh [`conda` environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For more details, see the [documentation](https://superscreen.readthedocs.io/en/latest/).

### Install via `pip`

From [PyPI](https://pypi.org/project/superscreen/), the Python Package Index:

```bash
pip install superscreen
```

From this [GitHub repository](https://github.com/loganbvh/superscreen/):

```bash
pip install git+https://github.com/loganbvh/superscreen.git
```

### Developer installation

```bash
git clone https://github.com/loganbvh/superscreen.git
cd superscreen
pip install -e .
```

## About `SuperScreen`

### Authors

- Primary author and maintainer: [@loganbvh](https://github.com/loganbvh/).

### Contributing

Want to contribute to `SuperScreen`? Check out our [contribution guidelines](CONTRIBUTING.md).

### BibTeX citation

Please cite this paper if you use `SuperScreen` in your research.

    @article{
        Bishop-Van_Horn2022-sy,
        title    = "{SuperScreen}: An open-source package for simulating the magnetic
                    response of two-dimensional superconducting devices",
        author   = "Bishop-Van Horn, Logan and Moler, Kathryn A",
        journal  = "Comput. Phys. Commun.",
        volume   =  280,
        pages    = "108464",
        month    =  nov,
        year     =  2022,
        url      = "https://doi.org/10.1016/j.cpc.2022.108464"
    }
