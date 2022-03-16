# SuperScreen

![SuperScreen Logo](docs/images/logo_currents_small.png)

SuperScreen is a package for simulating the magnetic response of multi-planar thin film
superconducting devices. SuperScreen solves the coupled Maxwell's and London equations
on a triangular mesh using a matrix inversion method described in the following references:

1. Phys. Rev. B 72, 024529 (2005) [[arXiv:cond-mat/0506144](https://arxiv.org/abs/cond-mat/0506144)]
2. Rev. Sci. Instrum. 87, 093702 (2016) [[arXiv:1605.09483](https://arxiv.org/abs/1605.09483)]
3. Supercond. Sci. Technol. 29 (2016) 124001 [[arXiv:1607.03950](https://arxiv.org/abs/1607.03950)]

See the [documentation](https://superscreen.readthedocs.io/en/latest/) for more details.

![PyPI](https://img.shields.io/pypi/v/superscreen) ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/loganbvh/superscreen/lint-and-test/main) [![Documentation Status](https://readthedocs.org/projects/superscreen/badge/?version=latest)](https://superscreen.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/loganbvh/superscreen/branch/main/graph/badge.svg?token=XW7LSY8WVD)](https://codecov.io/gh/loganbvh/superscreen) ![GitHub](https://img.shields.io/github/license/loganbvh/superscreen) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![DOI](https://zenodo.org/badge/376110557.svg)](https://zenodo.org/badge/latestdoi/376110557)

## Learn `SuperScreen`

The documentation for `SuperScreen` can be found at [superscreen.readthedocs.io](https://superscreen.readthedocs.io/en/latest/).

## Try `SuperScreen`

Click the badge below and navigate to `docs/notebooks/` to try `SuperScreen` interactively online via [Binder](https://mybinder.org/):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loganbvh/superscreen/HEAD)

## Install `SuperScreen`

`SuperScreen` requires `python >=3.7, <3.10`. We recommend installing `SuperScreen` in a fresh [`conda` environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). For more details, see the
[documentation](https://superscreen.readthedocs.io/en/latest/).

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

Want to contribute to `superscreen`? Check out our [contribution guidelines](CONTRIBUTING.md).
