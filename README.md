# SuperScreen

SuperScreen is a package for simulating the magnetic response of multiplanar thin film
superconducting devices. SuperScreen solves the coupled Maxwell's and London equations
on a triangular mesh using a matrix inversion method described in the following references:

1. Phys. Rev. B 72, 024529 (2005) [[arXiv:cond-mat/0506144](https://arxiv.org/abs/cond-mat/0506144)]
2. Rev. Sci. Instrum. 87, 093702 (2016) [[arXiv:1605.09483](https://arxiv.org/abs/1605.09483)]
3. Supercond. Sci. Technol. 29 (2016) 124001 [[arXiv:1607.03950](https://arxiv.org/abs/1607.03950)]

See the [documentation](https://superscreen.readthedocs.io/en/latest/) for more details.

![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/loganbvh/superscreen/lint-and-test/main) [![Documentation Status](https://readthedocs.org/projects/superscreen/badge/?version=latest)](https://superscreen.readthedocs.io/en/latest/?badge=latest) ![GitHub](https://img.shields.io/github/license/loganbvh/superscreen) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Learn `SuperScreen`

The documentation for `SuperScreen` can be found at [superscreen.readthedocs.io](https://superscreen.readthedocs.io/en/latest/).

## Try `SuperScreen`

Click the badge below to try `SuperScreen` interactively online via [Binder](https://mybinder.org/):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loganbvh/superscreen/HEAD?filepath=docs%2Fnotebooks)

(Note: Binder instances are [limited to 2 GB of memory](https://mybinder.readthedocs.io/en/latest/about/about.html#how-much-memory-am-i-given-when-using-binder), so you can only solve models with up to approximately
10,000 triangles online.)

## Install `SuperScreen`

`SuperScreen` requires `python >= 3.7`. For more details, see the
[documentation](https://superscreen.readthedocs.io/en/latest/).

### Installing from source

- Clone or download this repository
- `pip install -e .` in the superscreen directory
