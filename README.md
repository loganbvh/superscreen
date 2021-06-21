# SuperScreen

SuperScreen is a package for simulating Meissner screening in multiplanar thin film
superconducting devices. SuperScreen solves the coupled Maxwell's and London equations
on a triangular mesh using a matrix inversion method described in the following references:

1. Phys. Rev. B 72, 024529 (2005) [Brandt].
2. Rev. Sci. Instrum. 87, 093702 (2016). [Kirtley1]
3. Supercond. Sci. Technol. 29 (2016) 124001. [Kirtley2].

Pre-prints of the above references can be found in the [references](docs/references) directory
in this repo.

## Installing from source

- Clone or download this repository
- Create a new conda environment (Python >= 3.6):
    - `conda env create superscreen python=3.9`
    - `conda activate superscreen`
    - `pip install -e .` in the superscreen directory
