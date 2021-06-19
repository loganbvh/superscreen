.. SuperScreen documentation master file, created by
   sphinx-quickstart on Fri Jun 18 16:11:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********
SuperScreen
***********

``SuperScreen`` is an open-source Python package for simulating Meissner screening
in 2D superconductors and multiplanar superconducting devices of arbitrary geometry.
Using a matrix inversion method introduced by Brandt [ref], ``SuperScreen`` solves
the coupled London's and Maxwell's equations in and around superconducting films
with spatially-varying penetration depth in the presence of applied magnetic fields
and circulating currents.

``SuperScreen`` can be used to calculcate:

- The self-inductance of thin film superconducting structures
- The mutual-inductance between different thin film superconducting structures
- Flux-focusing and Meissner screening effects in thin film superconducting devices

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation.rst
   brandt.rst

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/device.rst
   api/fem.rst
   api/solver.rst
   api/visualization.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
