.. SuperScreen documentation master file, created by
   sphinx-quickstart on Fri Jun 18 16:11:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********
SuperScreen
***********

.. image:: https://img.shields.io/github/workflow/status/loganbvh/superscreen/lint-and-test/main
   :alt: GitHub Workflow Status (branch)

.. image:: https://readthedocs.org/projects/superscreen/badge/?version=latest
   :target: https://superscreen.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/loganbvh/superscreen
   :alt: GitHub

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black


``SuperScreen`` is an open-source Python package for simulating Meissner screening
in 2D superconductors and multiplanar superconducting devices of arbitrary geometry.
Using a matrix inversion method introduced by Brandt [Brandt-PRB-2005]_, ``SuperScreen`` solves
the coupled London's and Maxwell's equations in and around superconducting films
with spatially-varying penetration depth in the presence of applied magnetic fields
and circulating currents.

``SuperScreen`` can be used to calculcate:

- Self- and mutual-inductances in thin film superconducting structures
- Flux-focusing and Meissner screening effects in superconducting devices
- Vector magnetic fields resulting from currents in 2D superconducting structures

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation.rst
   background.rst
   notebooks/quickstart.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Users Guide

   notebooks/guide-device-meshing.ipynb
   notebooks/guide-self-inductance.ipynb
   notebooks/guide-mutual-inductance.ipynb
   notebooks/guide-parameters.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/device.rst
   api/fem.rst
   api/brandt.rst
   api/visualization.rst

.. toctree::
   :maxdepth: 2
   :caption: About SuperScreen

   license.rst
   references.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
