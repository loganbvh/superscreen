.. SuperScreen documentation master file, created by
   sphinx-quickstart on Fri Jun 18 16:11:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

***********
SuperScreen
***********

.. image:: images/logo_currents.png
   :alt: SuperScreen Logo
   :target: https://github.com/loganbvh/superscreen
   :height: 200px
   :align: center

.. image:: https://img.shields.io/pypi/v/superscreen
   :target: https://pypi.org/project/superscreen/
   :alt: PyPI

.. image:: https://img.shields.io/github/actions/workflow/status/loganbvh/superscreen/lint-and-test.yml?branch=main
   :alt: GitHub Workflow Status

.. image:: https://readthedocs.org/projects/superscreen/badge/?version=latest
   :target: https://superscreen.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/loganbvh/superscreen/branch/main/graph/badge.svg?token=XW7LSY8WVD
   :target: https://codecov.io/gh/loganbvh/superscreen
   :alt: Test Coverage

.. image:: https://img.shields.io/github/license/loganbvh/superscreen
   :alt: GitHub

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://zenodo.org/badge/376110557.svg
   :target: https://zenodo.org/badge/latestdoi/376110557


`SuperScreen <https://github.com/loganbvh/superscreen>`_ is an open-source Python package for simulating
the magnetic response of 2D superconductors and thin film superconducting devices of arbitrary geometry.
Using a matrix inversion method introduced by Brandt [Brandt-PRB-2005]_, ``SuperScreen`` solves
the coupled London and Maxwell's equations in and around superconducting thin films in the presence of
inhomogeneous applied magnetic fields, trapped flux, and bias currents.

.. tip::

   ``SuperScreen`` is described in detail in the following paper:

      SuperScreen: An open-source package for simulating the magnetic response of two-dimensional superconducting devices,
      Computer Physics Communications, Volume 280, 2022, 108464
      `https://doi.org/10.1016/j.cpc.2022.108464 <https://doi.org/10.1016/j.cpc.2022.108464>`_.

   The accepted version of the paper can also be found on arXiv: `arXiv:2203.13388 <https://doi.org/10.48550/arXiv.2203.13388>`_.

``SuperScreen`` can be used to calculate:

- Self- and mutual-inductances in thin film superconducting structures
- Flux-focusing and Meissner screening effects in superconducting devices
- Vector magnetic fields resulting from currents in 2D superconducting structures

Click :ref:`here <background>` to read more about the way ``SuperScreen`` solves
models of superconducting devices.

For quick demonstration of ``SuperScreen``, see the :ref:`quickstart notebook </notebooks/quickstart.ipynb>`.
Better yet, click the badge below and navigate to ``docs/notebooks/`` to try ``SuperScreen``
interactively online via `Binder <https://mybinder.org/>`_:


.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/loganbvh/superscreen/HEAD

If you use ``SuperScreen`` in your research, please cite the paper linked above.

.. code-block::

   % BibTeX citation
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
   

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation.rst
   notebooks/quickstart.ipynb
   background.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/polygons.ipynb
   notebooks/terminal-currents.ipynb
   notebooks/field-sources.ipynb
   notebooks/logo.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/device.rst
   api/solve.rst
   api/fem.rst
   api/visualization.rst
   api/sources.rst

.. toctree::
   :maxdepth: 2
   :caption: About SuperScreen

   about/changelog.rst
   about/license.rst
   about/contributing.rst
   about/references.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
