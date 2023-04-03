**********
Change Log
**********

View release history on `PyPI <https://pypi.org/project/superscreen/#history>`_,
`GitHub <https://github.com/loganbvh/superscreen/releases>`_, or `Zenodo <https://zenodo.org/badge/latestdoi/376110557>`_.

.. note::

    ``SuperScreen`` uses `semantic versioning <https://semver.org/>`_, with version numbers specified as
    ``MAJOR.MINOR.PATCH``. In particular, note that:

    - Major version zero (0.y.z) is for initial development. Anything MAY change at any time.
      The public API SHOULD NOT be considered stable.
    - Version 1.0.0 defines the public API.

----

Version 0.8.1
-------------

Release date: 2023-04-03

Changes
=======

- Evaluating the magnetic field within a ``Layer`` is no longer supported in :meth:`superscreen.Solution.field_at_position` and must be done using :meth:`superscreen.Solution.interp_fields` (`#91 <https://github.com/loganbvh/superscreen/pull/91>`_).

Version 0.8.0
-------------

Release date: 2022-12-15

Changes
=======

- Removed dependency on ``optimesh``, as it is not longer open source.

  - :meth:`superscreen.Device.make_mesh` and :meth:`superscreen.Polygon.make_mesh` now take an integer argument, ``smooth``, which specifies the number of Laplacian mesh smoothing iterations to perform.


Version 0.7.0
-------------

Release date: 2022-08-29

Changes
=======

- Added support for GPU-acceleration in :func:`superscreen.solve`, with `JAX <https://github.com/google/jax>`_
  as an optional dependency (`#75 <https://github.com/loganbvh/superscreen/pull/75>`_).
- Added :class:`superscreen.TransportDevice`, a subclass of :class:`superscreen.Device` on which one can define
  source/drain terminals for current biasing (`#78 <https://github.com/loganbvh/superscreen/pull/78>`_).
- Updated :meth:`superscreen.Solution.field_at_position` to use the 2D Biot-Savart directly,
  via :meth:`superscreen.sources.biot_savart_2d` (`#78 <https://github.com/loganbvh/superscreen/pull/78>`_).
- Updated :meth:`superscreen.fluxoid.find_fluxoid_solution` to use the mutual inductance matrix to solve for fluxoid states,
  which is much more efficient than the previous least-squares method, especially for multiple holes
  (`#78 <https://github.com/loganbvh/superscreen/pull/78>`_).

Version 0.6.1
-------------

Release date: 2022-07-02

Changes
=======

- Fixed an off-by-one error in the :math:`k`-space coordinates used to
  calculate the field from a Pearl vortex in :meth:`superscreen.sources.pearl_vortex` (`#74 <https://github.com/loganbvh/superscreen/pull/74>`_).

Version 0.6.0
-------------

Release date: 2022-05-20.

**Note**: On GitHub, this version was accidentally tagged as ``v0.6.6`` rather than ``v0.6.0``
(see `here <https://github.com/loganbvh/superscreen/releases/tag/v0.6.6>`_.)

Changes
=======

- Added ``Solution.vector_potential_at_position()`` (`#73 <https://github.com/loganbvh/superscreen/pull/73>`_).

----

Version 0.5.0
-------------

Release date: 2022-04-13

Changes
=======

- Added :math:`\vec{\nabla}\Lambda(x, y)` term and clarified documentation about the model in the context of inhomogeneous films
  (`#72 <https://github.com/loganbvh/superscreen/pull/72>`_).

----

Version 0.4.0
-------------

Release date: 2022-03-15

Changes
=======

- Remove support for Python 3.6, which has reached `end-of-life <https://www.python.org/downloads/release/python-3615/>`_
  (`#69 <https://github.com/loganbvh/superscreen/pull/69>`_).

----

Version 0.3.0
-------------

Release date: 2022-01-27

Changes
=======

- Use ``__slots__`` for ``Layers``, ``Polygons``, and ``Parameters`` (`#57 <https://github.com/loganbvh/superscreen/pull/57>`_).
- Add affine transformations for ``Polygon`` and ``Device``
  (`#59 <https://github.com/loganbvh/superscreen/pull/60>`_, `#60 <https://github.com/loganbvh/superscreen/pull/60>`_).
- Allow ``Parameters`` to return scalar or vector quantities (`# 61 <https://github.com/loganbvh/superscreen/pull/61>`_).
- Allow explicitly setting ``num_cpus`` in ``solve_many()`` (`#62 <https://github.com/loganbvh/superscreen/pull/62>`_).
- Add ``SheetCurrentField`` source and move mesh generation into its own module to enable ``Polygon.make_mesh()``
  (`#65 <https://github.com/loganbvh/superscreen/pull/65>`_).
- Use ``scipy.linalg.lu_solve()`` in ``superscreen.solve()`` instead of ``numpy.linalg.inv()`` (`#67 <https://github.com/loganbvh/superscreen/pull/67>`_).

----

Version 0.2.0 (initial development release)
-------------------------------------------

Release date: 2021-11-28
