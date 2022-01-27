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

Version 0.2.0 (initial development release)
-------------------------------------------

Release date: 2021-11-28.
