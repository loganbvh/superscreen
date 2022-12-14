.. superscreen

.. _api-solve:


******
Solver
******

The ``superscreen.solve`` module contains the actual implementation
of Brandt's method, as described :ref:`here <background>`.

Solve
-----

.. autofunction:: superscreen.solve

Solve Many
----------

.. autofunction:: superscreen.solve_many


Solution
--------

.. autoclass:: superscreen.Solution
    :members:

Fluxoid
-------

.. autoclass:: superscreen.Fluxoid
    :show-inheritance:

.. autoclass:: superscreen.Vortex
    :show-inheritance:

.. autofunction:: superscreen.fluxoid.make_fluxoid_polygons

.. autofunction:: superscreen.fluxoid.find_fluxoid_solution


IO
--

.. autofunction:: superscreen.io.save_solutions

.. autofunction:: superscreen.io.load_solutions

.. autofunction:: superscreen.io.iload_solutions

.. autofunction:: superscreen.io.zip_solution

.. autofunction:: superscreen.io.unzip_solution


Supporting Functions
--------------------

Brandt Core
===========

.. autofunction:: superscreen.solve.solve_layer

.. autofunction:: superscreen.solve.q_matrix

.. autofunction:: superscreen.solve.C_vector

.. autofunction:: superscreen.solve.Q_matrix

.. autofunction:: superscreen.solve.convert_field

.. autofunction:: superscreen.solve.field_conversion_factor


Parallel Processing
===================

.. autofunction:: superscreen.parallel.create_models

.. autofunction:: superscreen.parallel.solve_many_serial

.. autofunction:: superscreen.parallel.solve_many_mp

.. autofunction:: superscreen.parallel.solve_many_ray
