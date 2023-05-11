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


Supporting Functions
--------------------

.. autofunction:: superscreen.solver.solve_film.solve_film

.. autofunction:: superscreen.solver.convert_field

.. autofunction:: superscreen.solver.field_conversion_factor
