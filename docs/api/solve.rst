.. superscreen

.. _api-solve:


******
Solver
******

The ``superscreen.solve`` module contains the actual implementation
of Brandt's method, as described :ref:`here <background>`.

.. contents::
    :depth: 2

Solve
-----

.. autofunction:: superscreen.solve

.. autoclass:: superscreen.Vortex

Pre-factorizing models
----------------------

.. autofunction:: superscreen.factorize_model

.. autoclass:: superscreen.FactorizedModel
    :members:

Solution
--------

.. autoclass:: superscreen.Solution
    :members:

.. autoclass:: superscreen.FilmSolution
    :members:

Fluxoid
-------

.. autoclass:: superscreen.Fluxoid
    :show-inheritance:

.. autofunction:: superscreen.fluxoid.make_fluxoid_polygons

.. autofunction:: superscreen.fluxoid.find_fluxoid_solution


Supporting functions
--------------------

.. autofunction:: superscreen.solver.solve_film

.. autofunction:: superscreen.solver.solve_for_terminal_current_stream

.. autofunction:: superscreen.solver.factorize_linear_systems

.. autofunction:: superscreen.solver.convert_field

.. autofunction:: superscreen.solver.field_conversion_factor


Supporting classes
------------------

.. autoclass:: superscreen.solver.FilmInfo
    :members:

.. autoclass:: superscreen.solver.LambdaInfo
    :members:

.. autoclass:: superscreen.solver.LinearSystem
    :members:

.. autoclass:: superscreen.solver.TerminalSystems
    :members:
