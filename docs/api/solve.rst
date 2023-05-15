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

.. autoclass:: superscreen.FilmSolution
    :members:

Fluxoid
-------

.. autoclass:: superscreen.Fluxoid
    :show-inheritance:

.. autoclass:: superscreen.Vortex

.. autofunction:: superscreen.fluxoid.make_fluxoid_polygons

.. autofunction:: superscreen.fluxoid.find_fluxoid_solution


Supporting functions
--------------------

.. autofunction:: superscreen.solver.solve_film

.. autofunction:: superscreen.solver.solve_for_solve_for_terminal_current_stream

.. autofunction:: superscreen.solver.factorize_model

.. autofunction:: superscreen.solver.factorize_linear_systems

.. autofunction:: superscreen.solver.convert_field

.. autofunction:: superscreen.solver.field_conversion_factor


Supporting classes
------------------

.. autoclass:: superscreen.solver.FactorizedModel
    :members:

.. autoclass:: superscreen.solver.FilmInfo
    :members:

.. autoclass:: superscreen.solver.LambdaInfo
    :members:

.. autoclass:: superscreen.solver.LinearSystem
    :members:

.. autoclass:: superscreen.solver.TerminalSystems
    :members:
