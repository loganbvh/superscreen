.. superscreen

.. _api-brandt:


******
Solver
******

The ``superscreen.brandt`` module contains the actual implementation
of Brandt's method, as described :ref:`here <background>`.

Brandt Solver
-------------

.. autofunction:: superscreen.brandt.solve

.. autofunction:: superscreen.brandt.brandt_layer

Brandt Solution
---------------

.. autoclass:: superscreen.solution.BrandtSolution
    :members:

.. autofunction:: superscreen.io.save_solutions

.. autofunction:: superscreen.io.load_solutions

Supporting Functions
--------------------

.. autofunction:: superscreen.brandt.convert_field

.. autofunction:: superscreen.brandt.field_conversion_factor

.. autofunction:: superscreen.brandt.q_matrix

.. autofunction:: superscreen.brandt.C_vector

.. autofunction:: superscreen.brandt.Q_matrix