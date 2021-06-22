.. superscreen

.. _api-brandt:


******
Brandt
******

The ``superscreen.brandt`` module contains the actual implementation
of Brandt's method, as described :ref:`here <background>`.

Brandt Solver
-------------

.. autofunction:: superscreen.brandt.solve


BrandtSolution
--------------

.. autoclass:: superscreen.brandt.BrandtSolution
    :members:

Supporting Functions
--------------------

.. autofunction:: superscreen.brandt.field_conversion

.. autofunction:: superscreen.brandt.brandt_layer

.. autofunction:: superscreen.brandt.q_matrix

.. autofunction:: superscreen.brandt.C_vector

.. autofunction:: superscreen.brandt.Q_matrix