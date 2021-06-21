.. superscreen

.. _api-fem:


**********************
Finite Element Methods
**********************

The ``superscreen.fem`` module contains functions useful for performing computations
on a triangular mesh. Many of the functions take as arguments two arrays, ``points``
and ``triangles``, which together define the mesh. ``points``, which has shape ``(n, 2)``,
defines the coordinates of the triangle vertices. Each row in ``triangles``, which has shape ``(m, 3)``
gives the indices of the three vertices in a single triangle, such that ``points[triangles]``
is a shape ``(m, 3, 2)`` array where each row gives the coordinates of the three vertices
of a triangle in the mesh.

The weight matrix, mass matrix, and Laplacian operator are all sparse, meaning that most of their
entries are zero. In order to save memory, functions that generate these arrays give
the option to return a `scipy.sparse matrix <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_.
All matrices are converted to dense `numpy arrays <https://numpy.org/doc/stable/reference/arrays.ndarray.html>`_
when simulating a :class:`superscreen.device.Device`, however.

.. automodule:: superscreen.fem
    :members: