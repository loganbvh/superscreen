.. superscreen

.. _api-device:


******************
Devices & Geometry
******************

The classes defined in ``superscreen.device``, ``superscreen.geometry``,
and ``superscreen.parameter`` are used to set up the inputs to a
``SuperScreen`` simulation, namely:

- The geometry and penetration depth of all superconducting films
- The spatial distribution of the applied magnetic field

.. contents::
    :depth: 2

Devices
-------

Device
======

.. autoclass:: superscreen.Device
    :members:

Layer
=====

.. autoclass:: superscreen.Layer
    :members:

Polygon
=======

.. autoclass:: superscreen.Polygon
    :members:

Meshing
-------

.. autoclass:: superscreen.Mesh
    :members:

.. autoclass:: superscreen.device.EdgeMesh
    :members:

.. autoclass:: superscreen.device.MeshOperators
    :members:

.. autofunction:: superscreen.device.generate_mesh

.. autofunction:: superscreen.device.smooth_mesh

Parameters
----------

Parameter
=========

.. autoclass:: superscreen.Parameter
    :members:

CompositeParameter
==================

.. autoclass:: superscreen.parameter.CompositeParameter
    :members:
    :show-inheritance:
    :inherited-members:

Constant
========

.. autoclass:: superscreen.Constant
    :members:
    :show-inheritance:
    :inherited-members:

Geometry
--------

.. autofunction:: superscreen.geometry.rotate

.. autofunction:: superscreen.geometry.translate

.. autofunction:: superscreen.geometry.ellipse

.. autofunction:: superscreen.geometry.circle

.. autofunction:: superscreen.geometry.box
