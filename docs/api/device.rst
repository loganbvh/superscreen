.. superscreen

.. _api-device:


******************
Devices & Geometry
******************

The classes defined in ``superscreen.device``, ``superscreen.geometry``,
and ``superscreen.parameters`` are used to set up the inputs to a
``SuperScreen`` simulation, namely:

- The geometry and penetration depth of all superconducting films
- The spatial distribution of the applied magnetic field

Devices
-------

Device
======

.. autoclass:: superscreen.device.device.Device
    :members:

TransportDevice
===============

.. autoclass:: superscreen.device.TransportDevice
    :show-inheritance:
    :members:

Layer
=====

.. autoclass:: superscreen.device.components.Layer
    :members:

Polygon
=======

.. autoclass:: superscreen.device.components.Polygon
    :members:

Meshing
-------

.. autofunction:: superscreen.device.mesh.generate_mesh

.. autofunction:: superscreen.device.mesh.optimize_mesh

Parameters
----------

Parameter
=========

.. autoclass:: superscreen.parameter.Parameter
    :members:

CompositeParameter
==================

.. autoclass:: superscreen.parameter.CompositeParameter
    :members:
    :show-inheritance:
    :inherited-members:

Constant
========

.. autoclass:: superscreen.parameter.Constant
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
