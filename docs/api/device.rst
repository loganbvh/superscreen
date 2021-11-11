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

Layer
=====

.. autoclass:: superscreen.device.components.Layer
    :members:

Polygon
=======

.. autoclass:: superscreen.device.components.Polygon
    :members:

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

.. autofunction:: superscreen.geometry.ellipse

.. autofunction:: superscreen.geometry.circle

.. autofunction:: superscreen.geometry.box
