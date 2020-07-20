**********************
Paintbox Documentation
**********************


Overview
========

Paintbox is a parametric modeling and fitting toolbox designed to build and
model ths spectral energy distribution of galaxies from any kind of
spectro-photometric observations. The paintbox is designed to allow easy
construction of very general models by the combination of a few basic
classes of objects that compose the spectral features, such as stellar
population continuum and emission lines.

Requirements
============

``paintbox`` has the followong requirements:

- `Python <https://www.python.org/>`_ 3.7 or later
- astropy 4.0 or later

Optional requirements for tutorials

- pymc3
- ppxf
- matplotlib
- emcee


Installation
============

Installation of ``paintbox`` is made with ``pip``::

    pip install paintbox

Using ``paintbox``
==================

Interpolating stellar poopulation models
----------------------------------------

.. toctree::
   :maxdepth: 2

   test.rst

Using templates
---------------

Extinction laws
----------------

Polynomials
-----------

Velocity distribution convolution
----------------------------------

Rebinning
---------

Composing models
----------------

Extending ``paintbox``
----------------------

Reference/API
=============
.. automodapi:: paintbox.sed_components

.. automodapi:: paintbox.extlaws

.. automodapi:: paintbox.likelihoods

.. automodapi:: paintbox.operators

.. automodapi:: paintbox.interfaces

