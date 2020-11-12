Documentation for paintbox
==========================


Overview
--------

**Paintbox** is a parametric modeling and fitting toolbox designed for the
modeling of the spectral energy distribution (SED) of astronomical
observations including spectroscopy, photometry and any combination of those.
The **paintbox** is designed to allow easy
construction of very general models by the combination of a few basic
classes of objects that compose the spectral features, such as stellar
populations, emission lines, extinction laws, etc, with a high degree of
customization.

Requirements
------------

**paintbox** has the followong requirements:

- `Python <https://www.python.org/>`_ 3.7 or later
- astropy 4.0 or later

Optional requirements for tutorials

- pymc3
- ppxf
- matplotlib
- emcee

Installation
------------

Currently, the installation of **paintbox** is available with pip::

    pip install paintbox

We recommend importing **paintbox** as::

    import paintbox as pb

Contribute
----------
**Paintbox** is a open-source project, and contributions to improve/ extend its
capabilities are welcome. Please visit the project's github page: [github
.com/cebarbosa/paintbox](github.com/cebarbosa/paintbox).

Tutorials
---------

.. toctree:: preparing_models.rst
   :maxdepth: 2
.. toctree:: building_models.rst
   :maxdepth: 2


Reference / API
---------------

.. toctree:: paintbox/index.rst
   :maxdepth: 2