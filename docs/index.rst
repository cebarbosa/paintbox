Documentation for **paintbox**
==============================

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

**Paintbox** is a open-source project, and contributions to improve/ extend its
capabilities are welcome. Please visit the project's
`github page <https://github.com/cebarbosa/paintbox>`_.

Requirements
------------

**paintbox** has the followong requirements:

- `Python <https://www.python.org/>`_ 3.7 or later
- `astropy <https://www.astropy.org/>`_ 4.0 or later
- `spectres <https://spectres.readthedocs.io/en/latest/>`_

Optional requirements for tutorials include also

- `pymc3 <https://docs.pymc.io/>`_
- `ppxf <http://www-astro.physics.ox.ac.uk/~mxc/software/>`_
- `matplotlib <https://matplotlib.org/>`_
- `emcee <https://emcee.readthedocs.io/en/stable/>`_

Installation
------------

Currently, the installation of **paintbox** is available with pip::

    pip install paintbox

Instalation from the source is also possible through the project's `github page
<https://github.com/cebarbosa/paintbox>`_.

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