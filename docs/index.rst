Paintbox documentation
======================

Overview
--------

``Paintbox`` is a parametric modeling and fitting toolbox designed for the
modeling of the spectral energy distribution (SED) of astronomical
observations including spectroscopy, photometry and any combination of those.
The ``paintbox`` is designed to allow easy
construction of very general models by the combination of a few basic
classes of objects that compose the spectral features, such as stellar
populations, emission lines, extinction laws, etc, with a high degree of
customization. The ``paintbox`` is a open-source project, and contributions to improve/ extend
its capabilities are welcome.

Installation
------------

Installation of ``paintbox`` is available with pip::

    pip install paintbox


Alternatively, it is also possible to install from the source from the  from project's `github page
<https://github.com/cebarbosa/paintbox>`_. Installation requires `numpy <https://numpy.org/>`_, `scipy <https://www
.scipy.org/>`_, `astropy <https://www.astropy.org/>`_, and `spectres
<https://spectres.readthedocs.io/en/latest/>`_. The code has been developed
in Python 3.7. The tutorials also require other packages including `ppxf
<http://www-astro.physics.ox.ac.uk/~mxc/software/>`_, `matplotlib
<https://matplotlib.org/>`_, and `emcee <https://emcee.readthedocs
.io/en/stable/>`_.

User guide
----------

.. toctree:: basic_classes.rst
   :maxdepth: 2
.. toctree:: preparing_models.rst
   :maxdepth: 2


Reference / API
---------------

.. toctree:: paintbox/index.rst
   :maxdepth: 2