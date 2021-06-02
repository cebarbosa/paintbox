Paintbox documentation
======================

Overview
--------

The ``paintbox`` package provides a flexible modeling toolbox for spectroscopic
and SED fitting of astronomical observations. The ``paintbox`` is designed to allow easy
construction of very general models by the combination of a few basic
classes of objects that compose the spectral features, such as stellar
populations, emission lines, extinction laws, etc, with a high degree of
customization.  This is a an open-source project, and contributions to
improve/ extend its capabilities are welcome! If you are using the code and /
or plans to contribute, please let me know by email at <mailto:kadu
.barbosa@gmail.com>. Please, also follow the code to keep tunned on new
features, bug fixes and other info at `www.github.com/cebarbosa/paintbox>`_.

Installation
------------

To obtain the most recent version of ``paintbox``, we recommend the
installation from the source code::

    git clone https://github.com/cebarbosa/paintbox.git
    cd paintbox
    python setup.py install

Alternatively, installation of ``paintbox`` is available with pip::

    pip install paintbox


Installation requires `numpy <https://numpy.org/>`_, `scipy <https://www
.scipy.org/>`_, `astropy <https://www.astropy.org/>`_, `spectres
<https://spectres.readthedocs.io/en/latest/>`_, and `tqdm <https://tqdm.github.io/>`_. The code has
been developed in Python 3.7. The tutorials also require other packages including `ppxf
<http://www-astro.physics.ox.ac.uk/~mxc/software/>`_, `matplotlib
<https://matplotlib.org/>`_, and `emcee <https://emcee.readthedocs
.io/en/stable/>`_.

User guide
----------

.. toctree:: basic_classes.rst
   :maxdepth: 2
.. toctree:: preparing_models.rst
   :maxdepth: 2
.. toctree:: operations.rst
   :maxdepth: 2



Reference / API
---------------

.. toctree:: paintbox/index.rst
   :maxdepth: 2