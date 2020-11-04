Documentation for paintbox
==========================


Overview
--------

**paintbox** is a parametric modeling and fitting toolbox designed for the
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

Tutorials
---------

.. toctree:: tutorials/getting_started.ipynb
   :maxdepth: 2
   :caption: Tutorials


   .. plot::
       :include-source:
       import numpy as np
       import matplotlib.pyplot as plt
       from astropy.modeling import models, fitting

       # define a model for a line
       line_orig = models.Linear1D(slope=1.0, intercept=0.5)

       # generate x, y data non-uniformly spaced in x
       # add noise to y measurements
       npts = 30
       np.random.seed(10)
       x = np.random.uniform(0.0, 10.0, npts)
       y = line_orig(x)
       y += np.random.normal(0.0, 1.5, npts)

       # initialize a linear fitter
       fit = fitting.LinearLSQFitter()

       # initialize a linear model
       line_init = models.Linear1D()

       # fit the data with the fitter
       fitted_line = fit(line_init, x, y)

       # plot the model
       plt.figure()
       plt.plot(x, y, 'ko', label='Data')
       plt.plot(x, fitted_line(x), 'k-', label='Fitted Model')
       plt.xlabel('x')
       plt.ylabel('y')
       plt.legend()

User guide
----------

.. toctree:: paintbox/index.rst
   :maxdepth: 2