Building models with `paintbox`
-------------------------------

In **paintbox**, the models used to describe the observed spectrum
and/or spectral energy distribution of a galaxy are build from a
combination of spectral components, including the stellar continuum,
emission lines for the gas, etc. Moreover, the parametrization of the
model, i.e., the specific details about how these spectral elements are
combined, are defined interactively. Below, we illustrate how to
generate these spectral components in practice and how to combine them
to make a model.

Using CvD stellar population models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models from the `Conroy and van Dokkum
(2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...747...69C/abstract>`__
and `Conroy et
al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..139C/abstract>`__,
a.k.a. CvD models, can be obtained under request to the authors, and are
**not** distributed together with ``paintbox``. Similar to the MILES
models, CvD are also distributed as SSP models with varying ages,
metallicities, and IMFs, but also provide response functions that allow
the variation of several individual elements, e.g., C, N, O, Mg, Si, Ca,
Ti, and Fe. In this cases, To handle these models, we use the utility
class ``CvD18``, built from the basic ``paintbox`` classes, to handle
the input files and produce spectra with any combination of parameters.

::

    import os
    import glob

    import numpy as np
    from paintbox.utils import CvD18, disp2vel
    import matplotlib.pyplot as plt

    # Defining an arbitrary wavelength region in the near IR
    w1, w2 = 8000, 13000 # Setting the wavelength window
    sigma = 300 # Velocity dispersion of the output models
    wave = disp2vel([w1, w2], sigma)
    outdir = os.path.join(os.getcwd(), "CvD18_tutorials")
    ssp = CvD18(wave, sigma=sigma, outdir=outdir)


.. parsed-literal::

    Processing SSP files: 100%|██████████| 35/35 [01:01<00:00,  1.75s/it]
    Preparing response functions: 100%|██████████| 25/25 [00:10<00:00,  2.30it/s]


Please check out the documentation for the ``CvD18`` class to set this
class to work in your computer using the ``libpath`` keyword.





