Paintbox
--------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

Paintbox is a parametric modeling and fitting toolbox designed to build and
model ths spectral energy distribution of galaxies from any kind of
spectro-photometric observations. The paintbox is designed to allow easy
construction of very general models by the combination of a few basic
classes of objects that compose the spectral features, such as stellar
population continuum and emission lines.


Features
--------
One of the premises of the Paintbox is to allow large flexibility to the
user in the modeling of data. Some of the main features already present in
the package include

- Modular structure that separates the task of building the model from the fitting process.
- Interpolation tools for any pre-computed stellar population models.
- Modeling using templates.
- Simple kinematic tool to model the LOSVD.
- Easy extension to any other type of SED features.
- Complete control over SED model properties


Installation
------------

Install Paintbox by running:

    pip install paintbox

Contribute
----------

- Source Code: github.com/cebarbosa/paintbox

License
-------

This project is Copyright (c) Carlos Eduardo Barbosa and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.
