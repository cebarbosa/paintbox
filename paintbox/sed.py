# -*- coding: utf-8 -*-
"""

Basic SED classes for handling stellar population and emission lines
based on precomputed templates and polynomials.

"""
from __future__ import print_function, division

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import legendre

from .operators import CompositeSED

__all__ = ["ParametricModel", "NonParametricModel", "Polynomial"]

class ParametricModel():
    """
    Interpolation of SED model templates parametrically.

    This class allows the linear interpolation of SED templates, such as SSP
    models, based on a table of parameters and their SEDs.

    Warning: The linear interpolation is currently based on
    scipy.RegularGridInterpolator for better performance with large number of
    input models, thus the input data must be in the form of a regular grid.

    Attributes
    ----------
    parnames: list
        Name of the variables of the SED model.

    """
    def __init__(self, wave, params, data):
        """
        Parameters
        ----------
        wave: ndarray, Quantity
            Wavelenght array of the model.
        params: astropy.table.Table
            Table with parameters of the models.
        data: 2D ndarray
            The SED templates with dimensions (len(params), len(wave))

        """
        self.wave = wave
        self.params = params
        self.data = data
        self.parnames = self.params.colnames.copy()
        self._n = len(wave)
        self._nparams = len(self.parnames)
        # Interpolating models
        x = self.params.as_array()
        pdata = x.view((x.dtype[0], len(x.dtype.names)))
        nodes = []
        for param in self.parnames:
            x = np.unique(self.params[param]).data
            nodes.append(x)
        coords = np.meshgrid(*nodes, indexing='ij')
        dim = coords[0].shape + (self._n,)
        data = np.zeros(dim)
        with np.nditer(coords[0], flags=['multi_index']) as it:
            while not it.finished:
                multi_idx = it.multi_index
                x = np.array([coords[i][multi_idx] for i in range(len(coords))])
                idx = (pdata == x).all(axis=1).nonzero()[0]
                data[multi_idx] = self.data[idx]
                it.iternext()
        self._interpolator = RegularGridInterpolator(nodes, data,
                             bounds_error=False, fill_value=0)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.parnames:
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
        self._thetamin = np.array(thetamin)
        self._thetamax = np.array(thetamax)
        self._inner_grid = inner_grid

    def __call__(self, theta):
        """ Call for interpolated model at a given point theta.

        Parameters
        ----------
        theta: ndarray
            Point where the model is computed, with parameters in
            the same order of parnames. Points outside of the convex hull of
            the models are set to zero.

        Returns
        -------
        SED model at location theta.
        """
        return self._interpolator(theta)[0]

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def gradient(self, theta, eps=1e-6):
        """ Gradient of models at a given point theta.

        Gradients are computed with simple finite difference. If the input
        point is among the points used for interpolation, the gradient is not
        defined, returning zero instead.

        Parameters
        ----------
        theta: ndarray
            Point where the gradient of the model is computed,
            with parameters in the same order of parnames. Points outside
            of the convex hull of the models are set to zero.
        eps: float or ndarray, optional
            Step used in the finite difference calculation. Default is 1e-6.

        """
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self._thetamin + 2 * eps)
        theta = np.minimum(theta, self._thetamax - 2 * eps)
        grads = np.zeros((self._nparams, self._n))
        for i,t in enumerate(theta):
            epsilon = np.zeros(self._nparams)
            epsilon[i] = eps
            # Check if data point is in inner grid
            in_grid = t in self._inner_grid[i]
            if in_grid:
                continue
            else:
                tp = theta + epsilon
                tm = theta - epsilon
                grads[i] = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
        return grads

class NonParametricModel():
    """
    Weighted linear combination of SED models.

    This class allows the combination of a set of templates based on
    different weights.

    Attributes
    ----------
    parnames: list
        Name of the templates.

    """
    def __init__(self, wave, templates, names=None):
        """ 
        Parameters
        ----------
        wave: ndarray, Quantity
            Common wavelenght array of all templates.
        templates: 2D ndarray
            SED models with dimensions (N, len(wave)), where N=number of
            templates.
        names: list
            Name of the templates. Defaults to [temp, ..., tempN]
        """
        self.wave = wave
        self.templates = np.atleast_2d(templates)
        self._nparams = len(self.templates)
        self._n = len(templates)
        names = ["temp{}".format(n+1) for n in range(self._n)] if \
                 names is None else names
        self.parnames = names
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Returns the dot product of a vector theta with the templates.

        Parameters
        ----------
        theta: ndarray
            Vector with weights of the templates.

        Returns
            Dot product of theta with templates.
        """
        return np.dot(theta, self.templates)

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def gradient(self, theta):
        """ Gradient of the dot product with weights theta.

        This routine returns simply the templates, but it has an argument
        theta only to keep calls consistently across different SED
        components.
        """
        return self.templates

class Polynomial():
    """
    Polynomial SED component.

    This class produces Legendre polynomials that can be either added or
    multiplied with other SED components.

    Attributes
    ----------
    wave: ndarray, Quantity
        Wavelength array of the polynomials
    degree: int
        Order of the Legendre polynomial.
    poly: 2D ndarray
        Static Legendre polynomials array used in the calculations.
    parnames: list
        List with the name of the individual polynomials, set to
        [p0, p1, ..., pdegree] at initialization.

    """

    def __init__(self, wave, degree):
        """
        Parameters
        ----------
        wave: ndarray, Quantity
            Wavelength array of the polynomials
        degree: int
            Order of the Legendre polynomial.fg
        """
        self.wave = wave
        self.degree = degree
        self._x = 2 * ((self.wave - self.wave.min()) /
                       (self.wave.max() - self.wave.min()) - 0.5)
        self.poly = np.zeros((self.degree + 1, len(self._x)))
        for i in np.arange(self.degree + 1):
            self.poly[i] = legendre(i)(self._x)
        self.parnames = ["p{}".format(i) for i in np.arange(degree+1)]
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Calculation of the polynomial with weights theta. """
        return np.dot(theta, self.poly)

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def gradient(self, theta):
        """ Gradient of the polynomial with weights theta. """
        return self.poly