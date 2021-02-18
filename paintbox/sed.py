# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

Basic classes to build the SED/spectra of galaxies

"""
from __future__ import print_function, division

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import legendre

from .operators import SEDMul, SEDSum

__all__ = ["ParametricModel", "NonParametricModel", "Polynomial"]

class ParametricModel():
    """
    Class used for interpolation and call of templates parametrically.

    This class allows the linear interpolation of SED templates, such as SSP
    models, based on a table of parameters and their SEDs.
    Warning: The linear interpolation is currently based on
    scipy.RegularGridInterpolator for better performance with large number of
    input models, thus the input data must be in the form of a regular grid.

    Attributes
    ----------
    parnames: list
        Name of the variables of the SED model.

    Methods
    -------
    __call__
        Computation of interpolated model at a point, with parameters in the
        order provided by the parnames list.
    gradient
        Gradient of the interpolated model at a given point.

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
        self.parnames = self.params.colnames
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
        """ Multiplication between SED components. """
        return SEDSum(self, o)

    def __mul__(self, o):
        """ Sum of SED components. """
        return SEDMul(self, o)

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
    def __init__(self, wave, templates, em_names=None):
        self.wave = wave
        self.templates = np.atleast_2d(templates)
        self.nparams = len(self.templates)
        self.n_em = len(templates)
        self.em_names = ["emission{}".format(n) for n in range(self.n_em)] if \
            em_names is None else em_names
        self.parnames = self.em_names
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.templates)

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        return self.templates

class Polynomial():
    def __init__(self, wave, degree):
        self.wave = wave
        self.degree = degree
        self.x = 2 * ((self.wave - self.wave.min()) /
                      (self.wave.max() - self.wave.min()) - 0.5)
        self.poly = np.zeros((self.degree + 1, len(self.x)))
        for i in np.arange(self.degree + 1):
            self.poly[i] = legendre(i)(self.x)
        self.parnames = ["p{}".format(i) for i in np.arange(degree+1)]
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.poly)

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        return self.poly