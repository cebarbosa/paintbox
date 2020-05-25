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

__all__ = ["StPopInterp", "EmissionLines", "Polynomial"]

class StPopInterp():
    """ Linearly interpolated line-strength indices."""
    def __init__(self, wave, params, data):
        self.wave = wave
        self.params = params
        self.data = data
        self.parnames = self.params.colnames
        self._n = len(wave)
        self.nparams = len(self.parnames)
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
        self.f = RegularGridInterpolator(nodes, data, bounds_error=False,
                                         fill_value=0)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.parnames:
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)[0]

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta, eps=1e-6):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        grads = np.zeros((self.nparams, self._n))
        for i,t in enumerate(theta):
            epsilon = np.zeros(self.nparams)
            epsilon[i] = eps
            # Check if data point is in inner grid
            in_grid = t in self.inner_grid[i]
            if in_grid:
                tp1 = theta + 2 * epsilon
                tm1 = theta + epsilon
                grad1 = (self.__call__(tp1) - self.__call__(tm1)) / (2 * eps)
                tp2 = theta - epsilon
                tm2 = theta - 2 * epsilon
                grad2 = (self.__call__(tp2) - self.__call__(tm2)) / (2 * eps)
                grads[i] = 0.5 * (grad1 + grad2)
            else:
                tp = theta + epsilon
                tm = theta - epsilon
                grads[i] = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
        return grads

class EmissionLines():
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
        self.x = np.linspace(-1, 1, len(self.wave))
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
