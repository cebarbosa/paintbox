# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from scipy.ndimage import convolve1d
from scipy.interpolate import LinearNDInterpolator
from scipy.special import legendre
from spectres import spectres

class SSP():
    """ Linearly interpolated SSP models."""
    def __init__(self, wave, params, templates):
        self.wave = wave
        self.params = params
        self.templates = templates
        self.nparams = len(self.params.colnames)
        self.parnames = self.params.colnames
        ########################################################################
        # Interpolating models
        x = self.params.as_array()
        a = x.view((x.dtype[0], len(x.dtype.names)))
        self.f = LinearNDInterpolator(a, templates, fill_value=0.)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.params.colnames:
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)

    def gradient(self, theta, eps=1e-6):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        grads = np.zeros((self.nparams, self.templates.shape[1]))
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

class LOSVDConv():
    def __init__(self, wave, model, velscale):
        self.wave = wave
        self.model = model
        if not hasattr(velscale, "unit"):
            velscale = velscale * u.Unit("km/s")
        self.velscale = velscale.to("km/s").value
        self.parnames = model.parnames + ["V", "sigma"]
        self.nparam = len(self.parnames)
        self.shape = (self.nparam, len(self.wave))

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

    def __call__(self, theta):
        z = self.model(theta[:-2])
        y, k = self.kernel_arrays(theta[-2:])
        return convolve1d(z, k)

    def gradient(self, theta):
        p1 = theta[:-2]
        p2 = theta[-2:]
        grad = np.zeros(self.shape)
        model = self.model(theta[:-2])
        modelgrad = self.model.gradient(p1)
        y, k = self.kernel_arrays(p2)
        for i in range(len(modelgrad)):
            grad[i] = convolve1d(modelgrad[i], k)
        grad[-2] = convolve1d(model, y * k / p2[1])
        grad[-1] = convolve1d(model, (y * y - 1.) * k / p2[1])
        return grad

class Rebin():
    def __init__(self, wave, model):
        self.model = model
        self.inwave= self.model.wave
        self.wave = wave
        self.parnames = self.model.parnames
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        model = self.model(theta)
        rebin = spectres(self.wave, self.inwave, model)
        return rebin

    def gradient(self, theta):
        grads = self.model.gradient(theta)
        grads = spectres(self.wave, self.inwave, grads)
        return grads

class EmissionLines():
    def __init__(self, wave, templates, em_names=None):
        self.wave = wave
        self.templates = templates
        self.nparams = len(self.templates)
        self.n_em = len(templates)
        self.em_names = ["emission{}".format(n) for n in range(self.n_em)] if \
            em_names is None else em_names
        self.parnames = self.em_names
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.templates)

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

    def gradient(self, theta):
        return self.poly