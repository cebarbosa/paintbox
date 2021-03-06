# -*- coding: utf-8 -*-
"""

Created on 27/11/19

Author : Carlos Eduardo Barbosa

Basic classes to build the SED/spectra of galaxies

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from scipy.ndimage import convolve1d
from spectres import spectres

__all__ = ["LOSVDConv", "Resample", "SEDSum", "SEDMul", "ConstrainModel"]

class LOSVDConv():
    def __init__(self, obj, velscale, losvdpars=None):
        self.obj = obj
        self.wave = obj.wave
        losvdpars = ["V", "sigma"] if losvdpars is None else losvdpars
        if not hasattr(velscale, "unit"):
            velscale = velscale * u.Unit("km/s")
        self.velscale = velscale.to("km/s").value
        self.parnames = obj.parnames + losvdpars
        self.nparam = len(self.parnames)
        self.shape = (self.nparam, len(self.wave))

    def _kernel_arrays(self, p):
        x0, sigx = p / self.velscale
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

    def __call__(self, theta):
        z = self.obj(theta[:-2])
        y, k = self._kernel_arrays(theta[-2:])
        return convolve1d(z, k)

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        p1 = theta[:-2]
        p2 = theta[-2:]
        grad = np.zeros(self.shape)
        model = self.obj(theta[:-2])
        modelgrad = self.obj.gradient(p1)
        y, k = self._kernel_arrays(p2)
        for i in range(len(modelgrad)):
            grad[i] = convolve1d(modelgrad[i], k)
        grad[-2] = convolve1d(model, y * k / p2[1])
        grad[-1] = convolve1d(model, (y * y - 1.) * k / p2[1])
        return grad

class Resample():
    def __init__(self, wave, obj):
        self.obj = obj
        self.inwave= self.obj.wave
        self.wave = wave
        self.parnames = self.obj.parnames
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        model = self.obj(theta)
        rebin = spectres(self.wave, self.inwave, model)
        return rebin

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        grads = self.obj.gradient(theta)
        grads = spectres(self.wave, self.inwave, grads)
        return grads

class SEDSum():
    def __init__(self, o1, o2):
        msg = "Components with different wavelenghts cannot be added!"
        assert np.all(o1.wave == o2.wave), msg
        self.o1 = o1
        self.o2 = o2
        self.wave = self.o1.wave
        self.parnames = self.o1.parnames + self.o2.parnames
        self.nparams = len(self.parnames)
        self._grad_shape = (self.nparams, len(self.wave))

    def __call__(self, theta):
        theta1 = theta[:self.o1.nparams]
        theta2 = theta[self.o1.nparams:]
        return self.o1(theta1) + self.o2(theta2)

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        n = self.o1.nparams
        theta1 = theta[:n]
        theta2 = theta[n:]
        grad = np.zeros(self._grad_shape)
        grad[:n, :] = self.o1.gradient(theta1)
        grad[n:, :] = self.o2.gradient(theta2)
        return grad

class SEDMul():
    def __init__(self, o1, o2):
        msg = "Components with different wavelenghts cannot be multiplied!"
        assert np.all(o1.wave == o2.wave), msg
        self.o1 = o1
        self.o2 = o2
        self.wave = self.o1.wave
        self.parnames = self.o1.parnames + self.o2.parnames
        self.nparams = len(self.parnames)
        self._grad_shape = (self.nparams, len(self.wave))

    def __call__(self, theta):
        theta1 = theta[:self.o1.nparams]
        theta2 = theta[self.o1.nparams:]
        return self.o1(theta1) * self.o2(theta2)

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        n = self.o1.nparams
        theta1 = theta[:n]
        theta2 = theta[n:]
        grad = np.zeros(self._grad_shape)
        grad[:n, :] = self.o1.gradient(theta1) * self.o2(theta2)
        grad[n:, :] = self.o2.gradient(theta2) * self.o1(theta1)
        return grad

class ConstrainModel():
    def __init__(self, sed):
        self.sed = sed
        self.parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self.nparams = len(self.parnames)
        self._shape = len(self.sed.parnames)
        self._idxs = {}
        for param in self.parnames:
            self._idxs[param] = np.where(np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        t = np.zeros(self._shape)
        for param, val in zip(self.parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)