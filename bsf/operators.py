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

class LOSVDConv():
    def __init__(self, obj, velscale):
        self.obj = obj
        self.wave = obj.wave
        if not hasattr(velscale, "unit"):
            velscale = velscale * u.Unit("km/s")
        self.velscale = velscale.to("km/s").value
        self.parnames = obj.parnames + ["V", "sigma"]
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

class Rebin():
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
        if not (o1.wave == o2.wave).all():
            raise Exception("Components with different wavelenghts cannot be "
                            "added!")
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
        if not np.all(o1.wave == o2.wave):
            raise Exception("Components with different wavelenghts cannot be "
                            "multiplied!")
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

class CCM89:
    """ Cardelli, Clayton and Mathis (1989)"""
    def __init__(self, wave, unit=None):
        if hasattr(wave, "unit"):
            self.wave = wave.value
            self.unit = wave.unit
        else:
            self.wave = wave
            self.unit = u.AA if unit is None else unit
        x = 1 / (self.wave * self.unit).to(u.micrometer).value
        self.parnames = ["Av", "Rv"]
        self.nparams = 2

        def anir(x):
            return 0.574 * np.power(x, 1.61)

        def bnir(x):
            return -0.527 * np.power(x, 1.61)

        def aopt(x):
            y = x - 1.82
            return 1 + 0.17699 * y - 0.50447 * np.power(y, 2) \
                   - 0.02427 * np.power(y, 3) + 0.7208 * np.power(y, 4) \
                   + 0.0197 * np.power(y, 5) - 0.7753 * np.power(y, 6) \
                   + 0.32999 * np.power(y, 7)

        def bopt(x):
            y = x - 1.82
            return 1.41338 * y + 2.28305 * np.power(y, 2) + \
                   1.07233 * np.power(y, 3) - 5.38434 * np.power(y, 4) - \
                   0.62251 * np.power(y, 5) + 5.30260 * np.power(y, 6) - \
                   2.09002 * np.power(y, 7)

        def auv(x):
            Fa = - 0.04473 * np.power(x - 5.9, 2) - 0.009779 * np.power(x - 5.9,
                                                                        3)
            a = 1.752 - 0.316 * x - 0.104 / (np.power(x - 4.67, 2) + 0.341)
            return np.where(x < 5.9, a, a + Fa)

        def buv(x):
            Fb = 0.2130 * np.power(x - 5.9, 2) + 0.1207 * np.power(x - 5.9, 3)
            b = -3.090 + 1.825 * x + 1.206 / (np.power(x - 4.62, 2) + 0.263)
            return np.where(x < 5.9, b, b + Fb)

        nir = (0.3 <= x) & (x <= 1.1)
        optical = (1.1 < x) & (x <= 3.3)
        uv = (3.3 < x) & (x <= 8)
        self.a = np.where(nir, anir(x), np.where(optical, aopt(x),
                          np.where(uv, auv(x), 0)))
        self.b = np.where(nir, bnir(x), np.where(optical, bopt(x),
                          np.where(uv, buv(x), 0)))

    def __call__(self, theta):
        """ theta = (Av, Rv)"""
        return np.power(10, -0.4 * theta[0] * (self.a + self.b / theta[1]))

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        grad[0] = -0.4 * np.log(10) * (self.a + self.b / theta[1]) * A
        grad[1] = 0.4 * np.log(10) * theta[0] * self.b * \
                  np.power(theta[1], -2) * A
        return grad