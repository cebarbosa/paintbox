# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

Classes containing extinction laws.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u

from .operators import SEDSum, SEDMul

__all__ = ["CCM89", "C2000"]

class CCM89():
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

class C2000():
    """ Calzetti et al. (2000)"""
    def __init__(self, wave, unit=None):
        if hasattr(wave, "unit"):
            self.wave = wave.value
            self.unit = wave.unit
        else:
            self.wave = wave
            self.unit = u.AA if unit is None else unit
        x = 1 / (self.wave * self.unit).to(u.micrometer).value
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                              2.659 * (-1.857 + 1.040 * x), \
                              2.659 * (-2.156 + 1.509 * x - 0.198 * x * x
                                       + 0.011 * (x * x * x)))

    def __call__(self, theta):
        return np.power(10, -0.4 * theta[0] * (1. + self.kappa / theta[1]))

    def __add__(self, o):
        return SEDSum(self, o)

    def __mul__(self, o):
        return SEDMul(self, o)

    def gradient(self, theta):
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        # dAw / dAv
        grad[0] = A * np.log(10) * (-0.4 * (1. + self.kappa / theta[1]))
        # dAw / dRv
        grad[1] = A * 0.4 * theta[0] * self.kappa * np.log(10) * \
                  np.power(theta[1], -2.)
        return grad