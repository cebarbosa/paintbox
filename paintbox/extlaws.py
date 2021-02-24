# -*- coding: utf-8 -*-
""" 
Extinction laws used for dust attenuation.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u

from .operators import CompositeSED

__all__ = ["CCM89", "C2000"]

class CCM89():
    r""" Cardelli, Clayton and Mathis (1989) extinction law.

    The extinction laws are calculated using a dust screen model, returning
    the ratio

    .. math::
        :nowrap:

        \begin{equation}
            \frac{f_\lambda}{f_\lambda^0}= 10^{-0.4 A_V
            \left (a(x) + b(x) / R_V\right )}
        \end{equation}

    where :math:`x=1/\lambda`, :math:`a(x)` and :math:`b(x)` are fixed
    polynomials, :math:`A_V` is the total extinction, and :math:`R_V` is the
    total-to-selective extinction ratio.
    """
    def __init__(self, wave, unit=None):
        """
        Parameters
        ----------
        wave: numpy.ndarray, astropy.quantities.Quantity
            Wavelength array
        unit: str, optional
            Units can be specified in the form of a string. Default is
            Angstrom. This parameter is only used if the input wavelength
            array is not an astropy.quantities.Quantity.

        Attributes
        ----------
        parnames: list
            Name of the free parameters (Av, Rv) of the extinction law.


        """
        if hasattr(wave, "unit"):
            self.wave = wave.value
            self.unit = wave.unit
        else:
            self.wave = wave
            self.unit = u.AA if unit is None else unit
        x = 1 / (self.wave * self.unit).to(u.micrometer).value
        self.parnames = ["Av", "Rv"]
        self._nparams = 2

        def _anir(x):
            return 0.574 * np.power(x, 1.61)

        def _bnir(x):
            return -0.527 * np.power(x, 1.61)

        def _aopt(x):
            y = x - 1.82
            return 1 + 0.17699 * y - 0.50447 * np.power(y, 2) \
                   - 0.02427 * np.power(y, 3) + 0.7208 * np.power(y, 4) \
                   + 0.0197 * np.power(y, 5) - 0.7753 * np.power(y, 6) \
                   + 0.32999 * np.power(y, 7)

        def _bopt(x):
            y = x - 1.82
            return 1.41338 * y + 2.28305 * np.power(y, 2) + \
                   1.07233 * np.power(y, 3) - 5.38434 * np.power(y, 4) - \
                   0.62251 * np.power(y, 5) + 5.30260 * np.power(y, 6) - \
                   2.09002 * np.power(y, 7)

        def _auv(x):
            Fa = - 0.04473 * np.power(x - 5.9, 2) - 0.009779 * np.power(x - 5.9,
                                                                        3)
            a = 1.752 - 0.316 * x - 0.104 / (np.power(x - 4.67, 2) + 0.341)
            return np.where(x < 5.9, a, a + Fa)

        def _buv(x):
            Fb = 0.2130 * np.power(x - 5.9, 2) + 0.1207 * np.power(x - 5.9, 3)
            b = -3.090 + 1.825 * x + 1.206 / (np.power(x - 4.62, 2) + 0.263)
            return np.where(x < 5.9, b, b + Fb)

        nir = (0.3 <= x) & (x <= 1.1)
        optical = (1.1 < x) & (x <= 3.3)
        uv = (3.3 < x) & (x <= 8)
        self._a = np.where(nir, _anir(x), np.where(optical, _aopt(x),
                                                   np.where(uv, _auv(x), 0)))
        self._b = np.where(nir, _bnir(x), np.where(optical, _bopt(x),
                                                   np.where(uv, _buv(x), 0)))

    def __call__(self, theta):
        """ Returns the dust screen model attenuation.

        Parameters
        ----------
        theta: numpy.ndarray
            Array with values of Av and Rv.
        """
        return np.power(10, -0.4 * theta[0] * (self._a + self._b / theta[1]))

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def gradient(self, theta):
        """ Gradient of the extinction law.

        Parameters
        ----------
        theta: numpy.ndarray
            Array with values of Av and Rv.
        """
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        grad[0] = -0.4 * np.log(10) * (self._a + self._b / theta[1]) * A
        grad[1] = 0.4 * np.log(10) * theta[0] * self._b * \
                  np.power(theta[1], -2) * A
        return grad

class C2000():
    """ Calzetti et al. (2000) extinction law.

    The extinction laws are calculated using a dust screen model, returning
    the ratio

    .. math::
        :nowrap:

        \begin{equation}
        \frac{f_\lambda}{f_\lambda^0}= 10^{-0.4 A_V \left (1 +
                                       \kappa_\lambda / R_V\right )}
        \end{equation}
    """
    def __init__(self, wave, unit=None):
        """
        Parameters
        ----------
        wave: numpy.ndarray, astropy.quantities.Quantity
            Wavelength array
        unit: str, optional
            Units can be specified in the form of a string. Defalt is Angstrom.

        """
        if hasattr(wave, "unit"):
            self.wave = wave.value
            self.unit = wave.unit
        else:
            self.wave = wave
            self.unit = u.AA if unit is None else u.Quantity(unit)
        x = 1 / (self.wave * self.unit).to(u.micrometer).value
        self._kappa = np.where(self.wave > 0.63 * u.micrometer,
                               2.659 * (-1.857 + 1.040 * x), \
                               2.659 * (-2.156 + 1.509 * x - 0.198 * x * x
                                       + 0.011 * (x * x * x)))

    def __call__(self, theta):
        """ Returns the dust screen model attenuation.

        Parameters
        ----------
        theta: numpy.ndarray
            Array with values of Av and Rv.
        """
        return np.power(10, -0.4 * theta[0] * (1. + self._kappa / theta[1]))

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def gradient(self, theta):
        """ Gradient of the extinction law.

        Parameters
        ----------
        theta: numpy.ndarray
            Array with values of Av and Rv.
        """
        grad = np.zeros((2, len(self.wave)))
        A = self.__call__(theta)
        # dAw / dAv
        grad[0] = A * np.log(10) * (-0.4 * (1. + self._kappa / theta[1]))
        # dAw / dRv
        grad[1] = A * 0.4 * theta[0] * self._kappa * np.log(10) * \
                  np.power(theta[1], -2.)
        return grad