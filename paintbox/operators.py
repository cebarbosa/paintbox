# -*- coding: utf-8 -*-
"""
Miscelaneous classes for combination and modification of other paintbox model
classes.
"""
from __future__ import print_function, division

import numpy as np
import astropy.constants as const
from scipy.ndimage import convolve1d
from spectres import spectres

__all__ = ["LOSVDConv", "Resample", "CompositeSED", "Constrain"]

class LOSVDConv():
    """ Convolution of a SED model with the LOS Velocity Distribution.

    This class allows the convolution of a SED model with the line--of-sight
    velocity distribution. This class requires that the input SED model has a
    logarithmic-scaled wavelength dispersion. Currently, this operation only
    supports LOSVDs with two moments (velocity and velocity dispersion).
    
    Attributes
    ----------
    velscale: Quantity
        Velocity scale of the wavelength array.
    parnames: list
        Updated list of parameters, including the input SED object parameter
        names and the LOSVD parameters.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.
    """
    def __init__(self, obj, losvdpars=None, vunit="km/s"):
        """
        Parameters
        ----------
        obj: SED model
            Input paintbox SED model to be convolved with the LOSVD.
        losvdpars: list
            Name of the LOSVD parameters appended to the input paranames.
            Defaults to [V, sigma].
        vunit: str
            Units used for velocity variables. Defaults to km/s.

        """
        self.obj = obj
        self.wave = obj.wave
        # Check if velocity scale is constant
        velscale = np.diff(np.log(self.wave) * const.c.to(vunit))
        msg = "LOSVD convolution requires wavelength array with constant " \
              "velocity scale."
        assert np.all(np.isclose(velscale, velscale[0])), msg
        self.velscale = velscale[0]
        self._v = self.velscale.value
        self.losvdpars = ["V", "sigma"] if losvdpars is None else losvdpars
        self.parnames = obj.parnames + self.losvdpars
        self._nparams = len(self.parnames)
        self._shape = (self._nparams, len(self.wave))

    def _kernel_arrays(self, p):
        """ Produces kernels used in the convolution. """
        x0, sigx = p / self._v
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

    def __call__(self, theta):
        """ Performs convolution of input model with LOSVD."""
        z = self.obj(theta[:-2])
        y, k = self._kernel_arrays(theta[-2:])
        return convolve1d(z, k)

    def gradient(self, theta):
        """ Gradient of the convolved model. """
        p1 = theta[:-2]
        p2 = theta[-2:]
        grad = np.zeros(self._shape)
        model = self.obj(theta[:-2])
        modelgrad = self.obj.gradient(p1)
        y, k = self._kernel_arrays(p2)
        for i in range(len(modelgrad)):
            grad[i] = convolve1d(modelgrad[i], k)
        grad[-2] = convolve1d(model, y * k / p2[1])
        grad[-1] = convolve1d(model, (y * y - 1.) * k / p2[1])
        return grad

    def __add__(self, o):
        """ Addition of this output model with other SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """ Multiplication of this output model with other SED components. """
        return CompositeSED(self, o, "*")

class Resample():
    """ Resampling of SED model to a new wavelength dispersion.

    The resample can be performed to arbitrary dispersions based on the
    'spectres <https://spectres.readthedocs.io/en/latest/>'_ package.

    Attributes
    ----------
    parnames: list
        List of parameter names.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.
    """
    def __init__(self, wave, obj):
        """
        Parameters
        ----------
        wave: ndarray, Quantity
            New wavelength array of the SED model.
        obj: SED model
            SED model to be resampled.
        """
        self.obj = obj
        self._inwave= self.obj.wave
        self.wave = wave
        self.parnames = self.obj.parnames
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Performs the resampling. """
        model = self.obj(theta)
        rebin = spectres(self.wave, self._inwave, model)
        return rebin

    def gradient(self, theta):
        """ Calculation the the gradient of the resampled model. """
        grads = self.obj.gradient(theta)
        grads = spectres(self.wave, self._inwave, grads)
        return grads

    def __add__(self, o):
        """ Addition of this model with other SED models. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """ Multiplication of this model with other SED models. """
        return CompositeSED(self, o, "*")
    
class CompositeSED():
    """
    Combination of SED models.

    The CompositeSED class allows the combination of any number of SED model
    components using addition and / or multiplication, as long as the input
    classes have the same wavelength dispersion.

    Attributes
    ----------
    parnames: list
        The new parnames list is a concatenation of the input SED models.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.
    """
    def __init__(self, o1, o2, op):
        """
        Parameters
        ----------
        o1, o2: SED model components
            Input SED models to be combined either by multiplication or
            addition.
        op: str
            Operation of the combination, either "+" or "*".
        """
        msg = "Components with different wavelenghts cannot be combined!"
        assert np.all(o1.wave == o2.wave), msg
        self.__op = op
        msg = "Operations allowed in combination of SED components are + and *."
        assert self.__op in ["+", "*"], msg
        self.o1 = o1
        self.o2 = o2
        self.wave = self.o1.wave
        self.parnames = self.o1.parnames + self.o2.parnames
        self._nparams = len(self.parnames)
        self._grad_shape = (self._nparams, len(self.wave))

    def __call__(self, theta):
        """ SED model for combined components at point theta. """
        theta1 = theta[:self.o1._nparams]
        theta2 = theta[self.o1._nparams:]
        if self.__op == "+":
            return self.o1(theta1) + self.o2(theta2)
        elif self.__op == "*":
            return self.o1(theta1) * self.o2(theta2)
    
    def gradient(self, theta):
        """ Gradient of the combined SED model at point theta. """
        n = self.o1._nparams
        theta1 = theta[:n]
        theta2 = theta[n:]
        grad = np.zeros(self._grad_shape)
        if self.__op == "+":
            grad[:n, :] = self.o1.gradient(theta1)
            grad[n:, :] = self.o2.gradient(theta2)
        elif self.__op == "*":
            grad[:n, :] = self.o1.gradient(theta1) * self.o2(theta2)
            grad[n:, :] = self.o2.gradient(theta2) * self.o1(theta1)
        return grad

    def __add__(self, o):
        """ Addition of SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """ Multiplication of SED components. """
        return CompositeSED(self, o, "*")

class Constrain():
    """ Constrain parameters of an SED model.

    The combination of SED models may result in models with repeated
    parameters at different locations of the parnames list. This class allows
    the simplification of the input model by finding and constraining all
    instances of repeated parameters to the same value.

    Attributes
    ----------
    parnames: list
        The new parnames list is a concatenation of the input SED models,
        simplified in relation to the input model.
    wave: numpy.ndarray, astropy.quantities.Quantity
        Wavelength array.

    Methods
    -------
    __call__(theta)
        Returns the SED model according the parameters given in theta.

    """
    def __init__(self, sed):
        """


        """
        self.sed = sed
        self.parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self._nparams = len(self.parnames)
        self._shape = len(self.sed.parnames)
        self._idxs = {}
        for param in self.parnames:
            self._idxs[param] = np.where( \
                                np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        """ Calculates the constrained model. """
        t = np.zeros(self._shape)
        for param, val in zip(self.parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)

    def gradient(self, theta):
        raise NotImplementedError