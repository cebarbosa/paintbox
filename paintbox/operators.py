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

from .sed import PaintboxBase

__all__ = ["LOSVDConv", "Resample", "Constrain", "FixParams"]

class LOSVDConv(PaintboxBase):
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
        self.losvdpars = ["Vsyst", "sigma"] if losvdpars is None else losvdpars
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

class Resample(PaintboxBase):
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
        rebin = spectres(self.wave, self._inwave, model, fill=0, verbose=False)
        return rebin

    def gradient(self, theta):
        """ Calculation the the gradient of the resampled model. """
        grads = self.obj.gradient(theta)
        grads = spectres(self.wave, self._inwave, grads, fill=0, verbose=False)
        return grads

class Constrain(PaintboxBase):
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
        self._parnames = list(dict.fromkeys(sed.parnames))
        self.wave = self.sed.wave
        self._nparams = len(self._parnames)
        self._ntot = len(self.sed.parnames)
        self._idxs = {}
        for param in self._parnames:
            self._idxs[param] = np.where( \
                                np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        """ Calculates the constrained model. """
        t = np.zeros(self._ntot)
        for param, val in zip(self._parnames, theta):
            t[self._idxs[param]] = val
        return self.sed(t)

    def gradient(self, theta):
        raise NotImplementedError

class FixParams(PaintboxBase):
    def __init__(self, sed, fixed_vals):
        """ Fix parameters of SED model. """
        self.sed = sed
        self.wave = self.sed.wave
        self.fixed_vals = fixed_vals
        self._parnames = [_ for _ in sed.parnames if _ not in fixed_vals]
        self._free_idxs = {}
        self._ntot = len(self.sed.parnames)
        for param in self._parnames:
            self._free_idxs[param] = np.where( \
                                np.array(self.sed.parnames) == param)[0]
        self._fixed_idxs = {}
        for param, val in self.fixed_vals.items():
            self._fixed_idxs[param] = np.where(
                                      np.array(self.sed.parnames) == param)[0]

    def __call__(self, theta):
        """ Calculates the constrained model. """
        t = np.zeros(self._ntot)
        for param, val in zip(self.parnames, theta):
            t[self._free_idxs[param]] = val
        for param, val in self.fixed_vals.items():
            idx = self._fixed_idxs[param]
            t[idx] = val
        return self.sed(t)
