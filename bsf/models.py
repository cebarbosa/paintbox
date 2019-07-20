# -*- coding: utf-8 -*-
""" 

Created on 18/07/19

Author : Carlos Eduardo Barbosa

Classes for the different models used in the fitting

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from scipy.ndimage import convolve1d
from scipy.interpolate import LinearNDInterpolator
from spectres import spectres

class SEDModel():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, nssps=2,
                 velscale=None, wave_out=None, wave_unit=None,
                 em_templates=None, em_components=None,
                 em_names=None):
        """ SED model based on combination of SSPs, emission lines, sky lines
        and their LOSVD.

        This is the main routine to produce a Spectral Energy Distribution
        model using a combination of stellar populations and emission lines.

        Parameters
        ----------
        wave: array_like
           Common wavelength array for the stellar population templates. For
           kinematic fitting, wave has to be binned at a fixed velocity
           scale prior to the modelling.

        params: astropy.table.Table
            Parameters of the SSPs.

        templates: array_like
            SED templates for the single stellar population models.

        nssps: int or 1D array_like, optional
            Number of stellar populations to be used. Negative values ignore
            the kinematics. Negative values ignore the LOSVD convolution.

            Examples:
                - 2 SSPs with common kinematics (default) nssps = 2 or
                nssps = np.array([2])
                - 2 SSPs with decoupled kinematics:
                  nssps = np.array([1,1])
                - 2 SSPS with 2 kinematic components
                  nssps = np.array([2,2])
                - Ignore LOSVD convolution in the fitting of 2 stellar
                populations (multi-band SED fitting)
                  nssps = -2 or nssps = np.array([2])

        velscale: astropy.unit, optional
            Determines the velocity difference between adjacent pixels in the
            wavelength array. Default value is 1 km/s/pixel.

        wave_out: array_like, optional
            Wavelength array of the output. In case the output is not
            determined, it is assumed that wave_out is the same as wave. If
            necessary, the rebbining if performed with spectres
            (https://spectres.readthedocs.io/en/latest/)

        wave_unit: astropy.units, optional
            Units of the wavelength array. Assumed to be Angstrom if not
            specified.

        em_templates: array_like, optional
            Emission line templates, which can be used either to model gas
            emission lines or sky lines. If

        em_components: int or 1D array_like, optional
            Similar to nssps, negative components do not make LOSVD convolution.


        em_names: array_like, optional
            Names of the emission lines to be used in the outputs.

        Returns
        -------
            "SEDModel" callable class


        Methods
        -------
            __call__(p): Returns the SED model of parameters p

            gradient(p): Gradient of the SED function with parameters p

         """
        ########################################################################
        # Setting wavelength units
        self.wave_unit = u.angstrom if wave_unit is None else wave_unit
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * self.wave_unit
        self.wave_out = self.wave if wave_out is None else wave_out
        if not hasattr(self.wave_out, "unit"):
            self.wave_out *= self.wave_unit
        ########################################################################
        # Verify if rebinning is necessary to modeling
        self.rebin = not np.array_equal(self.wave, self.wave_out)
        ########################################################################
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.params = params
        self.templates = templates
        self.nssps = np.atleast_1d(nssps).astype(np.int)
        self.em_templates = em_templates
        self.em_names = em_names
        # Dealing with SSP components
        self.pops = []
        self.idxs = [0]
        self.parnames = []
        for i, comp in enumerate(self.nssps):
            if comp > 0:
                csp = NSSPSConv(self.wave, self.params,
                                self.templates, self.velscale, npop=comp)
            else:
                csp = NSSPs(self.wave, self.params, self.templates,
                            npop=abs(comp))
            self.idxs.append(csp.nparams + self.idxs[-1])
            self.pops.append(csp)
            self.parnames.append(["sp_{}_{}".format(i, p) for p in
                                  csp.parnames])
        if self.em_templates is None:
            self.nparams = self.idxs[-1]
            return
        #######################################################################
        # Dealing with emission line components
        n_em = len(em_templates) # Number of emission lines
        em_components = np.ones(n_em) if em_components is None else \
                             em_components
        self.em_components = np.atleast_1d(em_components).astype(np.int)
        for i, comp in enumerate(np.unique(self.em_components)):
            idx = np.where(self.em_components == comp)[0]
            if comp < 0:
                em = SkyLines(self.wave, self.em_templates[idx],
                              em_names=self.em_names[idx])
            else:
                em = EmissionLines(self.wave, self.em_templates[idx],
                                   self.em_names[idx], self.velscale)
            self.idxs.append(em.nparams + self.idxs[-1])
            self.pops.append(em)
            self.parnames.append(["em_{}_{}".format(i, p) for p in em.parnames])
        self.nparams = self.idxs[-1]

    def __call__(self, theta):
        sed = np.zeros(len(self.wave))
        for i in range(len(self.pops)):
            t = theta[self.idxs[i]: self.idxs[i+1]]
            s = self.pops[i](t)
            sed += s
        if not self.rebin:
            return sed
        sed = spectres(self.wave_out.to("AA").value,
                       self.wave.to("AA").value, sed)
        return sed

    def gradient(self, theta):
        grads = []
        for i, pop in enumerate(self.pops):
            t = theta[self.idxs[i]: self.idxs[i+1]]
            grads.append(pop.gradient(t))
        grads = np.vstack(grads)
        if not self.rebin:
            return grads
        grads = spectres(self.wave_out.to("AA").value,
                       self.wave.to("AA").value, grads)
        return grads

class SSP():
    """ Linearly interpolated SSP models."""
    def __init__(self, wave, params, templates):
        self.wave = wave
        self.params = params
        self.templates = templates
        self.nparams = len(self.params.colnames)
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

class NSSPs():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, npop=2):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.ssp = SSP(self.wave, self.params, self.templates)
        self.ncols = len(self.params.colnames)
        self.nparams = self.npop * (len(self.params.colnames) + 1)  + 2
        self.shape = (self.nparams, len(self.wave))
        # Preparing array for redenning
        x = 1 / self.wave.to("micrometer") * u.micrometer
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                         2.659 * (-1.857 + 1.040 * x), \
                         2.659 * (-2.156 + 1.509 * x - 0.198 * x *x
                                  + 0.011 * (x * x * x)))
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))

    def extinction(self, Av, Rv):
        return np.power(10, -0.4  * Av * (1. + self.kappa / Rv))

    def __call__(self, theta):
        p = theta[2:].reshape(self.npop, -1)
        return self.extinction(theta[0], theta[1]) * \
                np.dot(p[:,0], self.ssp(p[:, 1:]))

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        F = self.__call__(theta)
        # dF / dAv
        grad[0] = F * np.log(10) * (-0.4 * (1. + self.kappa / theta[1]))
        # dF / dRv
        grad[1] = F * 0.4 * theta[0] * self.kappa * np.log(10) * \
                  np.power(theta[1], -2.)
        ps = theta[2:].reshape(self.npop, -1)
        const = self.extinction(theta[0], theta[1])
        for i, p in enumerate(ps):
            idx = 2 + (i * (self.ncols + 1))
            # dF/dFi
            grad[idx] = const * self.ssp(p[1:])
            # dF / dSSPi
            grad[idx+1:idx+1+self.ncols] = const * p[0] * \
                                           self.ssp.gradient(p[1:])
        return grad

class NSSPSConv():
    def __init__(self, wave, params, templates, velscale, npop=1):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.velscale = velscale.to("km/s").value
        self.nssps = NSSPs(self.wave, self.params, self.templates,
                           npop=self.npop)
        self.nparams = self.nssps.nparams + 2
        self.shape = (self.nparams, len(self.wave))
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))
        self.parnames.append("V")
        self.parnames.append("sigma")

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
        p = theta[:self.nssps.nparams]
        sed = self.nssps(p)
        y, k = self.kernel_arrays(theta[self.nssps.nparams:])
        return convolve1d(sed, k)

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        sspgrad = self.nssps.gradient(theta[:self.nssps.nparams])
        p = theta[self.nssps.nparams:]
        y, k = self.kernel_arrays(p)
        for i in range(len(sspgrad)):
            grad[i] = convolve1d(sspgrad[i], k)
        sed = self.nssps(theta[:self.nssps.nparams])
        grad[-2] = convolve1d(sed, y * k / p[1])
        grad[-1] = convolve1d(sed, (y * y - 1.) * k / p[1])
        return grad

class EmissionLines():
    def __init__(self, wave, templates, em_names, velscale):
        self.wave = wave
        self.templates = templates
        self.em_names = em_names
        self.n_em = len(templates)
        self.velscale = velscale
        self.shape = (self.n_em+2, len(self.wave))
        self.parnames = ["flux_{}".format(name) for name in em_names]
        self.parnames.append("V")
        self.parnames.append("sigma")
        self.nparams = len(self.parnames)


    def __call__(self, theta):
        g = theta[:self.n_em]
        p = theta[self.n_em:]
        return convolve1d(np.dot(g, self.templates),
                          self.kernel_arrays(p)[1])

    def gradient(self, theta):
        g = theta[:self.n_em]
        p = theta[-2:]
        y, k = self.kernel_arrays(p)
        grad = np.zeros(self.shape)
        for i in range(self.n_em):
            grad[i] = convolve1d(self.templates[i], k)
        gas = np.dot(g, self.templates)
        grad[-2] = convolve1d(gas, y * k / p[1])
        grad[-1] = convolve1d(gas, (y * y - 1.) * k / p[1])
        return grad

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale.to("km/s").value
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

class SkyLines():
    def __init__(self, wave, templates, em_names=None):
        self.wave = wave
        self.templates = templates
        self.nparams = len(self.templates)
        self.n_em = len(templates)
        self.em_names = ["sky{}".format(n) for n in range(self.n_em)] if \
            em_names is None else em_names
        self.parnames = ["flux_{}".format(name) for name in self.em_names]
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.templates)

    def gradient(self, theta):
        return self.templates