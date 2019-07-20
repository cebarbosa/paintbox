# -*- coding: utf-8 -*-
"""

Created on 12/04/19

Author : Carlos Eduardo Barbosa

Bayesian spectrum fitting.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import gamma, digamma
import theano.tensor as tt
import pymc3 as pm

from models import SEDModel

class BSF():
    def __init__(self, wave, flux, twave, templates, params,
                 fluxerr=None, em_templates=None, em_names=None,
                 velscale=None, nssps=2, em_components=None,
                 wave_unit=None, z=0., loglike="studT"):
        """ Model observations with SSP models using hierarchical Bayesian
        approach and probabilisti programming.

        Parameters
        ----------
        wave: np.array
            The wavelength dispersion of the modeled data

        flux: np.array
            The flux of the modeled data

        twave: np.array
            The wavelength dispersion of the templates

        tflux: np.array
            The array of templates to be used in the fitting

        params: astropy.table.Table
            Table containing the parameters of the templates

        fluxerr: np.array
            Uncertainties for the modeled flux.

        """
        self.loglike = loglike
        # Observed parameters
        self.flux = np.atleast_1d(flux)
        self.fluxerr = np.ones_like(self.flux) if fluxerr is None else \
                       fluxerr
        self.z = z
        # SSP templates
        self.templates = templates
        self.params = params
        # Emission line templates
        self.em_templates = em_templates
        self.em_names = em_names
        self.em_components = em_components
        if self.em_components is not None:
            self.em_components = np.atleast_1d(em_components)
        ########################################################################
        # Setting wavelength units
        self.wave_unit = u.angstrom if wave_unit is None else wave_unit
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * self.wave_unit
        if hasattr(twave, "unit"):
            self.twave = twave
        else:
            self.twave = twave * self.wave_unit
        ########################################################################
        # Check if rebinning is necessary
        self.rebin = False if np.array_equal(self.wave, self.twave) else True
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.nssps = np.atleast_1d(nssps)
        self.sed = SEDModel(self.twave, self.params, self.templates,
                            nssps=self.nssps, velscale=velscale,
                            wave_out=wave, em_templates=self.em_templates,
                            em_names=em_names,
                            em_components=self.em_components)
        self.parnames = [item for sublist in self.sed.parnames for item in
                         sublist]

    def build_model(self):
        print("Generating model...")
        # Estimating scale to be used for magnitudes
        m0 = -2.5 * np.log10(np.median(self.flux) / np.median(self.templates))
        # Estimating scale for emission lines
        if self.em_templates is not None:
            m0em = np.median(np.max(self.em_templates, axis=1))
        # Estimating velocity from input redshift
        beta = (np.power(self.z + 1, 2) - 1) / (np.power(self.z + 1, 2) + 1)
        V0 = const.c.to(self.velscale.unit) * beta
        # Building statistical model
        self.model = pm.Model()
        with self.model:
            theta = []
            for par in self.parnames:
                comptype = par.split("_", 1)[0]
                vartype = par.split("_")[2]
                if comptype == "sp":
                    if vartype == "Av":
                        Av = pm.HalfNormal(par, sd=1.)
                        theta.append(Av)
                    elif vartype == "Rv":
                        BNormal = pm.Bound(pm.Normal, lower=0)
                        Rv = BNormal(par, mu=4.05, sd=0.8)
                        theta.append(Rv)
                    elif vartype == "flux":
                        magkey = par.replace("flux", "mag")
                        mag = pm.Normal(magkey, mu=m0, sd=3.)
                        flux = pm.Deterministic(par, pm.math.exp(-0.4 * mag *
                                                             np.log(10)))
                        theta.append(flux)
                    elif vartype in self.params.colnames:
                        param = pm.Uniform(par, lower=self.params[
                            vartype].min(), upper=self.params[vartype].max())
                        theta.append(param)
                    elif vartype == "V":
                        V = pm.Normal(par, mu=V0.value, sd=1000.)
                        theta.append(V)
                    elif vartype == "sigma":
                        sigma = pm.HalfNormal(par, sd=300)
                        theta.append(sigma)
                elif comptype == "em":
                    if vartype == "flux":
                        magkey = par.replace("flux", "mag")
                        mag = pm.Normal(magkey, mu=m0em, sd=3.)
                        flux = pm.Deterministic(par, pm.math.exp(-0.4 * mag *
                                                             np.log(10)))
                        theta.append(flux)
                    elif vartype == "V":
                        V = pm.Normal(par, mu=V0.value, sd=1000.)
                        theta.append(V)
                    elif vartype == "sigma":
                        sigma = pm.HalfNormal(par, sd=80)
                        theta.append(sigma)
            # Setting degrees-of-freedom of likelihood
            BGamma = pm.Bound(pm.Gamma, lower=2.01)
            nu = BGamma("nu", alpha=2., beta=.1, testval=10.)
            theta.append(nu)
            theta = tt.as_tensor_variable(theta)
            logl = LogLikeWithGrad(self.flux, self.wave, self.fluxerr,
                                   self.sed, loglike=self.loglike)
            # use a DensityDist
            pm.DensityDist('likelihood', lambda v: logl(v),
                           observed={'v': theta})

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop, loglike=None):
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.loglike = "studT" if loglike is None else "normal"
        if self.loglike == "studT":
            self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)
        elif self.loglike == "normal":
            self.likelihood = NormalLogLike(self.data, self.sigma, self.stpop)
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, likelihood):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = likelihood

    def perform(self, node, inputs, outputs):
        theta, = inputs
        # calculate gradients
        grads = self.likelihood.gradient(theta)
        outputs[0][0] = grads

class StudTLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams + 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.func(theta[:-1]) - self.data
        x = 1. + np.power(e_i / self.sigma, 2.) / (nu - 2)
        LLF = self.N * np.log(gamma(0.5 * (nu + 1)) /
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu))  \
             - 0.5 * (nu + 1) * np.sum(np.log(x)) \
             - 0.5 * np.sum(np.log(self.sigma**2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.func.nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.func(theta[:-1]) - self.data
        x = np.power(e_i / self.sigma, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.sigma**2) / (nu-2)
        term12 = term1 * term2
        sspgrad = self.func.gradient(theta[:-1])
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.sigma, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class NormalLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams

    def __call__(self, theta):
        e_i = self.func(theta) - self.data
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.log(self.sigma ** 2)) + \
              - 0.5 * np.sum(np.power(e_i / self.sigma, 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta[:-1]) - self.data
        grad = - np.sum(np.power(e_i / self.sigma, 2.)[np.newaxis, :] *
                        self.func.gradient(theta), axis=1)
        return grad