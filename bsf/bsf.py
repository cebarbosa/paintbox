# -*- coding: utf-8 -*-
"""

Created on 12/04/19

Author : Carlos Eduardo Barbosa

Bayesian spectrum fitting.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
from scipy.special import gamma, digamma
import theano.tensor as tt
import pymc3 as pm

from models import SEDModel

class BSF():
    def __init__(self, wave, flux, twave, templates, params,
                 fluxerr=None, em_templates=None, em_names=None,
                 velscale=None, components=2, em_components=None,
                 wave_unit=None):
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
        self.wave = wave
        self.flux = np.atleast_2d(flux)
        self.twave = twave
        self.templates = templates
        self.params = params

        # Check if rebinning is necessary
        self.rebin = False if np.array_equal(self.wave, self.twave) else True

        self.fluxerr = 1. if fluxerr is None else np.atleast_2d(fluxerr)

        self.nloc = len(self.flux)

        self.em_templates = em_templates
        self.em_names = em_names
        self.ngas = len(self.em_names) if self.em_names is not None else 0
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.components = np.atleast_1d(components)
        self.em_components = em_components
        if self.em_components is not None:
            self.em_components = np.atleast_1d(em_components)
        self.spm = SEDModel(self.wave_temp, self.params, self.templates,
                            components=self.components, velscale=velscale,
                            wave_out=wave, em_templates=self.em_templates,
                            em_names=em_names,
                            em_components=self.em_components)

    def build_model(self):
        print("Generating model...")
        # Setting prior parameters for stellar populations
        plower = np.array([self.params[col].min() for col in
                          self.params.colnames])
        pupper = np.array([self.params[col].max() for col in
                          self.params.colnames])
        pdelta = 0.5 * (pupper - plower)
        pmean = 0.5 * (pupper + plower)
        # Setting properties of priors for the logarithm of the flux
        m0 = -2.5 * np.log10(np.median(self.data) / np.median(self.templates))
        m0sd = np.sqrt(np.log10(self.data.std())**2 +
                        np.log10(self.templates.std())**2)
        # Setting prior parameters for LOSVD
        V = 0
        deltaV = 500
        sigmax = 500
        self.model = pm.Model()
        with self.model:
            for k, npop in enumerate(self.components):
                # Extinction
                Rv = pm.Normal("Rv{}".format(k), mu=4.05, sd=0.8,
                               shape=self.nloc, testval=4.05)
                MAv = pm.HalfNormal("MAv", sd=0.5, testval=0.2)
                SAv = pm.Gamma("SAv", alpha=2., beta=1., testval=0.2)
                BNormal = pm.Bound(pm.Normal, lower=0.)
                Av = BNormal("Av", mu=MAv, sd=SAv, shape=self.nloc, testval=0.2)
                theta = [Av, Rv]
                # Stellar population parameters
                for j in range(npop):
                    # Unobscured flux
                    mag = pm.Normal("mag{}_{}".format(k, j),
                                    mu=m0, sd=2. * m0sd, shape=self.nloc)
                    flux = pm.math.exp(-0.4 * mag * np.log(10))
                    theta.append(flux)
                    for i, param in enumerate(self.params.colnames):
                        BNormal = pm.Bound(pm.Normal, lower=plower[i],
                                           upper=pupper[i])
                        M = BNormal("M{}{}{}".format(param, k, j), mu=pmean[i],
                                    sd=pdelta[i])
                        S = pm.Gamma("S{}{}{}".format(param, k, j), alpha=2.,
                                     beta=1 / pdelta[i])
                        v = BNormal("{}{}{}".format(param, k, j), mu=M,
                                    sd=S, shape=self.nloc)
                        theta.append(v)
                if npop > 0:
                    # Systemic velocity
                    Vsyst = pm.Normal("MV{}".format(k), mu=V, sd=deltaV)
                    Svsyst = pm.Gamma("SV{}".format(k), alpha=2.,
                                   beta=1/deltaV)
                    vsyst = pm.Normal("vsyst{}".format(k), mu=Vsyst,
                                      sd=Svsyst, shape=self.nloc)
                    theta.append(vsyst)
                    # Velocity dispersion
                    Msigma = pm.HalfNormal("Msigma{}".format(k),
                                           sd=sigmax)
                    Ssigma = pm.Gamma("Ssigma{}".format(k), alpha=2.,
                                      beta=1./sigmax)
                    BNormal = pm.Bound(pm.Normal, lower=0)
                    sigma = BNormal("sigma{}".format(k,j), mu=Msigma,
                                    sd=Ssigma, shape=self.nloc)
                    theta.append(sigma)
            # Priors for the emission line templates
            if self.em_components is not None:
                for k in np.unique(self.em_components):
                    idx = np.where(self.em_components == k)[0]
                    em_templates = self.em_templates[idx]
                    em_names = self.em_names[idx]
                    peaks = np.max(em_templates, axis=1)
                    m0s = -2.5 * np.log10(np.median(self.data) / peaks)
                    for i, emline in enumerate(em_names):
                        mag = pm.Normal(emline, mu=m0s[i], sd=3.,
                                        shape=self.nloc)
                        flux = pm.math.exp(-0.4 * mag * np.log(10))
                        theta.append(flux)
                    if k >= 0:
                        # Systemic velocity
                        Vsyst = pm.Normal("MVgas{}".format(k), mu=V, sd=deltaV)
                        Svsyst = pm.Gamma("SVgas{}".format(k), alpha=2.,
                                       beta=1/deltaV)
                        vsyst = pm.Normal("vgas{}".format(k), mu=Vsyst,
                                          sd=Svsyst, shape=self.nloc)
                        theta.append(vsyst)
                        # Velocity dispersion
                        Msigma = pm.HalfNormal("Msgas{}".format(k),
                                               sd=sigmax)
                        Ssigma = pm.Gamma("Ssgas{}".format(k), alpha=2.,
                                          beta=1./sigmax)
                        BNormal = pm.Bound(pm.Normal, lower=0)
                        sigma = BNormal("sgas{}".format(k), mu=Msigma,
                                        sd=Ssigma, shape=self.nloc)
                        theta.append(sigma)
            # Setting degrees-of-freedom of likelihood
            BGamma = pm.Bound(pm.Gamma, lower=2.01)
            nu = BGamma("nu", alpha=2., beta=.1, shape=self.nloc,
                          testval=10.)
            theta.append(nu)
            theta = tt.as_tensor_variable(theta).T
            # Building the likelihood
            for i in range(self.nloc):
                logl = LogLikeWithGrad(self.data[i],
                                       self.wave, self.errors[i], self.spm)
                # use a DensityDist
                pm.DensityDist('likelihood{}'.format(i), lambda v: logl(v),
                               observed={'v': theta[i]})

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop):
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)
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
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) + \
              - 0.5 * (nu + 1) * np.sum(np.log(x))
        # - 0.5 * np.sum(np.log(sigma**2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.func.nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        const = -0.5 * (nu + 1)
        e_i = self.func(theta[:-1]) - self.data
        x = np.power(e_i / self.sigma, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.sigma**2) / (nu-2)
        term12 = term1 * term2
        sspgrad = self.func.gradient(theta[:-1])
        grad[:-1] = const * np.sum(term12[np.newaxis, :] * sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.sigma, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad