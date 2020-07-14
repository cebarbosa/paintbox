# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import numpy as np
from scipy.special import gamma, digamma
import theano.tensor as tt

__all__ = ["TheanoLogLikeInterface", "StudTLogLike", "NormalLogLike",
           "Normal2LogLike"]

class TheanoLogLikeInterface(tt.Op):
    """
    Produces a theano.tensor operator for the data log-likelihood to be used
    with pymc3.

    Parameters
    ----------
    observed : ndarray
        Observed SED / spectrum.
    model : BSF model
        Parametric model for the observed data.
    loglike : str, optional
        Log-likelihood function. Options are 'normal', 'normal2' or 'studt'.
        Default is normal.
    obserr : ndarray, optional
        Uncertainties in the observed SED / spectrum.

    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, observed, model, loglike=None, obserr=None):
        self.observed = observed
        self.model = model
        self.loglike = "studt" if loglike is None else loglike
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        if self.loglike == "studt":
            self.likelihood = StudTLogLike(self.observed, self.model,
                                             obserr=self.obserr)
        elif self.loglike == "normal":
            self.likelihood = NormalLogLike(self.observed, self.model,
                                             obserr=self.obserr)
        elif self.loglike == "normal2":
            self.likelihood = Normal2LogLike(self.observed, self.model,
                                             obserr=self.obserr)
        # initialise the gradient Op (below)
        self.logpgrad = _LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        theta, = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]

class _LogLikeGrad(tt.Op):

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
    def __init__(self, observed, model, obserr=None):
        self.observed = observed
        self.model = model
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        self.N = len(observed)
        self.nparams = self.model.nparams + 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.model(theta[:-1]) - self.observed
        x = 1. + np.power(e_i / self.obserr, 2.) / (nu - 2)
        LLF = self.N * np.log(gamma(0.5 * (nu + 1)) /
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu))  \
             - 0.5 * (nu + 1) * np.sum(np.log(x)) \
             - 0.5 * np.sum(np.log(self.obserr ** 2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.model.nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.model(theta[:-1]) - self.observed
        x = np.power(e_i / self.obserr, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.obserr ** 2) / (nu - 2)
        term12 = term1 * term2
        sspgrad = self.model.gradient(theta[:-1])
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.obserr, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class NormalLogLike():
    def __init__(self, observed, model, obserr=None):
        self.observed = observed
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        self.model = model
        self.N = len(observed)
        self.nparams = self.model.nparams

    def __call__(self, theta):
        e_i = self.model(theta) - self.observed
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / self.obserr, 2)) \
              - 0.5 * np.sum(np.log(self.obserr ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.model(theta) - self.observed
        grad = - np.sum(e_i / np.power(self.obserr, 2.)[np.newaxis, :] *
                        self.model.gradient(theta), axis=1)
        return grad

class Normal2LogLike():
    def __init__(self, observed, model, obserr=None):
        self.observed = observed
        self.model = model
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        self.N = len(observed)
        self.nparams = self.model.nparams

    def __call__(self, theta):
        model = self.model(theta[:-1])
        if np.all(model) == 0:
            return -np.infty
        e_i = model - self.observed
        S = theta[-1]
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / (S * self.obserr), 2)) \
              - 0.5 * np.sum(np.log((S * self.obserr) ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.model(theta[:-1]) - self.observed
        S = theta[-1]
        A = e_i / np.power(S * self.obserr, 2.)
        B = self.model.gradient(theta[:-1])
        C = -np.sum(A[np.newaxis,:] * B, axis=1)
        grad = np.zeros(len(theta))
        grad[:-1] = C
        grad[-1] = - self.N / S + \
                   np.power(S, -3) * np.sum(np.power(e_i / self.obserr, 2))
        return grad