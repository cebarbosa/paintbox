# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import numpy as np
from scipy.special import gamma, digamma
import theano.tensor as tt

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop, loglike=None):
        self.loglike = "studt" if loglike is None else loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.loglike = loglike
        if self.loglike == "studt":
            self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)
        elif self.loglike == "normal":
            self.likelihood = NormalLogLike(self.data, self.sigma, self.stpop)
        elif self.loglike == "normal2":
            self.likelihood = NormalWithErrorsLogLike(self.data, self.sigma,
                                                      self.stpop)
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
              - 0.5 * np.sum(np.power(e_i / self.sigma, 2)) \
              - 0.5 * np.sum(np.log(self.sigma ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta) - self.data
        grad = - np.sum(e_i / np.power(self.sigma, 2.)[np.newaxis, :] *
                        self.func.gradient(theta), axis=1)
        return grad

class NormalWithErrorsLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams

    def __call__(self, theta):
        e_i = self.func(theta[:-1]) - self.data
        S = theta[-1]
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / (S * self.sigma) , 2)) \
              - 0.5 * np.sum(np.log((S * self.sigma )** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta[:-1]) - self.data
        S = theta[-1]
        A = e_i / np.power(S * self.sigma, 2.)
        B = self.func.gradient(theta[:-1])
        C = -np.sum(A[np.newaxis,:] * B, axis=1)
        grad = np.zeros(len(theta))
        grad[:-1] = C
        grad[-1] = - self.N / S + \
                   np.power(S, -3) * np.sum(np.power(e_i / self.sigma, 2))
        return grad