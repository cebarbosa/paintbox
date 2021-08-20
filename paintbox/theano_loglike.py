# -*- coding: utf-8 -*-
"""

Created on 17/07/20

Author : Carlos Eduardo Barbosa

"""
import numpy as np
import theano.tensor as tt

from likelihoods import *

__all__ = ["TheanoLogLike"]

class TheanoLogLike(tt.Op):
    """
    Produces a theano.tensor operator for the data log-likelihood to be used
    with pymc3.

    Parameters
    ----------
    loglike : paintbox.Loglike
        Log-likelihood function.

    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, likelihood):
        self.likelihood = likelihood
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
        Initialise gradient of loglike.

        Parameters
        ----------
        loglike:
            The log-likelihood
        """
        self.likelihood = likelihood

    def perform(self, node, inputs, outputs):
        theta, = inputs
        # calculate gradients
        grads = self.likelihood.gradient(theta)
        outputs[0][0] = grads