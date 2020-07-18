# -*- coding: utf-8 -*-
"""

Created on 17/07/20

Author : Carlos Eduardo Barbosa

"""
import numpy as np
import theano.tensor as tt

from .likelihoods import *

__all__ = ["TheanoLogLikeInterface"]

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