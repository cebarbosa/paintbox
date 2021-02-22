# -*- coding: utf-8 -*-
""" 

Created on 27/11/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import numpy as np
from scipy.special import gamma, digamma


__all__ = ["StudTLogLike", "StudT2LogLike", "NormalLogLike", "Normal2LogLike",
           "JointLogLike"]

class LogLike:
    """ Base class for the loglikelihood classes.

    Attributes
    ----------

    """
    def __init__(self, observed, model, obserr=None, mask=None):
        """
        Parameters
        ----------
        observed: numpy.ndarray
            Observed SED of astronomical object.
        model:
            SED model used in the modelling.
        obserr: numpy.ndarray, optional
            Uncertainties in the observed SED fitting to be used in the
            weighting of the log-likelihood.
        mask: numpy.ndarray, optional
            Boolean mask for the observed data. The mask uses the Python
            convention, such that False indicate points to be masked,
            and True indicate the points to be used.

        """
        self.observed = observed
        self.model = model
        self.obserr = np.ones_like(self.observed) if obserr is None else obserr
        self.mask = np.full(len(self.observed), True) if mask is None \
                    else mask
        self._N = len(self.observed)
        self.parnames = self.model.parnames
        self._nparams = len(self.parnames)

class NormalLogLike(LogLike):
    """ Normal loglikelihood for SED modeling.

    The normal log-likelihood is given by

    ..math::

    L(y,\sigma_y|theta) = -\frac{N}{2}

    """
    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)

    def __call__(self, theta):
        e_i = (self.model(theta) - self.observed)[self.mask]
        LLF = - 0.5 * self._N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / self.obserr[self.mask], 2)) \
              - 0.5 * np.sum(np.log(self.obserr[self.mask] ** 2))
        return float(LLF)

    def gradient(self, theta):
        e_i = self.model(theta) - self.observed
        grad = - np.sum(e_i / np.power(self.obserr, 2.)[np.newaxis, :] *
                        self.model.gradient(theta), axis=1)
        return grad

class Normal2LogLike(LogLike):
    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["eta"]
        self._nparams += 1

    def __call__(self, theta):
        model = self.model(theta[:-1])
        if np.all(model) == 0:
            return -np.infty
        e_i = model - self.observed
        S = theta[-1]
        LLF = - 0.5 * self._N * np.log(2 * np.pi) + \
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
        grad[-1] = - self._N / S + \
                   np.power(S, -3) * np.sum(np.power(e_i / self.obserr, 2))
        return grad

class StudTLogLike(LogLike):
    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["nu"]
        self._nparams += 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.model(theta[:-1])[self.mask] - self.observed[self.mask]
        x = 1. + np.power(e_i / self.obserr[self.mask], 2.) / (nu - 2)
        LLF = self._N * np.log(gamma(0.5 * (nu + 1)) /
                               np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) \
              - 0.5 * (nu + 1) * np.sum(np.log(x)) \
              - 0.5 * np.sum(np.log(self.obserr ** 2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.model._nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.model(theta[:-1])[self.mask] - self.observed[self.mask]
        x = np.power(e_i / self.obserr[self.mask], 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.obserr[self.mask] ** 2) / (nu - 2)
        term12 = term1 * term2
        sspgrad = self.model.gradient(theta[:-1])[:, self.mask]
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self._N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self._N / (nu - 2)
        nuterm3 = -0.5 * self._N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.obserr, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class StudT2LogLike(LogLike):
    def __init__(self, observed, model, obserr=None, mask=None):
        super().__init__(observed, model, obserr=obserr, mask=mask)
        self.parnames += ["eta", "nu"]
        self._nparams += 2

    def __call__(self, theta):
        S, nu = theta[-2:]
        e_i = self.model(theta[:-2])[self.mask] - self.observed[self.mask]
        x = 1. + np.power(e_i / S / self.obserr[self.mask], 2.) / (nu - 2)
        LLF = self._N * np.log(gamma(0.5 * (nu + 1)) /
                               np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) \
              - 0.5 * (nu + 1) * np.sum(np.log(x)) \
              - 0.5 * np.sum(np.log((S * self.obserr[self.mask]) ** 2))
        return float(LLF)

    def gradient(self, theta):
        raise NotImplementedError("Gradients not supported for Studt2 "
                                  "loglikelyhood.")

class JointLogLike():
    def __init__(self, logp1, logp2):
        self.logp1 = logp1
        self.logp2 = logp2
        self.parnames = list(dict.fromkeys(logp1.parnames + logp2.parnames))
        self._idxs = []
        for parlist in [logp1.parnames, logp2.parnames]:
            idxs = []
            for p in parlist:
                idxs.append(self.parnames.index(p))
            self._idxs.append(np.array(idxs))

    def __call__(self, theta):
        return self.logp1(theta[self._idxs[0]]) + \
               self.logp2(theta[self._idxs[1]])