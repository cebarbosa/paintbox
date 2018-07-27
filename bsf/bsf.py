# -*- coding: utf-8 -*-
""" 

Created on 18/05/18

Author : Carlos Eduardo Barbosa

TMCSP: A full Bayesian Template-Matching based modeling of Composite Stellar
Populations.

"""

from __future__ import print_function, division

from builtins import range
from builtins import object
import pickle

import numpy as np
from scipy.special import legendre
import pymc3 as pm
import theano.tensor as T
import matplotlib.pyplot as plt

class NonParametric(object):
    def __init__(self, wave, flux, templates, adegree=None, mdegree=None,
                 reddening=False):
        """ Model CSP with bayesian model. """
        self.wave = wave
        self.flux = flux
        self.templates = templates
        self.ntemplates = len(templates)
        self.adegree = adegree
        # Construct additive polynomial
        if self.adegree is not None:
            _ = np.linspace(-1, 1, len(self.wave))
            self.apoly = np.zeros((adegree+1, len(_)))
            for i in range(adegree+1):
                self.apoly[i] = legendre(i)(_)
        else:
            self.apoly = np.zeros(1)
        # Build statistical model
        with pm.Model() as self.model:
            self.flux0 = pm.Normal("f0", mu=1, sd=5) # Multiplicative constant
            self.w = pm.Dirichlet("w", np.ones(self.ntemplates) /
                                 self.ntemplates)
            self.wpoly = pm.Deterministic("wpoly",
                                          pm.math.zeros_like(self.flux0))  \
                         if self.adegree is None  else \
                         pm.Normal("wpoly", mu=0, sd=1, shape=self.adegree)
            self.bestfit = pm.Deterministic("bestfit",
                      self.__call__(self.w, wpoly=self.wpoly,
                                    f0=self.flux0, math=pm.math))
            self.sigma = pm.Exponential("sigma", lam=0.01)
            self.like = pm.Normal('like', mu=self.bestfit, sd=self.sigma,
                                observed=flux)
            # pm.Cauchy("like", alpha=bestfit, beta=sigma, observed=flux)

    def __call__(self, w, wpoly=None, f0=1., math=np):
        if self.adegree is None:
            wpoly = np.zeros(1)
        else:
            wpoly = np.zeros(self.adegree) if self.wpoly is None else wpoly
        return f0 * (math.dot(w.T, self.templates) +
                     math.dot(wpoly.T, self.apoly))

    def NUTS_sampling(self, nsamp=2000, tune=1000, target_accept=0.9):
        """ Sampling the model using the NUTS method. """
        with self.model:
            self.trace = pm.sample(nsamp, tune=tune,
                                   nuts_kwargs={"target_accept": target_accept})

    def save(self, dbname):
        """ Save trace."""
        trace = self.trace
        vars = ["f0", "w", "sigma", "wpoly"]
        d = dict([(v, trace[v]) for v in vars])
        with open(dbname, 'wb') as f:
            pickle.dump(d, f)
        return

class Parametric(object):
    def __init__(self, wave, flux, templates, adegree=None, mdegree=None,
                 reddening=False):
        with pm.Model() as hierarchical_model:
            # Hyperpriors
            mu_age = pm.Normal("Age", mu=9, sd=1)
            mu_metal = pm.Normal("Metal", mu=0, sd=0.1)
            mu_alpha = pm.Normal("Alpha", mu=0.2, sd=.1)
            sigma_age = pm.Exponential('SAge', lam=1)
            sigma_metal = pm.Exponential('SMetal', lam=.1)
            sigma_alpha = pm.Exponential('SAlpha', lam=.1)
            ages = pm.Normal("age", mu=mu_age, sd=sigma_age, shape=N)
            metals = pm.Normal("metal", mu=mu_metal, sd=sigma_metal, shape=N)
            alphas = pm.Normal("alpha", mu=mu_alpha, sd=sigma_alpha, shape=N)
            eps = pm.Exponential("eps", lam=1, shape=N)
            w = [pm.Deterministic("w_{}".format(i), \
                                  T.exp(-0.5 * T.pow(
                                      (metals[i] - ssps.metals1D) / sigma_metal,
                                      2)) *
                                  T.exp(-0.5 * T.pow(
                                      (ages[i] - ssps.ages1D) / sigma_age, 2)) *
                                  T.exp(
                                      -0.5 * T.pow((alphas[i] - ssps.alphas1D) /
                                                   sigma_alpha, 2))) for i in
                 range(N)]
            bestfit = [pm.math.dot(w[i].T / T.sum(w[i]), ssps.flux) for i in
                       range(N)]
            like = [
                pm.Cauchy('like_{}'.format(i), alpha=bestfit[i], beta=eps[i],
                          observed=obs[i]) for i in range(N)]
        with hierarchical_model:
            trace = pm.sample(500, tune=500)
        vars = ["Age", "Metal", "Alpha", "SAge", "SMetal", "SAlpha", "age",
                "metal", "alpha", "eps"]
        d = dict([(v, trace[v]) for v in vars])
        with open(dbname, 'wb') as f:
            pickle.dump(d, f)

if __name__ == "__main__":
    pass