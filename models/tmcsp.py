# -*- coding: utf-8 -*-
""" 

Created on 18/05/18

Author : Carlos Eduardo Barbosa

TMCSP: A full Bayesian Template-Matching based modeling of Composite Stellar
Populations.

"""

from __future__ import print_function, division

import os
import pickle

import numpy as np
from scipy.special import legendre
import pymc3 as pm
import matplotlib.pyplot as plt

class TMCSP():
    def __init__(self, wave, flux, templates, adegree=None):
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

if __name__ == "__main__":
    pass