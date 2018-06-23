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
        self.ntemplates = len(templates)
        with pm.Model() as self.model:
            flux0 = pm.Normal("f0", mu=1, sd=5) # Multiplicative constant
            w = pm.Dirichlet("w", np.ones(self.ntemplates) / self.ntemplates)
        if adegree is not None:
            x = np.linspace(-1, 1, len(wave))
            apoly = np.zeros((adegree, len(x)))
            for i in range(adegree):
                apoly[i] = legendre(i)(x)
            with self.model:
                wpoly = pm.Normal("wpoly", mu=0, sd=1, shape=adegree)
                bestfit = pm.Deterministic("bestfit", flux0 * (pm.math.dot(w.T,\
                             templates) + pm.math.dot(wpoly.T, apoly)))
        else:
            with self.model:
                bestfit = pm.Deterministic("bestfit", flux0 * (pm.math.dot(w.T,\
                                                                   templates)))
        with self.model:
            sigma = pm.Exponential("sigma", lam=0.01)
            pm.Normal('like', mu=bestfit, sd=sigma, observed=flux)
            # pm.Cauchy("like", alpha=bestfit, beta=sigma, observed=flux)

    def NUTS_sampling(self, nsamp=2000, tune=1000, target_accept=0.9):
        """ Sampling the model using the NUTS method. """
        with self.model:
            self.trace = pm.sample(nsamp, tune=tune, nuts_kwargs={
                "target_accept": target_accept})

    def save(self, dbname):
        """ Save trace."""
        results = {'model': self.model, 'trace': self.trace}
        with open(dbname, 'wb') as buff:
            pickle.dump(results, buff)
        return

if __name__ == "__main__":
    pass