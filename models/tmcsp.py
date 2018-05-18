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

def tmcsp(wave, flux, templates, dbname, redo=False, adegree=10):
    """ Model CSP with bayesian model. """
    if os.path.exists(dbname) and not redo:
        return

    with pm.Model() as model:
        flux0 = pm.Normal("f0", mu=1, sd=5)
        w = pm.Dirichlet("w", np.ones(len(templates)))

    if adegree is not None:
        x = np.linspace(-1, 1, len(wave))
        apoly = np.zeros((adegree, len(x)))
        for i in range(adegree):
            apoly[i] = legendre(i)(x)
        with model:
            wpoly = pm.Normal("wpoly", mu=0, sd=1, shape=adegree)
            bestfit = pm.Deterministic("bestfit", flux0 * (pm.math.dot(w.T, \
                                    templates) + pm.math.dot(wpoly.T, apoly)))
    else:
        with model:
            bestfit = pm.Deterministic("bestfit", flux0 * (pm.math.dot(w.T, \
                                                                   templates)))

    with model:
        sigma = pm.Exponential("sigma", lam=1)
        pm.Normal('like', mu=bestfit, sd = sigma, observed=flux)
        # pm.Cauchy("like", alpha=bestfit, beta=sigma, observed=flux)
    with model:
        trace = pm.sample(1000, tune=1000)
    results = {'model': model, "trace": trace}
    with open(dbname, 'wb') as buff:
        pickle.dump(results, buff)
    return

if __name__ == "__main__":
    pass