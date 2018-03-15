# -*- coding: utf-8 -*-
""" 

Created on 06/03/18

Author : Carlos Eduardo Barbosa

Simple version of pPXF using Bayesian approach.

"""
from __future__ import print_function, division

import numpy as np
import pymc3 as pm
from theano import tensor as tt
import matplotlib.pyplot as plt

def bppxf(templates, spec, K=5):
    """ Simplified version of pPXF using Bayesian approach. """
    def stick_breaking(beta):
        portion_remaining = tt.concatenate(
            [[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining
    templates = templates[:5]
    ntemp = len(templates)
    fakeobs = 0.6 * templates[0] + 0.5 * templates[3]
    with pm.Model() as model:
        alpha = pm.DiscreteUniform("t", 0, ntemp)
        beta = pm.Beta('beta', 1., alpha, shape=K)
        w = pm.Deterministic('w', stick_breaking(beta))
        # eps = pm.HalfCauchy(r"eps", 5)
        # likelihood = pm.Normal('y', alpha=m, beta=eps, observed=fakeobs)

        # beta = pm.Beta('beta', 1., alpha, shape=ntemp)
        # w = pm.Deterministic('w', stick_breaking(beta))

    with model:
        trace = pm.sample(10)
    for w in trace["w"]:
        print(w.sum())
    print(trace["w"].shape)
    return