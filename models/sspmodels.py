# -*- coding: utf-8 -*-
""" 

Created on 06/03/18

Author : Carlos Eduardo Barbosa

Simple version of pPXF using Bayesian approach.

"""
from __future__ import print_function, division

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

def single_stellar_population(templates, spec):
    """ Simple template matching. """
    fakeobs = templates[10]
    norm = np.median(fakeobs)
    shape = templates.shape[:-1]
    matrix = T.as_tensor_variable(templates)
    alpha = np.ones(shape)
    with pm.Model() as model:
        idx = pm.Categorical("idx", alpha, shape=1)
        like = pm.Normal("like", mu=matrix[idx], sd=1., observed=fakeobs)
    with model:
        trace = pm.sample(1000)
    ibest = np.median(trace["idx"]).astype(int)
    plt.plot(fakeobs, "-")
    plt.plot(templates[ibest], "-")
    plt.show()


def composite_stellar_population(templates, spec, K=2):
    """ Simplified version of pPXF using Bayesian approach.

     TODO: This is just a prototype for the function, requires more work.
     """
    matrix = T.as_tensor_variable(templates)
    alpha = np.ones(len(templates)) / len(templates)
    fakeobs = 0.6 * templates[10] + 0.4 * templates[100]

    with pm.Model() as model:
        idx = pm.Categorical("idx", alpha, shape=K)
        w = pm.Dirichlet("w", np.ones(K))
        y = T.dot(w, matrix[idx])
        y = pm.Deterministic("y", y)
        like = pm.Normal("like", mu=y, sd=1., observed=fakeobs)
    with model:
        trace = pm.sample(1000)
    pm.traceplot(trace)
    plt.show()
    # plt.plot(fakeobs, "-")
    # plt.show()

