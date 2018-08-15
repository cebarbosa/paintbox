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

import numpy as np
from scipy.special import legendre
import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import astropy.units as u

class BSF(object):
    def __init__(self, wave, flux, templates, adegree=None, params=None,
                 reddening=True, statmodel=None):
        """ Model CSP with bayesian model. """
        self.wave = wave
        self.flux = flux
        self.templates = templates
        self.ntemplates = len(templates)
        self.adegree = adegree
        self.reddening = reddening
        self.params = params
        self.statmodel = "npfit" if statmodel is None else statmodel
        self.models = {"npfit": self.build_nonparametric_model,
                       "pfit": self.build_parametric_model,
                       "nssps": self.build_nssps_model}
        self.plotfunc = {"npfit": self.plot_nonparametric_model,
                       "pfit": self.plot_parametric_model,
                       "nssps": self.plot_nssps_model}
        build = self.models.get(self.statmodel)
        self.plot = self.plotfunc.get(self.statmodel)
        #######################################################################
        # Preparing array for redenning
        if not hasattr(self.wave, "_unit"):
            self.wave = self.wave * u.angstrom
        elif (self.wave.unit == u.dimensionless_unscaled) or \
             (self.wave.unit is None):
            self.wave = self.wave * u.angstrom
        x = 1 / self.wave.to("micrometer") * u.micrometer
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                              2.659 * (-1.857 + 1.040 * x),
                      2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 +0.011*x**3))
        ########################################################################
        # Construct additive polynomial
        if self.adegree is not None:
            _ = np.linspace(-1, 1, len(self.wave))
            self.apoly = np.zeros((adegree+1, len(_)))
            for i in range(adegree+1):
                self.apoly[i] = legendre(i)(_)
        else:
            self.apoly = np.zeros(1)
        build()

    def build_nonparametric_model(self):
        """ Build a non-parametric model for the fitting. """
        self.model = pm.Model()
        with self.model:
            self.w = pm.Dirichlet("w", np.ones(self.ntemplates))
            self.flux0 = pm.Normal("f0", mu=1, sd=5)  #
            ####################################################################
            # Handling Additive polynomials
            if self.adegree is None:
                self.wpoly = pm.math.zeros_like(self.flux0)
            else:
                self.wpoly = pm.Normal("wpoly", mu=0, sd=1, shape=self.adegree)
            ####################################################################
            # Handling Reddening law
            if self.reddening is None:
                self.extinction = pm.math.ones_like(self.flux0)
            else:
                self.Rv = pm.Normal("Rv", mu=3.1, sd=1)
                self.ebv = pm.Exponential("ebv", lam=2)
                self.extinction = T.pow(10, -0.4 * (self.kappa + self.Rv) *
                                                    self.ebv)
            ####################################################################
            self.bestfit = self.flux0 * self.extinction * \
                          (pm.math.dot(self.w.T, self.templates) + \
                           pm.math.dot(self.wpoly.T, self.apoly))
            self.sigma = pm.Exponential("sigma", lam=1)
            self.residuals = pm.Normal('residuals', mu=self.bestfit,
                                        sd=self.sigma, observed=self.flux)
            # pm.Cauchy("like", alpha=bestfit, beta=sigma, observed=flux)

    def build_parametric_model(self):
        """ Build a parametric model for the fitting. """
        self.model = pm.Model()
        with self.model:
            self.flux0 = pm.Normal("f0", mu=1, sd=5)
            mus, stds = [], []
            wps = [] # Partial weights
            for i, p in enumerate(self.params.colnames):
                vals = np.array(self.params[p], dtype=np.float32)
                mus.append(pm.Normal("{}_mean".format(p),
                                          mu=vals.mean(),
                                          sd=vals.std()))
                stds.append(pm.Exponential("{}_std".format(p),
                                                lam=1/vals.std()))
                wps.append(pm.math.exp(-0.5 * T.pow(
                                    (mus[i] - vals) / stds[i],  2)))
            ws = pm.math.prod(wps)
            weights = ws / T.sum(ws)
            csp = pm.math.dot(weights.T / T.sum(weights), self.templates)
            ####################################################################
            # Handling Reddening law
            if self.reddening:
                Rv = pm.Normal("Rv", mu=3.1, sd=1)
                ebv = pm.Exponential("ebv", lam=2)
                extinction = T.pow(10, -0.4 * (self.kappa + Rv) * ebv)
            else:
                extinction = pm.math.ones_like(self.flux0)
            ####################################################################
            bestfit =  self.flux0 * extinction * \
                       (pm.math.dot(weights.T / T.sum(weights), self.templates))
            eps = pm.Exponential("eps", lam=1)
            self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=eps, observed=self.flux)

    def build_nssps_model(self, N=10):
        """ Build a model assuming a number of SSPs. """
        self.Nssps = N
        self._idxs, self._values = [], []
        nparams = len(self.params.colnames)
        for par in self.params.colnames:
            vals = np.array(np.unique(self.params[par]))
            dict_ = dict([(x,i) for i,x in enumerate(vals)])
            self._idxs.append(dict_)
            self._values.append(vals)
        shape = [len(_) for _ in self._idxs] + [len(self.templates[0])]
        templates = np.zeros(shape)
        for j, p in enumerate(np.array(self.params)):
            idx = [self._idxs[i][v] for i,v in enumerate(p)] + \
                  [np.arange(len(self.templates[0]))]
            templates[tuple(idx)] = self.templates[j]
        self.templates = templates
        self.model = pm.Model()
        with self.model:
            w = pm.Dirichlet("w", np.ones(N))
            categs = []
            for i in range(nparams):
                categs.append(pm.Categorical("{}_idx".format(
                              self.params.colnames[i]),
                              np.ones_like(self._values[i]) / len(self._values[i]),
                    shape=N))
            ssp =  theano.shared(self.templates)[categs[0], categs[1],
                                                categs[2], categs[3],
                                                categs[4], :]
            ####################################################################
            # Handling Reddening law
            if self.reddening:
                Rv = pm.Normal("Rv", mu=3.1, sd=1)
                ebv = pm.Exponential("ebv", lam=2)
                extinction = T.pow(10, -0.4 * (self.kappa + Rv) * ebv)
            else:
                extinction = pm.math.ones_like(self.flux0)
            ###################################################################
            self.flux0 = pm.Normal("f0", mu=1, sd=5)
            bestfit =  self.flux0 * extinction * T.dot(w.T, ssp)
            eps = pm.Exponential("eps", lam=1)
            self.residuals = pm.Cauchy("resid", alpha=bestfit, beta=eps,
                                observed=self.flux)

    def plot_nssps_model(self):
        """ Produces plot for model with N SSPs."""
        def calc_bins(vals):
            """ Returns the bins to be used for a discrete set of parameters."""
            delta = 0.49 * np.diff(vals).min()
            return np.unique(np.column_stack((vals - delta, vals + delta)))
        npars = len(self.params.colnames)
        with self.model:
            w = self.trace["w"]
            fig = plt.figure()
            for i, pi in enumerate(self.params.colnames):
                for j, pj in enumerate(self.params.colnames):
                    if i < j:
                        continue
                    ax = plt.subplot2grid((npars, npars),
                                          (i,j))
                    tracei = self.trace["{}_idx".format(pi)]
                    x = self._values[i][tracei].flatten()
                    chains = tracei.shape[0]
                    binsx = calc_bins(self._values[i])
                    if i == j:
                        ax.hist(x, weights=w.flatten() / chains, bins=binsx)
                        ax.set_xlabel(pi)
                        ax.set_xlim(binsx[0], binsx[-1])
                    elif i > j:
                        tracej = self.trace["{}_idx".format(pj)]
                        y = self._values[j][tracej].flatten()
                        binsy = calc_bins(self._values[j])
                        H, xedges, yedges = np.histogram2d(x, y,
                                            weights=w.flatten() / chains,
                                            bins=(binsx, binsy))
                        Y, X = np.meshgrid(xedges, yedges)
                        ax.pcolormesh(X.T, Y.T, H, cmap="Blues")
                        if i == npars - 1:
                            ax.set_xlabel(pj)
                        if j == 0:
                            ax.set_ylabel(pi)

            plt.show()


    def plot_nonparametric_model(self):
        pass

    def plot_parametric_model(self):
        pass

if __name__ == "__main__":
    pass