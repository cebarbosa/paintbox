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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import astropy.units as u

class BSF(object):
    def __init__(self, wave, flux, templates, adegree=None, mdegree=0,
                 params=None, reddening=False, statmodel=None, Nssps=10,
                 fluxerr=None, robust_fitting=True):
        """ Model stellar populations with bayesian model. """
        self.wave = wave
        self.flux = flux
        self.templates = templates
        self.ntemplates = len(templates)
        self.adegree = adegree
        self.mdegree = mdegree
        self.reddening = reddening
        self.params = params
        self.fluxerr = fluxerr
        self.Nssps = Nssps
        self.robust_fitting = robust_fitting
        self.statmodel = "npfit" if statmodel is None else statmodel
        self.models = {"npfit": self.build_nonparametric_model,
                       "pfit": self.build_parametric_model,
                       "nssps": self.build_nssps_model}
        self.plotcorner = {"npfit": self.plot_corner_nonparametric,
                       "pfit": self.plot_corner_parametric,
                       "nssps": self.plot_corner_nssps}
        build = self.models.get(self.statmodel)
        self.plot_corner = self.plotcorner.get(self.statmodel)
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
            self.apoly = np.zeros((adegree, len(_)))
            for i in range(adegree):
                self.apoly[i] = legendre(i+1)(_)
        else:
            self.apoly = 0.
        ########################################################################
        # Construct multiplicative polynomial
        if self.mdegree > 0:
            _ = np.linspace(-1, 1, len(self.wave))
            self.mpoly = np.zeros((self.mdegree+1, len(_)))
            for i in range(self.mdegree+1):
                self.mpoly[i] = legendre(i)(_)
        else:
            self.mpoly = 1.
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
            bestfit =  self.flux0 * (extinction *
                                     pm.math.dot(weights.T / T.sum(weights),
                                    self.templates))
            eps = pm.Exponential("eps", lam=1)
            self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=eps, observed=self.flux)

    def build_nssps_model(self):
        """ Build a model assuming a number of SSPs. """
        N = self.Nssps
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
            categs = [pm.Categorical("{}_idx".format(self.params.colnames[i]),
                        np.ones_like(self._values[i]) / len(self._values[i]),
                        shape=N) for i in range(nparams)]
            s = "".join(["categs[{}],".format(i) for i in range(nparams)])
            ssp = eval("theano.shared(self.templates)[{}:]".format(s))
            # Handling Reddening law
            if self.reddening:
                Rv = pm.Normal("Rv", mu=3.1, sd=1, shape=N)
                ebv = pm.Exponential("ebv", lam=2, shape=N)
                extinction = [T.pow(10, -0.4 * ebv[i] * (self.kappa + Rv[i]))
                              for i in range(N)]
                csp = T.dot(w.T, [ssp[i] * extinction[i] for i in range(N)])
            else:
                csp = T.dot(w.T, ssp)
            ####################################################################
            # Handling multiplicative polynomial
            if self.mdegree >1:
                mpoly = pm.Normal("mpoly", mu=0, sd=10,
                                  shape=self.mdegree + 1)
                continuum = T.dot(mpoly, self.mpoly)
            else:
                continuum = pm.Normal("mpoly", mu=0, sd=10)
            ####################################################################
            bestfit = csp * continuum
            if self.fluxerr is None:
                sigma_y = pm.Exponential("sigma_y", lam=1)
            else:
                sigma_y = theano.shared(np.asarray(self.fluxerr,
                                        dtype=theano.config.floatX),
                                        name='sigma_y')
            ####################################################################
            if self.robust_fitting:
                nu = pm.Uniform("nu", lower=1, upper=100)
                self.residuals = pm.StudentT("residuals", mu=bestfit, nu=nu,
                                             sd=sigma_y, observed=self.flux)
            else:
                self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=sigma_y, observed=self.flux)


    def plot_corner_nssps(self, labels=None, cmap=None):
        """ Produces plot for model with N SSPs."""
        cmap = cm.get_cmap("cubehelix_r") if cmap is None else cm.get_cmap(cmap)
        def calc_bins(vals):
            """ Returns the bins to be used for a discrete set of parameters."""
            vin = vals[:-1] + 0.5 * np.diff(vals)
            v0 = 2 * vals[0] - vin[0]
            vf = 2 * vals[-1] - vin[-1]
            vs = np.hstack([v0, vin, vf])
            return vs
        npars = len(self.params.colnames)
        if labels is None:
            labels = self.params.colnames
        with self.model:
            w = self.trace["w"]
            fig = plt.figure(figsize=(3.32153, 3.32153))
            for i, pi in enumerate(self.params.colnames):
                for j, pj in enumerate(self.params.colnames):
                    if i < j:
                        continue
                    ax = plt.subplot2grid((npars, npars),
                                          (i,j))
                    ax.tick_params(right=True, top=True, axis="both",
                                   direction='in', which="both",
                                   width=0.5, pad=1, labelsize=6)
                    ax.minorticks_on()
                    tracei = self.trace["{}_idx".format(pi)]
                    x = self._values[i][tracei]
                    chains = tracei.shape[0]
                    binsx = calc_bins(self._values[i])
                    median = np.percentile(np.sum(w * x, axis=1), 50)
                    p05 = np.percentile(np.sum(w * x, axis=1), 50-34.14)
                    p95 = np.percentile(np.sum(w * x, axis=1), 50+34.14)
                    if i == j:
                        N, bins, patches = ax.hist(x.flatten(),
                            weights=w.flatten() / chains,
                            color="b", ec="k",
                            bins=binsx, density=True, edgecolor="k",
                                                   histtype='bar')
                        fracs = N.astype(float) / N.max()
                        norm = Normalize(-.2 * fracs.max(), 1.5 * fracs.max())
                        for thisfrac, thispatch in zip(fracs, patches):
                            color = cmap(norm(thisfrac))
                            thispatch.set_facecolor(color)
                            thispatch.set_edgecolor("none")
                        ax.tick_params(labelleft=False)
                        ax.set_xlim(binsx[0], binsx[-1])
                        ax.set_title("{0}=${1:.2f}_{{-{2:.2f}}}^{{+{"
                                     "3:.2f}}}$".format(
                                    labels[i], median, median - p05,
                                    p95 - median), fontsize=4, pad=1.5)
                        for perc in [p05, median, p95]:
                            ax.axvline(perc, ls="--", c="k", lw=0.3)
                    elif i > j:
                        tracej = self.trace["{}_idx".format(pj)]
                        y = self._values[j][tracej]
                        binsy = calc_bins(self._values[j])
                        H, xedges, yedges = np.histogram2d(x.flatten(),
                                                           y.flatten(),
                                            weights=w.flatten() / chains,
                                            bins=(binsx, binsy))
                        Y, X = np.meshgrid(xedges, yedges)
                        ax.pcolormesh(X.T, Y.T, H, cmap=cmap)
                        if j == 0:
                            ax.set_ylabel(labels[i], fontsize=8)
                        else:
                            ax.set_yticklabels([])
                    if i == npars - 1:
                        ax.set_xlabel(labels[j], fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    for axis in ['top', 'bottom', 'left', 'right']:
                        ax.spines[axis].set_linewidth(0.5)
            plt.subplots_adjust(hspace=0.04, wspace=0.04, right=0.985, top=0.96,
                                left=0.10, bottom=0.09)
            fig.align_labels()


    def plot_corner_nonparametric(self):
        pass

    def plot_corner_parametric(self):
        pass

if __name__ == "__main__":
    pass