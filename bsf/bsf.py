# -*- coding: utf-8 -*-
"""

Created on 18/05/18

Author : Carlos Eduardo Barbosa

"""

from __future__ import print_function, division

from builtins import range
from builtins import object

import numpy as np
from scipy.special import legendre
import pymc3 as pm
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import LinearNDInterpolator
import astropy.units as u
from uncertainties import ufloat

class BSF(object):
    def __init__(self, wave, flux, templates, adegree=None, mdegree=0,
                 params=None, reddening=False, fluxerr=None,
                 robust_fitting=True):
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
        self.robust_fitting = robust_fitting
        self.model = pm.Model()
        #######################################################################
        # Preparing array for redenning
        if self.reddening:
            if not hasattr(self.wave, "_unit"):
                self.wave = self.wave * u.angstrom
            elif (self.wave.unit == u.dimensionless_unscaled) or \
                 (self.wave.unit is None):
                self.wave = self.wave * u.angstrom
            x = 1 / self.wave.to("micrometer") * u.micrometer
            self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                                  2.659 * (-1.857 + 1.040 * x),  \
                                  2.659 * (-2.156 + 1.509 * x - 0.198 * x**2
                                  +0.011*x**3))
            self.kappas= np.tile(self.kappa, (self.ntemplates, 1))
            with self.model:
                Rv = pm.Normal("Rv", mu=3.1, sd=1)
                ebv = pm.Exponential("ebv", lam=2, shape=self.ntemplates)
                extinction = tt.pow(10, -0.4 * tt.dot(ebv, self.kappas + Rv))
        else:
            extinction = 1.
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
            mpoly = np.zeros((self.mdegree+1, len(_)))
            for i in range(self.mdegree+1):
                mpoly[i] = legendre(i)(_)
            with self.model:
                mcoeff = pm.Normal("mpoly", mu=0, sd=10, shape=self.mdegree + 1)
                self.mpoly = tt.dot(mcoeff, mpoly)
        else:
            self.mpoly = 1.
        ########################################################################
        if self.fluxerr is None:
            with self.model:
                self.sigma_y = pm.Exponential("sigma_y", lam=1)
        else:
            self.sigma_y = theano.shared(np.asarray(self.fluxerr,
                                    dtype=theano.config.floatX), name='sigma_y')
        ########################################################################

        N = int(self.ntemplates)
        with self.model:
            ####################################################################
            # Priors for the weights
            a = pm.HalfCauchy("a", beta=1, shape=N)
            w = pm.Dirichlet("w", a, shape=N)
            csp = pm.math.dot(w, self.templates * extinction)
            bestfit = csp * self.mpoly + self.apoly
            ####################################################################
            if self.robust_fitting:
                nu = pm.Uniform("nu", lower=1, upper=100)
                self.residuals = pm.StudentT("residuals", mu=bestfit, nu=nu,
                                             sd=self.sigma_y,
                                             observed=self.flux)
            else:
                self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=self.sigma_y, observed=self.flux)

    def plot_corner(self):
        def calc_bins(vals):
            """ Returns the bins to be used for a discrete set of parameters."""
            vin = vals[:-1] + 0.5 * np.diff(vals)
            v0 = 2 * vals[0] - vin[0]
            vf = 2 * vals[-1] - vin[-1]
            vs = np.hstack([v0, vin, vf])
            return vs
        self.calc_estimates()

        fig = plt.figure(figsize=(3.32153, 3.32153))
        for i, pi in enumerate(self.params.colnames):
            for j, pj in enumerate(self.params.colnames):
                if i < j:
                    continue
                ax = plt.subplot2grid((self.npars, self.npars), (i, j))
                ax.tick_params(right=True, top=True, axis="both",
                               direction='in', which="both",
                               width=0.5, pad=1, labelsize=6)
                ax.minorticks_on()
                axis = tuple(np.setdiff1d(np.arange(self.npars + 1),
                                          np.array([i, j])))
                data = np.sum(self.weights, axis=axis) / self.nchains
                if i == j:
                    bins = calc_bins(self._values[i])
                    ax.bar(self._values[i], data, np.diff(bins))
                else:
                    binsx = calc_bins(self._values[i])
                    binsy =  calc_bins(self._values[j])
                    X, Y = np.meshgrid(self._values[j], self._values[i])
                    X, Y = np.meshgrid(binsy[:-1], binsx[:-1])
                    ax.pcolormesh(X, Y, data.T)
        plt.show()

    def calc_estimates(self):
        """ Uses trace to calculate parameters of interest. """
        self._idxs, self._values = [], []
        for par in self.params.colnames:
            vals = np.array(np.unique(self.params[par]))
            dict_ = dict([(x,i) for i,x in enumerate(vals)])
            self._idxs.append(dict_)
            self._values.append(vals)
        w = self.trace["w"].T
        self.nchains = len(w[0])
        shape = [len(_) for _ in self._idxs] + [self.nchains]
        self.weights = np.zeros(shape)
        for j, p in enumerate(np.array(self.params)):
            idx = tuple([self._idxs[i][v] for i,v in enumerate(p)] +
                        [np.arange(self.nchains)])
            self.weights[idx] = w[j]
        self.npars = len(self.params.colnames)
        self.means = {}
        self.estimates = {}
        for i,par in enumerate(self.params.colnames):
            axis = tuple(np.setdiff1d(np.arange(self.npars), np.array([i])))
            w = np.sum(self.weights, axis=axis).T
            x = np.tile(self._values[i], (len(w), 1))
            v1 = np.sum(w * x, axis=1)
            v2 = np.sqrt(np.sum(w * (x - v1[:,None])**2, axis=1))
            self.estimates[par] = {"mean" : ufloat(np.mean(v1), np.std(v1)),
                                   "std" :  ufloat(np.mean(v2), np.std(v2))}


if __name__ == "__main__":
    pass