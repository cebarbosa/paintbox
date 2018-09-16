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
        self.robust_fitting = robust_fitting
        # Defining statistical model
        self.statmodel = "nssps" if statmodel is None else statmodel
        self.Nssps = self.ntemplates if self.statmodel == "npfit" else Nssps
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
            with pm.model():
                self.Rv = pm.Normal("Rv", mu=3.1, sd=1, shape=self.Nssps)
                self.ebv = pm.Exponential("ebv", lam=2, shape=self.Nssps)
                self.extinction = [tt.pow(10, -0.4 * self.ebv[i] *
                                          (self.kappa + self.Rv[i]))
                                   for i in range(self.Nssps)]
        else:
            self.extinction = np.ones(self.Nssps)
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
            with self.model:
                mcoeff = pm.Normal("mpoly", mu=0, sd=10, shape=self.mdegree + 1)
                self.continuum = tt.dot(mcoeff, self.mpoly)
        else:
            self.continuum = 1.
        ########################################################################
        if self.fluxerr is None:
            with self.model:
                self.sigma_y = pm.Exponential("sigma_y", lam=1)
        else:
            self.sigma_y = theano.shared(np.asarray(self.fluxerr,
                                    dtype=theano.config.floatX), name='sigma_y')
        ########################################################################
        # Defining appropriate model and plots
        self.models_dict = {"npfit": self.build_nonparametric_model,
                            "pfit": self.build_parametric_model,
                            "nssps": self.build_nssps_model}
        self.plot_corner_dict = {"npfit": self.plot_corner_nonparametric,
                                 "pfit": self.plot_corner_parametric,
                                 "nssps": self.plot_corner_nssps}
        self.plot_corner = self.plot_corner_dict.get(self.statmodel)
        # Build model
        self.models_dict.get(self.statmodel, lambda: "Invalid")()


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
            idx = tuple([self._idxs[i][v] for i,v in enumerate(p)] + \
                  [np.arange(len(self.templates[0]))])
            templates[idx] = self.templates[j]
        self.templates = templates
        self.model = pm.Model()
        with self.model:
            w = pm.Dirichlet("w", np.ones(N))
            categs = [pm.Categorical("{}_idx".format(self.params.colnames[i]),
                        np.ones_like(self._values[i]) / len(self._values[i]),
                        shape=N) for i in range(nparams)]
            s = "".join(["categs[{}],".format(i) for i in range(nparams)])
            ssps = eval("theano.shared(self.templates)[{}:]".format(s))
            csp = tt.dot(w.T, [ssps[i] * self.extinction[i] for i in range(N)])
            bestfit = csp * self.continuum
            ####################################################################
            if self.robust_fitting:
                nu = pm.Uniform("nu", lower=1, upper=100)
                self.residuals = pm.StudentT("residuals", mu=bestfit, nu=nu,
                                             sd=self.sigma_y,
                                             observed=self.flux)
            else:
                self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=self.sigma_y, observed=self.flux)

    def build_nonparametric_model(self):
        """ Build a non-parametric model for the fitting. """
        with self.model:
            loga = pm.Uniform("loga", lower=-3, upper=3,
                              shape=int(self.ntemplates))
            # logb = pm.Uniform("logb", lower=-3, upper=2)
            # regul = pm.Gamma("regul", alpha=pm.math.exp(np.log(10.) * loga),
            #                  beta=pm.math.exp(np.log(10.) * logb),
            #                  shape=int(self.ntemplates))
            w = pm.Dirichlet("w", pm.math.exp(np.log(10.) * loga),
                             shape=int(self.ntemplates))
            csp = pm.math.dot(w, [self.templates[i] * self.extinction[i] for i
                                   in range(self.Nssps)])
            bestfit = csp * self.continuum
            ####################################################################
            if self.robust_fitting:
                nu = pm.Uniform("nu", lower=1, upper=100)
                self.residuals = pm.StudentT("residuals", mu=bestfit, nu=nu,
                                             sd=self.sigma_y,
                                             observed=self.flux)
            else:
                self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=self.sigma_y, observed=self.flux)

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
                wps.append(pm.math.exp(-0.5 * tt.pow(
                                    (mus[i] - vals) / stds[i],  2)))
            ws = pm.math.prod(wps)
            weights = ws / tt.sum(ws)
            csp = pm.math.dot(weights.T / tt.sum(weights), self.templates)
            ####################################################################
            # Handling Reddening law
            if self.reddening:
                Rv = pm.Normal("Rv", mu=3.1, sd=1)
                ebv = pm.Exponential("ebv", lam=2)
                extinction = tt.pow(10, -0.4 * (self.kappa + Rv) * ebv)
            else:
                extinction = np.array([1.])
            bestfit =  self.flux0 * (extinction *
                                     pm.math.dot(weights.T / tt.sum(weights),
                                                 self.templates))
            eps = pm.Exponential("eps", lam=1)
            self.residuals = pm.Normal('residuals', mu=bestfit,
                                        sd=eps, observed=self.flux)

    def build_nssps_model_interp(self):
        """ Build a model assuming a number of SSPs. """
        # Making linear interpolation of templates
        self.ssp = SSP(self.params, self.templates)
        N = self.Nssps
        nparams = len(self.params.colnames)
        self.model = pm.Model()
        self.lower = [self.params[col].min() for col in
                      self.params.colnames]
        self.upper = [self.params[col].max() for col in
                      self.params.colnames]
        with self.model:
            if N > 1:
                w = pm.Dirichlet("w", np.ones(N))
            else:
                w = np.array([1.])
            ssps = []
            for n in range(N):
                pars = [pm.Uniform("{}_{}".format(self.params.colnames[i], n),
                        lower=self.lower[i], upper=self.upper[i]) for i in
                        range(nparams)]
                ssps.append(self.ssp(tt.as_tensor_variable(pars)))
            ssps = tt.as_tensor_variable(ssps)
            # Handling Reddening law
            if self.reddening:
                Rv = pm.Normal("Rv", mu=3.1, sd=1, shape=N)
                ebv = pm.Exponential("ebv", lam=2, shape=N)
                extinction = [tt.pow(10, -0.4 * ebv[i] * (self.kappa + Rv[i]))
                              for i in range(N)]
                csp = tt.dot(w.T, [ssps[i] * extinction[i] for i in range(N)])
            else:
                csp = tt.dot(w.T, ssps)
            ####################################################################
            # Handling multiplicative polynomial
            if self.mdegree >1:
                mpoly = pm.Normal("mpoly", mu=0, sd=10,
                                  shape=self.mdegree + 1)
                continuum = tt.dot(mpoly, self.mpoly)
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
        cmap = cm.get_cmap("viridis") if cmap is None else cm.get_cmap(cmap)
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
                        ax.hist(x.flatten(), weights=w.flatten() / chains,
                                color="C0", bins=binsx, density=True,
                                histtype='stepfilled')
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
        def calc_bins(vals):
            """ Returns the bins to be used for a discrete set of parameters."""
            vin = vals[:-1] + 0.5 * np.diff(vals)
            v0 = 2 * vals[0] - vin[0]
            vf = 2 * vals[-1] - vin[-1]
            vs = np.hstack([v0, vin, vf])
            return vs
        self._idxs, self._values = [], []
        npars = len(self.params.colnames)
        for par in self.params.colnames:
            vals = np.array(np.unique(self.params[par]))
            dict_ = dict([(x,i) for i,x in enumerate(vals)])
            self._idxs.append(dict_)
            self._values.append(vals)
        w = self.trace["w"].T
        nchains = len(w[0])
        shape = [len(_) for _ in self._idxs] + [nchains]
        weights = np.zeros(shape)
        for j, p in enumerate(np.array(self.params)):
            idx = tuple([self._idxs[i][v] for i,v in enumerate(p)] +
                        [np.arange(nchains)])
            weights[idx] = w[j]
        fig = plt.figure(figsize=(3.32153, 3.32153))
        for i, pi in enumerate(self.params.colnames):
            for j, pj in enumerate(self.params.colnames):
                if i < j:
                    continue
                print(i, j)
                ax = plt.subplot2grid((npars, npars),
                                      (i, j))
                ax.tick_params(right=True, top=True, axis="both",
                               direction='in', which="both",
                               width=0.5, pad=1, labelsize=6)
                ax.minorticks_on()
                axis = tuple(np.setdiff1d(np.arange(npars + 1),
                                          np.array([i, j])))
                data = np.sum(weights, axis=axis) / nchains
                if i == j:
                    print(self._values[i])
                    bins = calc_bins(self._values[i])
                    ax.bar(self._values[i], data, np.diff(bins))
                else:
                    binsx = calc_bins(self._values[i])
                    binsy =  calc_bins(self._values[j])
                    X, Y = np.meshgrid(self._values[j], self._values[i])
                    X, Y = np.meshgrid(binsy[:-1], binsx[:-1])
                    ax.pcolormesh(X, Y, data.T)
        plt.show()





    def build_parametric_model(self):
        pass

    def plot_corner_parametric(self):
        pass

class SSP(tt.Op):
    """ Class for linear interpolation of SSPs."""
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, params, templates):
        x = params.as_array()
        a = x.view((x.dtype[0], len(x.dtype.names)))
        self.func = LinearNDInterpolator(a, templates)
        # self.sspgrad = SSPGrad(self.func)

    def perform(self, node, inputs, outputs):
        theta, = inputs  # this will contain my variables
        outputs[0][0] = self.func(*theta)

    # def grad(self, inputs, g):
    #     theta, = inputs  # our parameters
    #     return [tt.dot(g[0], self.sspgrad(theta))]

class SSPGrad(tt.Op):
    """ Calculates the Jacobian of the SSPs. """
    itypes = [tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, func, eps=1e-6):
        self.func = func
        self.eps = eps

    def perform(self, node, inputs, outputs):
        theta, = inputs  # this will contain my variables
        jacob = approx_jacobian(self.func, theta)
        outputs[0][0] = jacob

def approx_jacobian(func, coord, eps=1e-6):
    """ Estimates the  Jacobian using finite-diference approximation. """
    grads = []
    x = np.array(coord)
    for e in np.eye(len(x)) * eps:
        partial = (func(x + e) - func(x - e)) / (2 * eps)
        grads.append(partial[0])
    grads = np.array(grads).T
    return grads

if __name__ == "__main__":
    pass