# -*- coding: utf-8 -*-
"""

Created on 12/04/19

Author : Carlos Eduardo Barbosa

Hierarchical Bayesian modeling of multi-band/index observations.

"""
from __future__ import print_function, division

import warnings

import numpy as np
from astropy.table import Table
import astropy.units as u
from scipy.interpolate import LinearNDInterpolator
from scipy.special import gamma, digamma
from scipy.optimize import least_squares
import theano.tensor as tt
import pymc3 as pm

class HBfit():
    def __init__(self, wave, data, errors, templates, params, npop=1,
                 gas_templates=None, gas_names=None):
        """ Model observations with SSP models using hierarchical Bayesian
        approach and probabilisti programming. """
        self.wave = wave
        self.data = data
        self.errors = errors
        self.templates = templates
        self.params = params
        self.nloc, self.nbands = data.shape
        self.npop = int(npop)
        self.gas_templates = gas_templates
        self.gas_names = gas_names
        self.ngas = len(self.gas_names) if self.gas_names is not None else 0.
        self.spm = CSPModel(self.wave, self.params, self.templates,
                            npop=self.npop, gas_templates=self.gas_templates,
                            gas_names=self.gas_names)

    def build_model(self):
        print("Generating model...")
        # Using weakly informative priors for parameters for stellar populations
        lower = np.array([self.params[col].min() for col in
                          self.params.colnames])
        upper = np.array([self.params[col].max() for col in
                          self.params.colnames])
        delta = 0.5 * (upper - lower)
        mus = 0.5 * (upper + lower)
        n = len(mus)
        pert = np.random.uniform(0., 1., size=self.npop * n).reshape((n,
                                                                    self.npop))
        vtest = lower[:,None] + (upper - lower)[:,None] * pert
        # Setting properties of priors for the logarithm of the flux
        data_median = np.median(self.data)
        lnfm = np.log(data_median / np.median(self.templates))
        lnfsd = np.sqrt(np.log(self.data.std())**2 +
                        np.log(self.templates.std())**2)
        lnf_testvals = np.random.normal(lnfm, lnfsd, size=self.npop)
        self.model = pm.Model()
        with self.model:
            # Extinction
            Av = pm.HalfCauchy("Av", beta=1., shape=self.nloc, testval=0.1)
            theta = [Av]
            # Stellar population parameters
            for j in range(self.npop):
                # Unobscured flux
                lnfj = pm.Normal("lnf{}".format(j), mu=lnfm, sd= 3.* lnfsd,
                                  shape=self.nloc, testval=lnf_testvals[j])
                fj = pm.math.exp(lnfj)
                theta.append(fj)
                for i, param in enumerate(self.params.colnames):
                    BNormal = pm.Bound(pm.Normal, lower=lower[i], upper=upper[i])
                    M = BNormal("M{}{}".format(param, j), mu=mus[i],
                                sd=delta[i], testval=vtest[i][j])
                    v = BNormal("{}{}".format(param,j), mu=M, sd=delta[i],
                                shape=self.nloc, testval=vtest[i][j])
                    theta.append(v)
            for k in range(self.ngas):
                lnfk = pm.Exponential("ln{}".format(self.gas_names[k]),
                                 lam=1., shape=self.nloc, testval=0.1)
                fk = pm.math.exp(lnfk)
                theta.append(fk)
            # Setting degrees-of-freedom of likelihood
            BGamma = pm.Bound(pm.Gamma, lower=2.01)
            nu = BGamma("nu", alpha=2., beta=.1, shape=self.nloc,
                          testval=10.)
            theta.append(nu)
            theta = tt.as_tensor_variable(theta).T
            # Building the likelihood
            for i in range(self.nloc):
                logl = LogLikeWithGrad(self.data[i],
                                       self.wave, self.errors[i], self.spm)
                # use a DensityDist
                pm.DensityDist('likelihood{}'.format(i), lambda v: logl(v),
                               observed={'v': theta[i]})

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta)

        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, likelihood):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.likelihood = likelihood

    def perform(self, node, inputs, outputs):
        theta, = inputs
        # calculate gradients
        grads = self.likelihood.gradient(theta)
        outputs[0][0] = grads

class CSPModel():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, gas_templates=None,
                 gas_names=None, Rv=None, npop=2):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.gas_templates = gas_templates
        self.gas_names = gas_names
        self.ssp = SSPInterpolator(self.params, self.templates)
        self.ncols = len(self.params.colnames)
        self.nparams = self.npop * (len(self.params.colnames) + 1)  + 1
        self.ngas = len(self.gas_names) if self.gas_names is not None else 0.
        self.shape = (self.nparams + self.ngas, len(self.wave))
        ########################################################################
        # Preparing array for redenning
        x = 1 / self.wave.to("micrometer") * u.micrometer
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                         2.659 * (-1.857 + 1.040 * x), \
                         2.659 * (-2.156 + 1.509 * x - 0.198 * x *x
                                  + 0.011 * (x * x * x)))
        self.Rv = 4.05 if Rv is None else Rv
        ########################################################################

    def extinction(self, Av):
        return np.power(10, -0.4  * Av * (1. + self.kappa / self.Rv))

    def __call__(self, theta):
        p = theta[1:self.nparams].reshape(self.npop, -1)
        stpop = self.extinction(theta[0]) * np.dot(p[:,0], self.ssp(p[:,1:]))
        if self.ngas == 0:
            return stpop
        g = theta[self.nparams:]
        gas = np.dot(g, self.gas_templates)
        return stpop + gas

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        # dF / dAv
        grad[0] = self.__call__(theta) * np.log(10) * \
                  (-0.4 * (1. + self.kappa / self.Rv))
        #
        ps = theta[1:self.nparams].reshape(self.npop, -1)
        gs = theta[self.nparams:]
        const = self.extinction(theta[0])
        for i, p in enumerate(ps):
            idx = 1 + (i * (self.ncols + 1))
            # dF/dFi
            grad[idx] = const * self.ssp(p[1:])
            # dF / dSSPi
            grad[idx+1:idx+1+self.ncols] = const * p[0] * \
                                           self.ssp.gradient(p[1:])
        grad[self.nparams:] = self.gas_templates
        return grad

class SSPInterpolator():
    """ Linear interpolation of models based on templates."""
    def __init__(self, params, templates):
        self.params = params
        self.templates = templates
        self.nparams = len(self.params.colnames)
        ########################################################################
        # Interpolating models
        x = self.params.as_array()
        a = x.view((x.dtype[0], len(x.dtype.names)))
        self.f = LinearNDInterpolator(a, templates, fill_value=0.)
        ########################################################################
        # Get grid points to handle derivatives
        inner_grid = []
        thetamin = []
        thetamax = []
        for par in self.params.colnames:
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)

    def gradient(self, theta, eps=1e-5):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        # Avoinding inner grid nodes where the gradient is indefined
        for i in range(self.nparams):
            if np.abs(np.min(theta[i] - self.inner_grid[i])) < eps:
                return 0.
        epsilons = np.eye(self.nparams) * eps
        grads = []
        for i, epsilon in enumerate(epsilons):
            tp = theta + epsilon
            tm = theta - epsilon
            diff = (self.__call__(tp) - self.__call__(tm)) / (2 * eps)
            grads.append(diff.flatten())
        grads = np.array(grads)
        return grads


class StudTLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams + 1

    def __call__(self, theta):
        nu = theta[-1]
        e_i = self.func(theta[:-1]) - self.data
        x = 1. + np.power(e_i / self.sigma, 2.) / (nu - 2)
        LLF = self.N * np.log(gamma(0.5 * (nu + 1)) /
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu)) + \
              - 0.5 * (nu + 1) * np.sum(np.log(x))
        # - 0.5 * np.sum(np.log(sigma**2)) # Constant
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.func.nparams + self.func.ngas + 1)
        nu = theta[-1]
        # d loglike / d theta
        const = -0.5 * (nu + 1)
        e_i = self.func(theta[:-1]) - self.data
        x = np.power(e_i / self.sigma, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.sigma**2) / (nu-2)
        term12 = term1 * term2
        sspgrad = self.func.gradient(theta[:-1])
        grad[:-1] = const * np.sum(term12[np.newaxis, :] * sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.sigma, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

if __name__ == "__main__":
    ############################################################################
    # Load file with templates and make normalization
    import context
    import misc
    from make_gas_templates import make_gas_templates
    params, templates, norm = misc.load_templates()
    line_names, gas_templates = make_gas_templates(z=0.)
    p0 = np.array([0.05, 0.7, 1., 0.05, 1.2, 0.5, -0.2])
    p1 = np.append(p0, np.zeros(len(line_names)))
    p2 = np.append(p0, 0.1 * np.ones(len(line_names)))
    wave = np.array([context.wave_eff[band] for band in context.bands]) \
                                                                 * u.angstrom
    M = CSPModel(wave, params, templates, gas_templates=gas_templates,
                 gas_names=line_names)
    context.plt.plot(wave, M(p1))
    context.plt.plot(wave, M(p2))
    context.plt.show()
    ############################################################################
    # Making some mock data (1D)
    data = M(p0) + np.random.normal(0, 0.04, 12)
    sigma = np.full_like(data, 0.04)
    ############################################################################
    logl = StudTLogLike(data, sigma, M)
    p1 = np.append(p0, [5.])
    ###########################################################################
    # Making some mock data (2D)
    data = M(p0)[np.newaxis,:] + np.random.normal(0, 0.02, (10, 12))
    errors = np.full_like(data, 0.01)
    npop = 2
    hbfit = HBfit(wave, data, sigma, templates, params, npop=npop)
    hbfit.build_model()
    # Fit simple model to test
    if False:
        lower = [params[col].min() for col in params.colnames]
        upper = [params[col].max() for col in params.colnames]
        bounds = [[0.] + ([0.] + lower) * npop,
                  [10] + ([100] + upper) * npop]
        p1 = np.array([0.1, 1., 1.1, 0., 0.8, 0.6, 0.])
        MAP = []
        for i, (sed, err) in enumerate(zip(data, errors)):
            def residual(p):
                return (M(p) - sed) / err
            sol = least_squares(residual, p1, loss="soft_l1", bounds=bounds)["x"]
            # context.plt.plot(wave, sed, "o")
            # context.plt.plot(wave, M(sol), "-")
            # context.plt.show()
            MAP.append(sol)
        MAP = np.array(MAP)
    ############################################################################