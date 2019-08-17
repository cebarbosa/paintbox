# -*- coding: utf-8 -*-
"""

Created on 12/04/19

Author : Carlos Eduardo Barbosa

Bayesian spectrum fitting.

"""
from __future__ import print_function, division

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.special import gamma, digamma, legendre
from scipy.ndimage import convolve1d
from scipy.interpolate import LinearNDInterpolator
from spectres import spectres
import theano.tensor as tt
import pymc3 as pm

class BSF():
    def __init__(self, wave, flux, twave, templates, params,
                 fluxerr=None, em_templates=None, em_names=None,
                 velscale=None, nssps=2, em_components=None,
                 wave_unit=None, z=0., loglike="studt", adegree=-1):
        """ Model observations with SSP models using hierarchical Bayesian
        approach and probabilisti programming.

        Parameters
        ----------
        wave: np.array
            The wavelength dispersion of the modeled data

        flux: np.array
            The flux of the modeled data

        twave: np.array
            The wavelength dispersion of the templates

        tflux: np.array
            The array of templates to be used in the fitting

        params: astropy.table.Table
            Table containing the parameters of the templates

        fluxerr: np.array
            Uncertainties for the modeled flux.

        """
        self.loglike = loglike.lower()
        self.adegree = adegree
        # Observed parameters
        self.flux = np.atleast_1d(flux)
        self.fluxerr = np.ones_like(self.flux) if fluxerr is None else \
                       fluxerr
        self.z = z
        # SSP templates
        self.templates = templates
        self.params = params
        # Emission line templates
        self.em_templates = em_templates
        self.em_names = em_names
        self.em_components = em_components
        if self.em_components is not None:
            self.em_components = np.atleast_1d(em_components)
        ########################################################################
        # Setting wavelength units
        self.wave_unit = u.angstrom if wave_unit is None else wave_unit
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * self.wave_unit
        if hasattr(twave, "unit"):
            self.twave = twave
        else:
            self.twave = twave * self.wave_unit
        ########################################################################
        # Check if rebinning is necessary
        self.rebin = False if np.array_equal(self.wave, self.twave) else True
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.nssps = np.atleast_1d(nssps)
        self.sed = SEDModel(self.twave, self.params, self.templates,
                            nssps=self.nssps, velscale=velscale,
                            wave_out=wave, em_templates=self.em_templates,
                            em_names=em_names,
                            em_components=self.em_components,
                            adegree = self.adegree)
        self.parnames = [item for sublist in self.sed.parnames for item in
                         sublist]

    def build_model(self):
        print("Generating model...")
        # Estimating scale to be used for magnitudes
        fmean = np.nanmedian(self.flux)
        m0 = -2.5 * np.log10(fmean / np.median(self.templates))
        # Estimating scale for emission lines
        if self.em_templates is not None:
            m0em = -2.5 * np.log10(fmean / np.max(self.em_templates))
        # Estimating velocity from input redshift
        beta = (np.power(self.z + 1, 2) - 1) / (np.power(self.z + 1, 2) + 1)
        V0 = const.c.to(self.velscale.unit) * beta
        vscale = self.velscale.value
        # Building statistical model
        self.model = pm.Model()
        with self.model:
            theta = []
            for par in self.parnames:
                psplit = par.split("_")
                pname = psplit[0]
                comp = int(psplit[1])
                if comp < len(self.nssps): # SSP components
                    if pname == "Av":
                        Av = pm.HalfNormal(par, sd=.4, testval=0.2)
                        theta.append(Av)
                    elif pname == "Rv":
                        BNormal = pm.Bound(pm.Normal, lower=0)
                        Rv = BNormal(par, mu=4.05, sd=0.8, testval=4.05)
                        theta.append(Rv)
                    elif pname == "flux":
                        magkey = par.replace("flux", "mag")
                        mag = pm.Normal(magkey, mu=m0, sd=1., testval=m0)
                        flux = pm.Deterministic(par, pm.math.exp(-0.4 * mag *
                                                             np.log(10)))
                        theta.append(flux)
                    elif pname in self.params.colnames:
                        lower = self.params[pname].min()
                        upper = self.params[pname].max()
                        param = pm.Uniform(par, lower=lower, upper=upper,
                                           testval=0.5 * (lower + upper))
                        theta.append(param)
                    elif pname == "V":
                        V = pm.Normal(par, mu=V0.value, sd=1000.)
                        theta.append(V)
                    elif pname == "sigma":
                        BHNormal = pm.Bound(pm.HalfNormal, lower= 0.5 * vscale,
                                            upper=1000.)
                        sigma = BHNormal(par, sd=200,
                                         testval=2 * vscale)
                        theta.append(sigma)
                else:
                    if pname == "flux":
                        magkey = par.replace("flux", "mag")
                        mag = pm.Normal(magkey, mu=m0em, sd=1., testval=m0em)
                        flux = pm.Deterministic(par, pm.math.exp(-0.4 * mag *
                                                             np.log(10)))
                        theta.append(flux)
                    elif pname == "V":
                        V = pm.Normal(par, mu=V0.value, sd=1000.)
                        theta.append(V)
                    elif pname == "sigma":
                        BHNormal = pm.Bound(pm.HalfNormal, lower= 0.5 * vscale,
                                            upper=1000.)
                        sigma = BHNormal(par, sd=80, testval=2 * vscale)
                        theta.append(sigma)
            # Setting degrees-of-freedom of likelihood
            if self.loglike == "studt":
                BGamma = pm.Bound(pm.Gamma, lower=2.01)
                nu = BGamma("nu", alpha=2., beta=.1, testval=10.)
                theta.append(nu)
            theta = tt.as_tensor_variable(theta)
            logl = LogLikeWithGrad(self.flux, self.wave, self.fluxerr,
                                   self.sed, loglike=self.loglike)
            # use a DensityDist
            pm.DensityDist('likelihood', lambda v: logl(v),
                           observed={'v': theta})

# define a theano Op for our likelihood function
class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, data, x, sigma, stpop, loglike=None):
        self.data = data
        self.x = x
        self.sigma = sigma
        self.stpop = stpop
        self.loglike = loglike
        if self.loglike == "studt":
            self.likelihood = StudTLogLike(self.data, self.sigma, self.stpop)
        elif self.loglike == "normal":
            self.likelihood = NormalLogLike(self.data, self.sigma, self.stpop)
        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        logl = self.likelihood(theta)
        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
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
                         np.sqrt(np.pi * (nu - 2)) / gamma(0.5 * nu))  \
             - 0.5 * (nu + 1) * np.sum(np.log(x)) \
             - 0.5 * np.sum(np.log(self.sigma**2)) # Constant
        if np.isnan(LLF):
            print("Nan loglikelihood: {}".format(theta))
        return float(LLF)

    def gradient(self, theta):
        grad = np.zeros(self.func.nparams + 1)
        nu = theta[-1]
        # d loglike / d theta
        e_i = self.func(theta[:-1]) - self.data
        x = np.power(e_i / self.sigma, 2.) / (nu - 2.)
        term1 = 1 / (1 + x)
        term2 = 2 * e_i / (self.sigma**2) / (nu-2)
        term12 = term1 * term2
        sspgrad = self.func.gradient(theta[:-1])
        grad[:-1] = -0.5 * (nu + 1) * np.sum(term12[np.newaxis, :] *
                                             sspgrad, axis=1)
        # d loglike / d nu
        nuterm1 = 0.5 * self.N * digamma(0.5 * (nu + 1))
        nuterm2 = - 0.5 * self.N / (nu - 2)
        nuterm3 = -0.5 * self.N * digamma(0.5 * nu)
        nuterm4 = -0.5 * np.sum(np.log(1 + x))
        nuterm5 = 0.5 * (nu + 1) * np.power(nu - 2, -2) * \
                  np.sum(np.power(e_i / self.sigma, 2) * term1)
        grad[-1] = nuterm1 + nuterm2 + nuterm3 + nuterm4 + nuterm5
        return grad

class NormalLogLike():
    def __init__(self, data, sigma, func):
        self.data = data
        self.sigma = sigma
        self.func = func
        self.N = len(data)
        self.nparams = self.func.nparams

    def __call__(self, theta):
        e_i = self.func(theta) - self.data
        LLF = - 0.5 * self.N * np.log(2 * np.pi) + \
              - 0.5 * np.sum(np.power(e_i / self.sigma, 2)) \
              - 0.5 * np.sum(np.log(self.sigma ** 2))
        #
        if np.isnan(LLF):
            print(theta)
        #     return -np.infty
        return float(LLF)

    def gradient(self, theta):
        e_i = self.func(theta) - self.data
        grad = - np.sum(e_i / np.power(self.sigma, 2.)[np.newaxis, :] *
                        self.func.gradient(theta), axis=1)
        return grad

class SEDModel():
    """ Stellar population model. """

    def __init__(self, wave, params, templates, nssps=2,
                 velscale=None, wave_out=None, wave_unit=None,
                 em_templates=None, em_components=None,
                 em_names=None, adegree=-1):
        """ SED model based on combination of SSPs, emission lines, sky lines
        and their LOSVD.

        This is the main routine to produce a Spectral Energy Distribution
        model using a combination of stellar populations and emission lines.

        Parameters
        ----------
        wave: array_like
           Common wavelength array for the stellar population templates. For
           kinematic fitting, wave has to be binned at a fixed velocity
           scale prior to the modelling.

        params: astropy.table.Table
            Parameters of the SSPs.

        templates: array_like
            SED templates for the single stellar population models.

        nssps: int or 1D array_like, optional
            Number of stellar populations to be used. Negative values ignore
            the kinematics. Negative values ignore the LOSVD convolution.

            Examples:
                - 2 SSPs with common kinematics (default) nssps = 2 or
                nssps = np.array([2])
                - 2 SSPs with decoupled kinematics:
                  nssps = np.array([1,1])
                - 2 SSPS with 2 kinematic components
                  nssps = np.array([2,2])
                - Ignore LOSVD convolution in the fitting of 2 stellar
                populations (multi-band SED fitting)
                  nssps = -2 or nssps = np.array([2])

        velscale: astropy.unit, optional
            Determines the velocity difference between adjacent pixels in the
            wavelength array. Default value is 1 km/s/pixel.

        wave_out: array_like, optional
            Wavelength array of the output. In case the output is not
            determined, it is assumed that wave_out is the same as wave. If
            necessary, the rebbining if performed with spectres
            (https://spectres.readthedocs.io/en/latest/)

        wave_unit: astropy.units, optional
            Units of the wavelength array. Assumed to be Angstrom if not
            specified.

        em_templates: array_like, optional
            Emission line templates, which can be used either to model gas
            emission lines or sky lines. If

        em_components: int or 1D array_like, optional
            Similar to nssps, negative components do not make LOSVD convolution.


        em_names: array_like, optional
            Names of the emission lines to be used in the outputs.

        Returns
        -------
            "SEDModel" callable class


        Methods
        -------
            __call__(p): Returns the SED model of parameters p

            gradient(p): Gradient of the SED function with parameters p

         """
        self.adegree = adegree
        ########################################################################
        # Setting wavelength units
        self.wave_unit = u.angstrom if wave_unit is None else wave_unit
        if hasattr(wave, "unit"):
            self.wave = wave
        else:
            self.wave = wave * self.wave_unit
        self.wave_out = self.wave if wave_out is None else wave_out
        if not hasattr(self.wave_out, "unit"):
            self.wave_out *= self.wave_unit
        ########################################################################
        # Verify if rebinning is necessary to modeling
        self.rebin = not np.array_equal(self.wave, self.wave_out)
        ########################################################################
        self.velscale = 1. * u.km / u.s if velscale is None else velscale
        self.params = params
        self.templates = templates
        self.nssps = np.atleast_1d(nssps).astype(np.int)
        self.em_templates = em_templates
        self.em_names = em_names
        # Dealing with SSP components
        self.pops = []
        self.idxs = [0]
        self.parnames = []
        for i, comp in enumerate(self.nssps):
            if comp > 0:
                csp = NSSPSConv(self.wave, self.params,
                                self.templates, self.velscale, npop=comp)
            else:
                csp = NSSPs(self.wave, self.params, self.templates,
                            npop=abs(comp))
            parnames = []
            for p in csp.parnames:
                psplit = p.split("_")
                psplit[0] = "{}_{}".format(psplit[0], i)
                newp = "_".join(psplit)
                parnames.append(newp)
            self.idxs.append(csp.nparams + self.idxs[-1])
            self.pops.append(csp)
            self.parnames.append(parnames)
        if self.em_templates is not None:
            #####################################################################
            # Dealing with emission line components
            n_em = len(em_templates)  # Number of emission lines
            em_components = np.ones(n_em) if em_components is None else \
                em_components
            self.em_components = np.atleast_1d(em_components).astype(np.int)
            for i, comp in enumerate(np.unique(self.em_components)):
                idx = np.where(self.em_components == comp)[0]
                if comp < 0:
                    em = SkyLines(self.wave, self.em_templates[idx],
                                  em_names=self.em_names[idx])
                else:
                    em = EmissionLines(self.wave, self.em_templates[idx],
                                       self.em_names[idx], self.velscale)
                parnames = []
                for p in em.parnames:
                    psplit = p.split("_")
                    psplit[0] = "{}_{}".format(psplit[0], i + len(self.nssps))
                    newp = "_".join(psplit)
                    parnames.append(newp)
                self.idxs.append(em.nparams + self.idxs[-1])
                self.pops.append(em)
                self.parnames.append(parnames)
        if self.adegree >= 0:
            apoly = APoly(self.wave, self.adegree)
            self.pops.append(apoly)
            self.parnames.append(apoly.parnames)
            self.idxs.append(apoly.nparams + self.idxs[-1])
        self.nparams = self.idxs[-1]

    def __call__(self, theta):
        sed = np.zeros(len(self.wave))
        for i in range(len(self.pops)):
            t = theta[self.idxs[i]: self.idxs[i + 1]]
            s = self.pops[i](t)
            sed += s
        if not self.rebin:
            return sed
        sed = spectres(self.wave_out.to("AA").value,
                       self.wave.to("AA").value, sed)
        return sed

    def gradient(self, theta):
        grads = []
        for i, pop in enumerate(self.pops):
            t = theta[self.idxs[i]: self.idxs[i + 1]]
            grads.append(pop.gradient(t))
        grads = np.vstack(grads)
        if not self.rebin:
            return grads
        grads = spectres(self.wave_out.to("AA").value,
                         self.wave.to("AA").value, grads)
        return grads

class SSP():
    """ Linearly interpolated SSP models."""

    def __init__(self, wave, params, templates):
        self.wave = wave
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
            thetamin.append(np.min(self.params[par].data))
            thetamax.append(np.max(self.params[par].data))
            inner_grid.append(np.unique(self.params[par].data)[1:-1])
        self.thetamin = np.array(thetamin)
        self.thetamax = np.array(thetamax)
        self.inner_grid = inner_grid

    def __call__(self, theta):
        return self.f(theta)

    def gradient(self, theta, eps=1e-6):
        # Clipping theta to avoid border problems
        theta = np.maximum(theta, self.thetamin + 2 * eps)
        theta = np.minimum(theta, self.thetamax - 2 * eps)
        grads = np.zeros((self.nparams, self.templates.shape[1]))
        for i, t in enumerate(theta):
            epsilon = np.zeros(self.nparams)
            epsilon[i] = eps
            # Check if data point is in inner grid
            in_grid = t in self.inner_grid[i]
            if in_grid:
                tp1 = theta + 2 * epsilon
                tm1 = theta + epsilon
                grad1 = (self.__call__(tp1) - self.__call__(tm1)) / (
                        2 * eps)
                tp2 = theta - epsilon
                tm2 = theta - 2 * epsilon
                grad2 = (self.__call__(tp2) - self.__call__(tm2)) / (
                        2 * eps)
                grads[i] = 0.5 * (grad1 + grad2)
            else:
                tp = theta + epsilon
                tm = theta - epsilon
                grads[i] = (self.__call__(tp) - self.__call__(tm)) / (
                        2 * eps)
        return grads

class NSSPs():
    """ Stellar population model. """
    def __init__(self, wave, params, templates, npop=2):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.ssp = SSP(self.wave, self.params, self.templates)
        self.ncols = len(self.params.colnames)
        self.nparams = self.npop * (len(self.params.colnames) + 1) + 2
        self.shape = (self.nparams, len(self.wave))
        # Preparing array for redenning
        x = 1 / self.wave.to("micrometer") * u.micrometer
        self.kappa = np.where(self.wave > 0.63 * u.micrometer,
                              2.659 * (-1.857 + 1.040 * x), \
                              2.659 * (-2.156 + 1.509 * x - 0.198 * x * x
                                       + 0.011 * (x * x * x)))
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))

    def extinction(self, Av, Rv):
        return np.power(10, -0.4 * Av * (1. + self.kappa / Rv))

    def __call__(self, theta):
        p = theta[2:].reshape(self.npop, -1)
        return self.extinction(theta[0], theta[1]) * \
               np.dot(p[:, 0], self.ssp(p[:, 1:]))

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        F = self.__call__(theta)
        # dF / dAv
        grad[0] = F * np.log(10) * (-0.4 * (1. + self.kappa / theta[1]))
        # dF / dRv
        grad[1] = F * 0.4 * theta[0] * self.kappa * np.log(10) * \
                  np.power(theta[1], -2.)
        ps = theta[2:].reshape(self.npop, -1)
        const = self.extinction(theta[0], theta[1])
        for i, p in enumerate(ps):
            idx = 2 + (i * (self.ncols + 1))
            # dF/dFi
            grad[idx] = const * self.ssp(p[1:])
            # dF / dSSPi
            grad[idx + 1:idx + 1 + self.ncols] = const * p[0] * \
                                                 self.ssp.gradient(p[1:])
        return grad

class NSSPSConv():
    def __init__(self, wave, params, templates, velscale, npop=1):
        self.params = params
        self.wave = wave
        self.templates = templates
        self.npop = npop
        self.velscale = velscale.to("km/s").value
        self.nssps = NSSPs(self.wave, self.params, self.templates,
                           npop=self.npop)
        self.nparams = self.nssps.nparams + 2
        self.shape = (self.nparams, len(self.wave))
        # Set parameter names
        self.parnames = ["Av", "Rv"]
        for n in range(self.npop):
            for p in ["flux"] + self.params.colnames:
                self.parnames.append("{}_{}".format(p, n))
        self.parnames.append("V")
        self.parnames.append("sigma")

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

    def __call__(self, theta):
        p = theta[:self.nssps.nparams]
        sed = self.nssps(p)
        y, k = self.kernel_arrays(theta[self.nssps.nparams:])
        return convolve1d(sed, k)

    def gradient(self, theta):
        grad = np.zeros(self.shape)
        sspgrad = self.nssps.gradient(theta[:self.nssps.nparams])
        p = theta[self.nssps.nparams:]
        y, k = self.kernel_arrays(p)
        for i in range(len(sspgrad)):
            grad[i] = convolve1d(sspgrad[i], k)
        sed = self.nssps(theta[:self.nssps.nparams])
        grad[-2] = convolve1d(sed, y * k / p[1])
        grad[-1] = convolve1d(sed, (y * y - 1.) * k / p[1])
        return grad

class EmissionLines():
    def __init__(self, wave, templates, em_names, velscale):
        self.wave = wave
        self.templates = templates
        self.em_names = em_names
        self.n_em = len(templates)
        self.velscale = velscale
        self.shape = (self.n_em + 2, len(self.wave))
        self.parnames = ["flux_{}".format(name) for name in em_names]
        self.parnames.append("V")
        self.parnames.append("sigma")
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        g = theta[:self.n_em]
        p = theta[self.n_em:]
        return convolve1d(np.dot(g, self.templates),
                          self.kernel_arrays(p)[1])

    def gradient(self, theta):
        g = theta[:self.n_em]
        p = theta[-2:]
        y, k = self.kernel_arrays(p)
        grad = np.zeros(self.shape)
        for i in range(self.n_em):
            grad[i] = convolve1d(self.templates[i], k)
        gas = np.dot(g, self.templates)
        grad[-2] = convolve1d(gas, y * k / p[1])
        grad[-1] = convolve1d(gas, (y * y - 1.) * k / p[1])
        return grad

    def kernel_arrays(self, p):
        x0, sigx = p / self.velscale.to("km/s").value
        dx = int(np.ceil(np.max(abs(x0) + 5 * sigx)))
        n = 2 * dx + 1
        x = np.linspace(-dx, dx, n)
        y = (x - x0) / sigx
        y2 = np.power(y, 2.)
        k = np.exp(-0.5 * y2) / (sigx * np.sqrt(2 * np.pi))
        return y, k

class SkyLines():
    def __init__(self, wave, templates, em_names=None):
        self.wave = wave
        self.templates = templates
        self.nparams = len(self.templates)
        self.n_em = len(templates)
        self.em_names = ["sky{}".format(n) for n in range(self.n_em)] if \
            em_names is None else em_names
        self.parnames = self.em_names
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.templates)

    def gradient(self, theta):
        return self.templates

class APoly():
    def __init__(self, wave, degree):
        self.wave = wave
        self.degree = degree
        self.x = np.linspace(-1, 1, len(self.wave))
        self.mpoly = np.zeros((self.degree + 1, len(self.x)))
        for i in np.arange(self.degree + 1):
            self.mpoly[i] = legendre(i)(self.x)
        self.parnames = ["acoeff_{}".format(i) for i in np.arange(degree+1)]
        self.nparams = len(self.parnames)

    def __call__(self, theta):
        return np.dot(theta, self.poly)

    def gradient(self, theta):
        return self.poly