# -*- coding: utf-8 -*-
""" 

Created on 05/03/18

Author : Carlos Eduardo Barbosa

Determination of LOSVD of a given spectrum similarly to pPXF.

"""

import os
import pickle

import context

import numpy as np
import pymc3 as pm
from theano import tensor as tt
import matplotlib.pyplot as plt
from specutils.io.read_fits import read_fits_spectrum1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

from ppxf.ppxf_util import log_rebin
from ppxf.miles_util import Miles
from der_snr import DER_SNR

class MilesCategorical():
    """ Uses Capellari's program to load templates and flatten arrays. """
    def __init__(self, velscale, fwhm=2.51):
        self.velscale = velscale
        self.fwhm = fwhm
        path = os.path.join(context.basedir, "ppxf/miles_models")
        pathname = "{}/Mun1.3*.fits".format(path)
        self.miles = Miles(pathname, velscale, fwhm)
        self.templates = self.miles.templates.T
        self.templates = self.templates.reshape(-1, self.templates.shape[-1])
        self.ages = self.miles.age_grid.T.reshape(-1)
        self.metals = self.miles.metal_grid.T.reshape(-1)
        self.grid_shape = self.miles.age_grid.T.shape
        self.age_range = [self.ages.min(), self.ages.max()]
        self.metal_range = [self.metals.min(), self.metals.max()]
        return

def example_csp(redo=False, plot_weights=False, plot_model=False,
                method="NUTS"):
    """ Produces a CSP spectraum using SSP and tries to recovery it using
    Bayesian modeling. """
    # Load templates
    dbname = os.path.join("csp_{}.db".format(method.lower()))
    miles = MilesCategorical(velscale=40)
    X = np.column_stack((miles.ages, miles.metals))
    mu_actual1 = np.array([8, -0.5])
    cov_actual1 = np.array([[1, 0.0], [0.0, 0.1]])
    var1 = multivariate_normal(mean=mu_actual1, cov=cov_actual1).pdf
    mu_actual2 = np.array([2, -1])
    cov_actual2 = np.array([[2, 0.2], [0.5, 0.2]])
    var2 = multivariate_normal(mean=mu_actual2, cov=cov_actual2).pdf
    w = np.zeros_like(miles.ages)

    for i,x in enumerate(X):
        w[i] = var1(x) + var2(x)
    w /= w.sum()
    if plot_weights:
        plt.pcolormesh(miles.ages.reshape(miles.grid_shape),
                       miles.metals.reshape(miles.grid_shape),
                       w.reshape(miles.grid_shape))
        plt.colorbar()
        plt.show()
    yhat = np.dot(w, miles.templates)
    y = yhat + np.random.normal(0, np.median(yhat) * 0.01, len(yhat))
    if os.path.exists(dbname) and not redo:
        return dbname, y
    if plot_model:
        plt.plot(yhat, "-")
        plt.show()

    def stick_breaking(beta):
        portion_remaining = tt.concatenate(
            [[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

        return beta * portion_remaining
    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(len(miles.templates)))
        bestfit = pm.math.dot(w.T, miles.templates)
        sigma = pm.Exponential("sigma", lam=1)
        likelihood = pm.Normal('like', mu=bestfit,
                                   sd = sigma, observed=y)
    if method == "NUTS":
        with model:
            trace = pm.sample(1000, tune=500)
        results = {'model': model, "trace": trace}
        with open(dbname, 'wb') as buff:
            pickle.dump(results, buff)
    if method == "svgd":
        with model:
            approx = pm.fit(300, method='svgd')
        results = {'model': model, "approx": approx}
        with open(dbname, 'wb') as buff:
            pickle.dump(results, buff)
    return dbname, y

def example_hydra():
    """ Fist example using Hydra cluster data to make ppxf-like model.

    The observed spectrum is one of the central spectra the Hydra I cluster
    core observed with VLT/FORS2 presented in Barbosa et al. 2016. This
    spectrum is not flux calibrated, hence the need of a polynomial term.

    In this example, we use a single spectrum with high S/N to derive the
    line-of-sight velocity distribution using a Gauss-Hermite distribution.

    """
    # Constants and instrumental properties
    velscale = 40 # km/s; this is the resolution used for the binning
    fwhm_fors2 = 2.1 # FWHM (Angstrom) of FORS2 data
    fwhm_miles = 2.51 # FWHM (Angstrom) of MILES stellar library
    ###########################################################################
    # Preparing the data for the fitting
    specfile =  os.path.join(os.getcwd(), "data/fin1_n3311cen1_s29a.fits")
    spec = read_fits_spectrum1d(specfile)
    disp = spec.dispersion[1] - spec.dispersion[0]
    fwhm_dif = np.sqrt((fwhm_miles ** 2 - fwhm_fors2 ** 2))
    sigma = fwhm_dif / 2.355 / disp
    galaxy = gaussian_filter1d(spec.flux, sigma)
    lamrange = [spec.dispersion[0], spec.dispersion[-1]]
    galaxy, wave = log_rebin(lamrange, galaxy, velscale=velscale)[:2]
    noise = np.median(spec.flux) / DER_SNR(spec.flux) * np.ones_like(galaxy)
    # TODO: crop spectrum to use only good wavelength range.
    ############################################################################
    miles = MilesCategorical(velscale)
    dbname, spec = example_csp()
    with open(dbname, 'rb') as buff:
        mcmc = pickle.load(buff)
    miles = MilesCategorical(velscale=40)
    # approx = mcmc["approx"]
    # weights = approx.sample(1000)["w"].mean(axis=0)
    trace = mcmc["trace"]
    weights = trace["w"].mean(axis=0)
    weights = weights.reshape(miles.grid_shape)
    # plt.pcolormesh(miles.ages.reshape(miles.grid_shape),
    #                miles.metals.reshape(miles.grid_shape),
    #                weights)
    # plt.colorbar()
    # plt.show()
    ages = miles.miles.age_grid.T
    metal = miles.miles.metal_grid.T
    plt.plot(ages.sum())
    # print(np.average(ages, weights=weights))
    # print(np.average(metal, weights=weights))
    # plt.pcolormesh(miles.ages.reshape(miles.grid_shape),
    #                miles.metals.reshape(miles.grid_shape),
    #                weights * ages)
    # plt.colorbar()
    # plt.show()
    # mcmcmodel = composite_stellar_population(spec, velscale)
    #
    # c = constants.c.to("km/s").value
    # dv = np.log(np.exp(miles.log_lam_temp[0]) / lamrange[0]) * c
    # start = [4000, 300]
    # goodpixels = np.argwhere(np.logical_and(np.exp(wave) > 4800, np.exp(wave) <
    #                                     5800 )).T[0]
    # pp = ppxf(stars_templates, galaxy, noise, velscale, start,
    #           plot=True, moments=4, degree=8, mdegree=-1, vsyst=dv,
    #           lam=np.exp(wave), clean=False, goodpixels=goodpixels)
    # plt.show()
    # print(pp)


if __name__ == "__main__":
    example_hydra()