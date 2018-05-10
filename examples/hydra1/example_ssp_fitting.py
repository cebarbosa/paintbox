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
    """ Uses Capellari's program to load templates. """
    def __init__(self, velscale, fwhm=2.51):
        self.velscale = velscale
        self.fwhm = fwhm
        path = os.path.join(context.basedir, "ppxf/miles_models")
        pathname = "{}/Mun1.3*.fits".format(path)
        self.miles = Miles(pathname, velscale, fwhm)
        self.templates3D = self.miles.templates.T
        self.ages2D = self.miles.age_grid.T
        self.metals2D = self.miles.metal_grid.T
        self.grid_shape = self.ages2D.shape
        # Reshaping arrays
        self.templates2D = self.templates3D.reshape(-1, self.templates3D.shape[
            -1])
        self.ages1D = self.ages2D.reshape(-1)
        self.metals1D = self.metals2D.reshape(-1)
        # Get ranges for parameters
        self.age_range = [self.ages1D.min(), self.ages1D.max()]
        self.metal_range = [self.metals1D.min(), self.metals1D.max()]
        return

def example_csp(redo=False, plot_weights=True, plot_model=False,
                plot_results=True, method="NUTS", velscale=300):
    """ Produces a CSP spectraum using SSP and tries to recovery it using
    Bayesian modeling. """
    # Producing a CSP spectrum
    miles = MilesCategorical(velscale=velscale)
    mu_actual1 = np.array([10, -0.5])
    cov_actual1 = np.array([[0.5, 0.0], [0.02, 0.05]])
    w1 = multivariate_normal(mean=mu_actual1, cov=cov_actual1).pdf
    mu_actual2 = np.array([2, -1.4])
    cov_actual2 = np.array([[0.2, 0.02], [0.01, 0.03]])
    w2 = multivariate_normal(mean=mu_actual2, cov=cov_actual2).pdf
    X = np.column_stack((miles.ages1D, miles.metals1D))
    W1, W2 = np.zeros(len(X)), np.zeros(len(X))
    for i,x in enumerate(X):
        W1[i] = w1(x)
        W2[i] = w2(x)
    W1 /= W1.sum()
    W2 /= W2.sum()
    f1 = 0.88
    f2 = 1 - f1
    model1 = f1 * np.dot(W1, miles.templates2D)
    model2 = f2 * np.dot(W2, miles.templates2D)
    model = model1 + model2
    if plot_model:
        plt.plot(model1, label="Old CSP")
        plt.plot(model2, label="New CSP")
        plt.plot(model, label="Model")
    W = (f1 * W1 + f2 * W2).reshape(miles.grid_shape)
    if plot_weights:
        plt.figure(10)
        ax = plt.subplot(2, 2, 1)
        m = ax.pcolormesh(miles.ages2D, miles.metals2D, W, vmax=0.1)
        plt.colorbar(m)
        ax = plt.subplot(2, 2, 2)
        ax.plot(miles.ages2D[0], (W).sum(axis=0))
        ax = plt.subplot(2, 2, 3)
        ax.plot(miles.metals2D[:, 0], W.sum(axis=1))
    dbname = csp_modeling(model, miles.templates2D, redo=redo, method=method)
    with open(dbname, 'rb') as buff:
        mcmc = pickle.load(buff)
    if method == "NUTS":
        trace = mcmc["trace"]
        weights = trace["w"].mean(axis=0)
    elif method in ["svgd", "advi"]:
        approx = mcmc["approx"]
        weights = approx.sample(1000)["w"].mean(axis=0)
    weights = weights.reshape(miles.grid_shape)
    if plot_results:
        plt.figure(5)
        ax = plt.subplot(2, 2, 1)
        m = ax.pcolormesh(miles.ages2D, miles.metals2D, weights, vmax=0.1)
        plt.colorbar(m)
        ax = plt.subplot(2, 2, 2)
        ax.plot(miles.ages2D[0], (weights).sum(axis=0))
        ax = plt.subplot(2, 2, 3)
        ax.plot(miles.metals2D[:, 0], weights.sum(axis=1))
    plt.show()

def csp_modeling(obs, templates, redo=False, method="NUTS"):
    """ Model a CSP with bayesian model. """
    dbname = os.path.join("csp_{}.db".format(method.lower()))
    if os.path.exists(dbname) and not redo:
        return dbname
    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(len(templates)))
        bestfit = pm.math.dot(w.T, templates)
        sigma = pm.Exponential("sigma", lam=1)
        likelihood = pm.Normal('like', mu=bestfit, sd = sigma, observed=obs)
    if method == "NUTS":
        with model:
            trace = pm.sample(1000, tune=500)
        results = {'model': model, "trace": trace}
        with open(dbname, 'wb') as buff:
            pickle.dump(results, buff)
    elif method in ["svgd", "advi"]:
        with model:
            approx = pm.fit(10000, method=method, obj_n_mc=10)
        results = {'model': model, "approx": approx}
        with open(dbname, 'wb') as buff:
            pickle.dump(results, buff)
    return dbname

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


    # print(np.average(ages, weights=weights))
    # print(np.average(metal, weights=weights))
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
    example_csp(redo=False, method="svgd")