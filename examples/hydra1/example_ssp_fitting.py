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
import matplotlib.pyplot as plt
from specutils.io.read_fits import read_fits_spectrum1d
from scipy.ndimage.filters import gaussian_filter1d
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

def example_two_csps(redo=False, plot=True, velscale=300, sn=30):
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
    signal = np.median(model)
    noise = np.random.normal(0, signal / sn, size=len(model))
    W = (f1 * W1 + f2 * W2).reshape(miles.grid_shape)

    observed = model + noise
    dbname = os.path.join(context.workdir,
                          "example_two_csps_sn{}.db".format(sn))
    csp_modeling(model, miles.templates2D, dbname, redo=redo)
    with open(dbname, 'rb') as buff:
        mcmc = pickle.load(buff)
    trace = mcmc["trace"]
    weights = trace["w"].mean(axis=0)
    weights = weights.reshape(miles.grid_shape)
    lam = np.exp(miles.miles.log_lam_temp)
    if plot:
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax1.minorticks_on()
        ax1.plot(lam, model1, label="CSP 1")
        ax1.plot(lam, model2, label="CSP 2")
        ax1.plot(lam, observed, label="Model (S/N={})".format(sn))
        ax1.set_xlabel("$\lambda$ (\AA)")
        ax1.legend()
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.minorticks_on()
        ax2.set_title("Model")
        vmax = np.percentile(W, 98.8)
        m = ax2.pcolormesh(np.log10(miles.ages2D), miles.metals2D, W,
                           vmax=vmax, vmin=0, cmap="cubehelix_r")
        ax2.set_ylabel("[Fe/H]")
        ax2.set_xlabel("log Age (Gyr)")
        cbar2 = plt.colorbar(m)
        cbar2.set_label("Weight")
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax3.minorticks_on()
        m = ax3.pcolormesh(np.log10(miles.ages2D), miles.metals2D, weights,
                           vmax=vmax, vmin=0, cmap="cubehelix_r")
        ax3.set_ylabel("[Fe/H]")
        ax3.set_xlabel("log Age (Gyr)")
        cbar3 = plt.colorbar(m)
        cbar3.set_label("Weight")
        ax3.set_title("Observed (S/N={})".format(sn))
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        ax4.minorticks_on()
        ages = np.log10(miles.ages2D[0])
        dw = 0.025
        w1 = W.sum(axis=0)
        w2 = weights.sum(axis=0)
        ax4.bar(ages- dw, w1, label="Model", width=0.05, alpha=0.8)
        ax4.bar(ages + dw, w2, label="Observed (S/N={})".format(sn),
                width=0.05, alpha=0.8)
        ax4.legend()
        ax4.set_xlabel("Age (Gyr)")
        ax4.set_ylabel("SFH")
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        ax5.minorticks_on()
        metal= miles.metals2D[:,0]
        w1 = W.sum(axis=1)
        w2 = weights.sum(axis=1)
        dw = 0.05
        ax5.bar(metal - dw, w1, label="Model", alpha=0.8, width=0.1)
        ax5.bar(metal + dw, w2, label="Observed (S/N={})".format(sn),
                alpha=0.8, width=0.1)
        ax5.legend()
        ax4.set_xlabel("[Fe/H]")
    plt.show()

def csp_modeling(obs, templates, dbname, redo=False,):
    """ Model a CSP with bayesian model. """
    if os.path.exists(dbname) and not redo:
        return dbname
    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(len(templates)))
        bestfit = pm.math.dot(w.T, templates)
        sigma = pm.Exponential("sigma", lam=1)
        likelihood = pm.Normal('like', mu=bestfit, sd = sigma, observed=obs)
    with model:
        trace = pm.sample(1000, tune=1000)
    results = {'model': model, "trace": trace}
    with open(dbname, 'wb') as buff:
        pickle.dump(results, buff)
    return

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
    # example_two_ssps()
    example_two_csps(redo=False, sn=25)
