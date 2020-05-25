# -*- coding: utf-8 -*-
""" 

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import sys
import time

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
import matplotlib.pyplot as plt
import pymc3 as pm
from tqdm import tqdm
import theano.tensor as tt
import scipy.optimize as opt

sys.path.append("../bsf")

import bsf

def make_emission_lines(wave):
    FWHM = 2.95 # Data resolution
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    Hbeta = 4862.721
    OIII_1 = 4958.911
    OIII_2 = 5008.239
    # Create Gaussian1D models for each of the Hbeta and OIII lines.
    h_beta = models.Gaussian1D(amplitude=1., mean=Hbeta, stddev=sigma)
    o3_2 = models.Gaussian1D(amplitude=1., mean=OIII_2, stddev=sigma)
    o3_1 = models.Gaussian1D(amplitude=1. /3.1, mean=OIII_1, stddev=sigma)
    # Tie the ratio of the intensity of the two OIII lines.
    def tie_ampl(model):
        return model.amplitude_2 / 3.1
    o3_1.amplitude.tied = tie_ampl
    o3 = o3_1 + o3_2
    emission = np.array([h_beta(wave), o3(wave)])
    line_names = ["Hbeta", "OIII5007d"]
    return emission, line_names

def build_model(wave, flux, fluxerr, spec, params, emlines, porder,
                loglike=None):
    loglike = "normal2" if loglike is None else loglike
    model = pm.Model()
    flux = flux.astype(np.float)
    with model:
        theta = []
        for param in params.colnames:
            vmin = params[param].data.min()
            vmax = params[param].data.max()
            v = pm.Uniform(param, lower=vmin, upper=vmax)
            theta.append(v)
        Av = pm.Exponential("Av", lam=1 / 0.4, testval=0.1)
        theta.append(Av)
        BNormal = pm.Bound(pm.Normal, lower=0)
        Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
        theta.append(Rv)
        for em in emlines:
            v = pm.HalfNormal(em, sigma=1)
            theta.append(v)
        BoundedNormal = pm.Bound(pm.Normal, lower=3500, upper=4200)
        V = BoundedNormal("V", mu=3800., sigma=100.)
        theta.append(V)
        sigma = pm.HalfNormal("sigma", sd=200)
        theta.append(sigma)
        p0 = pm.HalfNormal("p0", sd=2.)
        theta.append(p0)
        for n in range(porder):
            pn = pm.Normal("p{}".format(n+1), mu=0., sd=1.)
            theta.append(pn)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.TheanoLogLikeInterface(flux, spec, loglike=loglike,
                                          obserr=fluxerr)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
    return model

def example_full_spectral_fitting():
    """ Example of application of BSF for full spectral fitting of a single
    spectrum using E-MILES models.

    Both the spectrum and the models used in this models are homogeneized to
    a resolution FWHM=2.95 for MUSE data and templates have been rebinned to
    a velocity scale of 200 km/s.
    """
    velscale = 200 # km/s
    data = Table.read("ngc3311_center.fits")
    wave = data["wave"].data
    flux = data["flux"].data
    fluxerr = data["fluxerr"].data
    idx = np.where((wave >= 4500) & (wave <= 5995))[0]
    flux = flux[idx]
    fluxerr = fluxerr[idx]
    wave = wave[idx]
    norm = np.median(flux)
    flux /= norm
    fluxerr /= norm
    # Preparing templates
    templates_file = "emiles_velscale200.fits"
    templates = fits.getdata(templates_file, ext=0)
    tnorm = np.median(templates)
    templates /= tnorm
    params = Table.read(templates_file, hdu=1)
    limits = {}
    for param in params.colnames:
        limits[param] = (params[param].min(), params[param].max())
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave)
    ssp = bsf.StPopInterp(twave, params, templates)
    ssppars = ssp.params
    # Adding extinction to the stellar populations
    extinction = bsf.CCM89(twave)
    limits["Av"] = (0., 5)
    limits["Rv"] = (0., 5)
    stars = ssp * extinction
    p0_ssp = np.array([.15, 10., .2, .2])
    p0_ext = np.array([0.1, 3.8])
    p0_stars = np.hstack([p0_ssp, p0_ext])
    # Loading templates for the emission lines
    emwave = np.linspace(4499, 6001, 2000) # Oversampled dispersion
    gas_templates, line_names = make_emission_lines(emwave)
    emission = bsf.Rebin(twave, bsf.EmissionLines(emwave, gas_templates,
                                                  line_names))
    p0_em = np.ones(len(line_names), dtype=np.float)
    for lname in line_names:
        limits[lname] = (0, 10.)
    # Adding a polynomial
    porder = 10
    poly = bsf.Polynomial(wave, porder)
    p0_poly = np.zeros(porder + 1, dtype=np.float)
    p0_poly[0] = 1.8
    p0_losvd = np.array([3800, 280])
    limits["p0"] = (0, 10)
    for i in range(porder):
        limits["p{}".format(i+1)] = (-1, 1)
    ############################################################################
    # Creating a model including LOSVD
    spec = bsf.Rebin(wave, bsf.LOSVDConv((stars + emission), velscale)) * poly
    limits["V"] = (3600, 4100)
    limits["sigma"] = (50, 500)
    bounds = np.array([limits[par] for par in spec.parnames])
    p0 = np.hstack([p0_stars, p0_em, p0_losvd, p0_poly,])
    loglike = bsf.NormalLogLike(flux, spec, obserr=fluxerr)
    # func = lambda p: -loglike(p)
    # sol = opt.dual_annealing(func, bounds, x0=p0, maxiter=3)


    model = build_model(wave, flux, fluxerr, spec, ssppars, emission.parnames,
                        porder)
    with model:
        sol = pm.find_MAP()
        p1 = np.array([sol[par] for par in spec.parnames])
    plt.plot(wave, flux)
    plt.plot(wave, spec(p1))
    for par, pval in zip(spec.parnames, p1):
        print(par, pval)
    plt.show()

if __name__ == "__main__":
    example_full_spectral_fitting()
#
