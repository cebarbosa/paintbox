# -*- coding: utf-8 -*-
""" 

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import sys
import copy

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.modeling import models
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
import scipy.optimize as opt

sys.path.append("../bsf")

import paintbox as pb

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
    polynames = ["p{}".format(i+1) for i in range(porder)]
    with model:
        theta = []
        for param in spec.parnames:
            pname = param.split("_")[0]
            if pname == "w":
                w = pm.Normal(param, mu=0.5, sd=0.5)
                theta.append(w)
            elif pname in params.colnames:
                vmin = params[pname].data.min()
                vmax = params[pname].data.max()
                v = pm.Uniform(param, lower=vmin, upper=vmax,
                               testval=0.5 * (vmin + vmax))
                theta.append(v)
            # Extinction parameters
            elif pname == "Av":
                Av = pm.Uniform("Av", lower=0, upper=1., testval=0.1)
                theta.append(Av)
            elif pname == "Rv":
                Rv = pm.Uniform("Rv", lower=2.8, upper=4.8, testval=3.8)
                theta.append(Rv)
            # Emission lines
            elif param in emlines:
                v = pm.Uniform(param, lower=0, upper=10., testval=1.)
                theta.append(v)
            # Kinematic parameters
            elif pname == "V":
                V = pm.Uniform("V", lower=3600, upper=4000, testval=3800)
                theta.append(V)
            elif pname == "sigma":
                sigma = pm.Uniform("sigma", lower=100, upper=400, testval=200.)
                theta.append(sigma)
            # Polynomia parameters
            elif pname == "p0":
                p0 = pm.Normal("p0", mu=1, sd=0.1, testval=1.)
                theta.append(p0)
            elif param in polynames:
                pn = pm.Normal(param, mu=0, sd=0.1, testval=0.)
                theta.append(pn)
        # Appending additional parameters for likelihood
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Uniform("x", lower=0, upper=2.)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = pb.TheanoLogLikeInterface(flux, spec, loglike=loglike,
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
    ############################################################################
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
    ssp1 = pb.StPopInterp(twave, params, templates)
    ssp2 = copy.deepcopy(ssp1)
    w1 = pb.Polynomial(twave, 0)
    w1.parnames = ["w_1"]
    w2 = pb.Polynomial(twave, 0)
    w2.parnames = ["w_2"]
    ssp1.parnames = ["{}_1".format(_) for _ in ssp1.parnames]
    ssp2.parnames = ["{}_2".format(_) for _ in ssp2.parnames]
    ssp = w1 * ssp1 + w2 * ssp2
    ssppars = ssp1.params
    p0 = [0.5, .2, 10.2, .2, .2]
    p0_ssp = np.array(p0 * 2)
    ############################################################################
    # Adding extinction to the stellar populations
    extinction = pb.CCM89(twave)
    limits["Av"] = (0., 5)
    limits["Rv"] = (0., 5)
    p0_ext = np.array([0.1, 3.8])
    ############################################################################
    # Combining SSP with extinction
    stars = ssp * extinction
    p0_stars = np.hstack([p0_ssp, p0_ext])
    ############################################################################
    # Loading templates for the emission lines
    emwave = np.linspace(4499, 6001, 2000) # Oversampled dispersion
    gas_templates, line_names = make_emission_lines(emwave)
    emission = pb.Rebin(twave, pb.EmissionLines(emwave, gas_templates,
                                                  line_names))
    p0_em = np.ones(len(line_names), dtype=np.float)
    ###########################################################################
    # Multiplicative polynomial for continuum
    porder = 10
    poly = pb.Polynomial(wave, porder)
    p0_poly = np.zeros(porder + 1, dtype=np.float)
    p0_poly[0] = 1.8
    limits["p0"] = (0, 10)
    for i in range(porder):
        limits["p{}".format(i+1)] = (-1, 1)
    for lname in line_names:
        limits[lname] = (0, 10.)
    ############################################################################
    # Creating a model including LOSVD
    spec = pb.Rebin(wave, pb.LOSVDConv((stars + emission), velscale)) * poly
    limits["V"] = (3600, 4100)
    limits["sigma"] = (50, 500)
    p0_losvd = np.array([3800, 280])
    # bounds = np.array([limits[par] for par in spec.parnames])
    p0 = np.hstack([p0_stars, p0_em, p0_losvd, p0_poly])
    model = build_model(wave, flux, fluxerr, spec, ssppars, emission.parnames,
                        porder)
    db = "mcmc_db"
    summary = "summary_mcmc.csv"
    with model:
        trace = pm.sample(200, step=pm.Metropolis())
        df = pm.stats.summary(trace)
        df.to_csv(summary)
    pm.save_trace(trace, db, overwrite=True)

if __name__ == "__main__":
    example_full_spectral_fitting()