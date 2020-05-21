# -*- coding: utf-8 -*-
""" 

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm
from tqdm import tqdm
from spectres import spectres
import ppxf
from ppxf.ppxf_util import log_rebin
import theano.tensor as tt

from ppxf.ppxf_util import emission_lines

sys.path.append("../bsf")

import bsf

def build_model(wave, flux, fluxerr, spec, params, emlines, porder,
                loglike=None):
    loglike = "normal2" if loglike is None else loglike
    model = pm.Model()
    flux = flux.astype(np.float)
    with model:
        theta = []
        for param in params.colnames:
            vmin = params[param].observed.min()
            vmax = params[param].observed.max()
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
        logl = bsf.LogLike(flux, wave, fluxerr, spec, loglike=loglike)
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
    wave = data["wave"].observed
    flux = data["flux"].observed
    fluxerr = data["fluxerr"].observed
    # Resampling to fixed velocity scale
    # logLam = log_rebin([wave[0], wave[-1]], flux, velscale=velscale)[1]
    # lam = np.exp(logLam)
    # idx = np.where((lam >= 4500) & (lam <= 5995))[0][1:-1]
    # newwave = lam[idx]
    # flux, fluxerr = spectres(newwave, wave, flux, spec_errs=fluxerr)
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
    logwave = Table.read(templates_file, hdu=2)["loglam"].observed
    twave = np.exp(logwave)
    ssp = bsf.SSP(twave, params, templates)
    ssppars = ssp.params
    # for param in ssp.params.colnames:
    #     print(param, ssp.params[param].min(), ssp.params[param].max())
    # input()
    extinction = bsf.CCM89(twave)
    stars = ssp * extinction
    p0_ssp = np.array([.15, 10., .2, .2])
    p0_ext = np.array([0.1, 3.8])
    p0_stars = np.hstack([p0_ssp, p0_ext])
    # Loading templates for the emission lines
    emission, line_names, line_wave = emission_lines(logwave,
                                [wave.min(), wave.max()], 2.95)
    line_names = [_.replace("[", "").replace("]", "").replace("_", "") for _ in
                  line_names]
    enorm = np.nanmax(emission)
    emission /= enorm
    line_names = list(line_names)
    gas_templates = emission.T
    emission = bsf.EmissionLines(twave, gas_templates, line_names)
    p0_em = np.ones(len(line_names), dtype=np.float)
    # Adding a polynomial
    porder = 10
    poly = bsf.Polynomial(wave, porder)
    p0_poly = np.zeros(porder + 1, dtype=np.float)
    p0_poly[0] = 1.8
    p0_losvd = np.array([3800, 280])
    ############################################################################
    # Creating a model
    spec = bsf.Rebin(wave, bsf.LOSVDConv((stars + emission), velscale)) * poly
    p0 = np.hstack([p0_stars, p0_em, p0_losvd, p0_poly,])
    # plt.plot(wave, flux)
    # plt.plot(spec.wave, spec(p0))
    # plt.show()
    model = build_model(wave, flux, fluxerr, spec, ssppars, emission.parnames,
                        porder)
    with model:
        trace = pm.sample()
        df = pm.stats.summary(trace)
        df.to_csv("summary_nuts.txt")
        pm.traceplot(trace)
    plt.show()
    input()

    pm.save_trace(trace, output, overwrite=True)
    trace = load_traces(output, bsf.parnames)
    models = np.zeros((len(trace), len(bsf.wave)))
    for i, t in enumerate(tqdm(trace)):
        models[i] = bsf.sed(t)
    ax = plt.subplot(111)
    ax.plot(bsf.wave, bsf.flux + bsf.fluxerr)
    ax.plot(bsf.wave, bsf.flux - bsf.fluxerr, c="C0")
    ax.plot(bsf.wave, models.mean(axis=0))
    plt.show()
    ###########################################################################

if __name__ == "__main__":
    example_full_spectral_fitting()
#
