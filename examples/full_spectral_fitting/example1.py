# -*- coding: utf-8 -*-
""" 

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import sys

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm
from tqdm import tqdm
from spectres import spectres
from ppxf.ppxf_util import log_rebin
import theano.tensor as tt

from ppxf.ppxf_util import emission_lines

sys.path.append("../bsf")

from bsf.sed_models import SSP, EmissionLines, LOSVDConv, Rebin
from bsf.likelihoods import LogLike

class SSPPlusEmission():
    def __init__(self, wave, ssp, emission, velscale):
        self.wave = wave
        self.ssp = ssp
        self.emission = emission
        self.velscale = velscale
        self.wave = self.ssp.wave
        self.parnames = self.ssp.parnames + self.emission.parnames
        self.shape_grad = (len(self.parnames), len(self.wave))
        self._n =  len(self.ssp.parnames)

    def __call__(self, theta):
        p1 = theta[:self._n]
        p2 = theta[self._n:]
        ssp = self.ssp(p1)[0]
        emission = self.emission(p2)
        return ssp + emission

    def gradient(self, theta):
        p1 = theta[:self._n]
        p2 = theta[self._n:]
        ssp = self.ssp(p1)
        emission = self.emission(p2)
        sspgrad = self.ssp.gradient(p1)
        emission_grad = self.emission.gradient(p2)
        grad = np.zeros(self.shape_grad)
        grad[:self._n, :] = emission * sspgrad
        grad[self._n:, :] = ssp * emission_grad
        return grad

def build_model(wave, flux, fluxerr, spec, params, loglike=None):
    loglike = "normal2" if loglike is None else loglike
    model = pm.Model()
    emlines = ['Hbeta', 'Halpha', 'SII6716', 'SII6731',
               'OIII5007d', 'OI6300d', 'NII6583d']
    with model:
        theta = []
        for param in params.colnames:
            vmin = params[param].data.min()
            vmax = params[param].data.max()
            v = pm.Uniform(param, lower=vmin, upper=vmax)
            theta.append(v)
        for em in emlines:
            v = pm.HalfNormal(em, sigma=1)
            theta.append(v)
        BoundedNormal = pm.Bound(pm.Normal, lower=3500, upper=4200)
        V = BoundedNormal("V", mu=3800., sigma=100.)
        theta.append(V)
        sigma = pm.HalfNormal("sigma", sd=200)
        theta.append(sigma)
        if loglike == "studt":
            nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
            theta.append(nu)
        if loglike == "normal2":
            x = pm.Normal("x", mu=0, sd=1)
            s = pm.Deterministic("S", 1. + pm.math.exp(x))
            theta.append(s)
        theta = tt.as_tensor_variable(theta).T
        logl = LogLike(flux, wave, fluxerr, spec, loglike=loglike)
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})

    return model

def plot_MAP(bsf, mapfile):
    """ Plot the SED model for MAP estimate"""
    x = np.load(mapfile).item() # see: https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
    sol = []
    for par in bsf.parnames:
        print(par, float(x[par]))
        sol.append(x[par])
    sol = np.array(sol)
    ax = plt.subplot(111)
    plt.plot(bsf.wave, bsf.flux + bsf.fluxerr)
    plt.plot(bsf.wave, bsf.flux - bsf.fluxerr, c="C0")
    ax.plot(bsf.wave, bsf.sed(sol), "-", c="C1")
    plt.show()

def load_traces(db, params, alpha=15.865):
    if not os.path.exists(db):
        return None
    ntraces = len(os.listdir(db))
    data = [np.load(os.path.join(db, _, "samples.npz")) for _ in
            os.listdir(db)]
    traces = []
    for param in params:
        v = np.vstack([data[num][param] for num in range(ntraces)]).flatten()
        traces.append(v)
    traces = np.column_stack(traces)
    return traces

def example_full_spectral_fitting():
    """ Example of application of BSF for full spectral fitting of a single
    spectrum using E-MILES models.

    Both the spectrum and the models used in this models are homogeneized to
    a resolution FWHM=2.95 for MUSE data and templates have been rebinned to
    a velocity scale of 200 km/s.
    """
    velscale = 200
    data = Table.read("ngc3311_center.fits")
    wave = data["wave"].data
    flux = data["flux"].data
    fluxerr = data["fluxerr"].data
    # Resampling to fixed velocity scale
    logLam = log_rebin([wave[0], wave[-1]], flux, velscale=velscale)[1]
    lam = np.exp(logLam)
    idx = np.where((lam >= 4500) & (lam <= 5995))[0][1:-1]
    newwave = lam[idx]
    flux, fluxerr = spectres(newwave, wave, flux, spec_errs=fluxerr)
    norm = np.nanmedian(flux)
    flux /= norm
    fluxerr /= norm
    # Reading templates
    templates_file = "emiles_velscale200.fits"
    templates = fits.getdata(templates_file, ext=0)
    tnorm = np.median(templates)
    templates /= tnorm
    params = Table.read(templates_file, hdu=1)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave)
    ssp = SSP(twave, params, templates)
    # Loading templates for the emission lines
    emission, line_names, line_wave = emission_lines(logwave,
                                [wave.min(), wave.max()], 2.95)
    enorm = np.nanmax(emission)
    emission /= enorm
    line_names = list(line_names)
    gas_templates = emission.T
    emission = EmissionLines(twave, gas_templates, line_names)
    ############################################################################
    # Creating a model
    spec0 = SSPPlusEmission(twave, ssp, emission, velscale)
    p0 = np.array([0, 10, 0.3, 0.3, 1, 1, 1, 1, 1, 1, 1, 3800, 200])
    spec1 = LOSVDConv(twave, spec0, velscale)
    spec = Rebin(newwave, spec1)
    # plt.plot(spec0.wave, spec0(p0[:-2]))
    # plt.plot(spec1.wave, spec1(p0))
    # plt.plot(spec.wave, spec(p0))
    # plt.show()
    model = build_model(wave, flux, fluxerr, spec, params)
    with model:
        trace = pm.sample()
        df = pm.stats.summary(trace, alpha=0.3173)
        df.to_csv("summary.txt")
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
