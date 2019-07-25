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
import theano
import emcee
import pymc3 as pm
from multiprocessing import Pool


from ppxf.ppxf_util import emission_lines

sys.path.append("../../bsf")

from bsf.bsf import BSF

if __name__ == "__main__":
    # Reading input data
    # Data from central region of NGC 3311 observed with MUSE
    # Resolution have been homogenized to 2.95 Angstrom
    specfile = os.path.join(os.getcwd(), "data/fieldA_sn80_0001.fits")
    data = Table.read(specfile)
    wave = data["wave"].data * u.angstrom
    flux = data["flux"].data
    fluxerr = data["fluxerr"].data
    # Reading EMILES templates
    # The templates have been rebined to a logarithmic scale with dispersion
    # of 30 km/s/pixel, which is much better than the resolution of the
    # MUSE data
    # Templates have been already convolved to match the resolution of the
    # observations
    templates_file = os.path.join(os.getcwd(),
                           "data/emiles_muse_vel50_w4500_10000_test.fits")
    templates = fits.getdata(templates_file, ext=0)
    table = Table.read(templates_file, hdu=1)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave) * u.angstrom
    velscale = 50 * u.km / u.s
    # In this subset of templates, the IMF is constant, so we are not use
    # that as a input parameter
    # All spectra have been normalized to a similar flux, and this normalization
    # is in the last column of the table, and it is not a parameter for the
    # fitting.
    params = table[table.colnames[:-1]]
    # Loading templates for the emission lines
    emission, line_names, line_wave = emission_lines(logwave,
                                [wave.value.min(), wave.value.max()], 2.95)
    gas_templates = emission.T
    em_components = np.ones(7)
    nlines = line_names
    ############################################################################
    # Generating models with SEDModel
    nssps = 1
    bsf = BSF(wave, flux, twave, templates, params, fluxerr=fluxerr,
              em_templates=gas_templates, em_names=line_names,
              velscale=velscale, nssps=nssps, em_components=em_components,
              z=0.012759, loglike="normal")
    bsf.build_model()
    ############################################################################
    # Testing EMCEE
    with bsf.model as model:
        f = theano.function(model.vars, [model.logpt] + model.deterministics)


        def log_prob_func(params):
            dct = model.bijection.rmap(params)
            args = (dct[k.name] for k in model.vars)
            results = f(*args)
            return tuple(results)
    with bsf.model as model:
        # First we work out the shapes of all of the deterministic variables
        res = pm.find_MAP()
        vec = model.bijection.map(res)
        initial_blobs = log_prob_func(vec)[1:]
        dtype = [(var.name, float, np.shape(b)) for var, b in
                 zip(model.deterministics, initial_blobs)]

        # Then sample as usual
        coords = vec + 1e-5 * np.random.randn(40, len(vec))
        nwalkers, ndim = coords.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func,
                                        blobs_dtype=dtype)
        sampler.run_mcmc(coords, 5000, progress=True)
    # Running NUTS
    # hmap_file = os.path.join(os.getcwd(), "hmap.npy")
    # nuts_dir = os.path.join(os.getcwd(), "NUTS")
    # NUTS_summary = os.path.join(os.getcwd(), "NUTS_summary.txt")
    # with bsf.model:
    #     trace = pm.sample(njobs=1)
    #     df = pm.stats.summary(trace, alpha=0.3173)
    #     df.to_csv(NUTS_summary)