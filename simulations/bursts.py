# -*- coding: utf-8 -*-
""" 

Created on 18/05/18

Author : Carlos Eduardo Barbosa

Test the recovery of stellar populations in the case of single bursts

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
import matplotlib.pyplot as plt

from misc import array_from_header
from models.tmcsp import tmcsp

def make_mock_spectra(outdir, nsim=10, nbursts=3, sigma=350, outw1=4700,
                      outw2=9100,
                    dw=10):
    """ Produces mock spectra simulating stellar populations produced in
    bursts. """
    velscale = 30
    templates_dir = "/home/kadu/Dropbox/basket/templates"
    tempfile = os.path.join(templates_dir, "emiles_velscale{}.fits".format(
        velscale))
    wave = np.exp(array_from_header(tempfile, axis=1, extension=0))
    wregrid = np.arange(outw1, outw2, dw)
    ssps = fits.getdata(tempfile, 0)
    params = Table.read(tempfile, hdu=2)
    ntemp = len(ssps)
    # Normalization of the SSPs to get fractions in terms of light
    norms = np.zeros(ntemp)
    for i,ssp in enumerate(ssps):
        norms[i] = np.median(ssp)
        ssps[i] = ssp / norms[i]
    for i in range(nsim):
        outtable = os.path.join(outdir, "pars_{:04d}.fits".format(i+1))
        outspec = os.path.join(outdir, "spec_{:04d}.fits".format(i+1))
        if os.path.exists(outtable) and os.path.exists(outspec):
            continue
        weights = np.random.dirichlet(np.ones(nbursts))
        idx = np.random.choice(ntemp, size=nbursts)
        simtable = Table([idx, weights], names=["idx", "weights"])
        simtable = hstack([params[idx], simtable])
        simtable.write(outtable, overwrite=True)
        ssps_sim = np.dot(weights, ssps[idx])
        flux = gaussian_filter1d(ssps_sim, sigma / velscale, mode="constant",
                                 cval=0.0)
        fsim = spectres(wregrid, wave, flux)
        spec = Table([wregrid, fsim], names=["wave", "flux"])
        spec.write(outspec, overwrite=True)

def prepare_templates(outw1, outw2, dw, sigma=350, redo=False, velscale=30):
    """ Resample templates for full spectral fitting. """
    tempfile = os.path.join(templates_dir, "emiles_velscale{}.fits".format(
        velscale))
    output = os.path.join(templates_dir,
             "emiles_sigma{}_dw{}.fits".format(sigma, dw))
    if os.path.exists(output) and not redo:
        templates = fits.getdata(output, 0)
        wave = fits.getdata(output, 1)
        params = fits.getdata(output, 2)
        return wave, params, templates
    wave = np.exp(array_from_header(tempfile, axis=1,
                                                  extension=0))
    ssps = fits.getdata(tempfile, 0)
    params = Table.read(tempfile, hdu=2)
    newwave = np.arange(outw1, outw2, dw)
    templates = np.zeros((len(ssps), len(newwave)))
    norms = np.zeros(len(ssps))
    for i in np.arange(len(ssps)):
        sigma_pix = sigma / velscale
        flux = gaussian_filter1d(ssps[i], sigma_pix, mode="constant",
                                 cval=0.0)
        norm = np.median(flux)
        flux /= norm
        templates[i] = spectres(newwave, wave, flux)
        norms[i] = norm
    norms = Table([norms], names=["norm"])
    params = hstack([params, norms])
    hdu1 = fits.PrimaryHDU(templates)
    hdu2 = fits.ImageHDU(newwave)
    hdu3 = fits.BinTableHDU(params)
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return newwave, params, templates

if __name__ == "__main__":
    # Simulation parameters
    outw1 = 4700
    outw2 = 9100
    dw = 10
    sigma = 350
    nsim = 100
    home = "/home/kadu/Dropbox/basket"
    templates_dir = os.path.join(home, "templates")
    simulation_dir = os.path.join(home, "sim_sigma{}_dw{}".format(sigma, dw))
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)
    make_mock_spectra(simulation_dir, nsim=nsim)
    wave, params, templates = prepare_templates(outw1, outw2, dw)
    templates = np.array(templates, dtype=np.float)
    for i in range(nsim):
        specfile = os.path.join(simulation_dir, "spec_{:04d}.fits".format(i+1))
        specdata = Table.read(specfile, format="fits")
        dbname = specfile.replace(".fits", ".db")
        tmcsp(wave, specdata["flux"], templates, dbname=dbname, adegree=None)