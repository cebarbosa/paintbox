# -*- coding: utf-8 -*-
""" 

Created on 18/05/18

Author : Carlos Eduardo Barbosa

Test the recovery of stellar populations in the case of single bursts

"""
from __future__ import print_function, division

import os
import pickle

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

def plot(flux, dbname, outw1, outw2, dw, tabfile):
    """ Plot results from model. """
    wave, params, templates = prepare_templates(outw1, outw2, dw)
    with open(dbname, 'rb') as buff:
        mcmc = pickle.load(buff)
    trace = mcmc["trace"]
    # Recovering information from table
    table = Table.read(tabfile)
    what = np.zeros(len(templates))
    for line in table:
        what[line["idx"]] = line["weights"]
    # make_corner_plot(trace, params)
    bestfit = trace["bestfit"].mean(axis=0)
    bf05 = np.percentile(trace["bestfit"], 5, axis=0)
    bf95 = np.percentile(trace["bestfit"], 95, axis=0)
    # Plot
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax1.minorticks_on()
    ax1.plot(wave, flux, "o-")
    ax1.plot(wave, bestfit, "o-")
    ax1.fill_between(wave, bf05, bf95, color="C1", alpha=0.5)
    ax1.set_xlabel("$\lambda$ (\AA)")
    ax1.set_ylabel("Flux ($10^{-20}$ erg s $^{\\rm -1}$ cm$^{\\rm -2}$ \\r{"
                   "A}$^{\\rm -1}$)")
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax2.minorticks_on()
    ax2.plot(wave, bestfit - flux, c="C1")
    ax2.fill_between(wave,  (bf05-flux), bf95 - flux, color="C1", alpha=0.5)
    ax3 = plt.subplot2grid((4, 1), (3, 0))
    weights = trace["w"].mean(axis=0)
    print(weights[np.argsort(weights)[::-1][:3]])
    print(params[np.argsort(weights)[::-1][:3]])
    plt.plot(what)
    plt.plot(weights, "-")
    plt.show()

def make_corner_plot(trace, params):
    """ Produces corner plot for relevant variables. """
    weights = trace["w"].mean(axis=0)
    parnames = params.colnames[:-1]
    npars = len(params.colnames[:-1])
    fig = plt.figure(1)
    idxs = np.arange(npars)
    ij = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    widths = [0.1, 0.1, 0.8, 0.15, 0.2]
    for i, j in ij:
        if i == j:
            ax = plt.subplot(npars, npars, j + npars * i + 1)
            ax.minorticks_on()
            values = np.unique(params[parnames[i]])
            w = np.zeros(len(values))
            for k,val in enumerate(values):
                idx = np.where(params[parnames[i]] == val)
                if not len(idx):
                    continue
                w[k] = np.sum(weights[idx] / params["norm"][idx])
                # w[k] = np.sum(weights[idx])
            w /= w.sum()
            ax.bar(values, w, width=widths[i])
        else:
            ax = plt.subplot(npars, npars, j + npars * i + 1)
            ax.minorticks_on()
            v1s = np.unique(params[parnames[i]])
            v2s = np.unique(params[parnames[j]])
            w = np.zeros((len(v1s), len(v2s)))
            for l,v1 in enumerate(v1s):
                idx1 = np.where(params[parnames[i]] == v1)[0]
                for m,v2 in enumerate(v2s):
                    idx2 = np.where(params[parnames[j]] == v2)[0]
                    idx = np.intersect1d(idx1, idx2)
                    w[l,m] = np.sum(weights[idx] / params["norm"][idx])
                    # w[l, m] = np.sum(weights[idx])
            x, y = np.meshgrid(v1s, v2s)
            ax.pcolormesh(y.T, x.T, w, shading="gouraud")
            ax.contour(y.T, x.T, w, colors="k")
    plt.show()

if __name__ == "__main__":
    # Simulation parameters
    outw1 = 4700
    outw2 = 9100
    dw = 5
    sigma = 350
    nsim = 1
    redo=True
    home = "/home/kadu/Dropbox/basket"
    templates_dir = os.path.join(home, "templates")
    simulation_dir = os.path.join(home, "sim_sigma{}_dw{}".format(sigma, dw))
    if not os.path.exists(simulation_dir):
        os.mkdir(simulation_dir)
    make_mock_spectra(simulation_dir, nsim=nsim, dw=dw)
    wave, params, templates = prepare_templates(outw1, outw2, dw)
    templates = np.array(templates, dtype=np.float)
    for i in range(nsim):
        specfile = os.path.join(simulation_dir, "spec_{:04d}.fits".format(i+1))
        tabfile = os.path.join(simulation_dir, "pars_{:04d}.fits".format(i+1))
        specdata = Table.read(specfile, format="fits")
        flux = specdata["flux"]
        dbname = specfile.replace(".fits", ".db")
        if not os.path.exists(dbname) or redo:
            tmcsp(wave, flux, templates, dbname=dbname,
                  adegree=None, redo=redo)
        plot(flux, dbname, outw1, outw2, dw, tabfile)
