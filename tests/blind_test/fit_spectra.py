# -*- coding: utf-8 -*-
""" 

Created on 31/08/18

Author : Carlos Eduardo Barbosa

Using BSF to fit the spectra in the blind_test.

"""
from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from specutils.io.read_fits import read_fits_spectrum1d
import pymc3 as pm

import matplotlib.pyplot as plt

import context
from bsf.bsf import BSF

def load_templates():
    """ Load set of templates for a given """
    templates_file = os.path.join(context.home, "templates",
                                  "emiles_templates.fits")
    templates = fits.getdata(templates_file, 0)
    templates = np.array(templates, dtype=np.float)
    params = fits.getdata(templates_file, extname="PARAMS")
    params = Table(params)
    norm = params["norm"]
    del params["norm"]
    wave = fits.getdata(templates_file, extname="DISPERSION")
    return templates, params, norm, wave


def plot_fitting(bsf, norm=1., spec=None):
    """ Plot the best fit results in comparison with the data. """
    spec = "Data" if spec is None else spec
    with bsf.model:
        # wp = np.array([bsf.trace["mpoly_{}".format(i)] for i in range(
        #                bsf.mdegree + 1)]).T
        wp = bsf.trace["mpoly"]
        w = bsf.trace["w"]
    idxs = []
    for i, p in enumerate(bsf.params.colnames):
        idxs.append(bsf.trace["{}_idx".format(p)])
    idxs = np.array(idxs)
    nchains = idxs.shape[1]
    bestfits = np.zeros((nchains, len(bsf.wave)))
    mpoly = np.zeros_like(bestfits)
    csps = np.zeros_like(bestfits)
    for i in np.arange(nchains):
        idx = idxs[:, i, :]
        ssps =  bsf.templates[idx[0], idx[1], idx[2], idx[3], idx[4], :]
        csps[i] = np.dot(w[i].T, ssps)
        mpoly[i] = np.dot(wp[i].T, bsf.mpoly)
        bestfits[i] = mpoly[i] *  csps[i]
    bestfit = np.percentile(bestfits, 50, axis=0)
    percupper = np.percentile(bestfits, 50 + 34.14, axis=0)
    perclower = np.percentile(bestfits, 50 - 34.14, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   gridspec_kw = {'height_ratios':[3, 1]},
                                   figsize=(3.32153, 2.5))
    for ax in (ax1, ax2):
        ax.tick_params(right=True, top=True, axis="both",
                       direction='in', which="both")
        ax.minorticks_on()
    ax1.fill_between(bsf.wave, norm * (bsf.flux + bsf.fluxerr),
                     norm * (bsf.flux - bsf.fluxerr), linestyle="-",
                     color="C0", label=spec)
    ax1.fill_between(bsf.wave, norm * perclower, norm * percupper,
                     linestyle="-", color="C1",
                     label="BSF model")
    ax1.set_xticklabels([])
    ax1.set_ylabel("Flux ($10^{-20}$ erg/cm$^{2}$/\\r{A}/s)",
                   fontsize=8)
    ax1.legend(loc=2, prop={'size': 6}, frameon=False)
    ax2.plot(bsf.wave, 100 * (bsf.flux - bestfit) / bsf.flux)
    ax2.axhline(y=0, c="k", ls="--")
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel("Resid. (\%)", fontsize=8)
    ax2.set_xlabel("Rest Wavelength (\\r{A})", fontsize=8)
    plt.subplots_adjust(left=0.14, right=0.985, hspace=0.08, bottom=.16,
                        top=.98)
    return

def fit_spectra(plot=False):
    wnorm = 5635
    dnorm = 40
    filenames = sorted(os.listdir(context.data_dir))
    outdir = os.path.join(context.home, "bsf")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for filename in filenames:
        print("Processing file {}".format(filename))
        spec = read_fits_spectrum1d(os.path.join(context.data_dir, filename))
        dbname = os.path.join(outdir, filename.replace(".fits", ""))
        summary = os.path.join(outdir, filename.replace(".fits", ".txt"))
        cornerplot = os.path.join(outdir, filename.replace(".fits", ".png"))
        templates, params, norm, wave = load_templates()
        idx_norm = np.where(np.logical_and(wave > wnorm - dnorm,
                                           wave < wnorm + dnorm))[0]
        norm = np.median(spec.flux[idx_norm])
        flux = spec.flux / norm
        bsf = BSF(wave, flux, templates, params=params, statmodel="npfit",
<<<<<<< HEAD
                  reddening=False, mdegree=1, robust_fitting=False)
=======
                  reddening=False, mdegree=1, Nssps=10, robust_fitting=False)
>>>>>>> 403f86be6be1a4dd585f19af4601b32ad6323a0f
        if not os.path.exists(dbname):
            with bsf.model:
                db = pm.backends.Text(dbname)
                bsf.trace = pm.sample(trace=db, njobs=4,
                                      nuts_kwargs={"target_accept": 0.90})
                df = pm.stats.summary(bsf.trace)
                df.to_csv(summary)
        with bsf.model:
            bsf.trace = pm.backends.text.load(dbname)
        if not plot:
            continue
        print("Producing corner figure...")
        bsf.plot_corner(labels=[r"$\alpha - 1$", "[Z/H]", "Age (Gyr)",
                                r"[$\alpha$/Fe]", "[Na/Fe]"])
        plt.savefig(cornerplot, dpi=300)
        plt.show()
        # plt.close()
        # bsf.fluxerr = 0
        # plot_fitting(bsf)
        # plt.show()



if __name__ == "__main__":
    fit_spectra(plot=True)


