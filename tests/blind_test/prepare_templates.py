# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Prepare stellar population model as templates for pPXF and SSP fitting.

"""
from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.table import Table, hstack
from astropy.io import fits

from specutils.io import read_fits
from scipy.ndimage.filters import gaussian_filter1d

import ppxf.ppxf_util as util
from spectres import spectres

import context

class EMiles():
    """ Class to handle data from the EMILES SSP models. """
    def __init__(self):
        # Location of the EMILES SSP models
        self.path = "/home/kadu/Dropbox/hydraimf/models/EMILES_BASTI_INTERPOLATED"
        self.fhwm = 2.51
        self.gamma = np.array([0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 2.3,
                               2.5, 2.8, 3.0, 3.3, 3.5])
        self.Z = np.array([-0.96, -0.66, -0.35, -0.25, 0.06, 0.15, 0.26, 0.4])
        self.age = np.linspace(1., 14., 27)
        self.alphaFe = np.array([0., 0.2, 0.4])
        self.NaFe = np.array([0., 0.3, 0.6])

    def get_filename(self, imf, metal, age, alpha, na):
        """ Returns the name of files for the EMILES library. """
        msign = "p" if metal >= 0. else "m"
        esign = "p" if alpha >= 0. else "m"
        azero = "0" if age < 10. else ""
        nasign = "p" if na >= 0. else "m"
        return "Ebi{0:.2f}Z{1}{2:.2f}T{3}{4:02.4f}_Afe{5}{6:2.1f}_NaFe{7}{" \
               "8:1.1f}.fits".format(imf, msign, abs(metal), azero, age, esign,
                                     abs(alpha), nasign, na)

def prepare_templates(wave, sigma, output, deltaw=50):
    """ Pipeline for the preparation of the templates."""
    emiles = EMiles()
    wnorm = 5635
    dnorm = 40
    grid = np.array(np.meshgrid(emiles.gamma, emiles.Z,
                             emiles.age, emiles.alphaFe,
                             emiles.NaFe)).T.reshape(-1, 5)
    # Using first spectrum to build reference arrays
    filename = os.path.join(emiles.path, emiles.get_filename(*grid[0]))
    spec0 = read_fits.read_fits_spectrum1d(filename)
    idx = np.where(np.logical_and(spec0.dispersion > wave[0] - deltaw,
                                  spec0.dispersion < wave[-1] + deltaw))[0]
    idx_norm = np.where(np.logical_and(wave > wnorm - dnorm,
                                  wave < wnorm + dnorm))[0]
    wave_templates = spec0.dispersion[idx]
    flux0 = spec0.flux[idx]
    wrange = [wave_templates[0], wave_templates[-1]]
    newflux, logLam, velscale = util.log_rebin(wrange, flux0)
    sigma_diff = sigma / velscale
    ssps = np.zeros((len(grid), len(wave)))
    params = np.zeros((len(grid), 5))
    norms = np.ones(len(grid))
    for i, args in enumerate(grid):
        print("Spectrum {} ({}/{})".format(filename.split("/")[-1], i+1,
                                           len(grid)))
        filename = os.path.join(emiles.path, emiles.get_filename(*args))
        spec = read_fits.read_fits_spectrum1d(filename)
        flux = spec.data[idx]
        fluxrebin, logLam, velscale = util.log_rebin(wrange, flux,
                                                     velscale=velscale)
        fluxbroad = gaussian_filter1d(fluxrebin, sigma_diff, mode="constant",
                                 cval=0.0)
        flux = spectres(wave, np.exp(logLam), fluxbroad)
        norm = np.median(flux[idx_norm])
        ssps[i] = flux / norm
        norms[i] = norm
        params[i] = args
    params = Table(params, names=["alpha", "Z", "age", "alphaFe",
                                  "NaFe"])
    norms = Table([norms], names=["norm"])
    params = hstack([params, norms])
    hdu1 = fits.PrimaryHDU(ssps)
    hdu2 = fits.ImageHDU(wave)
    hdu3 = fits.BinTableHDU(params)
    hdu1.header["CRVAL1"] = wave[0]
    hdu1.header["CD1_1"] = wave[1] - wave[0]
    hdu1.header["CRPIX1"] = 1.
    hdu1.header["EXTNAME"] = "TEMPLATES"
    hdu2.header["EXTNAME"] = "DISPERSION"
    hdu3.header["EXTNAME"] = "PARAMS"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return

def prepare_blind_test():
    """ Simple example of how to prepare the templates. """
    sigma = 360 # km/s
    testfile = os.path.join(context.data_dir, sorted(os.listdir(
        context.data_dir))[0])
    spec = read_fits.read_fits_spectrum1d(testfile)
    outdir = os.path.join(context.home, "templates")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    output = os.path.join(outdir, "emiles_ages_metal.fits")

    prepare_templates(spec.dispersion, sigma, output)

if __name__ == "__main__":
    prepare_blind_test()
