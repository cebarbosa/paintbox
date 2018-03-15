# -*- coding: utf-8 -*-
""" 

Created on 05/03/18

Author : Carlos Eduardo Barbosa

Determination of LOSVD of a given spectrum similarly to pPXF.

"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..')))

import numpy as np
import pymc3 as pm
from astropy import constants
import matplotlib.pyplot as plt
from specutils.io.read_fits import read_fits_spectrum1d

from miles_util import Miles
from ppxf import ppxf
from ppxf_util import log_rebin, gaussian_filter1d
from der_snr import DER_SNR

from models.bppxf import bppxf

def example_ppxf_noregul():
    """ Fist example using Hydra cluster data to make ppxf-like model.

    The observed spectrum is one of the central spectra the Hydra I cluster
    core observed with VLT/FORS2 presented in Barbosa et al. 2016. This
    spectrum is not flux calibrated, hence the need of a polynomial term.

    In this example, we use a single spectrum with high S/N to derive the
    line-of-sight velocity distribution using a Gauss-Hermite distribution.

    """
    specfile =  os.path.join(os.getcwd(), "hydra1/fin1_n3311cen1_s29a.fits")
    spec = read_fits_spectrum1d(specfile)
    disp = spec.dispersion[1] - spec.dispersion[0]
    velscale = 40
    fwhm_fors2 = 2.1
    fwhm_miles = 2.51
    fwhm_dif = np.sqrt((fwhm_miles ** 2 - fwhm_fors2 ** 2))
    sigma = fwhm_dif / 2.355 / disp
    galaxy = gaussian_filter1d(spec.flux, sigma)
    lamrange = [spec.dispersion[0], spec.dispersion[-1]]
    galaxy, wave = log_rebin(lamrange, galaxy, velscale=velscale)[:2]
    # Read templates
    filenames = os.path.join(os.getcwd(), "miles_models", 'Mun1.30*.fits')
    miles = Miles(filenames, velscale, 2.51)
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    snr = DER_SNR(spec.flux)
    noise = np.median(spec.flux) / snr * np.ones_like(galaxy)
    c = constants.c.to("km/s").value
    dv = np.log(np.exp(miles.log_lam_temp[0]) / lamrange[0]) * c
    start = [4000, 300]
    goodpixels = np.argwhere(np.logical_and(np.exp(wave) > 4800, np.exp(wave) <
                                        5800 )).T[0]
    # pp = ppxf(stars_templates, galaxy, noise, velscale, start,
    #           plot=True, moments=4, degree=8, mdegree=-1, vsyst=dv,
    #           lam=np.exp(wave), clean=False, goodpixels=goodpixels)
    # plt.show()
    # print(pp)
    bpp = bppxf(stars_templates.T, galaxy)

if __name__ == "__main__":
    example_ppxf_noregul()