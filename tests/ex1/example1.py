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
from scipy.optimize import least_squares
import pymc3 as pm

from ppxf.ppxf_util import emission_lines

sys.path.append("../../bsf")

from bsf.models import SEDModel
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
                           "data/emiles_muse_vel30_w4500_10000_kinematics.fits")
    templates = fits.getdata(templates_file, ext=0)
    table = Table.read(templates_file)
    logwave = Table.read(templates_file, hdu=2)["loglam"].data
    twave = np.exp(logwave) * u.angstrom
    velscale = 30 * u.km / u.s
    # In this subset of templates, the IMF is constant, so we are not use
    # that as a input parameter
    # All spectra have been normalized to a similar flux, and this normalization
    # is in the last column of the table, and it is not a parameter for the
    # fitting.
    params = table[table.colnames[1:-1]]
    params.rename_column("[Z/H]", "Z")
    params.rename_column("age", "T")
    params.rename_column("[alpha/Fe]", "alphaFe")
    params.rename_column("[Na/Fe]", "NaFe")
    # Loading templates for the emission lines
    emission, line_names, line_wave = emission_lines(logwave,
                                [wave.value.min(), wave.value.max()], 2.95)
    gas_templates = emission.T
    em_components = np.ones(7)
    nlines = line_names
    ############################################################################
    # Generating models with SEDModel
    nssps = [1]
    bsf = BSF(wave, flux, twave, templates, params, fluxerr=fluxerr,
              em_templates=gas_templates, em_names=line_names,
              velscale=velscale, nssps=nssps, em_components=em_components,
              z=0.012759)
    bsf.build_model()
    hmap_file = os.path.join(os.getcwd(), "hmap.npy")
    with bsf.model:
        m = pm.find_MAP()
        np.save(hmap_file, m)
    pfit = np.array([m[p] for p in bsf.parnames])
    plt.plot(wave.value, flux)
    plt.plot(wave.value, bsf.sed(pfit))
    plt.plot(wave.value, flux - bsf.sed(pfit))
    plt.show()
    # ##########################################################################
    # testvals_csp = {"Av": 0.1, "Rv": 4.05, "flux": 350, "Z": 0.1, "T": 10.2,
    #             "alphaFe": 0.22, "NaFe": 0.15, "V": 4000, "sigma": 300}
    # testvals_em = {"flux": 2000, "V": 4000, "sigma": 50}
    # lower_csp = {"Av": 0, "Rv": 2., "flux": 0, "Z": params["Z"].min(),
    #              "T": params["T"].min(),  "alphaFe": params["alphaFe"].min(),
    #              "NaFe": params["NaFe"].min(), "V": 3000, "sigma": 0}
    # upper_csp = {"Av": 4, "Rv": 6., "flux": np.infty, "Z": params["Z"].max(),
    #              "T": params["T"].max(),  "alphaFe": params["alphaFe"].max(),
    #              "NaFe": params["NaFe"].max(), "V": 5000, "sigma": 500}
    # lower_em = {"flux": 0, "V": 3000, "sigma": 0.}
    # upper_em = {"flux": np.infty, "V": 5000, "sigma": 100.}
    # # Performing least_squares fitting.
    # # Setting initial values according to testvals
    # p0 = []
    # lbounds = []
    # ubounds = []
    # for pnames in bsf.sed.parnames:
    #     poptype = pnames[0].split("_")[0]
    #     ps = ["_".join(p.split("_")[1:]).split("_")[0] for p in pnames]
    #     testvals = testvals_csp if poptype.startswith("pop") else testvals_em
    #     lower = lower_csp if poptype.startswith("pop") else lower_em
    #     upper = upper_csp if poptype.startswith("pop") else upper_em
    #     for p in ps:
    #         p0.append(testvals[p])
    #         lbounds.append(lower[p])
    #         ubounds.append(upper[p])
    # bounds = (np.array(lbounds), np.array(ubounds))
    # p0 = np.array(p0)
    # # Calculating best model
    # def residue(p):
    #     return (flux - bsf.sed(p)) / fluxerr
    # sol = least_squares(residue, p0, bounds=bounds, loss="linear")
    # p = sol["x"]
    # plt.plot(wave.value, flux)
    # plt.plot(wave.value, bsf.sed(p))
    # plt.plot(wave.value, flux - bsf.sed(p))
    # plt.show()

