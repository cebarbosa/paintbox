# -*- coding: utf-8 -*-
"""

Created on 29/09/2016

@Author: Carlos Eduardo Barbosa

Tests computation of Lick indices using SSPs from the MILES library.

"""

from __future__ import print_function, division
import os

import numpy as np
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lick import Lick

def test_lick():
    """ Validation of code for calculation of Lick indices. """
    ############################################################################
    # We use a set of SSP models from Vazdequis+ 2010 for comparison.
    # Set directory with MILES SSP models
    data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                            "MILES")
    # File containing information about Lick indices
    filename = os.path.join(data_dir, "lector_tmputH9bu.list_LINE")
    # Name of the spectra files
    stars = np.loadtxt(filename, usecols=(0,), dtype=str)
    # Read the values of the Lick indices
    lick_miles = np.loadtxt(filename,
             usecols=(2,3,4,5,6,7,8,9,14,15,16,17,18,24,25,26,
                      27,28,29,30,31,32,33,34,35))
    ############################################################################
    # Reading the definition of Lick indices in an ASCII table
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                             "tables/bands.txt")
    names = np.loadtxt(bandsfile, usecols=(0,), dtype=str)
    units = np.loadtxt(bandsfile, usecols=(9,), dtype=str).tolist()
    units = np.array([u.Unit(_.replace("Ang", "Angstrom")) for _ in units])
    bands = np.loadtxt(bandsfile, usecols=(2,3,4,5,6,7,)) * u.AA
    obs = []
    for i, star in enumerate(stars):
        filename = os.path.join(data_dir, "{}.fits".format(star))
        spec = fits.getdata(filename)
        h = fits.getheader(filename)
        w = h["CRVAL1"] + h["CDELT1"] * \
                            (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"])
        w *= u.AA
        ll = Lick(w, spec, bands, units=units)
        ll.classic_integration()
        obs.append(ll.classic)
    obs = np.array(obs)
    fig = plt.figure(1, figsize=(20,12))
    gs = GridSpec(5,5)
    gs.update(left=0.08, right=0.98, top=0.98, bottom=0.06, wspace=0.25,
              hspace=0.4)
    obs = obs.T
    ref = lick_miles.T
    for i in range(25):
        ax = plt.subplot(gs[i])
        plt.locator_params(axis="x", nbins=6)
        ax.minorticks_on()
        ax.plot(obs[i], (obs[i] - ref[i]), "o")
        ax.axhline(y=0, ls="--", c="k")
        lab = "median $= {0:.3f}$".format(
            np.nanmedian(obs[i] - ref[i])).replace("-0.00", "0.00")
        ax.axhline(y=np.nanmedian(obs[i] - ref[i]), ls="--", c="C1", label=lab)
        ax.set_xlabel("{0} ({1})".format(names[i].replace("_", " "), units[i]))
        ax.legend(loc=1,prop={'size':15})
        if units[i] == u.Unit("Angstrom"):
            ax.set_ylim(-0.05, 0.05)
        else:
            ax.set_ylim(-0.002, 0.002)
    fig.text(0.02, 0.5, '$\Delta I$',
             va='center',
             rotation='vertical', size=40)
    plt.show()

if __name__ == "__main__":
    test_lick()
