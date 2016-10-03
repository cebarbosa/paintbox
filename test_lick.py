# -*- coding: utf-8 -*-
"""

Created on 29/09/2016

@Author: Carlos Eduardo Barbosa

Tests computation of Lick indices using SSPs from the MILES library.

"""
import os

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lick import Lick

def test_lick():
    """ Validation of code for calculation of Lick indices. """
    # Reading files containing bands
    bands = os.path.join(os.getcwd(), "tables/bands.txt")
    workdir = "/home/kadu/Dropbox/groups/MILES"
    os.chdir(workdir)
    filename = "lector_tmputH9bu.list_LINE"
    stars = np.loadtxt(filename, usecols=(0,),
                       dtype=str)
    ref = np.loadtxt(filename,
             usecols=(2,3,4,5,6,7,8,9,14,15,16,17,18,24,25,26,
                      27,28,29,30,31,32,33,34,35))
    obs = []
    for i, star in enumerate(stars):
        print star + ".fits"
        spec = pf.getdata(star + ".fits")
        h = pf.getheader(star + ".fits")
        w = h["CRVAL1"] + h["CDELT1"] * \
                            (np.arange(h["NAXIS1"]) + 1 - h["CRPIX1"])
        ll = Lick(w, spec, np.loadtxt(bands, usecols=(2,3,4,5,6,7,)))
        ll.classic_integration()
        obs.append(ll.classic)
    obs = np.array(obs)
    fig = plt.figure(1, figsize=(20,12))
    gs = GridSpec(5,5)
    gs.update(left=0.08, right=0.98, top=0.98, bottom=0.06, wspace=0.25,
              hspace=0.4)
    obs = obs.T
    ref = ref.T
    names = np.loadtxt(bands, usecols=(0,), dtype=str)
    units = np.loadtxt(bands, usecols=(9,), dtype=str).tolist()
    # units = [x.replace("Ang", "\AA") for x in units]
    for i in range(25):
        ax = plt.subplot(gs[i])
        plt.locator_params(axis="x", nbins=6)
        ax.minorticks_on()
        ax.plot(obs[i], (obs[i] - ref[i]), "o", color="0.5")
        ax.axhline(y=0, ls="--", c="k")
        lab = "median $= {0:.3f}$".format(
            np.nanmedian(obs[i] - ref[i])).replace("-0.00", "0.00")
        ax.axhline(y=np.nanmedian(obs[i] - ref[i]), ls="--", c="r", label=lab)
        ax.set_xlabel("{0} ({1})".format(names[i].replace("_", " "), units[i]))
        ax.legend(loc=1,prop={'size':15})
        if units[i] == "Ang":
            ax.set_ylim(-0.05, 0.05)
        else:
            ax.set_ylim(-0.002, 0.002)
    fig.text(0.02, 0.5, '$\Delta I$',
             va='center',
             rotation='vertical', size=40)
    plt.show()

if __name__ == "__main__":
    test_lick()
