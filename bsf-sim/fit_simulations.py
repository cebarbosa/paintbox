# -*- coding: utf-8 -*-
""" 

Created on 19/06/18

Author : Carlos Eduardo Barbosa

Fit the simulated spectra using basket routines.

"""
from __future__ import print_function, division, absolute_import

import os
import pickle
import sys

import yaml
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm

import context
from bsf.bsf import BSF
from simulations.make_simulations import Templates


def fit_simulations(simnum, config_file, redo=False, plot=False):
    """ Fitting simulations at different levels using TMCSP"""
    config = yaml.load(open(config_file))

    base_dir = os.path.join(context.workdir, "simulations",
                      "{}_sigma{}".format(config["simtype"], config["sigma"]))
    sim_dir = os.path.join(base_dir, "data")
    fit_dir = os.path.join(base_dir, "nssps_sn{}".format(config["sn"]))
    if not os.path.exists(fit_dir):
        os.mkdir(fit_dir)
    ############################################################################
    # Loading templates
    templates_file = os.path.join(sim_dir, "templates.pkl")
    with open(templates_file, "rb") as f:
        templates = pickle.load(f)
    params = Table([np.log10(templates.ages1D), templates.metals1D], names=[
        "Age", "Z"])
    ############################################################################
    simfiles = [_ for _ in sorted(os.listdir(sim_dir)) if _.endswith(".pkl")
                   and _.startswith("sim")]
    simfile = "sim{:04d}.pkl".format(int(simnum))
    print("Simulation {}".format(len(simfiles)))
    with open(os.path.join(sim_dir, simfile), "rb") as f:
        sim = pickle.load(f)

    dbname = os.path.join(fit_dir, simfile.replace(".pkl", ".db"))
    summary = os.path.join(fit_dir, simfile.replace(".pkl", ".txt"))
    flux = sim["spec"]
    noise = np.random.normal(0, np.median(flux) / config["sn"], size=len(
        flux))
    fsim = flux + noise
    bsf = BSF(templates.wave, fsim, templates.templates, reddening=False,
              statmodel="nssps", params=params, mdegree=0,
              robust_fitting=False)
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
        return
    cornerplot = os.path.join(fit_dir, simfile.replace(".pkl", ".png"))
    if not os.path.exists(cornerplot) or redo:
        bsf.plot_corner()
        plt.savefig(cornerplot, dpi=300)
        plt.close()
    return



if __name__ == "__main__":
    config_file = "simpars/sim0001.yaml"
    if len(sys.argv) == 1:
        sys.argv.append("1")
    fit_simulations(sys.argv[1], config_file)
