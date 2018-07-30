# -*- coding: utf-8 -*-
""" 

Created on 02/07/18

Author : Carlos Eduardo Barbosa

Plot results of the simulations

"""
from __future__ import print_function, division, absolute_import

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import context
from simulations.make_simulated_csps import Templates

def plot_unimodal_simulations(sigma, sn):
    """ Make comparison of simulated values with those obtained with TMCSP. """
    base_dir =  os.path.join(context.workdir, "simulations",
                          "unimodal_sigma{}".format(sigma))
    sim_dir = os.path.join(base_dir, "data")
    fit_dir = os.path.join(base_dir, "npfit_sn{}".format(sn))
    output = os.path.join(context.plots_dir,
                          "sim_unimodal_sigma{}_sn{}.png".format(sigma, sn))
    ############################################################################
    # Loading templates
    templates_file = os.path.join(sim_dir, "templates.pkl")
    with open(templates_file) as f:
        templates = pickle.load(f)
    simfiles = [_ for _ in sorted(os.listdir(sim_dir)) if
                _.endswith(".pkl")  and _.startswith("sim")]
    datasim, datafit, errors = [], [], []
    for j, simfile in enumerate(simfiles):
        print("Processing simulation {}/{}".format(j+1, len(simfiles)))
        # Check if simulations have been fitted.
        dbname = os.path.join(fit_dir, simfile.replace(".pkl", ".db"))
        if not os.path.exists(dbname):
            continue
        #######################################################################
        # Getting results from the simulations
        with open(os.path.join(sim_dir, simfile)) as f:
            sim = pickle.load(f)
        Wsim = sim["weights"]
        Tsim = np.average(templates.ages2D, weights=Wsim)
        Zsim =  np.average(templates.metals2D, weights=Wsim)
        datasim.append([Tsim, Zsim])
        #######################################################################
        # Processing results from the fitting
        with open(dbname, 'rb') as buff:
            trace = pickle.load(buff)
        wfit = trace["w"]
        W3D = np.zeros((wfit.shape[0], templates.ages2D.shape[0],
                        templates.ages2D.shape[1]))
        for i in np.arange(len(W3D)):
            W3D[i] = wfit[i].reshape(templates.ages2D.shape)
        Wfit = W3D.mean(axis=0)
        Wfit_std = W3D.std(axis=0)
        Tfit = np.average(templates.ages2D, weights=Wfit)
        Zfit= np.average(templates.metals2D, weights=Wfit)
        Tfit_err = np.sqrt(np.sum(np.square(templates.ages2D * Wfit_std)))
        Zfit_err = np.sqrt(np.sum(np.square(templates.metals2D * Wfit_std)))
        datafit.append([Tfit, Zfit])
        errors.append([Tfit_err, Zfit_err])
        if i == 5:
            break
    datasim = np.array(datasim).T
    datafit = np.array(datafit).T
    errors = np.array(errors).T
    npars = len(datasim)
    xlim = [[0, templates.ages[-1]],
            [templates.metals[0], templates.metals[-1]]]
    ylim = [[-8, 8], [-1,1]]
    labels = ["Age", "[Z/H]"]
    units = ["Gyr", "dex"]
    fig = plt.figure(1, figsize=(6.9, 2.5))
    for i in range(npars):
        ax = plt.subplot(1, npars, i+1)
        ax.minorticks_on()
        ax.set_xlim(xlim[i])
        ax.set_ylim(ylim[i])
        diff = datafit[i] - datasim[i]
        bias = np.median(diff)
        scatter = np.std(diff)
        plt.errorbar(datasim[i], diff, yerr=errors[i],
                     color="C0",
                     ecolor="0.8", fmt="o", label="S/N={}".format(sn))
        ax.legend(loc=0)
        ax.axhline(0, ls="--", c="k")
        ax.set_ylabel("$\Delta${} ({})".format(labels[i], units[i]))
        ax.set_xlabel("{} ({})".format(labels[i], units[i]))
        ax.text(0.08, 0.12, "bias={0:.2f} {2}    scatter={1:.2f} {2}".format(
            bias,
                                                                  scatter,
                                                                        units[i]),
                transform=ax.transAxes)
    plt.subplots_adjust(left=0.08, right=0.98, wspace=0.25, bottom=0.16,
                        top=0.97)
    plt.savefig(output, dpi=250)

if __name__ == "__main__":
    sigma = 300
    sn = 30
    plot_unimodal_simulations(sigma, sn)