# -*- coding: utf-8 -*-
""" 

Created on 07/09/18

Author : Carlos Eduardo Barbosa

"""
import pickle

import numpy as np
import matplotlib.pyplot as plt

def plot_tmscp(templates, dbname, sim, sn, fsim):
    Wsim = sim["weights"]
    with open(dbname, 'rb') as buff:
        trace = pickle.load(buff)
    ws = trace["w"]
    W3D = np.zeros((ws.shape[0], templates.ages2D.shape[0],
                    templates.ages2D.shape[1]))
    for i in np.arange(len(W3D)):
        W3D[i] = ws[i].reshape(templates.ages2D.shape)
    weights = W3D.mean(axis=0)
    wstd = W3D.std(axis=0)
    Wfit =  weights.reshape(templates.ages2D.shape)
    ############################################################################
    # Plot the results
    fig = plt.figure(1, figsize=(6.97, 6))
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.minorticks_on()
    ax1.plot(templates.wave, fsim, label="Model (S/N={})".format(sn))
    ax1.plot(templates.wave, weights.reshape(-1).dot(templates.templates),
             label="Best fit")
    ax1.set_xlabel("$\lambda$ (\AA)")
    ax1.set_ylabel("Norm. Flux")
    ax1.legend()
    ############################################################################
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.minorticks_on()
    ax2.set_title("Model")
    vmax = np.percentile(weights, 98.8)
    m = ax2.pcolormesh(np.log10(templates.ages2D), templates.metals2D, Wsim,
                       cmap="cubehelix_r", vmax=vmax, vmin=0)
    ax2.set_ylabel("[Fe/H]")
    ax2.set_xlabel("log Age (Gyr)")
    cbar2 = plt.colorbar(m, fraction=0.2, pad=0.01, aspect=9)
    cbar2.ax.set_title("fraction")
    ############################################################################
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.minorticks_on()
    m = ax3.pcolormesh(np.log10(templates.ages2D), templates.metals2D,
                       Wfit, vmax=vmax, vmin=0, cmap="cubehelix_r")
    ax3.set_ylabel("[Fe/H]")
    ax3.set_xlabel("log Age (Gyr)")
    cbar3 = plt.colorbar(m, fraction=0.2, pad=0.01, aspect=9)
    cbar3.ax.set_title("fraction")
    ax3.set_title("Recovered")
    ############################################################################
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.minorticks_on()
    ages = np.log10(templates.ages2D[0])
    dw = 0.025
    w1 = Wsim.sum(axis=0)
    w2 = Wfit.sum(axis=0)
    ax4.bar(ages- dw, w1, label="Model", width=0.05,
            alpha=0.8)
    ax4.bar(ages + dw, w2, label="Recovered", width=0.05, alpha=0.8)
    ax4.legend(prop={'size': 8})
    ax4.set_xlabel("log Age (Gyr)")
    ax4.set_ylabel("SFH")
    ############################################################################
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.minorticks_on()
    metal= templates.metals2D[:,0]
    w1 = Wsim.sum(axis=1)
    w2 = Wfit.sum(axis=1)
    dw = 0.05
    ax5.bar(metal - dw, w1, label="Model", alpha=0.8, width=0.1)
    ax5.bar(metal + dw, w2, label="Recovered".format(sn),
            alpha=0.8, width=0.1)
    ax5.legend(prop={'size': 8})
    ax5.set_xlabel("[Fe/H]")
    ############################################################################
    output = dbname.replace(".db", ".png")
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.08,
                        wspace=0.2, hspace=0.4)
    plt.savefig(output, dpi=250)
    plt.close(fig)
    ############################################################################