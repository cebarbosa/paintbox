"""

Created on 05/03/18

Author : Carlos Eduardo Barbosa

Determination of LOSVD of a given spectrum similarly to pPXF.

"""
from __future__ import print_function, division, absolute_import

import os
import pickle

import numpy as np
import pymc3 as pm
import theano.tensor as T
from scipy import stats
from specutils.io.read_fits import read_fits_spectrum1d
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
import ppxf.ppxf_util as util

from simulations import context

class Templates():
    def __init__(self, velscale, sigma, _noread=False):

        """ Load Miles templates for simulations. """
        self.velscale = velscale
        self.sigma = sigma
        miles_path = os.path.join(context.basedir, "miles_models")
        # Search for spectra and their properties
        fitsfiles = [_ for _ in os.listdir(miles_path) if _.endswith(".fits")]
        # Define the different values of metallicities and ages of templates
        self.metals = np.unique([float(_.split("Z")[1].split("T")[0].replace(
                                 "m","-").replace("p","+")) for _ in fitsfiles])
        self.ages = np.unique([float(_.split("T")[1].split("_iP")[0].replace(
                               "m", "-").replace("p","+")) for _ in fitsfiles])
        # Defining arrays
        self.ages2D, self.metals2D  = np.meshgrid(self.ages, self.metals)
        self.metals1D = self.metals2D.reshape(-1)
        self.ages1D = self.ages2D.reshape(-1)
        if _noread:
            return
        templates, norms= [], []
        for metal, age in zip(self.metals1D, self.ages1D):
            template_file = os.path.join(miles_path, self.miles_filename(
                                         metal, age))
            spec = read_fits_spectrum1d(template_file)
            wave = spec.dispersion
            flux = spec.flux
            speclog, logwave, _ = util.log_rebin([wave[0], wave[-1]], flux,
                                                velscale=self.velscale)
            speclog = gaussian_filter1d(speclog, sigma / velscale)
            wave = wave[1:-2]
            flux = spectres(wave, np.exp(logwave), speclog)
            norm = np.sum(flux)
            templates.append(flux / norm)
        self.templates = np.array(templates)
        self.norms = np.array(norms)
        self.wave = wave
        return

    def miles_filename(self, metal, age):
        """ Returns the name of files for the MILES library. """
        msign = "m" if metal < 0 else "p"
        mstr = "{1}{0:04.2f}".format(abs(metal), msign)
        astr = "{:07.4f}".format(age)
        return "Mun1.30Z{}T{}_iPp0.00_baseFe_linear_FWHM_2.51" \
                ".fits".format(mstr, astr)

def simulate_unimodal_distribution(templates, logdir, redo=False, nsim=100):
    """ Produces a CSP spectram using SSPs. """
    # Producing a CSP spectrum
    for sim in np.arange(nsim):
        logfile = os.path.join(logdir, "sim{:04d}.pkl".format(sim+1))
        if os.path.exists(logfile) and not redo:
            continue
        # Old population
        t = np.random.uniform(templates.ages.min(), templates.ages.max())
        tau = np.random.exponential(2)
        Z = np.random.uniform(templates.metals.min(),
                                      templates.metals.max())
        Z_std = np.random.exponential(0.50)
        sfh = stats.norm(loc=t, scale=tau).pdf(templates.ages2D)
        metal_dist = stats.norm(loc=Z, scale=Z_std).pdf(templates.metals2D)
        pop =  sfh * metal_dist
        weights = pop / pop.sum()
        spec = weights.reshape(-1).dot(templates.templates)
        log = {"t" : t, "tau" : tau, "metal" : Z,
               "metal_std" : Z_std, "sfh" : sfh,
               "metal_dist" : metal_dist, "weights" : weights,
               "spec" : spec}
        with open(logfile, "wb") as f:
            pickle.dump(log, f)
    return

def simulate_bimodal_csps(templates, logdir, redo=False, nsim=100):
    """ Produces a CSP spectram using SSPs. """
    # Producing a CSP spectrum
    for sim in np.arange(nsim):
        logfile = os.path.join(logdir, "sim{:04d}.pkl".format(sim+1))
        if os.path.exists(logfile) and not redo:
            continue
        # Old population
        t_old = np.random.uniform(8, 14)
        tau_old = np.random.exponential(3)
        metal_old = np.random.uniform(-1.7, 0.22)
        std_metal_old = np.random.exponential(0.30),
        # sfh1 = delayed_tau(t_old, tau_old, templates.ages2D)
        sfh1 = stats.norm(loc=t_old, scale=tau_old).pdf(
                          templates.ages2D)
        metal_dist = stats.norm(loc=metal_old, scale=std_metal_old).pdf(
                                templates.metals2D)
        pop1 =  sfh1 * metal_dist
        pop1 /= pop1.sum()
        # Young population
        t_young = np.random.uniform(0.1, 8)
        tau_young = np.random.exponential(0.5)
        metal_young = np.random.uniform(-1.7, 0.22)
        std_metal_young = np.random.exponential(0.2)
        metal_dist2 = stats.norm(loc=metal_young, scale=std_metal_young).pdf(
                                 templates.metals2D)
        # sfh2 = delayed_tau(t_young, tau_young, templates.ages2D)
        sfh2 = stats.norm(loc=t_young, scale=tau_young).pdf(templates.ages2D)
        pop2 = sfh2 * metal_dist2
        pop2 /= pop2.sum()
        # Weights of the populations
        w1 = np.random.uniform(0.5, 1)
        w2 = 1 - w1
        # Mixture populations
        w = w1 * pop1 + w2 * pop2
        model = w.reshape(-1).dot(templates.templates)
        log = {"t_old" : t_old, "tau_old" : tau_old, "metal_old" : metal_old,
               "metal_old_std" : std_metal_old, "sfh_old" : sfh1,
               "metal_dist_old" : metal_dist, "spec_old" : pop1,
               "t_young" : t_young, "tau_young" : tau_young,
               "metal_young" : metal_young, "metal_young_std" :
                   std_metal_young, "sfh_old" : sfh2,
               "metal_dist_young" : metal_dist2 , "spec_young" : pop2,
               "w_old" : w1, "w_young": w2, "sim_spec" : model}
        with open(logfile, "wb") as f:
            pickle.dump(log, f)
    return

def csp_modeling2(obs, templates, dbname, redo=False):
    """ Model a CSP with bayesian model. """
    if os.path.exists(dbname) and not redo:
        return dbname
    with pm.Model() as model:
        muZ = pm.Uniform("muZ", lower=templates.metals.min(),
                         upper=templates.metals.max())
        muT = pm.Uniform("muT", lower=templates.ages.min(),
                         upper=templates.ages.max())
        sigmaZ = pm.Uniform("sigmaZ", lower=0.05, upper=1)
        sigmaT = pm.Uniform("sigmaT", lower=0.1, upper=10)
        w = pm.Deterministic("w", T.pow(2  * np.pi * sigmaT * sigmaZ , -1) * \
                  T.exp(-0.5 * T.pow((muZ - templates.metals1D)/ sigmaZ, 2)) * \
                  T.exp(-0.5 * T.pow((muT - templates.ages1D)/ sigmaT, 2)))
        bestfit = pm.math.dot(w.T, templates.templates)
        sigma = pm.Exponential("sigma", lam=1)
        pm.Normal('like', mu=bestfit, sd = sigma, observed=obs)
    with model:
        trace = pm.sample(1000, tune=1000)
    results = {'model': model, "trace": trace}
    with open(dbname, 'wb') as buff:
        pickle.dump(results, buff)
    return

def make_unimodal_simulations():
    """ Produces the simulations for unimodal distributions of ages and
    metallicites. """
    sigma = 300
    velscale = sigma / 10
    nsim = 100
    logdir = os.path.join(context.workdir, "simulations",
                          "unimodal_sigma{}".format(sigma), "data")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    templates_file = os.path.join(logdir, "templates.pkl")
    if os.path.exists(templates_file):
        with open(templates_file, "rb") as f:
            templates = pickle.load(f)
    else:
        templates = Templates(velscale=velscale, sigma=sigma, _noread=False)
        with open(templates_file, "wb") as f:
            pickle.dump(templates, f)
    simulate_unimodal_distribution(templates, logdir, redo=False, nsim=nsim)

if __name__ == "__main__":
    make_unimodal_simulations()

