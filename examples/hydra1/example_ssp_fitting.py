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
import theano
import theano.tensor as T
from astropy import constants
import matplotlib.pyplot as plt
from specutils.io.read_fits import read_fits_spectrum1d
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.filters import convolve1d

from miles_util import Miles
from ppxf_util import log_rebin, gaussian_filter1d
from der_snr import DER_SNR



class MilesInterp():
    """ Produces interpolated model using the pPXF interface for the
        MILES data. """
    def __init__(self, velscale):
        filenames = os.path.join(os.getcwd(), "miles_models", 'Mun1.30*.fits')
        miles = Miles(filenames, velscale, 2.51)
        dim = miles.age_grid.shape
        grid = np.array(np.meshgrid(np.arange(dim[0]),
                                    np.arange(dim[1]))).T.reshape(-1, 2)
        self.pars = np.zeros((np.prod(dim), 2))
        self.templates = np.zeros((len(self.pars), len(miles.templates)))
        for i, (i1, i2) in enumerate(grid):
            self.pars[i] = [np.log10(miles.age_grid[i1, i2]), miles.metal_grid[
                i1, i2]]
            self.templates[i] = miles.templates[:, i1, i2]
        self.model = LinearNDInterpolator(self.pars, self.templates)
        self.ranges = np.array([self.pars.min(axis=0), self.pars.max(axis=0)]).T

    def __call__(self, *args):
        return self.model(*args)

class MilesCathegorical():
    """ Produces interpolated model using the pPXF interface for the
        MILES data. """
    def __init__(self, velscale):
        filenames = os.path.join(os.getcwd(), "miles_models", 'Mun1.30*.fits')
        miles = Miles(filenames, velscale, 2.51)
        dim = miles.age_grid.shape
        grid = np.array(np.meshgrid(np.arange(dim[0]),
                                    np.arange(dim[1]))).T.reshape(-1, 2)
        self.pars = np.zeros((np.prod(dim), 2))
        self.templates = np.zeros((len(self.pars), len(miles.templates)))
        for i, (i1, i2) in enumerate(grid):
            self.pars[i] = [miles.age_grid[i1, i2], miles.metal_grid[
                i1, i2]]
            self.templates[i] = miles.templates[:, i1, i2]
        self.ranges = np.array([self.pars.min(axis=0), self.pars.max(axis=0)]).T


    def __call__(self, args):
        return self.templates[args]

def single_stellar_population(spec, velscale):
    """ Model stellar populations. """
    miles = MilesInterp(velscale)

    @theano.compile.ops.as_op(itypes=[T.dscalar, T.dscalar], otypes=[T.dvector])
    def ssp(age, metal):
        return miles(age, metal)
    spec = miles(0.0,0.0)
    obs = spec
    with pm.Model() as model:
        age = pm.Uniform("age", lower=miles.ranges[0,0],
                         upper=miles.ranges[0,1])
        metal = pm.Uniform("metal", lower=miles.ranges[1,0],
                         upper=miles.ranges[1,1])
        beta = pm.HalfCauchy("beta", beta=1)
        like = pm.Normal("like", mu=ssp(age, metal), sd=beta,
                         observed=obs)
    with model:
        trace = pm.sample(10, tune=5, step=pm.Slice())
    pm.summary(trace)
    plt.plot(spec, "-")
    plt.plot(miles(trace["age"].mean(), trace["metal"].mean()), "-")
    plt.plot(miles(np.mean(trace["age"]), np.mean(miles(trace["metal"]))))
    plt.show()
    print(map)
    pm.traceplot(trace)
    plt.show()

def composite_stellar_population(spec, velscale, K=2):
    """ Simplified version of pPXF using Bayesian approach.

     TODO: This is just a prototype for the function, requires more work.
     """
    miles = MilesCathegorical(velscale)
    matrix = T.as_tensor_variable(miles.templates)
    alpha = np.ones(len(templates)) / len(templates)
    fakeobs = 0.6 * templates[10] + 0.4 * templates[100]

    with pm.Model() as model:
        idx = pm.Categorical("idx", alpha, shape=K)
        w = pm.Dirichlet("w", np.ones(K))
        y = T.dot(w, matrix[idx])
        y = pm.Deterministic("y", y)
        like = pm.Normal("like", mu=y, sd=1., observed=fakeobs)
    with model:
        trace = pm.sample(1000)
    pm.traceplot(trace)
    plt.show()


def example_ssps():
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
    mcmcmodel = composite_stellar_population(spec, velscale)
    # noise = np.median(spec.flux) / snr * np.ones_like(galaxy)
    # c = constants.c.to("km/s").value
    # dv = np.log(np.exp(miles.log_lam_temp[0]) / lamrange[0]) * c
    # start = [4000, 300]
    # goodpixels = np.argwhere(np.logical_and(np.exp(wave) > 4800, np.exp(wave) <
    #                                     5800 )).T[0]
    # pp = ppxf(stars_templates, galaxy, noise, velscale, start,
    #           plot=True, moments=4, degree=8, mdegree=-1, vsyst=dv,
    #           lam=np.exp(wave), clean=False, goodpixels=goodpixels)
    # plt.show()
    # print(pp)

if __name__ == "__main__":
    example_ssps()