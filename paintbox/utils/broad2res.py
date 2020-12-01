# -*- coding: utf-8 -*-
"""

Created on 04/05/16

@author: Carlos Eduardo Barbosa

Calculates and plot the spectral resolution of MUSE.

"""
from __future__ import print_function

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def broad2res(w, flux, obsres, outres, fluxerr=None):
    """ Broad resolution of observed spectra to a given resolution.

    Input Parameters
    ----------------
    w : np.array
        Wavelength array

    flux: np.array
        Spectrum to be broadened to the desired resolution.

    obsres : float or np.array
        Observed wavelength spectral resolution FWHM.

    outres: float
        Resolution FWHM  of the spectra after the broadening.

    fluxerr: np.array
        Spectrum errors whose uncertainties are propagated

    Output parameters
    -----------------
    np.array:
        Broadened spectra.

    """
    dws = np.diff(w)
    dw = np.median(dws)
    assert np.all(np.isclose(dws, dw)), \
        "Wavelength dispersion has to be constant!"
    sigma_diff = np.sqrt(outres ** 2 - obsres ** 2) / 2.3548 / dw
    diag = np.diag(flux)
    for j in range(len(w)):
        diag[j] = gaussian_filter1d(diag[j], sigma_diff[j], mode="constant",
                                 cval=0.0)
    newflux = diag.sum(axis=0)
    if fluxerr is None:
        return newflux
    errdiag = np.diag(fluxerr)
    for j in range(len(w)):
        errdiag[j] = gaussian_filter1d(errdiag[j]**2, sigma_diff[j],
                                       mode="constant", cval=0.0)
    newfluxerr = np.sqrt(errdiag.sum(axis=0))
    return newflux, newfluxerr