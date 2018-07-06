# -*- coding: utf-8 -*-
"""

Created on 30/10/2017

@Author: Carlos Eduardo Barbosa

Convolution routines for stellar populations analysis.

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

def broad2lick(wl, intens, obsres, vel=0):
    """ Convolve spectra to match the Lick/IDS instrumental  resolution.

    Broad a given spectra to the Lick/IDS system resolution. The resolution
    of the IDS varies as function of the wavelength, and we use the mean
    interpolated values from the appendix of Worthey and Ottaviani 1997.

    ================
    Input parameters
    ================
    wl (array) :
        Wavelenght 1-D array in Angstroms.

    intens (array):
        Intensity 1-D array of Intensity, in arbitrary units. The
        lenght has to be the same as wl.

    obsres (float or array):
        Value of the observed resolution Full Width at Half Maximum (FWHM) in
        Angstroms.

    vel: float
        Recession velocity of the measured spectrum.

    =================
    Output parameters
    =================
    array_like
        The convolved intensity 1-D array.

    """
    c = 299792.458  # Speed of light in km/s
    dw = wl[1] - wl[0]
    if not isinstance(obsres, np.ndarray):
        obsres = np.ones_like(wl) * obsres
    wlick = np.array([2000., 4000., 4400., 4900., 5400., 6000., 8000.]) * \
            np.sqrt((1 + vel/c)/(1 - vel/c))
    lickres = np.array([11.5, 11.5, 9.2, 8.4, 8.4, 9.8, 9.8])
    flick = interp1d(wlick, lickres, kind="linear", bounds_error=False,
                         fill_value="extrapolate")
    fwhm_lick = flick(wl)
    fwhm_broad = np.sqrt(fwhm_lick**2 - obsres**2)
    sigma_b = fwhm_broad/ 2.3548 / dw
    intens2D = np.diag(intens)
    for i in range(len(sigma_b)):
        intens2D[i] = gaussian_filter1d(intens2D[i], sigma_b[i],
                      mode="constant", cval=0.0)
    return intens2D.sum(axis=0)

def broad2res(wave, intens, inres, outres):
    """ Convolve input spectra to a given resolution.

    Warning: This routine does not handle units, please keep consistency.

    Input parameters
    ----------------
    wave : np.array
        Linear wavelenght 1-D array in Angstroms.

    intens : np.array
        Intensity 1-D array of Intensity, in arbitrary units. The
        lenght has to be the same as wl.

    inres : float
        Input resolution (FWHM) of the spectrum

    outres: float
        Recession velocity of the measured spectrum.

    Output parameters
    -----------------
    array_like
        The convolved intensity 1-D array.

    """
    fwhm_diff = np.sqrt(outres**2 - inres**2)
    dw = wave[1] - wave[0]
    dsigma = fwhm_diff / 2.3548 / dw
    intens_broad = gaussian_filter1d(intens, dsigma,
                      mode="constant", cval=0.0)
    return intens_broad

if __name__ == "__main__":
    pass
