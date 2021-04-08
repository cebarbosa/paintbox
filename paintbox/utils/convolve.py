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

if __name__ == "__main__":
    pass
