# -*- coding: utf-8 -*-
"""

Created on 16/05/16

@author: Carlos Eduardo Barbosa

Program to calculate lick indices

"""

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d

class Lick():
    """ Class to measure Lick indices.

    Computation of the Lick indices in a given spectrum. Position of the
    passbands are determined by redshifting the position of the bands
    to the systemic velocity of the galaxy spectrum.

    =================
    Input parameters:
    =================
        wave (array):
            Wavelength of the spectrum given.

        galaxy (array):
            Galaxy spectrum in arbitrary units.

        bands0 (array) :
            Definition of passbands for Lick indices at rest
            wavelengths. Units should be consistent with wavelength array.

        vel (float, optional):
            Systemic velocity of the spectrum in km/s. Defaults to zero.

        dw (float, optinal):
            Extra wavelength to be considered besides
            bands for interpolation. Defaults to 2 wavelength units.

    ===========
    Attributes:
    ===========
        bands (array):
            Wavelengths of the bands after shifting to the
            systemic velocity of the galaxy.

    """
    def __init__(self, wave, galaxy, bands0, vel=0, dw=2.):
        self.galaxy = galaxy
        self.wave = wave
        self.vel = vel
        self.bands0 = bands0
        self.dw = dw
        c = 299792.458 # Speed of light in km/s
        self.bands = self.bands0 * np.sqrt((1 + vel/c)/(1 - vel/c))


    def classic_integration(self):
        """ Calculation of Lick indices using spline integration.

        ===========
        Attributes:
        ===========
            R (array):
                Raw integration values for the Lick indices.

            Ia (array):
                Indices measured in equivalent widths.

            Im (array):
                Indices measured in magnitudes.

            classic (array):
                Indices measured according to the conventional
                units mixturing equivalent widths and magnitudes.
        """
        self.R = np.zeros(self.bands.shape[0])
        self.Ia = np.zeros_like(self.R)
        self.Im = np.zeros_like(self.R)
        for i, w in enumerate(self.bands):
            if (w[0] - self.dw < self.wave[0]) or \
               (w[-1] + self.dw > self.wave[-1]):
                self.R[i] = np.nan
            # Defining indices for each section
            idxb = np.where(((self.wave > w[0] - self.dw) &
                                 (self.wave < w[1] + self.dw)))
            idxr = np.where(((self.wave > w[4] - self.dw) &
                                (self.wave < w[5] + self.dw)))
            idxcen = np.where(((self.wave > w[2] - self.dw) &
                                (self.wave < w[3] + self.dw)))
            # Defining wavelenght samples
            wb = self.wave[idxb]
            wr = self.wave[idxr]
            wcen = self.wave[idxcen]
            # Defining intensity samples
            fb = self.galaxy[idxb]
            fr = self.galaxy[idxr]
            fcen = self.galaxy[idxcen]
            # Interpolation functions for pseudocontinuum
            sb = InterpolatedUnivariateSpline(wb, fb)
            sr = InterpolatedUnivariateSpline(wr, fr)
            # Calculating the mean fluxes for the pseudocontinuum
            fp1 = sb.integral(w[0], w[1]) / (w[1] - w[0])
            fp2 = sr.integral(w[4], w[5]) / (w[5] - w[4])
            # Making pseudocontinuum vector
            x1 = (w[0] + w[1])/2.
            x2 = (w[4] + w[5])/2.
            fc = fp1 + (fp2 - fp1)/ (x2 - x1) * (wcen - x1)
            # Calculating indices
            ffc = InterpolatedUnivariateSpline(wcen, fcen/fc/(w[3]-w[2]))
            self.R[i] =  ffc.integral(w[2], w[3])
            self.Ia[i] = (1 - self.R[i]) * (w[3]-w[2])
            self.Im[i] = -2.5 * np.log10(self.R[i])
        self.classic = np.copy(self.Ia)
        idx = np.array([2,3,14,15,23,24])
        self.classic[idx] = self.Im[idx]
        return

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

def bands_shift(bands, vel):
    c = 299792.458  # Speed of light in km/s
    return  bands * np.sqrt((1 + vel/c)/(1 - vel/c))

if __name__ == "__main__":
    pass
