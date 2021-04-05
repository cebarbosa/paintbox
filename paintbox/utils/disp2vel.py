import numpy as np
import astropy.units as u


def disp2vel(wave, velscale):
    """ Returns a log-rebinned wavelength dispersion with constant velocity.

    This code is an adaptation of pPXF's log_rebin routine, simplified to
    deal with the wavelength dispersion only.

    Parameters
    ----------
    wave: np.array or astropy.Quantity
        Input wavelength dispersion.

    velscale: float or astropy.Quantity
        Desired output velocity scale. Units are assumed to be km/s unless
        specified as an astropy.Quantity.

    """
    c = 299792.458  # Speed of light in km/s
    if hasattr(wave, "unit"):
        wunits = wave.unit
        lamRange = np.array([wave[0].value, wave[-1].value])
    else:
        wunits = 1
        lamRange = np.array([wave[0], wave[-1]])
    if hasattr(velscale, "unit"):
        velscale = velscale.to(u.km/u.s).value
    n = wave.shape[0]
    dLam = np.diff(lamRange) / (n - 1.)
    lim = lamRange / dLam + [+0.5, -0.5]  # Trimming wavelength
    logLim = np.log(lim)
    logScale = velscale / c
    m = int(np.diff(logLim) / logScale)  # Number of output pixels
    logLim[1] = logLim[0] + m * logScale
    newBorders = np.exp(np.linspace(*logLim, num=m + 1))  # Logarithmically
    logLam = np.log(np.sqrt(newBorders[1:] * newBorders[:-1]) * dLam)
    return np.exp(logLam) * wunits