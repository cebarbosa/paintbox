import numpy as np
import astropy.units as u
import astropy.constants as const

__all__ = ["disp2vel", "logspace_dispersion"]

def disp2vel(wrange, velscale):
    """ Returns a log-rebinned wavelength dispersion with constant velocity.

    This code is an adaptation of pPXF's log_rebin routine, simplified to
    deal with the wavelength dispersion only.

    Parameters
    ----------
    wrange: list, np.array or astropy.Quantity
        Input wavelength dispersion range with two elements.

    velscale: float or astropy.Quantity
        Desired output velocity scale. Units are assumed to be km/s unless
        specified as an astropy.Quantity.

    """
    c = 299792.458  # Speed of light in km/s
    if isinstance(wrange, list):
        wrange = np.array(wrange)
    wunits = wrange.unit if hasattr(wrange, "unit") else 1
    if hasattr(velscale, "unit"):
        velscale = velscale.to(u.km/u.s).value
    veldiff = np.log(np.max(wrange) / np.min(wrange)) * c
    n = veldiff / velscale
    m = int(n)
    dv = 0.5 * (n-m) * velscale
    v = np.arange(0, m * velscale, velscale) + dv
    w = wrange[0] * np.exp(v / c)
    return w * wunits

def logspace_dispersion(interval, velscale, vunit=None):
    """ Returns a wavelength dispersion array with constant velocity scale .

    This code is an adaptation of pPXF's log_rebin routine, simplified to
    deal with the wavelength dispersion only, and dealing with arbitrary units
    using astropy.

    Parameters
    ----------
    interval: list, np.array or astropy.Quantity
        Input wavelength dispersion range with two elements.

    velscale: float or astropy.Quantity
        Desired output velocity scale. Units are assumed to be km/s unless
        specified as an astropy.Quantity.

    vunit: astroppy.Quantity
        Units of the velocity scale. Defaults to km/s.

    Returns
    -------

    np.array
        Wavelength array in interval [w1, w2),

    """
    vunit = u.km / u.s if vunit is None else vunit
    c = const.c.to(vunit).value
    if isinstance(interval, list):
        interval = np.array(interval)
    wunits = interval.unit if hasattr(interval, "unit") else 1
    if hasattr(velscale, "unit"):
        velscale = velscale.to(vunit).value
    veldiff = np.log(np.max(interval) / np.min(interval)) * c
    n = veldiff / velscale
    m = int(n)
    v = np.arange(0, m * velscale, velscale)
    w = interval[0] * np.exp(v / c)
    return w * wunits