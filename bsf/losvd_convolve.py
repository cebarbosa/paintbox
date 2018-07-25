# -*- coding: utf-8 -*-
""" 

Created on 26/03/18

Author : Carlos Eduardo Barbosa

LOSVD spectral convolution routines.
"""
from __future__ import print_function, division

import numpy as np
from scipy.ndimage import convolve1d

def losvd_convolve(spec, losvd, velscale):
    """ Apply LOSVD to a given spectra given that both wavelength and spec
     arrays are log-binned. Adapted from pPXF code. """
    # Convert to pixel scale
    pars = np.copy(losvd)
    pars[:2] /= velscale
    dx = int(np.ceil(np.max(abs(pars[0]) + 5*pars[1])))
    nl = 2*dx + 1
    x = np.linspace(-dx, dx, nl) # Evaluate the Gaussian using steps of
    # 1/factor pixel
    vel = pars[0]
    w = (x - vel)/(pars[1])
    w2 = w**2
    gauss = np.exp(-0.5*w2)
    profile = gauss/gauss.sum()
    # Hermite polynomials normalized as in Appendix A of van der Marel & Franx (1993).
    # Coefficients for h5, h6 are given e.g. in Appendix C of Cappellari et al. (2002)
    if losvd.size > 2:        # h_3 h_4
        poly = 1 + pars[2]/np.sqrt(3)*(w*(2*w2-3)) \
                 + pars[3]/np.sqrt(24)*(w2*(4*w2-12)+3)
        if len(losvd) == 6:  # h_5 h_6
            poly += pars[4]/np.sqrt(60)*(w*(w2*(4*w2-20)+15)) \
                  + pars[5]/np.sqrt(720)*(w2*(w2*(8*w2-60)+90)-15)
        profile *= poly
    profile /= profile.sum() # Normalization not used in pPXF!
    return convolve1d(spec, profile)

if __name__ == "__main__":
    pass