# -*- coding: utf-8 -*-
""" 

Created on 23/05/18

Author : Carlos Eduardo Barbosa

Make simulations for the case of composite stellar populations with a
log-normal star formation and different metallicity.

"""
from __future__ import print_function, division

import os
import pickle

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from scipy.ndimage.filters import gaussian_filter1d
from spectres import spectres
import matplotlib.pyplot as plt

from misc import array_from_header
from models.basket import tmcsp

class Miles():
    """ Read templates files and prepare them for the fitting. """
    def __init__(self, sigma=300):
        # Defining the ranges of the ages and metallicities
        self.ages_str = np.array(["00.0631", "00.0794", "00.1000", "00.1259",
                              "00.1585",
                         "00.1995", "00.2512", "00.3162", "00.3981", "00.5012",
                         "00.6310", "00.7943", "01.0000", "01.2589", "01.5849",
                         "01.9953", "02.5119", "03.1623", "03.9811", "05.0119",
                         "06.3096", "07.9433", "10.0000", "12.5893", "15.8489"])
        self.metals_str = np.array(["-1.71", "-1.31", "-0.71", "-0.40", "+0.00",
                               "+0.22"])
        self.ages2D_str, self.metals2D_str = np.meshgrid(self.ages_str,
                                                         self.metals_str)
        print(self.ages2D_str)
        raw_input()

        self.ages = self.ages_str.astype(np.float)
        self.metals = self.metals_str.astype(np.float)

        self.filenames = []
        for age in self.ages_str:
            for metal in self.metals_str:
                filename = os.path.join(os.getcwd(), "miles_models",
                                        miles_filename(metal,age))
                self.filenames.append(filename)



def miles_filename(metal, age):
    """ Returns the name of files for the MILES library. """
    m = metal.replace("+", "p").replace("-", "m")
    return "Mun1.30Z{}T{}_iPp0.00_baseFe_linear_FWHM_2.51" \
            ".fits".format(m, age)

if __name__ == "__main__":
    sigma = 300
    miles = Miles(sigma)
    print(miles.metals)


