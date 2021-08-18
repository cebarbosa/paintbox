import os
import glob

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm
from spectres import spectres
from scipy.ndimage.filters import gaussian_filter1d
from paintbox.sed import ParametricModel, PaintboxBase

from .logspace_dispersion import logspace_dispersion

__all__ = ["MILES"]

class MILES(PaintboxBase):
    """ Utility class to prepare MILES SSP models.

    More info about MILES models can be found at their website [1]

    Parameters
    ----------
    spectral_range: str (default 'E")
        Define the spectral range (wavelength) used in the models. Options are:
            - E (EMILES):  1680-50000 Angstrom
            - M (MILES): 3540.5-7409.6 Angstrom
            - B (V99_Blue): 3855.50-4476.44 Angstrom
            - R (V99_Red): 4794.95-5465.09 Angstrom
            - C (CaT): 8348.85-8951.50 Angstrom

    imf_type: str (default "bi")
        Initial Mass Function (IMF) parametrization. Options are:
            - un (unimodal)
            - bi (bimodal)
            - ku (Kroupa universal)
            - kb (Kroupa revised)
            - ch (Chabrier)

    isochrones: str (default BasTI)
        Isochrones used for the SSP. Options are BasTI and Padova.




    [1]: http://research.iac.es/proyecto/miles/
    """
    def __init__(self, spectral_range=None, imf_type=None,
                 isochrones=None, wave=None,
                 libpath=None, resolution=None, store=None, norm=True):
        # Check spectral range
        self.spectral_range = "E" if spectral_range is None else spectral_range
        allowed_ranges = ["E", "M", "B", "R", "C"]
        msg = """Allowed spectral ranges are "E", "M", "B", "R" and "C". """
        assert self.spectral_range in allowed_ranges, msg
        # Check IMF type
        self.imf_type = "bi" if imf_type is None else imf_type
        msg = """ Allowed IMF types are "un", "bi", "ku", "kb" and "ch" """
        allowed_imfs = ["un", "bi", "ku", "kb", "ch"]
        assert self.imf_type in allowed_imfs, msg
        # Check isochones
        self.isochrones = "BasTI" if isochrones is None else isochrones
        msg = "Isochrones should be BasTI or Padova. "
        assert self.isochrones in ["BasTI", "Padova"], msg
        # Define allowed values for metallicity, ages and alpha-elements
        self._MHs_iso = {"Padova": [-2.32, -1.71, -1.31, -0.71, -0.40,
                                        0.00, 0.22],
                    "BasTI": [-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35,
                              -0.25, +0.06, 0.15, 0.26, 0.40]}
        self._ages_iso = {
                     "Padova":
                              [0.063, 0.071, 0.079, 0.089, 0.10, 0.11, 0.13,
                                0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32,
                                0.35, 0.40, 0.45, 0.50, 0.56, 0.63, 0.71, 0.79,
                                0.89, 1.00, 1.12, 1.26, 1.41, 1.58, 1.78, 2.00,
                                2.24, 2.51, 2.82, 3.16, 3.55, 3.98, 4.47, 5.01,
                                5.62, 6.31, 7.08, 7.94, 8.91, 10.00, 11.22,
                                12.59, 14.13, 15.85, 17.78],
                     "BasTI": [00.03, 00.04, 00.05, 00.06, 00.07, 00.08, 00.09,
                               00.10, 00.15, 00.20, 00.25, 00.30, 00.35, 00.40,
                               00.45, 00.50, 00.60, 00.70, 00.80, 00.90, 01.00,
                               01.25, 01.50, 01.75, 02.00, 02.25, 02.50, 02.75,
                               03.00, 03.25, 03.50, 03.75, 04.00, 04.50, 05.00,
                               05.50, 06.00, 06.50, 07.00, 07.50, 08.00, 08.50,
                               09.00, 09.50, 10.00, 10.50, 11.00, 11.50, 12.00,
                               12.50, 13.00, 13.50, 14.00]}
        self._alphaFes_iso = {"Padova": ["base"], "BasTI": ["base", 0, 0.4]}
        self.imf_slopes = {"un": [0.3, 0.5, 0.8, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5,
                                  2.8, 3.0, 3.3, 3.5],
                           "bi": [0.3, 0.5, 0.8, 1.3, 1.5, 1.8, 2.0, 2.3, 2.5,
                                  2.8, 3.0, 3.3, 3.5],
                           "ku": [1.3], "kb": [1.3], "ch": [1.3]}
        # self.resolution = resolution
        # self.store = store
        self.libpath = libpath
        # self.norm = norm
        # if os.path.isfile(self.store):
        #     self.load_templates()
        # else:
        #     self.set_wave(wave)
        #     self.process_ssps()
        #     if self.store is not None:
        #         self.save_templates()
        # self.build_model()

    @property
    def MHs(self):
        """ Defines allowed metallicities in the SSPs."""
        return self._MHs_iso[self.isochrones]

    @property
    def ages(self):
        """ Defines allowed ages in the SSPs. """
        return self._ages_iso[self.isochrones]

    @property
    def alphaFes(self):
        """ Defines allowed alpha/Fe in the SSPs."""
        return self._alphaFes_iso[self.isochrones]

    def get_filename(self, MH, age, alphaFe="base", imfslope=1.3):
        """ Retrieves the filename containing the spectrum of given ages,
        metallicities, alpha/Fe and IMF slope.

        Parameters
        ----------
        MH: float
            Metallicity of the SSP.
        age: float
            Age of the SSP.
        alphaFe: float or string
            Alpha-element abundances. Defaults to "base" models. For BasTI
            isochrones, models can have alpha element abundances of 0 and +0.4.
        imfslope: float
            Slope of the IMF. Defaults to 1.3.


        """
        msign = "p" if MH >= 0. else "m"
        azero = "0" if age < 10. else ""
        # filename = f"{self.spectral_range}{self.imf_type}{imfslope:.2f}Z{0
        # }{1:.2f}T{2}{3:02.4f}_iPp0.00_baseFe_linear""" \
        #            "_FWHM_2.51.fits".format(msign, abs(MH), azero, age)
        # return os.path.join(self.libpath, filename)




