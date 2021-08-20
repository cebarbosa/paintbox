import os
import glob

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm
from spectres import spectres
from scipy.ndimage.filters import gaussian_filter1d

from paintbox.sed import PaintboxBase, ParametricModel
from paintbox.logspace_dispersion import disp2vel,  logspace_dispersion

__all__ = ["CvD18", "MILES"]

np.seterr(divide='ignore', invalid='ignore')

class CvD18(PaintboxBase):
    """ Utility class for the use of CvD models.

    This class provides a convenience `paintbox` interface to CvD models,
    producing a callable object that can be used together with other classes
    to produce more detailed SED models.

    The CvD model files require pre-processing for use with `paintbox`,
    which may require an initial overhead of ~one minute, but the processed
    files can be reused for several files / runs if necessary it models are
    stored in disk.

    Parameters
    ----------
    wave: ndarray, astropy.units.Quantity
        Wavelength array of the model.

    libpath: str
        Path to the locations where CvD18 models are stored.

    sigma: float
        Velocity dispersion of the output in km/s. Default is 100 km/s
        (the minimum velocity dispersion allowed).

    store: str
        Directory where processed models are stored for reuse. If models
        are not found within store, models in the libpath are
        processed for use, and stored in this directory.

    elements: list
        Chemical abundances to be included in the model. Available elements are
        'Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe', 'K', 'Mg', 'Mn', 'N',
        'Na', 'Ni', 'Si', 'Sr', 'T', 'Ti', 'V', 'a/Fe', 'as/Fe'. Default is
        to use all elements except T (temperature of hot star) and 'a/Fe'.

    norm: bool
        Normalize the flux of the processed models to the median. Default is
        True.

    """
    def __init__(self, wave=None, libpath=None, sigma=100, store=None,
                 elements=None, norm=True):
        self.sigma = sigma
        self.store = store
        self.elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K",
                         "Cr", "Mn", "Ba", "Ni", "Co", "Eu", "Sr", "V", "Cu",
                         "a/Fe"] if elements is None else elements
        self.libpath = libpath
        self._all_elements = ['Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe',
                              'K', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'Si', 'Sr',
                              'T', 'Ti', 'V', 'a/Fe', 'as/Fe']
        self.norm = norm
        if os.path.isfile(self.store):
            self.load_templates()
        else:
            self.set_wave(wave)
            self.process_ssps()
            self.process_respfun()
            if self.store is not None:
                self.save_templates()
        self.build_model()

    def set_wave(self, wave):
        if wave is None:
            ssp_files = glob.glob(os.path.join(self.libpath, "VCJ*.s100"))
            wave = np.loadtxt(ssp_files[0], usecols=(0,))
        if hasattr(wave, "unit"):
            self.wave = wave.to(u.Angstrom).value
        else:
            self.wave = wave #Assumes units are Angstrom
        assert self.wave.min() >= 3501, "Minimum wavelength is 3501 Angstrom"
        assert self.wave.max() <= 25000, "Maximum wavelength is 25000 Angstrom"
        assert self.sigma >= 100, "Minumum velocity dispersion for models is " \
                                   "100 km/s."
        return

    def build_model(self):
        """ Build model with paintbox SED methods. """
        ssp = ParametricModel(self.wave, self.params, self.templates)
        self._limits = {}
        for p in self.params.colnames:
            vmin = self.params[p].data.min()
            vmax = self.params[p].data.max()
            self._limits[p] = (vmin, vmax)

        self._response_functions = {}
        for element in self.elements:
            rf = ParametricModel(self.wave, self.rfpars[element], self.rfs[
                element])
            self.response_functions[element] = rf
            ssp = ssp * rf
            vmin = rf.params[element].data.min()
            vmax = rf.params[element].data.max()
            self._limits[element] = (vmin, vmax)
        if len(self.elements) > 0: # Update limits in case response functions
            # are used.
            for p in ["Age", "Z"]:
                vmin = rf.params[p].data.min()
                vmax = rf.params[p].data.max()
                self._limits[p] = (vmin, vmax)
        self._interpolator = ssp.constrain_duplicates()
        self._parnames = self._interpolator.parnames
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Returns a model for a given set of parameters theta. """
        return self._interpolator(theta)

    def process_ssps(self):
        """ Process SSP models. """
        ssp_files = glob.glob(os.path.join(self.libpath, "VCJ*.s100"))
        if len(ssp_files) == 0:
            raise ValueError(f"Stellar populations not found in libpath: "
                             f"{self.libpath}")
        nimf = 16
        imfs = 0.5 + np.arange(nimf) / 5
        x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
        velscale = int(self.sigma / 4)
        kernel_sigma = np.sqrt(self.sigma ** 2 - 100 ** 2) / velscale
        ssps, params = [], []
        for fname in tqdm(ssp_files, desc="Processing SSP files"):
            spec = os.path.split(fname)[1]
            T = float(spec.split("_")[3][1:])
            Z = float(spec.split("_")[4][1:-8].replace("p", "+").replace(
                        "m", "-"))
            for i, (x1, x2) in enumerate(zip(x1s, x2s)):
                params.append(Table([[Z], [T], [x1], [x2]],
                                    names=["Z", "Age", "x1", "x2"]))
            data = np.loadtxt(fname)
            w = data[:,0]
            if self.sigma > 100:
                wvel = disp2vel(w, velscale)
            ssp = data[:, 1:].T
            if self.sigma <= 100:
                newssp = spectres(self.wave, w, ssp, fill=0, verbose=False)
            else:
                ssp_rebin = spectres(wvel, w, ssp, fill=0, verbose=False)
                ssp_broad = gaussian_filter1d(ssp_rebin, kernel_sigma,
                                              mode="constant", cval=0.0)
                newssp = spectres(self.wave, wvel, ssp_broad, fill=0,
                                  verbose=False)

            ssps.append(newssp)
        self.params = vstack(params)
        self.templates = np.vstack(ssps)
        self.fluxnorm = np.median(self.templates, axis=1) if self.norm else 1.
        self.templates /= self.fluxnorm[:, np.newaxis]
        return

    def process_respfun(self):
        """ Prepare response functions from CvD models. """
        self.rf_infiles = glob.glob(os.path.join(self.libpath,
                                                     "atlas_ssp*.s100"))
        # Read one spectrum to get name of columns
        with open(self.rf_infiles[0]) as f:
            header = f.readline().replace("#", "")
        fields = [_.strip() for _ in header.split(",")]
        fields[fields.index("C+")] = "C+0.15"
        fields[fields.index("C-")] = "C-0.15"
        fields[fields.index("T+")] = "T+50"
        fields[fields.index("T-")] = "T-50"
        fields = ["{}0.3".format(_) if _.endswith("+") else _ for _ in fields ]
        fields = ["{}0.3".format(_) if _.endswith("-") else _ for _ in fields]
        elements = set([_.split("+")[0].split("-")[0] for _ in fields if
                        any(c in _ for c in ["+", "-"])])
        signal = ["+", "-"]
        velscale = int(self.sigma / 4)
        kernel_sigma = np.sqrt(self.sigma**2 - 100**2) / velscale
        rfsout = dict([(element, []) for element in elements])
        parsout = dict([(element, []) for element in elements])
        desc = "Preparing response functions"
        for i, fname in enumerate(tqdm(self.rf_infiles, desc=desc)):
            spec = os.path.split(fname)[1]
            T = float(spec.split("_")[2][1:])
            Z = float(spec.split("_")[3].split(".abun")[0][1:].replace(
                "p", "+").replace("m", "-"))
            data = np.loadtxt(fname)
            w = data[:, 0]
            data = data.T
            if self.sigma > 100:
                wvel = disp2vel(w, velscale)
                rebin = spectres(wvel, w, data, fill=0, verbose=False)
                broad = gaussian_filter1d(rebin, kernel_sigma,
                                          mode="constant", cval=0.0)
                data = spectres(self.wave, wvel, broad, fill=0, verbose=False).T

            else:
                data = spectres(self.wave, w, data, fill=0, verbose=False).T
            fsun = data[:, 1]
            for element in elements:
                # Adding solar response
                p = Table([[Z], [T], [0.]], names=["Z", "Age", element])
                rfsout[element].append(np.ones(len(self.wave)))
                parsout[element].append(p)
                # Adding non-solar responses
                for sign in signal:
                    name = "{}{}".format(element, sign)
                    cols = [(i,f) for i, f in enumerate(fields) if
                            f.startswith(name)]
                    for i, col in cols:
                        val = float("{}1".format(sign)) * \
                              float(col.split(sign)[1])
                        t = Table([[Z], [T], [val]],
                                  names=["Z", "Age", element])
                        parsout[element].append(t)
                        rf = data[:, i] / fsun
                        rfsout[element].append(rf)
        self.rfs = dict([(e, np.array(rfsout[e])) for e in elements])
        self.rfpars = dict([(e, vstack(parsout[e])) for e in elements])
        return

    def save_templates(self):
        hdu0 = fits.PrimaryHDU()
        hdu1 = fits.BinTableHDU(Table([self.wave], names=["wave"]))
        hdu1.header["EXTNAME"] = "WAVE"
        hdu2 = fits.ImageHDU(self.templates * self.fluxnorm[:, None])
        hdu2.header["EXTNAME"] = "DATA.SSPS"
        params = Table(self.params)
        hdu3 = fits.BinTableHDU(params)
        hdu3.header["EXTNAME"] = "PARS.SSPS"
        hdulist = fits.HDUList([hdu0, hdu1, hdu2, hdu3])
        for element in self._all_elements:
            hdudata = fits.ImageHDU(self.rfs[element])
            hdudata.header["EXTNAME"] = f"DATA.{element}"
            hdulist.append(hdudata)
            hdutable = fits.BinTableHDU(self.rfpars[element])
            hdutable.header["EXTNAME"] = f"PARS.{element}"
            hdulist.append(hdutable)
        hdulist.writeto(self.store, overwrite=True)

    def load_templates(self):
        hdulist = fits.open(self.store)
        nhdus = len(hdulist)
        hdunum = np.arange(1, nhdus)
        hdunames = [hdulist[i].header["EXTNAME"] for i in hdunum]
        hdudict = dict(zip(hdunames, hdunum))
        self.wave = Table.read(self.store, hdu=hdudict["WAVE"])["wave"].data
        self.params = Table.read(self.store, hdu=hdudict["PARS.SSPS"])
        self.templates = hdulist[hdudict["DATA.SSPS"]].data

        self.rfs = {}
        self.rfpars = {}
        for e in self._all_elements:
            self.rfs[e] = hdulist[hdudict[f"DATA.{e}"]].data
            self.rfpars[e] = Table.read(self.store, hdu=hdudict[f"PARS.{e}"])
        self.fluxnorm = np.median(self.templates, axis=1) if self.norm else 1.
        self.templates /= self.fluxnorm[:, np.newaxis]
        return

    @property
    def limits(self):
        """ Lower and upper limits of the model parameters. """
        return self._limits


    @property
    def response_functions(self):
        """ Dictionary with all response functions in model. """
        return self._response_functions

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
