""" Calculates the M/L for CvD models. """
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

from .dispersion_constant_velscale import disp2vel

__all__ = ["CvD18"]

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