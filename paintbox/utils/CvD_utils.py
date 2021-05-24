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
from paintbox.sed import ParametricModel, CompositeSED, PaintboxBase
from paintbox.operators import Constrain

from .disp2vel import disp2vel

__all__ = ["CvD18"]

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
        Wavelenght array of the model.

    libpath: str
        Path to the locations where CvD18 models are stored. By default, it
        looks for files inside paintbox/paintbox/extern/CvD18. Both SSP and
        response function files are assumed to be uncompressed inside this
        directory.

    sigma: float
        Velocity dispersion of the output in km/s. Default is 100 km/s
        (the minimum velocity dispersion allowed).

    store: bool
        Option to store processed models in disk. Default is True.

    stored_dir: str
        Location where processed models are stored for reuse. If models with
        the same specifications are not found, models in the libpath are
        processed for use instead.

    use_stored: bool
        Option to use stored models. Default is True.

    elements: list
        Chemical abundances to be included in the model. Available elements are
        'Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe', 'K', 'Mg', 'Mn', 'N',
        'Na', 'Ni', 'Si', 'Sr', 'T', 'Ti', 'V', 'a/Fe', 'as/Fe'. Default is
        to use all elements except T (temperature of hot star) and 'a/Fe'.

    norm: bool
        Normalize the flux of the processed models to the median. Default is
        True.

    """
    def __init__(self, wave, libpath=None, sigma=100, store=True,
                 stored_dir=None, use_stored=True, elements=None, norm=True):
        if hasattr(wave, "unit"):
            self.wave = wave.to(u.Angstrom).value
        else:
            self.wave = wave #Assumes units are Angstrom
        assert wave.min() >= 3501, "Minimum wavelength is 3501 Angstrom"
        assert wave.max() <= 25000, "Maximum wavelength is 25000 Angstrom"
        assert sigma >= 100, "Minumum velocity dispersion for models is 100 " \
                             "km/s."
        self.sigma = sigma
        self.store = store
        self.elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K",
                         "Cr", "Mn", "Ba", "Ni", "Co", "Eu", "Sr", "V", "Cu",
                         "a/Fe"] if elements is None else elements
        # Defining output names
        wmin, wmax = int(self.wave.min()), int(self.wave.max())
        self.outdir = os.getcwd() if stored_dir is None else stored_dir
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.outprefix = os.path.join(self.outdir,
                         f"CvD18_sig{self.sigma}_w{wmin}-{wmax}")
        #
        ext_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
        default_path = os.path.join(ext_path, "extern/CvD18")
        self.libpath = default_path if libpath is None else libpath
        # Processing SSP files if necessary
        self.ssp_file = f"{self.outprefix}_SSPs.fits"
        if not os.path.exists(self.ssp_file) or not use_stored:
            self.ssp_files = glob.glob(os.path.join(self.libpath, "VCJ*.s100"))
            if len(self.ssp_files) == 0:
                raise ValueError("Stellar populations not found in libpath.")
            self.templates, self.params = self._prepare_CvD18_ssps()
            if store:
                self._write(self.params, self.templates,
                            self.ssp_file)
        else:
            self.params, self.templates = self._read(self.ssp_file)
        # Normalizing models if required
        self.norm = 1.
        if norm:
            self.norm = np.median(self.templates, axis=1)
            self.templates /= self.norm[:, np.newaxis]
        # Setting limits of the models
        self._limits = {}
        for param in self.params.colnames:
            vmin = self.params[param].data.min()
            vmax = self.params[param].data.max()
            self._limits[param] = (vmin, vmax)
        # Processing response functions
        self.rf_infiles = glob.glob(os.path.join(self.libpath,
                                                 "atlas_ssp*.s100"))
        self._all_elements = ['Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe',
                              'K', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'Si', 'Sr',
                              'T', 'Ti', 'V', 'a/Fe', 'as/Fe']
        self.rf_outfiles = ["{}_{}.fits".format(self.outprefix,
                            el.replace("/", ":")) for
                            el in self._all_elements]
        rfexist = all([os.path.exists(f) for f in self.rf_outfiles])
        if not rfexist or not use_stored:
            self.rfs, self.rfpars = self._prepare_CvD18_respfun()
            if self.store:
                for element, fname in zip(self._all_elements,
                                          self.rf_outfiles):
                    self._write(self.rfpars[element], self.rfs[
                               element], fname)
        else:
            self.rfs, self.rfpars = {}, {}
            for element, fname in zip(self._all_elements, self.rf_outfiles):
                self.rfpars[element], self.rfs[element] = self._read(fname)
        # Build model with paintbox
        ssp = ParametricModel(self.wave, self.params, self.templates)
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
        self._interpolator = Constrain(ssp)
        self._parnames = self._interpolator.parnames
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        """ Returns a model for a given set of parameters theta. """
        return self._interpolator(theta)

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def _prepare_CvD18_ssps(self):
        """ Process SSP models. """
        nimf = 16
        imfs = 0.5 + np.arange(nimf) / 5
        x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
        velscale = int(self.sigma / 4)
        kernel_sigma = np.sqrt(self.sigma ** 2 - 100 ** 2) / velscale
        ssps, params = [], []
        for fname in tqdm(self.ssp_files, desc="Processing SSP files"):
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
                newssp = spectres(self.wave, w, ssp)
            else:
                ssp_rebin = spectres(wvel, w, ssp)
                ssp_broad = gaussian_filter1d(ssp_rebin, kernel_sigma,
                                              mode="constant", cval=0.0)
                newssp = spectres(self.wave, wvel, ssp_broad)

            ssps.append(newssp)
        ssps = np.vstack(ssps)
        params = vstack(params)
        return ssps, params

    def _write(self, params, templates, output):
        """ Produces a MEF file for stellar populations and response
        functions. """
        hdu1 = fits.PrimaryHDU(templates)
        hdu1.header["EXTNAME"] = "TEMPLATES"
        params = Table(params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([self.wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)
        return

    def _read(self, filename):
        """ Read the MEF file with stellar populations and response
        functions. """
        templates = fits.getdata(filename)
        params = Table.read(filename, hdu=1)
        return params, templates

    def _prepare_CvD18_respfun(self):
        """ Prepare response functions from CvD models. """
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
                rebin = spectres(wvel, w, data)
                broad = gaussian_filter1d(rebin, kernel_sigma,
                                          mode="constant", cval=0.0)
                data = spectres(self.wave, wvel, broad).T

            else:
                data = spectres(self.wave, w, data).T
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
        rfs = dict([(e, np.array(rfsout[e])) for e in elements])
        rfpars = dict([(e, vstack(parsout[e])) for e in elements])
        return rfs, rfpars

    @property
    def limits(self):
        """ Lower and upper limits of the model parameters. """
        return self._limits


    @property
    def response_functions(self):
        """ Dictionary with all response functions in model. """
        return self._response_functions