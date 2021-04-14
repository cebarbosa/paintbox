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
from paintbox import ParametricModel, Constrain, CompositeSED

try:
    from .disp2vel import disp2vel
except:
    from disp2vel import disp2vel

class CvD18():
    def __init__(self, wave, ssp_files=None, rf_files=None, sigma=100,
                 store=True, outdir=None, outname=None, use_stored=True,
                 elements=None, wprefix=None, norm=True):
        if hasattr(wave, "unit"):
            wave = wave.to(u.Angstrom).value
        self.wave = wave
        self.ssp_files = ssp_files
        self.rf_files = [] if rf_files is None else rf_files
        self.wprefix = "w" if wprefix is None else wprefix
        assert sigma >= 100, "Minumum velocity dispersion for models is 100 " \
                             "km/s."
        self.sigma = sigma
        self.store = store
        self.outdir = os.getcwd() if outdir is None else outdir
        self.outname = "CvD18" if outname is None else outname
        # Processing and reading stellar population models
        self.ssp_file = os.path.join(self.outdir,
                                     "{}_sps.fits".format(self.outname))
        if not os.path.exists(self.ssp_file) or not use_stored:
            assert self.ssp_files is not None, "SSP files have to be provided."
            self.templates, self.params = self.prepare_CvD18_ssps(ssp_files,
                                          wave, self.ssp_file, sigma=self.sigma)
            if store:
                self.write(self.wave, self.params, self.templates,
                           self.ssp_file)
        else:
            self.params, self.templates = self._read(self.ssp_file)
        self.norm = 1.
        if norm:
            self.norm = np.median(self.templates, axis=1)
            self.templates /= self.norm[:, None]
        self.limits = {}
        for param in self.params.colnames:
            vmin = self.params[param].data.min()
            vmax = self.params[param].data.max()
            self.limits[param] = (vmin, vmax)
        # Processing and reading response functions
        self._all_elements = ['Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe',
                              'K', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'Si', 'Sr',
                              'T', 'Ti', 'V', 'a/Fe', 'as/Fe']
        self.elements = ['Ba', 'C', 'Ca', 'Co', 'Cr', 'Cu', 'Eu', 'Fe',
                         'K', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'Si', 'Sr',
                         'Ti', 'V'] if elements is None else elements
        rfprefix = os.path.join(self.outdir, self.outname)
        rfout = ["{}_{}.fits".format(rfprefix, el.replace("/", ":")) for el in
                 self.elements]
        if not all([os.path.exists(f) for f in rfout]):
            self.rfs, self.rfpars = self.prepare_CvD18_respfun(rf_files,
                                                             self.wave,
                                    outprefix=rfprefix, sigma=self.sigma)
            if store:
                for element, fname in zip(self.elements, rfout):
                    self.write(self.wave, self.rfpars[element], self.rfs[
                               element], fname)
        else:
            self.rfs, self.rfpars = {}, {}
            for element, fname in zip(self.elements, rfout):
                self.rfpars[element], self.rfs[element] = self._read(fname)
        # Build model with paintbox
        ssp = ParametricModel(self.wave, self.params, self.templates)
        for element in self.elements:
            rf = ParametricModel(self.wave, self.rfpars[element], self.rfs[
                element])
            ssp = ssp * rf
            vmin = rf.params[element].data.min()
            vmax = rf.params[element].data.max()
            self.limits[element] = (vmin, vmax)
        if len(self.elements) > 0: # Update limits in case response functions
            # are used.
            for p in ["Age", "Z"]:
                vmin = rf.params[p].data.min()
                vmax = rf.params[p].data.max()
                self.limits[p] = (vmin, vmax)
        self._interpolator = Constrain(ssp)
        self.parnames = self._interpolator.parnames
        self._nparams = len(self.parnames)

    def __call__(self, theta):
        return self._interpolator(theta)

    def __add__(self, o):
        """ Addition between two SED components. """
        return CompositeSED(self, o, "+")

    def __mul__(self, o):
        """  Multiplication between two SED components. """
        return CompositeSED(self, o, "*")

    def prepare_CvD18_ssps(self, ssp_files, wave, output, sigma=100):
        """ Prepare templates for SSP models from Villaume et al. (2017).

        Parameters
        ----------
        ssp_files: list
            List containing the full path of SSP models to be processed.
        wave: np.array or astropy.Quantity
            Wavelength dispersion. Default units in Angstrom is assumed if
            wavelength is provided as as numpy array.
        output: str
            Name of the output file (a multi-extension FITS file)
        sigma: float
            Velocity dispersion of the models, in km/s. Defaults to 100 km/s,
            the minimum resolution of the models.
        store: bool
            Option to store

        """
        if hasattr(wave, "unit"):
            wave = wave.to(u.Angstrom).value
        assert wave[0] >= 3501, "Minimum wavelength is 3501 Angstrom"
        assert wave[-1] <= 25000, "Maximum wavelength is 25000 Angstrom"
        assert all([os.path.exists(f) for f in ssp_files]), \
               "Some or all input files are missing."
        nimf = 16
        imfs = 0.5 + np.arange(nimf) / 5
        x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
        velscale = int(sigma / 4)
        kernel_sigma = np.sqrt(sigma ** 2 - 100 ** 2) / velscale
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
            if sigma > 100:
                wvel = disp2vel(w, velscale)
            ssp = data[:, 1:].T
            if sigma <= 100:
                newssp = spectres(wave, w, ssp)
            else:
                ssp_rebin = spectres(wvel, w, ssp)
                ssp_broad = gaussian_filter1d(ssp_rebin, kernel_sigma,
                                              mode="constant", cval=0.0)
                newssp = spectres(wave, wvel, ssp_broad)

            ssps.append(newssp)
        ssps = np.vstack(ssps)
        params = vstack(params)
        return ssps, params

    def write(self, wave, params, templates, output):
        """ Produces a MEF file for stellar populations and response
        functions. """
        hdu1 = fits.PrimaryHDU(templates)
        hdu1.header["EXTNAME"] = "TEMPLATES"
        params = Table(params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)
        return

    def _read(self, filename):
        """ Read the MEF file with stellar populations and response
        functions. """
        templates = fits.getdata(filename)
        params = Table.read(filename, hdu=1)
        wave = Table.read(filename, hdu=2)
        assert np.all(wave["wave"].data == self.wave), "Wavelength of input " \
                                                      "and " \
                                               "models do not match."
        return params, templates


    def prepare_CvD18_respfun(self, rf_files, wave, outprefix, overwrite=False,
                              sigma=100):
        """ Prepare response functions from CvD models.

        Parameters
        ----------
        rf_files: str
            List of paths for the response function files.
        wave: np.array
            Wavelength dispersion for the output.
        outprefix: str
            First part of the name of the response function output files. The
            response functions are stored in different files for different
            elements, named "{}_{}.fits".format(outprefix, element).
        overwrite: bool (optional)
            Overwrite output.
        """
        if hasattr(wave, "unit"):
            wave = wave.to(u.Angstrom).value
        assert wave[0] >= 3501, "Minimum wavelength is 3501 Angstrom"
        assert wave[-1] <= 25000, "Maximum wavelength is 25000 Angstrom"
        assert sigma >= 100, "Minimum velocity dispersion is 100 km/s"
        assert all([os.path.exists(f) for f in rf_files]), \
               "Some or all input files are missing."
        # Read one spectrum to get name of columns
        with open(rf_files[0]) as f:
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
        velscale = int(sigma / 4)
        kernel_sigma = np.sqrt(sigma**2 - 100**2) / velscale
        rfsout = dict([(element, []) for element in elements])
        parsout = dict([(element, []) for element in elements])
        desc = "Preparing response functions"
        for i, fname in enumerate(tqdm(rf_files, desc=desc)):
            spec = os.path.split(fname)[1]
            T = float(spec.split("_")[2][1:])
            Z = float(spec.split("_")[3].split(".abun")[0][1:].replace(
                "p", "+").replace("m", "-"))
            data = np.loadtxt(fname)
            w = data[:, 0]
            data = data.T
            if sigma > 100:
                wvel = disp2vel(w, velscale)
                rebin = spectres(wvel, w, data)
                broad = gaussian_filter1d(rebin, kernel_sigma,
                                          mode="constant", cval=0.0)
                data = spectres(wave, wvel, broad).T

            else:
                data = spectres(wave, w, data).T
            fsun = data[:, 1]
            for element in elements:
                # Adding solar response
                p = Table([[Z], [T], [0.]], names=["Z", "Age", element])
                rfsout[element].append(np.ones(len(wave)))
                parsout[element].append(p)
                # Adding non-solar responses
                for sign in signal:
                    name = "{}{}".format(element, sign)
                    cols = [(i,f) for i, f in enumerate(fields) if f.startswith(
                        name)]
                    for i, col in cols:
                        val = float("{}1".format(sign)) * float(col.split(sign)[1])
                        t = Table([[Z], [T], [val]], names=["Z", "Age", element])
                        parsout[element].append(t)
                        rf = data[:, i] / fsun
                        rfsout[element].append(rf)
        rfs = dict([(e, np.array(rfsout[e])) for e in elements])
        rfpars = dict([(e, vstack(parsout[e])) for e in elements])
        return rfs, rfpars


def example():
    """ Example of application for the preparation of CvD models. """
    import matplotlib.pyplot as plt
    sigma = 300
    data_dir = "/home/kadu/Dropbox/SSPs/CvD18"
    ssps_dir = os.path.join(data_dir, "VCJ_v8")
    ssp_files = glob.glob(os.path.join(ssps_dir, "VCJ*.s100"))
    rfs_dir = os.path.join(data_dir, "RFN_v3")
    rf_files = glob.glob(os.path.join(rfs_dir, "atlas_ssp*.s100"))
    wave = disp2vel(np.array([7000, 12000]), 100)
    outdir = os.path.join(data_dir, "CvD_test")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Determining the name of the output
    names = np.array([os.path.split(f)[-1].split("_") for f in
                      ssp_files])
    idx = np.all(names == names[0, np.newaxis], axis=0)
    outname = "_".join(names[0][idx]).replace("s100", "s{}".format(sigma))
    ssp = CvD18(wave, ssp_files=ssp_files, rf_files=rf_files, sigma=300,
                  outdir=outdir, outname=outname)
    w1 = 0.5
    w2 = 0.5
    # Making arrays with minimum and maximum available parameters
    pyoung= np.array([ssp.limits[p.split("_")[0]][0] for p in ssp.specpars])
    pold = np.array([ssp.limits[p.split("_")[0]][1] for p in ssp.specpars])
    pcomposite = np.hstack([w1, pyoung, w2, pold])
    plt.plot(wave, ssp(pcomposite), "-", label="Composite population")
    plt.plot(wave, w1 * ssp.get_spectrum(pold), label="Old populaiton")
    plt.plot(wave, w2 * ssp.get_spectrum(pyoung), label="Young population")
    plt.legend()
    plt.xlabel(r"$\lambda$ (Angstrom)")
    plt.ylabel(r"Flux (normalized)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example()