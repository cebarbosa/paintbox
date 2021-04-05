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

from .disp2vel import disp2vel

def prepare_CvD18(ssp_files, wave, output, overwrite=False, sigma=100):
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
    overwrite: bool (optional)
        Overwrite the output files if they already exist.
    sigma: float
        Velocity dispersion of the models, in km/s. Defaults to 100 km/s,
        the minimum resolution of the models.

    """
    if hasattr(wave, "unit"):
        wave = wave.to(u.Angstrom).value
    assert wave[0] >= 3500, "Minimum wavelength is 3500 Angstrom"
    assert wave[-1] <= 25000, "Maximum wavelength is 25000 Angstrom"
    assert sigma >= 100, "Minimum velocity dispersion is 100 km/s"
    if os.path.exists(output) and not overwrite:
        return
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
        ssps.append(newssp.T)
    ssps = np.array(ssps)
    params = vstack(params)
    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    params = Table(params)
    hdu2 = fits.BinTableHDU(params)
    hdu2.header["EXTNAME"] = "PARAMS"
    # Making wavelength array
    hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
    hdu3.header["EXTNAME"] = "WAVE"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return

def prepare_response_functions(rf_files, wave, outprefix, overwrite=False,
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
    assert wave[0] >= 3500, "Minimum wavelength is 3500 Angstrom"
    assert wave[-1] <= 25000, "Maximum wavelength is 25000 Angstrom"
    assert sigma >= 100, "Minimum velocity dispersion is 100 km/s"
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
    kernel_sigma = np.sqrt(sigma ** 2 - 100 ** 2) / velscale
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
    for element in elements:
        output = "{}_{}.fits".format(outprefix, element.replace("/", "_over_"))
        if os.path.exists(output) and not overwrite:
            continue
        rfdata = np.array(rfsout[element])
        params = vstack(parsout[element])
        hdu1 = fits.PrimaryHDU(rfdata)
        hdu1.header["EXTNAME"] = "SSPS"
        params = Table(params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)

def example():
    """ Example of application for the preparation of CvD models. """
    data_dir = "/home/kadu/Dropbox/SSPs/CvD18"
    ssps_dir = os.path.join(data_dir, "VCJ_v8")
    ssp_files = glob.glob(os.path.join(ssps_dir, "VCJ*.s100"))
    wave = np.linspace(4000, 20000, 2000)
    prepare_CvD18(ssp_files, wave, "CvD.fits", sigma=300, overwrite=True)
    rfs_dir = os.path.join(data_dir, "RFN_v3")
    rf_files = glob.glob(os.path.join(rfs_dir, "atlas_ssp*.s100"))
    outprefix = os.path.join(os.getcwd(), "rf")
    prepare_response_functions(rf_files, wave, "rf", sigma=300)


if __name__ == "__main__":
    example()