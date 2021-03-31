""" Calculates the M/L for CvD models. """
import os

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.io import fits
from tqdm import tqdm
from spectres import spectres

def prepare_VCJ17(data_dir, wave, output, overwrite=False):
    """ Prepare templates for SSP models from Villaume et al. (2017).

    Parameters
    ----------
    data_dir: str
        Path to the SSP models.
    wave: np.array or astropy.Quantity
        Wavelength dispersion. Default units in Angstrom is assumed if
        wavelength is provided as as numpy array.
    output: str
        Name of the output file (a multi-extension FITS file)
    overwrite: bool (optional)
        Overwrite the output files if they already exist.

    """
    if hasattr(wave, "unit"):
        wave = wave.to(u.Angstrom).value
    if os.path.exists(output) and not overwrite:
        return
    specs = sorted(os.listdir(data_dir))
    nimf = 16
    imfs = 0.5 + np.arange(nimf) / 5
    x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
    ssps, params = [], []
    for spec in tqdm(specs, desc="Processing SSP files"):
        T = float(spec.split("_")[3][1:])
        Z = float(spec.split("_")[4][1:-8].replace("p", "+").replace(
                    "m", "-"))
        data = np.loadtxt(os.path.join(data_dir, spec))
        w = data[:,0]
        for i, (x1, x2) in enumerate(zip(x1s, x2s)):
            params.append(Table([[Z], [T], [x1], [x2]],
                                names=["Z", "Age", "x1", "x2"]))
            ssp = data[:, i+1]
            newssp = spectres(wave, w, ssp)
            ssps.append(newssp)
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

def prepare_response_functions(data_dir, wave, outprefix, redo=False):
    """ Prepare response functions from CvD models.

    Parameters
    ----------
    data_dir: str
        Path to the response function files
    wave: np.array
        Wavelength dispersion.
    outprefix: str
        First part of the name of the response function output files. The
        response functions are stored in different files for different
        elements, named "{}_{}.fits".format(outprefix, element).
    redo: bool (optional)
        Overwrite output.

    """
    specs = sorted(os.listdir(data_dir))
    # Read one spectrum to get name of columns
    with open(os.path.join(data_dir, specs[0])) as f:
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
    for element in tqdm(elements, desc="Preparing response functions"):
        output = "{}_{}.fits".format(outprefix, element.replace("/", ""))
        if os.path.exists(output) and not redo:
            continue
        params = []
        rfs = []
        for spec in specs:
            T = float(spec.split("_")[2][1:])
            Z = float(spec.split("_")[3].split(".abun")[0][1:].replace(
                      "p", "+").replace("m", "-"))
            data = np.loadtxt(os.path.join(data_dir, spec))
            w = data[:,0]
            fsun = data[:,1]
            # Adding solar response
            p = Table([[Z], [T], [0.]], names=["Z", "Age", element])
            rf = np.ones(len(wave))
            rfs.append(rf)
            params.append(p)
            # Adding non-solar responses
            for sign in signal:
                name = "{}{}".format(element, sign)
                cols = [(i,f) for i, f in enumerate(fields) if f.startswith(
                    name)]
                for i, col in cols:
                    val = float("{}1".format(sign)) * float(col.split(sign)[1])
                    t = Table([[Z], [T], [val]], names=["Z", "Age", element])
                    params.append(t)
                    rf = data[:, i] / fsun
                    newrf= spectres(wave, w, rf)
                    rfs.append(newrf)
        rfs = np.array(rfs)
        params = vstack(params)
        hdu1 = fits.PrimaryHDU(rfs)
        hdu1.header["EXTNAME"] = "SSPS"
        params = Table(params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)

if __name__ == "__main__":
    pass
