Preparing stellar population models
-----------------------------------

Paintbox is particularly designed to model the observed spectrum, i.e.,
the flux from stars as a function of the wavelength, of galaxies where
stars are not individually resolved. In these cases, the most important
ingredient to describe the stellar features in observations are stellar
population models, which describe the properties of an emsemble of stars
with different properties, e.g., ages, metallicities, etc. Several
groups of astronomers distribute their models for free for users, either
publicily or under request. However, there is no single standard way of
distributing stellar population models, so we have to prepare the data
from the models *before* using **paintbox**. Moreover, in practical
applications, we also require the models to be tunned according to the
scientific requirements of the modeling as a way to minimize the (often
expensive) number of computations depending on the resolution of the
data.

USING EMILES stellar populations templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stellar populations models from the MILES library can be obtained in a
variety of ways in their
`website <http://research.iac.es/proyecto/miles//pages/stellar-libraries/miles-library.php>`__.
For this example, we will use the packages
`astropy <https://www.astropy.org>`__ to handle FITS fiels and tables,
and the [pPXF](`ppxf <https://pypi.org/project/ppxf/>`__ for rebinning
the data to a logarithmic scale.

::

    import os
    
    import numpy as np
    from astropy.io import fits 
    from astropy.table import Table
    from ppxf import ppxf_util

For this example, we will use a set of single stellar population (SSP)
templates of the E-MILES models (version 11) produced with BASTI
isochrones and assuming a Chabrier initial mass function, which can be
downloaded in a tarball from their public ftp link available `in their
website <http://miles.iac.es/>`__ (EMILES_BASTI_BASE_CH_FITS.tar.gz, 95
Mb). After downloading the data, it is necessary to unpack the tarfile
(preferentially into a subdirectory, which we name emiles_v11),
containing the 636 SSP spectra in this case.

::

    emiles_dir = os.path.join(os.getcwd(), "emiles_v11")
    w1 = 2600 # Minimum wavelength
    w2 = 10000 # Maximum wavelength

We can use the `MILES name
convention <http://research.iac.es/proyecto/miles/pages/ssp-models/name-convention.php>`__
to read the files with the models.

::

    def miles_filename(specrange, imf, imfslope, metal, age):
        """ Returns the name of a fits file in the MILES library according
        to the name convention. """
        msign = "p" if metal >= 0. else "m"
        azero = "0" if age < 10. else ""
        return "{0}{1}{2:.2f}Z{3}{4:.2f}T{5}{6:02.4f}" \
                "_iTp0.00_baseFe.fits".format(specrange, imf, \
                imfslope, msign, abs(metal), azero, age)

Below we produce a list containing all the spectra that we are going to
use in our analysis (filenames), and we also produce an astropy `Table
object <https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table>`__
storing the parameters of the files.

::

    specrange = "E" # options: "E", "M", "B", "R", "C"
    imf = "ch" # options: "un", "bi", "ku", "kb", "ch"
    imfslope = 1.3
    # Values of metallicities and ages available for BASTI isochrones
    Zs = np.array([-0.96, -0.66, -0.35, -0.25, 0.06, 0.15,  0.26,  0.4]) 
    Ts = np.linspace(1., 14., 27) # Using only ages > 1 Gyr
    ssps_grid = np.array(np.meshgrid(Ts, Zs)).T.reshape(-1, 2)
    nssps = len(ssps_grid)
    filenames = []
    for t, z in ssps_grid:
        filenames.append(miles_filename(specrange, imf, imfslope, z, t))
    params = Table(ssps_grid, names=["T", "Z"])

We use the information in the header of one spectrum to determine the
wavelength range (which is always the same for a given set of models).
Notice that the wavelength range covered by the EMILES models is large
(from the near-UV to the IR).

::

    h = fits.getheader(os.path.join(emiles_dir, filenames[0]))
    wave = (h['CRVAL1'] + h['CDELT1'] * (np.arange((h['NAXIS1'])) + 1 - h['CRPIX1']))

Finally, we need to trim and/or rebin the model spectra. We need to trim
the data to cover only the spectral region of the observed data. Notice,
however, that we should always have extra coverage in the models,
preferentially on both edges of the spectra, if we are also modeling the
kynematics of the galaxy, and also to avoid problems at the edges of the
models owing to convolutions (below I use 500 Angstrom, but this can be
optimized for a galaxy according to their redshift). We may also rebin
the data, either to have the same wavelength dispersion of the
observations, or to a logarithmic scale to model the kinematics. We use
the pPXF for this purpose, assuming a velocity scale for the rebinning
of 200 km/s.

::

    velscale = 200
    extra_wave = 500
    idx = np.where((wave >= w1 - extra_wave) & (wave <= w2 + extra_wave))
    wave = wave[idx] # Trimming wavelength array
    
    # Using first spectrum to get array size after rebbining
    flux = fits.getdata(os.path.join(emiles_dir, filenames[0]))[idx]
    wrange = [wave[0], wave[-1]]
    newflux, logLam, velscale = ppxf_util.log_rebin(wrange, flux, 
                                                     velscale=velscale)
    # Loop to trim and rebin spectra
    ssps = np.zeros((nssps, len(newflux)))
    for i, filename in enumerate(filenames):
        flambda = fits.getdata(os.path.join(emiles_dir, filename))[idx]
        flux_log = ppxf_util.log_rebin(wrange, flambda, velscale=velscale)[0]
        ssps[i] = flux_log

Now, we just need to store the processed data into a FITS file.

::

    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    hdu2 = fits.BinTableHDU(params)
    hdu2.header["EXTNAME"] = "PARAMS"
    hdu3 = fits.BinTableHDU(Table([logLam], names=["loglam"]))
    hdu3.header["EXTNAME"] = "WAVE"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    output = "emiles_chabrier_w{}_{}_vel{}.fits".format(w1, w2, velscale)
    hdulist.writeto(output, overwrite=True)

In this particular example, we will obtain a multi-extension FITS file
named “emiles_chabrier_w2600_10000_vel200.fits”, which contains the 2D
array with the models, a parameter table, and an 1D array with the
wavelength array. Notice that, in practice, if often necessary to
degrade the model spectra to match the resolution of the observations,
which can be performed with the task paintbox.utils.broad2res.

Preparing CvD models
~~~~~~~~~~~~~~~~~~~~

Models from the `Conroy and van Dokkum
(2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...747...69C/abstract>`__
and `Conroy et
al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..139C/abstract>`__,
a.k.a. CvD models, can be obtained under request to the authors. Similar
to the MILES models, CvD are also distributed as SSP models with varying
ages, metallicities, and IMFs, but also provide response functions that
allow the variation of several individual elements, e.g., C, N, O, Mg,
Si, Ca, Ti, and Fe. Below we show how to handle these models for
**paintbox**. For this example, we use the SSP models computed with the
`VCJ stellar
library <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...23V/abstract>`__
version 8, and the response functions from Conroy et al. (2018) version
3. In this example, we use
`SpectRes <https://spectres.readthedocs.io/en/latest/>`__ to perform the
rebinning of the models, as it can handle arbitrary wavelength
dispersions.

::

    import os
    
    import numpy as np
    from astropy.table import Table, vstack
    from astropy.io import fits
    from ppxf import ppxf_util
    from spectres import spectres
    from tqdm import tqdm

First we define a function to handle the SSP models.

::

    def prepare_VCJ17(data_dir, wave, output, overwrite=False):
        """ Prepare templates for SSP models from Villaume et al. (2017).
    
            Parameters
        ----------
        data_dir: str
            Path to the SSP models.
        wave: np.array
            Wavelength dispersion.
        output: str
            Name of the output file (a multi-extension FITS file)
        overwrite: bool (optional)
            Overwrite the output files if they already exist.
    
        """
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


Similarly, we define a function to produce the models for the response
functions.

::

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

For instance, for near-infrared observations, the above routines can be
used as follows:

::

    # Preparing SSP models
    w1, w2 = 8000, 13000 # Setting the wavelength window
    models_dir = "/home/kadu/Dropbox/SPINS/CvD18/" # Directory where models are stored
    ssps_dir = os.path.join(models_dir, "VCJ_v8")
    
    # Loading the wavelength dispersion from one of the models
    wave = np.loadtxt(os.path.join(ssps_dir, os.listdir(ssps_dir)[0]), usecols=(0,))
    idx = np.where((wave >= w1) & (wave <= w2))[0]
    wave = wave[idx] # Trimming wavelength range
    # Defining where the models should be stored
    outdir = os.path.join(os.getcwd(), "templates")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    output = os.path.join(outdir, "VCJ17_varydoublex.fits")
    prepare_VCJ17(ssps_dir, wave, output)
    # Preparing response functions
    rfs_dir = os.path.join(models_dir, "RFN_v3")
    outprefix = os.path.join(outdir, "C18_rfs")
    prepare_response_functions(rfs_dir, wave, outprefix)


.. parsed-literal::

    Processing SSP files: 100%|██████████| 35/35 [04:22<00:00,  7.51s/it]
    Preparing response functions: 100%|██████████| 21/21 [03:44<00:00, 10.69s/it]


