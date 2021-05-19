Stellar population models
--------------------------

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

EMILES
~~~~~~

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

    emiles_dir = "/home/kadu/Dropbox/SSPs/emiles_v11"
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

CvD models
~~~~~~~~~~

Models from the `Conroy and van Dokkum
(2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...747...69C/abstract>`__
and `Conroy et
al. (2018) <https://ui.adsabs.harvard.edu/abs/2018ApJ...854..139C/abstract>`__,
a.k.a. CvD models, can be obtained under request to the authors. Similar
to the MILES models, CvD are also distributed as SSP models with varying
ages, metallicities, and IMFs, but also provide response functions that
allow the variation of several individual elements, e.g., C, N, O, Mg,
Si, Ca, Ti, and Fe. Below we show how to handle these models using
**paintbox** utilities. For this example, we use the SSP models computed
with the `Extended IRTF Spectral
Library <https://ui.adsabs.harvard.edu/abs/2017ApJS..230...23V/abstract>`__
version 8, and the response functions from Conroy et al. (2018) version
3.

::

    import os
    import glob

    import numpy as np
    from paintbox.utils import CvD_utils

    base_dir = "/home/kadu/Dropbox/SSPs/CvD18"

We first need to point out the location of the models in our computer.
To make things simple, I store all SSP models from VCJ library in a
subdirectory.

::

    ssps_dir = os.path.join(base_dir, "VCJ_v8")
    ssp_files = glob.glob(os.path.join(ssps_dir, "VCJ*.s100"))

To prepare the data to a convenient wavelength dispersion and to store
the models in a single FITS file for later use, we use the
paintbox.prepare_VCJ routine.

::

    # Defining an arbitrary wavelength region in the near IR
    w1, w2 = 8000, 13000 # Setting the wavelength window
    wave = np.arange(w1, w2)
    output = os.path.join(os.getcwd(), "CvD18_varydoublex_test.fits")
    CvD_utils.prepare_CvD18(ssp_files, wave, output)


.. parsed-literal::

    Processing SSP files: 100%|██████████| 35/35 [00:55<00:00,  1.58s/it]


Similarly, we can prepare the response functions for the different
elements which can be later used in the fitting process.

::

    # Preparing response functions
    rfs_dir = os.path.join(base_dir, "RFN_v3")
    rf_files = glob.glob(os.path.join(rfs_dir, "atlas_ssp*.s100"))
    # Each element will be saved in a different file, thus we define a prefix for the RFs
    outprefix = os.path.join(os.getcwd(), "C18_rfs")
    CvD_utils.prepare_response_functions(rf_files, wave, outprefix)


.. parsed-literal::

    Preparing response functions: 100%|██████████| 25/25 [00:07<00:00,  3.26it/s]


