Preparing stellar population models for **paintbox**
----------------------------------------------------

Paintbox was initially designed to describe the observed spectrum, i.e.,
the flux from stars as a function of the wavelength, of galaxies where
stars are not individually resolved. In these cases, the most important
ingredient to describe the stellar features in observations are stellar
population models, which describe the properties of an emsemble of stars
with different properties, e.g., ages, metallicities, etc. These models
are incredibly complex to be generated, but several groups of
astronomers distribute their models for free for users, either publicily
or under request. However, there is no single standard way of
distributing stellar population models, so we have to prepare the data
from the models *before* using **paintbox**. Moreover, in practical
applications, we also require the models to be optimized for that
application, as a way to minimize the (often expensive) number of
computations depending on the resolution of the data.

Using MILES stellar populations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stellar populations models from the MILES library can be obtained in a
variety of ways in their
`website <http://research.iac
.es/proyecto/miles//pages/stellar-libraries/miles-library.php>`__. ::

    import os
    
    import numpy as np
    from astropy.io import fits 
    from astropy.table import Table
    
    emiles_dir = os.path.join(os.getcwd(), "EMILES_v11") # Edit here the location of the unpacked models
    w1 = 2600 # Minimum wavelength of the templates for application
    w2 = 10000 # Maximum wavelength

We can use the `name
convention <http://research.iac.es/proyecto/miles/pages/ssp-models/name-convention.php>`__
of the MILES models to find the model that we want. ::

    def miles_filename(specrange, imf, imfslope, metal, age):
        """ Returns the name of a fits file in the MILES library according
        to the name convention. """
        msign = "p" if metal >= 0. else "m"
        azero = "0" if age < 10. else ""
        return "{0}{1}{2:.2f}Z{3}{4:.2f}T{5}{6:02.4f}" \
                "_iTp0.00_baseFe.fits".format(specrange, imf, \
                imfslope, msign, abs(metal), azero, age)

For this example, we will use a set of single stellar population (SSP)
templates of the E-MILES models (version 11) produced with BASTI
isochrones and assuming a Chabrier initial mass function, which can be
downloaded in a tarball from their public ftp link available
`here <http://miles.iac.es/>`__ (95 Mb). After downloading the data, it
is necessary to unpack the tarfile (preferentially into a directory),
which contains 636 SSP spectra. ::

    specrange = "E" # options: "E", "M", "B", "R", "C"
    imf = "ch" # options: "un", "bi", "ku", "kb", "ch"
    imfslope = 1.3
    # All metallicities and ages available for BASTI isochrones
    Zs = np.array([-0.96, -0.66, -0.35, -0.25, 0.06, 
                      0.15,  0.26,  0.4]) 
    Ts = np.linspace(1., 14., 27)# Using only ages > 1 Gyr
    ssps_grid = np.array(np.meshgrid(Ts, Zs)).T.reshape(-1, 2)
    params = Table(ssps_grid, names=["T", "Z"])
    nssps = len(ssps_grid)
    filenames = []
    for t, z in ssps_grid:
        filenames.append(miles_filename(specrange, imf, imfslope, t, z))

We use the information in the header of one spectrum to determine the
wavelength range, we select the desired wavelength range, and put the
spetra in a wavelength array. ::

    h = fits.getheader(os.path.join(emiles_dir, filenames[0]))
    wave = (h['CRVAL1'] +   h['CDELT1'] * (np.arange((h['NAXIS1'])) + 1 -
                                      h['CRPIX1']))
    idx = np.where((wave >= wave_lims[0]) & (wave <=wave_lims[1]))
    wave = wave[idx]
    # Using first spectrum to get array size after rebbining
    flux = fits.getdata(filenames[0])[idx]
    wrange = [wave[0], wave[-1]]
    newflux, logLam, velscale = util.log_rebin(wrange, flux,
                                               velscale=velscale)
    ssps = np.zeros((nssps, len(newflux)))
    w = wave * u.angstrom
    for i, filename in enumerate(tqdm(filenames)):
        flambda = fits.getdata(os.path.join(emiles_dir, filename))[idx]
        flux_log, logLam, velscale = util.log_rebin(wrange, flambda,
                                                   velscale=velscale)
        ssps[i] = flux_log