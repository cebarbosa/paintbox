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
`website <http://research.iac.es/proyecto/miles//pages/stellar-libraries/miles-library.php>`__.
For this example, we will use a set of single stellar population (SSP)
templates of the E-MILES models produced with BASTI isochrones and
assuming a Chabrier initial mass function, which can be downloaded in a
tarball from this
`link <ftp://milespublic:phoShi4v@ftp.iac.es/E-MILES/EMILES_BASTI_BASE_CH_FITS.tar.gz>`__
(95 Mb). After downloading the data, it is necessary to unpack the
tarfile (preferentially into a directory), which contains 636 SSP
spectra.::

    import os
    import numpy as np
    from astropy.io import fits 
    
    emiles_dir = os.path.join(os.getcwd(), "emiles_basti_chabrier") # Edit here the location of the unpacked models

We can use the `name
convention <http://research.iac.es/proyecto/miles/pages/ssp-models/name-convention.php>`__
of the MILES models to find the model that we want::

    def miles_filename(specrange, imf, imfslope, metal, age):
        """ Returns the name of a fits file in the MILES library according to the name convention. """
        msign = "p" if metal >= 0. else "m"
        azero = "0" if age < 10. else ""
        return "{0}{1}{2:.2f}Z{3}{4:.2f}T{5}{6:02.4f}" \
                "_iTp0.00_baseFe.fits".format(specrange, imf, imfslope, msign, abs(metal), azero, age)
    
    specrange = "E" # options: "E", "M", "B", "R", "C"
    imf = "ch" # options: "un", "bi", "ku", "kb", "ch"
    imfslope = 1.3
    # All metallicities and ages available for BASTI isochrones
    metal = np.array([-0.96, -0.66, -0.35, -0.25, 0.06, 0.15,  0.26,  0.4]) 
    # Using only ages > 1 Gyr 
    ages = np.linspace(1., 14., 27)



::

    for Z in metal:
        for T in ages:
            fname = os.path.join(emiles_dir, miles_filename(specrange, imf, imfslope, Z, T))

