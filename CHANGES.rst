1.5 (2021-08-20)
----------------
- Refactored code, including previous utils module as part of the main
package.

1.4 (2021-08--18)
----------------
- Added method to fix parameters in paintbox models directly.
- Added method to constrain duplicated parameters in paintbox models.
- Refactored code for dispersion with constant velocity scale from disp2vel
to logspace_dispersion.


1.3.1 (2021-06-18)
------------------
- Fixed bug in wavelength comparison while combining models.

1.3.0 (2021-06-11)
------------------
- New operator to fix values in model.
- Changed behavior of mask in likelihoods to be according to conventions.
- Refactored CvD18 class.

1.2.1 (2021-05-25)
------------------
- Simplified interface for CvD models.
- Fixed documentation in the readthedocs.
- Included tutorial notebooks for more classes.
- Fixed bug reusing CvD models for disk.
- Included access to individual response functions in CvD18 class.

1.2.0 (2021-05-12)
------------------
- New, simple interface for CvD models.
- Fixed behaviour of disp2vel for list inputs.
- Included examples for simple classes and CvD models

1.1.1 (2021-04-07)
------------------
- Fixed bug in shape of CvD output after preparation of models.

1.1.0 (2021-04-05)
------------------
- Included tools to facilitate use of CvD models.
- Included option to broad CvD models to a given velocity dispersion.
- New routine to calculate wavelength dispersion with fixed velocity scale.

1.0 (2021-03-11)
------------------
- Initial release of the paintbox code.