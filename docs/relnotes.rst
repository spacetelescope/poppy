
Release Notes
===============


0.3
-----

 * Dependencies updated to use astropy.
 * Added documentation and examples for POPPY, separate from the WebbPSF documentation.
 * Improved configuration settings system, using astropy.config framework.

   * The astropy.config framework itself is in flux from astropy 0.3 to 0.4; some of the related functionality
     in poppy may need to change in the future.

 * Added support for rectangular subarray calculations. You can invoke these by setting fov_pixels or fov_arcsec with a 2-element iterable::

    >> nc = webbpsf.NIRCam()
    >> nc.calcPSF('F212N', fov_arcsec=[3,6])
    >> nc.calcPSF('F187N', fov_pixels=(300,100) )

   Those two elements give the desired field size as (Y,X) following the usual Python axis order convention.
 * Added support for pyFFTW in addition to PyFFTW3.
 * pyFFTW will auto save wisdom to disk for more rapid execution on subsequent invocations
 * InverseTransmission of an AnalyticElement is now allowed inside a CompoundAnalyticOptic
 * Added SecondaryObscuration optic to conveniently model an opaque secondary mirror and adjustible support spiders.
 * Added RectangleAperture. Added rotation keywords for RectangleAperture and SquareAperture.
 * Added AnalyticOpticalElement.sample() function to sample analytic functions onto a user defined grid. Refactored 
   the display() and toFITS() functions. Improved functionality of display for CompoundAnalyticOptics. 

0.2.8
----------
 * First release as a standalone package (previously was integrated as part of webbpsf). See the release notes for WebbPSF for prior verions.
 * switched package building to use `setuptools` instead of `distutils`/`stsci_distutils_hack`
 * new `Instrument` class in poppy provides much of the functionality previously in JWInstrument, to make it
   easier to model generic non-JWST instruments using this code.

