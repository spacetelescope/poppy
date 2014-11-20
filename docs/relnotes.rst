
Release Notes
===============

0.3.3
-------------------

:ref:`Bigger team!<about_team>`. This release log now includes github usernames of contributors: 
 
 * New classes for wavefront aberrations parameterized by Zernike polynomials (@josephoenix, @mperrin)
 * ThinLens class now reworked to require explicitly setting an outer radius over which the wavefront is normalized. *Note this is an API change for this class, and will require minor changes in code using this class*. ThinLens is now a subclass of CircularAperture.
 * Implement resizing of phasors to allow use of FITSOpticalElements with Wavefronts that have different spatial sampling. (@douglase)
 * Installation improvements and streamlining (@josephoenix, @cslocum)
 * Code cleanup and formatting (@josephoenix)
 * Improvements in unit testing (@mperrin, @josephoenix, @douglase)
 * Added normalize='exit_pupil' option; added documentation for normalization options. (@mperrin)
 * Bug fix for "FQPM on an obscured aperture" example. Thanks to Github user qisaiman for the bug report. (@mperrin)
 * Bug fix to compound optic display (@mperrin)
 * Documentation improvements (team)

0.3.2
-------------------
Released 2014 Sept 8

 * Bug fix: Correct pupil orientation for inverse transformed pupils using PyFFTW so that it is consistent with the result using numpy FFT.

0.3.1
-------------------
Released August 14 2014

 * Astropy compatibility updated to 0.4. 
        * Configuration system reworked to accomodate the astropy.configuration transition.
        * Package infrastructure updated to most recent `astropy package-template <https://github.com/astropy/package-template/>`_.
 * Several OpticalElements got renamed, for instance ``IdealCircularOcculter`` became just ``CircularOcculter``. (*All* the optics in ``poppy`` are 
   fairly idealized and it seemed inconsistent to signpost that for only some of them. The explicit 'Ideal' nametag is kept only for the FQPM to emphasize that one
   in particular uses a very simplified prescription and neglects refractive index variation vs wavelength.)
 * Substantially improved unit test system. 
 * Some new utility functions added in poppy.misc for calculating analytic PSFs such as Airy functions for comparison (and use in the test system).
 * Internal code reorganization, mostly which should not affect end users directly.
 * Packaging improvements and installation process streamlining, courtesy of Christine Slocum and Erik Bray
 * Documentation improvements, in particular adding an IPython notebook tutorial. 



0.3
----------

Released April 7, 2014

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

Contributors
================

See :ref:`about`.
