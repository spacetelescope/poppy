
Release Notes
===============

For a list of contributors, see :ref:`about`.

.. _whatsnew:

0.5.1
-----

*Unreleased*

 * Fix ConfigParser import (see `astropy/package-template#172 <https://github.com/astropy/package-template/pull/172>`_)
 * Fixes to formatting of ``astropy.units.Quantity`` values (`#171 <https://github.com/mperrin/poppy/issues/171>`_, `#174 <https://github.com/mperrin/poppy/pull/174>`_, `#179 <https://github.com/mperrin/poppy/pull/174>`_; @josephoenix, @neilzim)
 * Fixes to ``fftw_save_wisdom`` and ``fftw_load_wisdom`` (`#177 <https://github.com/mperrin/poppy/issues/177>`_, `#178 <https://github.com/mperrin/poppy/pull/178>`_; @mmecthley)
 * Add ``calc_datacube`` method to ``poppy.Instrument`` (`#182 <https://github.com/mperrin/poppy/issues/182>`_; @mperrin)
 * Test for Apple Accelerate more narrowly (`#176 <https://github.com/mperrin/poppy/issues/176>`_; @mperrin)
 * ``Wavefront.display()`` correctly handles ``vmin`` and ``vmax`` args (`#183 <https://github.com/mperrin/poppy/pull/183>`_; @neilzim)
 * Changes to Travis-CI configuration (`#197 <https://github.com/mperrin/poppy/pull/197>`_; @etollerud)
 * Warn on requested field-of-view too large for pupil sampling (`#180 <https://github.com/mperrin/poppy/issues/180>`_; reported by @mmechtley, addressed by @mperrin)
 * Bugfix for ``add_detector`` in ``FresnelOpticalSystem`` (`#193 <https://github.com/mperrin/poppy/pull/193>`_; @maciekgroch)
 * Fixes to unit handling and short-distance propagation in ``FresnelOpticalSystem`` (`#194 <https://github.com/mperrin/poppy/issues/194>`_; @maciekgroch, @douglase, @mperrin)
 * PEP8 renaming for ``poppy.fresnel`` for consistency with the rest of POPPY: ``propagateTo`` becomes ``propagate_to``, ``addPupil`` and ``addImage`` become ``add_pupil`` and ``add_image``, ``inputWavefront`` becomes ``input_wavefront``, ``calcPSF`` becomes ``calc_psf`` (@mperrin)
 * Fix ``display_psf(..., markcentroid=True)`` (`#175 <https://github.com/mperrin/poppy/issues/175>`_, @josephoenix)

.. _rel0.5.0:

0.5.0
-----

*2016 June 10:*

Several moderately large enhancements, involving lots of under-the-hood updates to the code. (*While we have tested this code extensively, it is possible that there may be
some lingering bugs. As always, please let us know of any issues encountered via `the github issues page 
<https://github.com/mperrin/poppy/issues/>`_.*)

 * Increased use of ``astropy.units`` to put physical units on quantities, in
   particular wavelengths, pixel scales, etc. Instead of wavelengths always being
   implicitly in meters, you can now explicitly say e.g. ``wavelength=1*u.micron``, 
   ``wavelength=500*u.nm``, etc. You can also generally use Quantities for 
   arguments to OpticalElement classes, e.g. ``radius=2*u.cm``. This is *optional*; the
   API still accepts bare floating-point numbers which are treated as implicitly in meters.
   (`#145 <https://github.com/mperrin/poppy/issues/145>`_, `#165 <https://github.com/mperrin/poppy/pull/165>`_;
        @mperrin, douglase)
 * The ``getPhasor`` function for all OpticalElements has been refactored to split it into 3
   functions: ``get_transmission`` (for electric field amplitude transmission), ``get_opd``
   (for the optical path difference affectig the phase), and ``get_phasor`` (which combines transmission 
   and OPD into the complex phasor). This division simplifies and makes more flexible the subclassing 
   of optics, since in many cases (such as aperture stops) one only cares about setting either the 
   transmission or the OPD.  Again, there are back compatibility hooks to allow existing code calling 
   the deprecated ``getPhasor`` function to continue working.
   (`#162 <https://github.com/mperrin/poppy/pull/162>`_; @mperrin, josephoenix)
 * Improved capabilities for handling complex coordinate systems:

     * Added new `CoordinateInversion` class to represent a change in orientation of axes, for instance the
       flipping "upside down" of a pupil image after passage through an intermediate image plane. 
     * ``OpticalSystem.input_wavefront()`` became smart enough to check for ``CoordinateInversion`` and ``Rotation`` planes,
       and, if the user has requested a source offset,  adjust the input tilts such that the source will move as requested in
       the final focal plane regardless of intervening coordinate transformations.
     * ``FITSOpticalElement`` gets new options ``flip_x`` and ``flip_y`` to flip orientations of the
       file data.

 * Update many function names for `PEP8 style guide compliance <https://www.python.org/dev/peps/pep-0008/>`_.
   For instance `calc_psf` replaces `calcPSF`.  This was done with back compatible aliases to ensure 
   that existing code continues to run with no changes required at this time, but *at some 
   future point* (but not soon!) the older names will go away, so users are encouranged to migrate to the new names. 
   (@mperrin, josephoenix)

And some smaller enhancements and fixes:

 * New functions for synthesis of OPDs from Zernike coefficients, iterative Zernike expansion on obscured
   apertures for which Zernikes aren't orthonormal, 2x faster optimized computation of Zernike basis sets,
   and computation of hexike basis sets using the alternate ordering of hexikes used by the JWST Wavefront Analysis System
   software.
   (@mperrin)
 * New function for orthonormal Zernike-like basis on arbitrary aperture 
   (`#166 <https://github.com/mperrin/poppy/issues/166>`_; Arthur Vigan)
 * Flip the sign of defocus applied via the ``ThinLens`` class, such that 
   positive defocus means a converging lens and negative defocus means 
   diverging. (`#164 <https://github.com/mperrin/poppy/issues/164>`_; @mperrin)
 * New ``wavefront_display_hint`` optional attribute on OpticalElements in an OpticalSystem allows customization of
   whether phase or intensity is displayed for wavefronts at that plane. Applies to ``calc_psf`` calls 
   with ``display_intermediates=True``. (@mperrin)
 * When displaying wavefront phases, mask out and don't show the phase for any region with intensity less than
   1/100th of the mean intensity of the wavefront. This is to make the display less visually cluttered with near-meaningless
   noise, especially in cases where a Rotation has sprayed numerical interpolation noise outside
   of the true beam. The underlying Wavefront values aren't affected at all, this just pre-filters a copy of
   the phase before sending it to matplotlib.imshow. (@mperrin)
 * remove deprecated parameters in some function calls 
   (`#148 <https://github.com/mperrin/poppy/issues/148>`_; @mperrin)

.. _rel0.4.1:

0.4.1
-----

2016 Apr 4:

Mostly minor bug fixes: 

 * Fix inconsistency between older deprecated ``angle`` parameter to some optic classes versus new ``rotation`` parameter for any AnalyticOpticalElement  (`#140 <https://github.com/mperrin/poppy/issues/140>`_; @kvangorkom, @josephoenix, @mperrin)
 * Update to newer API for ``psutil``  (`#139 <https://github.com/mperrin/poppy/issues/139>`_; Anand Sivaramakrishnan, @mperrin)
 * "measure_strehl" function moved to ``webbpsf`` instead of ``poppy``.  (`#138 <https://github.com/mperrin/poppy/issues/138>`_; Kathryn St.Laurent, @josephoenix, @mperrin)
 * Add special case to handle zero radius pixel in circular BandLimitedOcculter.  (`#137 <https://github.com/mperrin/poppy/issues/137>`_; @kvangorkom, @mperrin)
 * The output FITS header of an `AnalyticOpticalElement`'s `toFITS()` function is now compatible with the input expected by `FITSOpticalElement`. 
 * Better saving and reloading of FFTW wisdom. 
 * Misc minor code cleanup and PEP8 compliance. (`#149 <https://github.com/mperrin/poppy/issues/149>`_; @mperrin)

And a few more significant enhancements:

 * Added `MatrixFTCoronagraph` subclass for fast optimized propagation of coronagraphs with finite fields of view. This is a 
   related variant of the approach used in the `SemiAnalyticCoronagraph` class, suited for
   coronagraphs with a focal plane field mask limiting their field of view, for instance those
   under development for NASA's WFIRST mission. ( `#128 <https://github.com/mperrin/poppy/pull/128>`_; `#147 <https://github.com/mperrin/poppy/pull/147>`_; @neilzim)
 * The `OpticalSystem` class now has `npix` and `pupil_diameter` parameters, consistent with the `FresnelOpticalSystem`.  (`#141 <https://github.com/mperrin/poppy/issues/141>`_; @mperrin)
 * Added `SineWaveWFE` class to represent a periodic phase ripple.

.. _rel0.4.0:

0.4.0
-----

2015 November 20

 * **Major enhancement: the addition of Fresnel propagation** (
   `#95 <https://github.com/mperrin/poppy/issue/95>`_, 
   `#100 <https://github.com/mperrin/poppy/pull/100>`_, 
   `#103 <https://github.com/mperrin/poppy/issue/103>`_, 
   `#106 <https://github.com/mperrin/poppy/issue/106>`_, 
   `#107 <https://github.com/mperrin/poppy/pull/107>`_, 
   `#108 <https://github.com/mperrin/poppy/pull/108>`_, 
   `#113 <https://github.com/mperrin/poppy/pull/113>`_, 
   `#114 <https://github.com/mperrin/poppy/issue/114>`_, 
   `#115 <https://github.com/mperrin/poppy/pull/115>`_, 
   `#100 <https://github.com/mperrin/poppy/pull/100>`_, 
   `#100 <https://github.com/mperrin/poppy/pull/100>`_; @douglase, @mperrin, @josephoenix) *Many thanks to @douglase for the initiative and code contributions that made this happen.* 
 * Improvements to Zernike aberration models (
   `#99 <https://github.com/mperrin/poppy/pull/99>`_, 
   `#110 <https://github.com/mperrin/poppy/pull/110>`_, 
   `#121 <https://github.com/mperrin/poppy/pull/121>`_, 
   `#125 <https://github.com/mperrin/poppy/pull/125>`_; @josephoenix)
 * Consistent framework for applying arbitrary shifts and rotations to any AnalyticOpticalElement 
   (`#7 <https://github.com/mperrin/poppy/pull/7>`_, @mperrin)
 * When reading FITS files, OPD units are now selected based on BUNIT 
   header keyword instead of always being "microns" by default, 
   allowing the units of files to be set properly based on the FITS header.
 * Added infrastructure for including field-dependent aberrations at an optical 
   plane after the entrance pupil (
   `#105 <https://github.com/mperrin/poppy/pull/105>`_, @josephoenix)
 * Improved loading and saving of FFTW wisdom (
   `#116 <https://github.com/mperrin/poppy/issue/116>`_,
   `#120 <https://github.com/mperrin/poppy/issue/120>`_,
   `#122 <https://github.com/mperrin/poppy/issue/122>`_,
   @josephoenix)
 * Allow configurable colormaps and make image origin position consistent
   (`#117 <https://github.com/mperrin/poppy/pull/117>`_, @josephoenix)
 * Wavefront.tilt calls are now recorded in FITS header HISTORY lines 
   (`#123 <https://github.com/mperrin/poppy/pull/123>`_; @josephoenix)
 * Various improvements to unit tests and test infrastructure
   (`#111 <https://github.com/mperrin/poppy/pull/111>`_, 
   `#124 <https://github.com/mperrin/poppy/pull/124>`_, 
   `#126 <https://github.com/mperrin/poppy/pull/126>`_, 
   `#127 <https://github.com/mperrin/poppy/pull/127>`_; @josephoenix, @mperrin)

.. _rel0.3.5:

0.3.5
-----

2015 June 19

 * Now compatible with Python 3.4 in addition to 2.7!  (`#83 <https://github.com/mperrin/poppy/pull/82>`_, @josephoenix)
 * Updated version numbers for dependencies (@josephoenix)
 * Update to most recent astropy package template (@josephoenix)
 * :py:obj:`~poppy.optics.AsymmetricSecondaryObscuration` enhanced to allow secondary mirror supports offset from the center of the optical system. (@mperrin)
 * New optic :py:obj:`~poppy.optics.AnnularFieldStop` that defines a circular field stop with an (optional) opaque circular center region (@mperrin)
 * display() functions now return Matplotlib.Axes instances to the calling functions.
 * :py:obj:`~poppy.optics.FITSOpticalElement` will now determine if you are initializing a pupil plane optic or image plane optic based on the presence of a ``PUPLSCAL`` or ``PIXSCALE`` header keyword in the supplied transmission or OPD files (with the transmission file header taking precedence). (`#97 <https://github.com/mperrin/poppy/pull/97>`_, @josephoenix)
 * The :py:func:`poppy.zernike.zernike` function now actually returns a NumPy masked array when called with ``mask_array=True``
 * poppy.optics.ZernikeAberration and poppy.optics.ParameterizedAberration have been moved to poppy.wfe and renamed :py:obj:`~poppy.wfe.ZernikeWFE` and :py:obj:`~poppy.wfe.ParameterizedWFE`. Also, ZernikeWFE now takes an iterable of Zernike coefficients instead of (n, m, k) tuples.
 * Various small documentation updates
 * Bug fixes for: 

   * redundant colorbar display (`#82 <https://github.com/mperrin/poppy/pull/82>`_)
   * Unnecessary DeprecationWarnings in :py:func:`poppy.utils.imshow_with_mouseover` (`#53 <https://github.com/mperrin/poppy/issues/53>`_)
   * Error in saving intermediate planes during calculation (`#81 <https://github.com/mperrin/poppy/issues/81>`_)
   * Multiprocessing causes Python to hang if used with Apple Accelerate (`#23 <https://github.com/mperrin/poppy/issues/23>`_, n.b. the fix depends on Python 3.4)
   * Copy in-memory FITS HDULists that are passed in to FITSOpticalElement so that in-place modifications don't affect the caller's copy of the data (`#89 <https://github.com/mperrin/poppy/issues/89>`_)
   * Error in the :py:func:`poppy.utils.measure_EE` function produced values for the edges of the radial bins that were too large, biasing EE values and leading to weird interpolation behavior near r = 0. (`#96 <https://github.com/mperrin/poppy/pull/96>`_)

.. _rel0.3.4:

0.3.4
-----

2015 February 17

 * Continued improvement in unit testing (@mperrin, @josephoenix)
 * Continued improvement in documentation (@josephoenix, @mperrin)
 * Functions such as addImage, addPupil now also return a reference to the added optic, for convenience (@josephoenix)
 * Multiprocessing code and semi-analytic coronagraph method can now return intermediate wavefront planes (@josephoenix)
 * Display methods for radial profile and encircled energy gain a normalization keyword (@douglase)
 * matrixDFT: refactor into unified function for all centering types (@josephoenix)
 * matrixDFT bug fix for axes parity flip versus FFT transforms (Anand Sivaramakrishnan, @josephoenix, @mperrin)
 * Bug fix: Instrument class can now pass through dict or tuple sources to OpticalSystem calcPSF (@mperrin)
 * Bug fix: InverseTransmission class shape property works now. (@mperrin)
 * Refactor instrument validateConfig method and calling path (@josephoenix)
 * Code cleanup and rebalancing where lines had been blurred between poppy and webbpsf (@josephoenix, @mperrin)
 * Misc packaging infrastructure improvements (@embray)
 * Updated to Astropy package helpers 0.4.4
 * Set up integration with Travis CI for continuous testing. See https://travis-ci.org/mperrin/poppy
 

.. _rel0.3.3:

0.3.3
-----

2014 Nov

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

.. _rel0.3.2:

0.3.2
-----

Released 2014 Sept 8

 * Bug fix: Correct pupil orientation for inverse transformed pupils using PyFFTW so that it is consistent with the result using numpy FFT.

.. _rel0.3.1:

0.3.1
-----

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

.. _rel0.3.0:

0.3.0
-----

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

.. _rel0.2.8:

0.2.8
-----

 * First release as a standalone package (previously was integrated as part of webbpsf). See the release notes for WebbPSF for prior verions.
 * switched package building to use `setuptools` instead of `distutils`/`stsci_distutils_hack`
 * new `Instrument` class in poppy provides much of the functionality previously in JWInstrument, to make it
   easier to model generic non-JWST instruments using this code.
