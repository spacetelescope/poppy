.. _whatsnew:

Release Notes
===============

For a list of contributors, see :ref:`about`.

1.0.1
-----

.. _rel1.0.1:

*2021 December 9*

This is a very minor re-release, to fix some documentation formatting and release packaging issues with the 1.0.0 release. No changes in functionality.


1.0.0
-----

.. _rel1.0.0:

*2021 December 7*

This is a major release with significant enhancements and changes, in particular with regards to changes in wavefront sign convention representations. 

.. admonition:: Changes and Clarifications in Signs for Wavefront Error and Phase

    **Some sign conventions for wavefront error and optical phase have changed in this version of poppy**

    This release includes optical algorithm updates after a thorough audit and cross-check of sign conventions for phase and wavefront error, disambiguating portions of the
    sign conventions and code to ensure consistency with several other relevant optical modeling packages. Poppy now strictly follows the sign conventions as advocated in e.g.
    Wyant and Creath's `Basic Wavefront Aberration Theory for Optical Metrology <https://ui.adsabs.harvard.edu/abs/1992aooe...11....2W/abstract>`_ (or see `here <https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/03-BasicAberrations_and_Optical_Testing.pdf>`_). This makes poppy consistent with the convention more widely used in optical metrology and other optical software such as Code V; however this is not consistent with some other reference such as Goodman's classic text *Fourier Optics*.

    To achieve that consistency, *this is a partially back-incompatible release*, with
    changes in the signs of complex exponentials in some Fourier propagation calculations. Depending on your use case this may result in some changes in output PSFs or
    different signs or orientations from prior results.

    See `Sign Conventions for Coordinates, Phase, and Wavefront Error <https://poppy-optics.readthedocs.io/en/latest/sign_conventions_for_coordinates_and_phase.html>`_ for details, discussion, and demonstration. 
    
    Many thanks to Derek
    Sabatke (Ball Aerospace); Matthew Bergkoetter, Alden Jurling, and Tom Zielinski (NASA GSFC); and
    Randal Telfer (STScI) for invaluable discussions and aid in getting these
    details onto a more rigorous footing.


**API Changes:**
  * Several functions in the Zernike module were renamed for clarity, in particular the prior ``opd_expand`` is now :py:func:`~poppy.zernike.decompose_opd`, and ``opd_from_zernikes`` is now :py:func:`~poppy.zernike.compose_opd_from_basis`.
    The prior function names also continue to work as aliases for backwards compatibility.  (:pr:`471` by :user:`mperrin`)

**New Functionality:**
 * New class :py:obj:`~poppy.TipTiltStage`, which allows putting additional tip-tilt on any arbitrary optic, and adjusting/controlling the tip and tilt. See `here <https://poppy-optics.readthedocs.io/en/latest/available_optics.html#Tip-Tilt-Stage>`_ for example. (:pr:`414` by :user:`mperrin`)
 * New class :py:obj:`~poppy.CircularSegmentedDeformableMirror`, which models an aperture comprising several individually-controllable circular mirrors. See `here <https://poppy-optics.readthedocs.io/en/latest/available_optics.html#Circularly-Segmented-Deformable-Mirrors>`_ for example. (:pr:`407` and :pr:`424` by :user:`Teusia`)
 * New class :py:obj:`~poppy.KolmogorovWFE`, which models the phase distortions in a turbulent atmosphere. See `this notebook <https://github.com/spacetelescope/poppy/blob/develop/notebooks/Propagation%20through%20turbulent%20atmosphere.ipynb>`_ for details. (:pr:`437` by :user:`DaPhil`)
 * New class :py:obj:`~poppy.ThermalBloomingWFE`, which models the change in WFE from heating of air (or other transmission medium) due to high powered laser beams. See `this notebook <https://github.com/spacetelescope/poppy/blob/develop/notebooks/Thermal%20Blooming%20Demo.ipynb>`_ for details. (:pr:`438` by :user:`DaPhil`)


**Other enhancements and fixes:**
 * Wavefront instances gain a `.wfe` attribute for the wavefront error in meters (computed from phase, so it will wrap if wavefront error exceeds +- 0.5 waves), and the wavefront display method can display wfe as well as intensity and phase.
 * Faster algorithm for calculations in the :py:func:`~poppy.zernike.opd_from_zernikes` function (:pr:`400` by :user:`grbrady`). Run time of this function was reduced roughly in half.
 * Various performance enhancements in FFTs, array rotations, zero padding, and array indexing in certain cases (:pr:`394`, :pr:`398`, :pr:`411`, :pr:`413` by :user:`mperrin`)
 * Bug fix to a sign inconsistency in wavefront rotation: While the documentation states that positive rotations are counterclockwise, the code had the other sign. Updated code to match the documented behavior, which also matches the rotation convention for optical elements. (:pr:`411` by :user:`mperrin`)
 * More robust algorithm for offset sources in optical systems with coordinate rotations and inversions (:pr:`420` by :user:`mperrin`). This ensures the correct sign of tilt is applied in the entrance pupil plane to achieve the requested source position in the output image plane.
 * Added ``inwave=`` parameter to ``calc_psf`` and related functions, for both Fresnel and Fraunhofer propagation types, to allow providing a custom input wavefront, for instance the output of some prior upstream calculation. If provided, this is used instead of the default input wavefront (a plane wave of uniform intensity). (:pr:`402` by :user:`kian1377`)
 * Improved support for astropy Quantities, including being able to specify monochromatic wavelengths using Quantities of wavelength, and to specify optic shifts using Quantities in length or angular units as appropriate (:pr:`445`, :pr:`447` by :user:`mperrin`).



**Software Infrastructure Updates and Internals:**
 * Continuous integration system migrated to Github Actions, replacing previous use of Travis CI. (:pr:`434` by :user:`shanosborne`)
 * Updates to recommended (not minimum) dependency versions to track latest numpy, scipy, etc (various PRs by :user:`shanosborne`)
 * Updates to minimum dependency versions, generally to upstream releases as of mid-2020. (:pr:`415`, :pr:`472` by :user:`mperrin`)
 * Swap to use of base ``synphot`` rather than ``stsynphot`` package, to avoid dependency on many GBs of reference data. (:pr:`421` by :user:`mperrin`)


0.9.2
-----

.. _rel0.9.2:

*2021 Feb 11*

This release includes several updated optical element classes, bug fixes, and improved documentation. This is intended as a maintenance release shortly before v 1.0 which will introduce some backwards-incompatible changes. 

**New Functionality:**
 * New OpticalElement classes for ScalarOpticalPathDifference, LetterFAperture, and LetterFOpticalPathDifference. (:pr:`386` by :user:`mperrin`)
 * Improved `radial_profile` function to allow measurement of partial profiles for sources offset outside the FOV (:pr:`380` by :user:`mperrin`)
 * Improved the CompoundAnalyticOptic class to correctly handle OPDS for compound optics with multiple non-overlapping apertures. (:pr:`386` by :user:`mperrin`)

**Other enhancements and fixes:**
 * The ShackHartmannWavefrontSensor class was refactored and improved . (:pr:`369` by :user:`fanpeng-kong`). And a unit test case for this class was added (:pr:`376` by :user:`remorgan123` in collaboration with :user:`douglase`)
 * Expanded documentation and example code for usage of astropy Units. (:pr:`374`, :pr:`378` by :user:`mperrin`; with thanks to :user:`keflavich’ and  :user:`mcbeth`)
 * Made the HexagonalSegmentedDeformableMirror class consistent with ContinuousDeformableMirror in having an 'include_factor_of_two' parameter, for control in physical surface versus wavefront error units
 * Bug fix for influence functions of rotated hexagonally segmented deformable mirrors. (:pr:`371` by :user:`mperrin`)
 * Bug fix for FWHM measurement on integer data type images. (:pr:`368` by :user:`kjbrooks`)
 * Bug fix for StatisticalPSDWFE to avoid side effects from changing global numpy random generator state. (:pr:`377` by :user:`ivalaginja`)
 * Bug fix for image display in cases using angular coordinates in units other than arc seconds. (:pr:`378` by :user:`mperrin`; with thanks to :user:`mcbeth`)


**Software Infrastructure Updates and Internals:**
 * The minimum numpy version is now 1.16. (:pr:`356` by :user:`mperrin`)
 * The main branches were renamed/relabeled to ’stable’  (rather than ‘master’) and ‘develop’. (:pr:`361`, :pr:`370` by :user:`mperrin`)
 * Updates to Travis CI settings. (:pr:`367`, :pr:`395` by :user:`shanosborne`)
 * Avoid deprecated modification of matplotlib colormaps (:pr:`379` by :user:`spacegal-spiff`)
 * Minor doc string clarification for get_opd (:pr:`381` by :user:`douglase`)
 * Remove unused parameter to Detector class (:pr:`385` by :user:`mperrin`)
 * Updates to meet STScI INS's JWST Software Standards (:pr:`390` by :user:`shanosborne`)
 * Use Github's Dependabot to test and update dependencies (:pr:`391: by :user:`shanosborne`)



0.9.1
-----

.. _rel0.9.1:

*2020 June 22*

This is a minor release primarily for updates in packaging infrastructure, plus a handful of small enhancements related to datacubes, segmented apertures, and new functionality for subsampled optics.

**New Functionality:**
 * Adds new `Subapertures` class for modeling subsampled optics (i.e. optics that have multiple spatially disjoint output beams). Adds `ShackHartmannWavefrontSensor` class to model that type of sensor. See `this notebook <https://github.com/spacetelescope/poppy/blob/develop/notebooks/Shack%20Hartmann%20Wavefront%20Sensor%20Demo.ipynb>`_ for details and example codes. (:pr:`346` thanks to :user:`remorgan01` and :user:`douglase`)

**Other enhancements and fixes:**
 * `calc_datacube` function now allows `nwavelengths>100`, removing a prior limitation of this function. (:pr:`351` by :user:`ojustino`)
 * `radial_profile` function can now be applied to datacubes, with a `slice` keyword to specify which slice of the cube should be examined. (:pr:`352` by :user:`mperrin`)
 * Improved the Zernike basis expansion function for segmented apertures, `opd_expand_segments`, to allow optional masking out of pixels at the segment borders. This can be useful in some circumstances for avoiding edge effects from partially illuminated pixels or interpolation artifacts when evaluating Zernike or hexike coefficients per segment. (:pr:`353` by :user:`mperrin`)
 * Allows `Segmented_PTT_Basis` to pass through keyword arguments to parent class `MultiHexagonAperture`, in particular for selecting/excluding particular segments from the apreture geometry. (:pr:`357` by :user:`kjbrooks`)
 * Fix a log string formatting bug encountered in MFT propagation under certain conditions (:pr:`360` by :user:`mperrin`)

**Software Infrastructure Updates and Internals:**
 * Removed dependency on the deprecated astropy-helpers package framework. (:pr:`349` by :user:`shanosborne`). Fixes :issue:`355`.
 * Switched code coverage CI service to codecov.io. (:pr:`349` by :user:`shanosborne`)
 * The minimum Python version is now 3.6. (:pr:`356` by :user:`mperrin`)

0.9.0
-----

.. _rel0.9.0:

*2019 Nov 25*

**New Functionality:**
 * **Chaining together multiple propagations calculations:** Multiple `OpticalSystem` instances can now be chained together into a `CompoundOpticalSystem`. This includes mixed
   propagations that are partially Fresnel and partially Fraunhofer; Wavefront objects will be cast between types as
   needed. (:pr:`290` by :user:`mperrin`)
 * **Gray pixel subsampling of apertures:** Implemented "gray pixel" sampling for circular apertures and stops, providing more precise models of aperture edges.
   For circular apertures this is done  using a fast analytic geometry implementation adapted from open-source IDL code
   originally by Marc Buie. (:pr:`325` by :user:`mperrin`, using Python code contributed by :user:`astrofitz`).
   For subpixel / gray pixel sampling of other optics in general, a new function `fixed_sampling_optic` takes any
   AnalyticOpticalElement and returns an equivalent ArrayOpticalElement with fixed sampling. This is useful for instance
   for taking a computationally-slow optic such as MultiHexagonAperture and saving a discretized version for future
   faster use. (:pr:`307` by :user:`mperrin`)
 * **Modeling tilted optics:** New feature to model geometric projection (cosine scaling) of inclined optics, by setting an  `inclination_x` or
   `inclination_y` attribute to the tilt angle in degrees. For instance `inclination_x=30` will tilt an optic by 30
   degrees around the X axis, and thus compress its apparent size in the Y axis by cosine(30 deg). Note, this
   transformation only applies the cosine scaling to the optic's appearance, and does *not* introduce wavefront for
   tilt. (:pr:`329` by :user:`mperrin`)

 * **Many improvements to the Continuous Deformable Mirror class**: 

    * Enhance model of DM actuator influence functions for more precise subpixel spacing of DM actuators, rather than
      pokes separated by integer pixel spacing. This applies to the 'convolution by influence function' method for
      modeling DMs (:pr:`329` by :user:`mperrin`)
    * Support distinct radii for the active controllable mirror size and the reflective mirror size (:pr:`293` by :user:`mperrin`)
    * ContinuousDeformableMirror now supports `shift_x` and `shift_y` to translate / decenter the DM, consistent with
      other optical element classes. (:pr:`307` by :user:`mperrin`)
    * ContinuousDeformableMirror now also supports `flip_x` and `flip_y` attributes to flip its orientation along one or
      both axes, as well as the new `inclination_x` and `inclination_y` attributes for geometric projection.

 * **Improved models of certain kinds of wavefront error:**

   * New class `StatisticalPSDWFE` that models random wavefront errors described by a power spectral density, as is
     commonly used to specify and measure typical polishing residuals in optics. (:pr:`315` by :user:`ivalaginja`;
     :pr:`317` by :user:`mperrin`)
   * `FITSOpticalElement` can now support wavelength-independent phase maps defined in radians, for instance for modeling
     Pancharatnam-Berry phase as used in certain vector coronagraph masks. (:pr:`306` by :user:`joseph-long`)

 * `add_optic` in Fresnel systems can now insert optics at any index into an optical system, rather than just appending
   at the end (:pr:`298` by :user:`sdwill`)

**Software Infrastructure Updates and Internals:**
 * PR :pr:`290` for CompoundOpticalSystem involved refactoring the Wavefront and FresnelWavefront classes to both be child classes of a new abstract base class BaseWavefront. This change should be transparent for most/all users and requires no changes in calling code.
 * PR :pr:`306` for wavelength-independent phase subsequently required refactoring of the optical element display code to correctly handle all cases. As a result the display code internals were clarified and made more consistent. (:pr:`314` and :pr:`321`  by :user:`mperrin` with contributions from :user:`ivalaginja` and :user:`shanosborne`). Again this change should be transparent for users. 
 * Removed deprecated / unused decorator function in WFE classes, making their `get_opd` function API consistent with the rest of poppy. (:pr:`322` by :user:`mperrin`)
 * Accomodate some upstream changes in astropy (:pr:`294` by :user:`shanosborne`, :pr:`330` by :user:`mperrin`)
 * The `poppy.Instrument._get_optical_system` function, which has heretofore been an internal method (private, starting with
   underscore) of the Instrument class, has been promoted to a public part of the API as
   `Instrument.get_optical_system()`.
 * Note, minimum supported versions of some upstream packages such as numpy and matplotlib have been updated.

**Bug Fixes and Misc Improvements:**
 * Correctly assign BUNIT keyword after rescaling OPDs (:issue:`285`, :pr:`286` by :user:`laurenmarietta`).
 * New header keywords in output PSF files for `OPD_FILE` and `OPDSLICE` to more cleanly record the information
   previously stored together in the `PUPILOPD` keyword (:pr:`316` by :user:`mperrin`)
 * Update docs and example notebooks to replace deprecated function names with the current ones (:pr:`288` by :user:`corcoted`).
 * Improvements in resampling wavefronts onto Detector instances, particularly in cases where the wavefront is already at the right plane so no propagation is needed. (Part of :pr:`290` by :user:`mperrin`, then further improved in :pr:`304` by :user:`sdwill`)
 * Allow passthrough of "normalize" keyword to measure_ee and measure_radius_at_ee functions (:pr:`333` by
   :user:`mperrin`; :issue:`332` by :user:`ariedel`)
 * Fix `wavefront.as_fits` complex wavefront output option (:pr:`293` by :user:`mperrin`)
 * Stricter checking for consistent wavefront type and size parameters when summing wavefronts (:pr:`313` and :pr:`326` by :user:`mperrin`)
 * Fix an issue with MultiHexagonAperture in the specific case of 3 rings of hexes (:issue:`303` by :user:`LucasMarquis` and :user:`FredericCassaing`; :pr:`307` by :user:`mperrin`)
 * Fix an issue with BaseWavefront class refactor (:pr:`311` by :user:`douglase` and :user:`jlumbres`)
 * Fix an issue with indexing in HexSegmentedDeformableMirror when missing the center segment (:issue:`318` by :user:`ivalaginja`; :pr:`320` by :user:`mperrin`)
 * Fix title display by OpticalElement.display function (:pr:`299` by :user:`shanosborne`)
 * Fix display issue in SemiAnalyticCoronagraph class (:pr:`324` by :user:`mperrin`).
 * Small improvements in some display labels (:pr:`307` by :user:`mperrin`)

*Note*, the new functionality for gray pixel representation of circular apertures does not work precisely for elliptical
apertures such as from inclined optics. You may see warnings about this in cases when you use `inclination_y` or
`inclination_x` attributes on a circular aperture. This warning is generally benign; the calculation is still more
accurate than it would be without the subpixel sampling, though not perfectly precise. This known issue will likely be
improved upon in a future release. 


0.8.0
-----

.. _rel0.8.0:

*2018 December 15*

.. admonition:: Py2.7 support and deprecated function names removed

    As previously announced, support for Python 2 has been removed in this release,
    as have the deprecated non-PEP8-compliant function names.

**New Functionality:**

 * The `zernike` submodule has gained better support for dealing with wavefront error defined over
   segmented apertures. The `Segment_Piston_Basis` and `Segment_PTT_Basis` classes implement basis
   functions for piston-only or piston/tip/tilt motions of arbitrary numbers of hexagonal segments.
   The `opd_expand_segments` function implements a version of the `opd_expand_orthonormal` algorithm
   that has been updated to correctly handle disjoint (non-overlapping support) basis functions defined on
   individual segments. (mperrin)
 * Add new `KnifeEdge` optic class representing a sharp opaque half-plane, and a `CircularPhaseMask` representing a circular region with constant optical path difference. (#273, @mperrin)
 * Fresnel propagation can now automatically resample wavefronts onto the right pixel scales at Detector objects,
   same as Fraunhofer propagation. (#242, #264, @mperrin)
 * The `display_psf` function now can also handle datacubes produced by `calc_datacube` (#265, @mperrin)

**Documentation:**

 * Various documentation improvements and additions, in particular including a new "Available Optics" page showing
   visual examples of all the available optical element classes.

**Bug Fixes and Software Infrastructure Updates:**

 * Removal of Python 2 compatibility code, Python 2 test cases on Travis, and similar (#239, @mperrin)
 * Removal of deprecated non-PEP8 function names (@mperrin)
 * Fix for output PSF formatting to better handle variable numbers of extensions (#219, @shanosborne)
 * Fix for FITSOpticalElement opd_index parameter for selecting slices in datacubes (@mperrin)
 * Fix inconsistent sign of rotations for FITSOpticalElements vs. other optics (#275, @mperrin)
 * Cleaned up the logic for auto-choosing input wavefront array sizes (#274, @mperrin)
 * Updates to Travis doc build setup (#270, @mperrin, robelgeda)
 * Update package organization and documentation theme for consistency with current STScI package template (#267, #268, #278, @robelgeda)
 * More comprehensive unit tests for Fresnel propagation. (#191, #251, #264, @mperrin)
 * Update astropy-helpers to current version, and install bootstrap script too (@mperrin, @jhunkeler)
 * Minor: doc string correction in FresnelWavefront (@sdwill), fix typo in some error messages (#255, @douglase),
   update some deprecated logging function calls (@mperrin).

0.7.0
-----

.. _rel0.7.0:

*2018 May 30*

.. admonition:: Python version support: Future releases will require Python 3.

    Please note, this is the *final* release to support Python 2.7. All
    future releases will require Python 3.5+. See `here <https://python3statement.org>`_ for more information on migrating to Python 3.

.. admonition:: Deprecated function names will go away in next release.

    This is also the *final* release to support the older, deprecated
    function names with mixed case that are not compatible with the Python PEP8
    style guide (e.g. ``calcPSF`` instead of ``calc_psf``, etc). Future versions will
    require the use of the newer syntax.


**Performance Improvements:**

 * Major addition of GPU-accelerated calculations for FFTs and related operations in many
   propagation calculations. GPU support is provided for both CUDA (NVidia GPUs) and OpenCL (AMD
   GPUs); the CUDA implementation currently accelerates a slightly wider range of operations.
   Obtaining optimal performance, and understanding tradeoffs between numpy, FFTW, and CUDA/OpenCL,
   will in general require tests on your particular hardware. As part of this, much of the FFT
   infrastructure has been refactored out of the Wavefront classes and into utility functions in
   `accel_math.py`.  This functionality and the resulting gains in performance are described more in
   Douglas & Perrin, Proc. SPIE 2018.  (`#239 <https://github.com/spacetelescope/poppy/pull/239>`_,
   @douglase), (`#250 <https://github.com/spacetelescope/poppy/pull/250>`_, @mperrin and @douglase).
 * Additional performance improvements to other aspects of calculations using the `numexpr` package.
   Numexpr is now a *highly recommended* optional installation. It may well become a requirement in
   a future release.  (`#239 <https://github.com/spacetelescope/poppy/pull/239>`_, `#245
   <https://github.com/spacetelescope/poppy/pull/245>`_, @douglase)
 * More efficient display of AnalyticOptics, avoiding unnecessary repetition of optics sampling.
   (@mperrin)
 * Single-precision floating point mode added, for cases that do not require the default double
   precision floating point and can benefit from the increased speed. (Experimental / beta; some
   intermediate calculations may still be done in double precision, thus reducing speed gains).

**New Functionality:**

 * New `PhysicalFresnelWavefront` class that uses physical units for the wavefront (e.g.
   volts/meter) and intensity (watts). See `this notebook
   <https://github.com/spacetelescope/poppy/blob/stable/notebooks/Physical%20Units%20Demo.ipynb>`_ for
   examples and further discussion.  (`#248 <https://github.com/spacetelescope/poppy/pull/248>`, @daphil).
 * `calc_psf` gains a new parameter to request returning the complex wavefront (`#234
   <https://github.com/spacetelescope/poppy/pull/234>`_,@douglase).
 * Improved handling of irregular apertures in WFE basis functions (`zernike_basis`, `hexike_basis`,
   etc.) and the `opd_expand`/`opd_expand_nonorthonormal` fitting functions (@mperrin).
 * Added new function `measure_radius_at_ee` which finds the radius at which a PSF achieves some
   given amount of encircled energy; in some sense an inverse to `measure_ee`. (`#244
   <https://github.com/spacetelescope/poppy/pull/244>`_, @shanosborne)
 * Much improved algorithm for `measure_fwhm`: the function now works by fitting a Gaussian rather
   than interpolating between a radial profile on fixed sampling. This yields much better results on
   low-sampled or under-sampled PSFs. (@mperrin)
 * Add `ArrayOpticalElement` class, providing a cleaner interface for creating arbitrary optics at
   runtime by generating numpy ndarrays on the fly and packing them into an ArrayOpticalElement.
   (@mperrin)
 * Added new classes for deformable mirrors, including both `ContinuousDeformableMirror` and
   `HexSegmentedDeformableMirror` (@mperrin).

**Bug Fixes and Software Infrastructure Updates:**

 * The Instrument class methods and related API were updated to PEP8-compliant names. Old names
   remain for back compatibility, but are deprecated and will be removed in the next release.
   Related code cleanup for better PEP8 compliance. (@mperrin)
 * Substantial update to semi-analytic fast coronagraph propagation to make it more flexible about
   optical plane setup. Fixes #169 (`#169 <https://github.com/spacetelescope/poppy/issues/169>`_, @mperrin)
 * Fix for integer vs floating point division when padding array sizes in some circumstances (`#235
   <https://github.com/spacetelescope/poppy/issues/235>`_, @exowanderer, @mperrin)
 * Fix for aperture clipping in `zernike.arbitrary_basis` (`#241
   <https://github.com/spacetelescope/poppy/pull/241>`_, @kvangorkom)
 * Fix / documentation fix for divergence angle in the Fresnel code (`#237
   <https://github.com/spacetelescope/poppy/pull/237>`_, @douglase). Note, the `divergence` function now
   returns the *half angle* rather than the *full angle*.
 * Fix for `markcentroid` and `imagecrop` parameters conflicting in some cases in `display_psf`
   (`#231 <https://github.com/spacetelescope/poppy/pull/231>`_, @mperrin)
 * For FITSOpticalElements with both shift and rotation set, apply the rotation first and then the
   shift for more intuitive UI (@mperrin)
 * Misc minor doc and logging fixes  (@mperrin)
 * Increment minimal required astropy version to 1.3, and minimal required numpy version to 1.10;
   and various related Travis CI setup updates. Also added numexpr test case to Travis. (@mperrin)
 * Improved unit test for Fresnel model of Hubble Space Telescope, to reduce memory usage and avoid
   CI hangs on Travis.
 * Update `astropy-helpers` submodule to current version; necessary for compatibility with recent
   Sphinx releases. (@mperrin)

.. _rel0.6.1:

0.6.1
-----

*2017 August 11*

 * Update ``ah_bootstrap.py`` to avoid an issue where POPPY would not successfully install when pulled in as a dependency by another package (@josephoenix)

.. _rel0.6.0:

0.6.0
-----

*2017 August 10*

 * WavefrontError and subclasses now handle tilts and shifts correctly (`#229 <https://github.com/spacetelescope/poppy/issues/229>`_, @mperrin) Thanks @corcoted for reporting!
 * Fix the ``test_zernikes_rms`` test case to correctly take the absolute value of the RMS error, support ``outside=`` for ``hexike_basis``, enforce which arguments are required for ``zernike()``. (`#223 <https://github.com/spacetelescope/poppy/issues/223>`_, @mperrin) Thanks to @kvangorkom for reporting!
 * Bug fix for stricter Quantity behavior (``UnitTypeError``) in Astropy 2.0 (@mperrin)
 * Added an optional parameter "mergemode" to CompoundAnalyticOptic which provides two ways to combine AnalyticOptics: ``mergemode="and"`` is the previous behavior (and new default), ``mergemode="or"`` adds the transmissions of the optics, correcting for any overlap. (`#227 <https://github.com/spacetelescope/poppy/pull/227>`_, @corcoted)
 * Add HexagonFieldStop optic (useful for making hexagon image masks for JWST WFSC, among other misc tasks.) (@mperrin)
 * Fix behavior where ``zernike.arbitrary_basis`` would sometimes clip apertures (`#222 <https://github.com/spacetelescope/poppy/pull/222>`_, @kvangorkom)
 * Fix ``propagate_direct`` in fresnel wavefront as described in issue `#216 <https://github.com/spacetelescope/poppy/issues/216>_` (`#218 <https://github.com/mperrin/poppy/pull/218>`_, @maciekgroch)
 * ``display_ee()`` was not passing the ``ext=`` argument through to ``radial_profile()``, but now it does. (`#220 <https://github.com/spacetelescope/poppy/pull/220>`_, @josephoenix)
 * Fix displaying planes where ``what='amplitude'`` (`#217 <https://github.com/spacetelescope/poppy/pull/217>`_, @maciekgroch)
 * Fix handling of FITSOpticalElement big-endian arrays to match recent changes in SciPy (@mperrin) Thanks to @douglase for reporting!
 * ``radial_profile`` now handles ``nan`` values in radial standard deviations (`#214 <https://github.com/spacetelescope/poppy/pull/214>`_, @douglase)
 * The FITS header keywords that are meaningful to POPPY are now documented in :doc:`fitsheaders` and a new ``PIXUNIT`` keyword encodes "units of the pixels in the header, typically either *arcsecond* or *meter*" (`#205 <https://github.com/spacetelescope/poppy/pull/205>`_, @douglase)
 * A typo in the handling of the ``markcentroid`` argument to ``display_psf`` is now fixed (so the argument can be set ``True``) (`#211 <https://github.com/spacetelescope/poppy/pull/211>`_, @josephoenix)
 * ``radial_profile`` now accepts an optional ``pa_range=`` argument to specify the [min, max] position angles to be included in the radial profile. (@mperrin)
 * Fixes in POPPY to account for the fact that NumPy 1.12+ raises an ``IndexError`` when non-integers are used to index an array (`#203 <https://github.com/spacetelescope/poppy/pull/203>`_, @kmdouglass)
 * POPPY demonstration notebooks have been refreshed by @douglase to match output of the current code

.. _rel0.5.1:

0.5.1
-----

*2016 October 28*

 * Fix ConfigParser import (see `astropy/package-template#172 <https://github.com/astropy/package-template/pull/172>`_)
 * Fixes to formatting of ``astropy.units.Quantity`` values (`#171 <https://github.com/spacetelescope/poppy/issues/171>`_, `#174 <https://github.com/mperrin/poppy/pull/174>`_, `#179 <https://github.com/mperrin/poppy/pull/174>`_; @josephoenix, @neilzim)
 * Fixes to ``fftw_save_wisdom`` and ``fftw_load_wisdom`` (`#177 <https://github.com/spacetelescope/poppy/issues/177>`_, `#178 <https://github.com/mperrin/poppy/pull/178>`_; @mmecthley)
 * Add ``calc_datacube`` method to ``poppy.Instrument`` (`#182 <https://github.com/spacetelescope/poppy/issues/182>`_; @mperrin)
 * Test for Apple Accelerate more narrowly (`#176 <https://github.com/spacetelescope/poppy/issues/176>`_; @mperrin)
 * ``Wavefront.display()`` correctly handles ``vmin`` and ``vmax`` args (`#183 <https://github.com/spacetelescope/poppy/pull/183>`_; @neilzim)
 * Changes to Travis-CI configuration (`#197 <https://github.com/spacetelescope/poppy/pull/197>`_; @etollerud)
 * Warn on requested field-of-view too large for pupil sampling (`#180 <https://github.com/spacetelescope/poppy/issues/180>`_; reported by @mmechtley, addressed by @mperrin)
 * Bugfix for ``add_detector`` in ``FresnelOpticalSystem`` (`#193 <https://github.com/spacetelescope/poppy/pull/193>`_; @maciekgroch)
 * Fixes to unit handling and short-distance propagation in ``FresnelOpticalSystem`` (`#194 <https://github.com/spacetelescope/poppy/issues/194>`_; @maciekgroch, @douglase, @mperrin)
 * PEP8 renaming for ``poppy.fresnel`` for consistency with the rest of POPPY: ``propagateTo`` becomes ``propagate_to``, ``addPupil`` and ``addImage`` become ``add_pupil`` and ``add_image``, ``inputWavefront`` becomes ``input_wavefront``, ``calcPSF`` becomes ``calc_psf`` (@mperrin)
 * Fix ``display_psf(..., markcentroid=True)`` (`#175 <https://github.com/spacetelescope/poppy/issues/175>`_, @josephoenix)

.. _rel0.5.0:

0.5.0
-----

*2016 June 10*

Several moderately large enhancements, involving lots of under-the-hood updates to the code. (*While we have tested this code extensively, it is possible that there may be
some lingering bugs. As always, please let us know of any issues encountered via `the github issues page
<https://github.com/spacetelescope/poppy/issues/>`_.*)

 * Increased use of ``astropy.units`` to put physical units on quantities, in
   particular wavelengths, pixel scales, etc. Instead of wavelengths always being
   implicitly in meters, you can now explicitly say e.g. ``wavelength=1*u.micron``,
   ``wavelength=500*u.nm``, etc. You can also generally use Quantities for
   arguments to OpticalElement classes, e.g. ``radius=2*u.cm``. This is *optional*; the
   API still accepts bare floating-point numbers which are treated as implicitly in meters.
   (`#145 <https://github.com/spacetelescope/poppy/issues/145>`_, `#165 <https://github.com/mperrin/poppy/pull/165>`_; @mperrin, douglase)
 * The ``getPhasor`` function for all OpticalElements has been refactored to split it into 3
   functions: ``get_transmission`` (for electric field amplitude transmission), ``get_opd``
   (for the optical path difference affectig the phase), and ``get_phasor`` (which combines transmission
   and OPD into the complex phasor). This division simplifies and makes more flexible the subclassing
   of optics, since in many cases (such as aperture stops) one only cares about setting either the
   transmission or the OPD.  Again, there are back compatibility hooks to allow existing code calling
   the deprecated ``getPhasor`` function to continue working.
   (`#162 <https://github.com/spacetelescope/poppy/pull/162>`_; @mperrin, josephoenix)
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
   (`#166 <https://github.com/spacetelescope/poppy/issues/166>`_; Arthur Vigan)
 * Flip the sign of defocus applied via the ``ThinLens`` class, such that
   positive defocus means a converging lens and negative defocus means
   diverging. (`#164 <https://github.com/spacetelescope/poppy/issues/164>`_; @mperrin)
 * New ``wavefront_display_hint`` optional attribute on OpticalElements in an OpticalSystem allows customization of
   whether phase or intensity is displayed for wavefronts at that plane. Applies to ``calc_psf`` calls
   with ``display_intermediates=True``. (@mperrin)
 * When displaying wavefront phases, mask out and don't show the phase for any region with intensity less than
   1/100th of the mean intensity of the wavefront. This is to make the display less visually cluttered with near-meaningless
   noise, especially in cases where a Rotation has sprayed numerical interpolation noise outside
   of the true beam. The underlying Wavefront values aren't affected at all, this just pre-filters a copy of
   the phase before sending it to matplotlib.imshow. (@mperrin)
 * remove deprecated parameters in some function calls
   (`#148 <https://github.com/spacetelescope/poppy/issues/148>`_; @mperrin)

.. _rel0.4.1:

0.4.1
-----

2016 Apr 4:

Mostly minor bug fixes:

 * Fix inconsistency between older deprecated ``angle`` parameter to some optic classes versus new ``rotation`` parameter for any AnalyticOpticalElement  (`#140 <https://github.com/spacetelescope/poppy/issues/140>`_; @kvangorkom, @josephoenix, @mperrin)
 * Update to newer API for ``psutil``  (`#139 <https://github.com/spacetelescope/poppy/issues/139>`_; Anand Sivaramakrishnan, @mperrin)
 * "measure_strehl" function moved to ``webbpsf`` instead of ``poppy``.  (`#138 <https://github.com/spacetelescope/poppy/issues/138>`_; Kathryn St.Laurent, @josephoenix, @mperrin)
 * Add special case to handle zero radius pixel in circular BandLimitedOcculter.  (`#137 <https://github.com/spacetelescope/poppy/issues/137>`_; @kvangorkom, @mperrin)
 * The output FITS header of an `AnalyticOpticalElement`'s `toFITS()` function is now compatible with the input expected by `FITSOpticalElement`.
 * Better saving and reloading of FFTW wisdom.
 * Misc minor code cleanup and PEP8 compliance. (`#149 <https://github.com/spacetelescope/poppy/issues/149>`_; @mperrin)

And a few more significant enhancements:

 * Added `MatrixFTCoronagraph` subclass for fast optimized propagation of coronagraphs with finite fields of view. This is a
   related variant of the approach used in the `SemiAnalyticCoronagraph` class, suited for
   coronagraphs with a focal plane field mask limiting their field of view, for instance those
   under development for NASA's WFIRST mission. ( `#128 <https://github.com/spacetelescope/poppy/pull/128>`_; `#147 <https://github.com/mperrin/poppy/pull/147>`_; @neilzim)
 * The `OpticalSystem` class now has `npix` and `pupil_diameter` parameters, consistent with the `FresnelOpticalSystem`.  (`#141 <https://github.com/spacetelescope/poppy/issues/141>`_; @mperrin)
 * Added `SineWaveWFE` class to represent a periodic phase ripple.

.. _rel0.4.0:

0.4.0
-----

2015 November 20

 * **Major enhancement: the addition of Fresnel propagation** (
   `#95 <https://github.com/spacetelescope/poppy/issue/95>`_,
   `#100 <https://github.com/spacetelescope/poppy/pull/100>`_,
   `#103 <https://github.com/spacetelescope/poppy/issue/103>`_,
   `#106 <https://github.com/spacetelescope/poppy/issue/106>`_,
   `#107 <https://github.com/spacetelescope/poppy/pull/107>`_,
   `#108 <https://github.com/spacetelescope/poppy/pull/108>`_,
   `#113 <https://github.com/spacetelescope/poppy/pull/113>`_,
   `#114 <https://github.com/spacetelescope/poppy/issue/114>`_,
   `#115 <https://github.com/spacetelescope/poppy/pull/115>`_,
   `#100 <https://github.com/spacetelescope/poppy/pull/100>`_,
   `#100 <https://github.com/spacetelescope/poppy/pull/100>`_; @douglase, @mperrin, @josephoenix) *Many thanks to @douglase for the initiative and code contributions that made this happen.*
 * Improvements to Zernike aberration models (
   `#99 <https://github.com/spacetelescope/poppy/pull/99>`_,
   `#110 <https://github.com/spacetelescope/poppy/pull/110>`_,
   `#121 <https://github.com/spacetelescope/poppy/pull/121>`_,
   `#125 <https://github.com/spacetelescope/poppy/pull/125>`_; @josephoenix)
 * Consistent framework for applying arbitrary shifts and rotations to any AnalyticOpticalElement
   (`#7 <https://github.com/spacetelescope/poppy/pull/7>`_, @mperrin)
 * When reading FITS files, OPD units are now selected based on BUNIT
   header keyword instead of always being "microns" by default,
   allowing the units of files to be set properly based on the FITS header.
 * Added infrastructure for including field-dependent aberrations at an optical
   plane after the entrance pupil (
   `#105 <https://github.com/spacetelescope/poppy/pull/105>`_, @josephoenix)
 * Improved loading and saving of FFTW wisdom (
   `#116 <https://github.com/spacetelescope/poppy/issue/116>`_,
   `#120 <https://github.com/spacetelescope/poppy/issue/120>`_,
   `#122 <https://github.com/spacetelescope/poppy/issue/122>`_,
   @josephoenix)
 * Allow configurable colormaps and make image origin position consistent
   (`#117 <https://github.com/spacetelescope/poppy/pull/117>`_, @josephoenix)
 * Wavefront.tilt calls are now recorded in FITS header HISTORY lines
   (`#123 <https://github.com/spacetelescope/poppy/pull/123>`_; @josephoenix)
 * Various improvements to unit tests and test infrastructure
   (`#111 <https://github.com/spacetelescope/poppy/pull/111>`_,
   `#124 <https://github.com/spacetelescope/poppy/pull/124>`_,
   `#126 <https://github.com/spacetelescope/poppy/pull/126>`_,
   `#127 <https://github.com/spacetelescope/poppy/pull/127>`_; @josephoenix, @mperrin)

.. _rel0.3.5:

0.3.5
-----

2015 June 19

 * Now compatible with Python 3.4 in addition to 2.7!  (`#83 <https://github.com/spacetelescope/poppy/pull/82>`_, @josephoenix)
 * Updated version numbers for dependencies (@josephoenix)
 * Update to most recent astropy package template (@josephoenix)
 * :py:obj:`~poppy.optics.AsymmetricSecondaryObscuration` enhanced to allow secondary mirror supports offset from the center of the optical system. (@mperrin)
 * New optic :py:obj:`~poppy.optics.AnnularFieldStop` that defines a circular field stop with an (optional) opaque circular center region (@mperrin)
 * display() functions now return Matplotlib.Axes instances to the calling functions.
 * :py:obj:`~poppy.optics.FITSOpticalElement` will now determine if you are initializing a pupil plane optic or image plane optic based on the presence of a ``PUPLSCAL`` or ``PIXSCALE`` header keyword in the supplied transmission or OPD files (with the transmission file header taking precedence). (`#97 <https://github.com/spacetelescope/poppy/pull/97>`_, @josephoenix)
 * The :py:func:`poppy.zernike.zernike` function now actually returns a NumPy masked array when called with ``mask_array=True``
 * poppy.optics.ZernikeAberration and poppy.optics.ParameterizedAberration have been moved to poppy.wfe and renamed :py:obj:`~poppy.wfe.ZernikeWFE` and :py:obj:`~poppy.wfe.ParameterizedWFE`. Also, ZernikeWFE now takes an iterable of Zernike coefficients instead of (n, m, k) tuples.
 * Various small documentation updates
 * Bug fixes for:

   * redundant colorbar display (`#82 <https://github.com/spacetelescope/poppy/pull/82>`_)
   * Unnecessary DeprecationWarnings in :py:func:`poppy.utils.imshow_with_mouseover` (`#53 <https://github.com/spacetelescope/poppy/issues/53>`_)
   * Error in saving intermediate planes during calculation (`#81 <https://github.com/spacetelescope/poppy/issues/81>`_)
   * Multiprocessing causes Python to hang if used with Apple Accelerate (`#23 <https://github.com/spacetelescope/poppy/issues/23>`_, n.b. the fix depends on Python 3.4)
   * Copy in-memory FITS HDULists that are passed in to FITSOpticalElement so that in-place modifications don't affect the caller's copy of the data (`#89 <https://github.com/spacetelescope/poppy/issues/89>`_)
   * Error in the :py:func:`poppy.utils.measure_EE` function produced values for the edges of the radial bins that were too large, biasing EE values and leading to weird interpolation behavior near r = 0. (`#96 <https://github.com/spacetelescope/poppy/pull/96>`_)

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
 * Bug fix: Instrument class can now pass through dict or tuple sources to OpticalSystem calc_psf (@mperrin)
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
    >> nc.calc_psf('F212N', fov_arcsec=[3,6])
    >> nc.calc_psf('F187N', fov_pixels=(300,100) )

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
