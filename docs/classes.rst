.. _classes:

.. default-role:: obj

POPPY Class Listing
============================

The key classes for POPPY are `~poppy.OpticalSystem` and the various `~poppy.OpticalElement` classes (of which there are many). There is also a `~poppy.Wavefront` class that is used internally, but users will rarely
need to instantiate that directly. Results are returned as FITS files, specifically astropy.io.fits.HDUList objects.

`~poppy.OpticalSystem` is in essence a container for `~poppy.OpticalElement` instances, which handles creating input wavefronts, propagating them through the individual optics, and then combining the
results into a broadband output point spread function.


The `~poppy.Instrument` class provides a framework for developing high-level models of astronomical instruments.
An `~poppy.OpticalSystem` does not include any information about spectral bandpasses, filters, or light source properties,
it just propagates whatever specified list of wavelengths and weights it's provided with.  The
`~poppy.Instrument` class provides the machinery for handling filters and sources to generate weighted source spectra, as
well as support for configurable instruments with selectable mechanisms, and system-level impacts on PSFs such as pointing jitter.

Note that the `~poppy.Instrument` class should not be used directly but rather is subclassed to implement the details of your particular instrument. See its class documentation for more details.


Optical Systems
-----------------

 *  `~poppy.OpticalSystem` is the fundamental optical system class, that propagates `~poppy.Wavefront` objects between optics using Fourier transforms.
 *  `~poppy.SemiAnalyticCoronagraph` implements the semi-analytic coronagraphic propagation algorithm of Soummer et al.
 *  `~poppy.MatrixFTCoronagraph` enables efficient propagation calculations for Lyot coronagraphs with diaphragm-type focal plane masks, relevant to the WFIRST coronagraph and described by Zimmerman et al. (2016).



Optical Elements
-----------------

 * `~poppy.OpticalElement` is the fundamental building block
 * `~poppy.FITSOpticalElement` implements optics defined numerically on discrete grids read in from FITS files
 * `~poppy.AnalyticOpticalElement` implements optics defined analytically on any arbitrary sampling.  There are many of these.

     * `~poppy.ScalarTransmission` is a simple floating-point throughput factor.
     * `~poppy.CompoundAnalyticOptic` allows multiple analytic optics to be merged into one container object

 * Pupil plane analytic optics include:

     * `~poppy.CircularAperture`
     * `~poppy.SquareAperture`
     * `~poppy.RectangleAperture`
     * `~poppy.HexagonAperture`
     * `~poppy.MultiHexagonAperture`
     * `~poppy.NgonAperture`
     * `~poppy.SecondaryObscuration`
     * `~poppy.ThinLens`
     * `~poppy.FQPM_FFT_aligner`

 * Image plane analytic optics include:
     * `~poppy.RectangularFieldStop`
     * `~poppy.SquareFieldStop`
     * `~poppy.CircularOcculter`
     * `~poppy.BarOcculter`
     * `~poppy.BandLimitedCoron`
     * `~poppy.IdealFQPM`

 * `~poppy.InverseTransmission` allows any optic, whether analytic or discrete, to be flipped in sign, a la the Babinet principle.
 * `~poppy.Rotation` represents a rotation of the axes of the wavefront, for instance to change coordinate systems between two optics that are
   rotated with respect to one another. The axis of rotation must be the axis of optical propagation.
 * `_poppy.CoordinateInversion` represents a flip in orientation of the X or Y axis, or both at once.

 * `~poppy.Detector` represents a detector with some fixed sampling and pixel scale.

Wavefront Error Optical Elements
--------------------------------

 * `poppy.wfe.ZernikeWFE`
 * `poppy.wfe.SineWaveWFE`
