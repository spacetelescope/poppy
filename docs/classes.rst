.. JWST-PSFs documentation master file, created by
   sphinx-quickstart on Mon Nov 29 15:57:01 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. default-role:: obj

POPPY Class Documentation
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
     * `~poppy.IdealFieldStop`
     * `~poppy.IdealRectangularFieldStop`
     * `~poppy.IdealCircularOcculter`
     * `~poppy.IdealBarOcculter`
     * `~poppy.BandLimitedCoron`
     * `~poppy.IdealFQPM`

 * `~poppy.InverseTransmission` allows any optic, whether analytic or discrete, to be flipped in sign, a la the Babinet principle.
 * `~poppy.Rotation` represents a rotation of the axes of the wavefront, for instance to change coordinate systems between two optics that are 
   rotated with respect to one another. The axis of rotation must be the axis of optical propagation.

 * `~poppy.Detector` represents a detector with some fixed sampling and pixel scale.




Reference/API
=============

.. automodapi:: poppy


.. comment 
	#
	#.. inheritance-diagram:: poppy.Detector poppy.Wavefront poppy.OpticalSystem poppy.Rotation poppy.CircularAperture poppy.HexagonAperture poppy.SquareAperture poppy.IdealFieldStop poppy.IdealCircularOcculter poppy.IdealBarOcculter poppy.BandLimitedCoron poppy.IdealFQPM poppy.FQPM_FFT_aligner poppy.CompoundAnalyticOptic poppy.FITSOpticalElement poppy.Instrument poppy.SecondaryObscuration poppy.InverseTransmission poppy.NgonAperture poppy.MultiHexagonAperture
	#
	#
	#.. _Wavefront:
	#
	#Wavefront
	#---------
	#
	#.. autoclass:: poppy.Wavefront
	#    :members:
	#
	#.. OpticalSystem:
	#
	#Optical System 
	#--------------
	#
	#.. autoclass:: poppy.OpticalSystem
	#    :members:
	#
	#.. autoclass:: poppy.SemiAnalyticCoronagraph
	#    :members:
	#
	#
	#
	#
	#.. OpticalElement:
	#
	#Optical Elements
	#----------------
	#
	#.. autoclass:: poppy.OpticalElement
	#   :members:
	#
	#
	#General Optical Elements
	#^^^^^^^^^^^^^^^^^^^^^^^^^
	#
	#.. autoclass:: poppy.FITSOpticalElement
	#   :members:
	#
	#.. autoclass:: poppy.AnalyticOpticalElement
	#   :show-inheritance:
	#.. autoclass:: poppy.CompoundAnalyticOptic
	#   :show-inheritance:
	#.. autoclass:: poppy.InverseTransmission
	#   :show-inheritance:
	#.. autoclass:: poppy.Rotation
	#   :show-inheritance:
	#
	#
	#Pupil Plane Optical Elements
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#.. autoclass:: poppy.CircularAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.SquareAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.RectangleAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.HexagonAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.MultiHexagonAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.NgonAperture
	#   :show-inheritance:
	#.. autoclass:: poppy.SecondaryObscuration
	#   :show-inheritance:
	#
	#.. autoclass:: poppy.ThinLens
	#   :show-inheritance:
	#
	#
	#
	#.. autoclass:: poppy.FQPM_FFT_aligner
	#   :show-inheritance:
	#
	#
	#
	#
	#Image Plane Optical Elements
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#
	#
	#.. autoclass:: poppy.IdealFieldStop
	#   :show-inheritance:
	#.. autoclass:: poppy.IdealCircularOcculter
	#   :show-inheritance:
	#.. autoclass:: poppy.IdealBarOcculter
	#   :show-inheritance:
	#.. autoclass:: poppy.BandLimitedCoron
	#   :show-inheritance:
	#.. autoclass:: poppy.IdealFQPM
	#   :show-inheritance:
	#
	#
	#
	#The Detector Optical Element
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	#
	#.. autoclass:: poppy.Detector
	#   :show-inheritance:
	#
	#
	#
	#
	#.. Instrument:
	#
	#Instrument
	#----------------
	#
	#.. autoclass:: poppy.Instrument
	#   :members:
	#
	#------
	#

--------------

Documentation last updated on |today|

