Overview 
====================

The module ``poppy`` implements an object-oriented system for modeling physical optics
propagation with diffraction, particularly for telescopic and coronagraphic
imaging. Right now only image and pupil planes are supported; intermediate
planes are a future goal.  

Poppy also includes a system for modeling a complete instrument (including
optical propagation, synthetic photometry, and pointing jitter), and a variety
of useful utility functions for analysing and plotting PSFs, documented below. 


**Key Concepts:**

To model optical propagation, ``poppy`` implements an object-oriented system for
representing an optical train. There are a variety of `~poppy.OpticalElement` classes
representing both physical elements as apertures, mirrors, and apodizers, and
also implicit operations on wavefronts, such as rotations or tilts. Each
`~poppy.OpticalElement`  may be defined either via analytic functions (e.g. a simple
circular aperture) or by numerical input FITS files (e.g. the complex JWST
aperture with appropriate per-segment WFE). A series of such `~poppy.OpticalElement` objects is
chained together in order in an `~poppy.OpticalSystem` class. That class is capable of generating
`~poppy.Wavefront` instances suitable for propagation through the desired elements 
(with correct array size and sampling), and onto
the final image plane. 

There is an even higher level class `~poppy.Instrument` which adds support
for selectable instrument mechanisms (such as filter wheels, pupil stops, etc). In particular it adds support for computing via synthetic photometry the
appropriate weights for multiwavelength computations through a spectral bandpass filter, and for PSF blurring due to pointing jitter (neither of which effects are modeled by `~poppy.OpticalSystem`). 
Given a specified instrument configuration, an appropriate `~poppy.OpticalSystem` is generated, the appropriate wavelengths and weights are calculated based on the bandpass filter and target source spectrum, the PSF is calculated, and optionally is then convolved with a blurring kernel due to pointing jitter.  For instance, all of the WebbPSF instruments are implemented by subclassing `poppy.Instrument`.


.. _fraunhofer:

Fraunhofer domain calculations
--------------------------------

``poppy``'s default mode assumes that optical propagation can be modeled using
Fraunhofer diffraction (the "far field" approximation), such that the
relationship between pupil and image plane optics is given by two-dimensional
Fourier transforms.  (Fresnel propagation is also available, :ref:`with slightly
different syntax <fresnel>`.)

Two different algorithmic flavors of Fourier transforms are used in Poppy. The
familiar FFT algorithm is used for transformations between pupil and image
planes in the general case. This algorithm is relatively fast (*O(N log(N))*)
but imposes strict constraints on the relative sizes and samplings of pupil and
image plane arrays. Obtaining fine sampling in the image plane requires very
large oversized pupil plane arrays and vice versa, and image plane pixel
sampling becomes wavelength dependent. To avoid these constraints, for
transforms onto the final `~poppy.Detector` plane, instead a Matrix Fourier Transform
(MFT) algorithm is used (See `Soummer et al. 2007 Optics Express <http://adsabs.harvard.edu/abs/2007OExpr..1515935S>`_).  This allows
computation of the PSF directly on the desired detector pixel scale or an
arbitrarily finely subsampled version therof. For equivalent array sizes *N*,
the MFT is slower than the FFT(*O(N^3)*), but in practice the ability to freely
choose a more appropriate *N* (and to avoid the need for post-FFT interpolation
onto a common pixel scale) more than makes up for this and the MFT is faster.


.. note::

        This code makes use of the python standard module ``logging`` for
        output information. Top-level details of the calculation are output at
        level ``logging.INFO``, while details of the propagation through each
        optical plane are printed at level ``logging.DEBUG``. See the Python
        logging documentation for an explanation of how to redirect the
        ``poppy`` logger to the screen, a textfile, or any other log
        destination of your choice.




Working with OpticalElements
----------------------------

OpticalElements can be instantiated from FITS files, or created by one of a large number of analytic function definitions implemented as `~poppy.AnalyticOpticalElement` subclasses. 
Typically these classes take some number of arguments to set their properties. 
Once instantiated, any analytic function can be displayed on screen, sampled onto a numerical grid, and/or saved to disk.::

    >>> ap = poppy.CircularAperture(radius=2)      # create a simple circular aperture
    >>> ap.display(what='both')                    # display both intensity and phase components

    >>> values = ap.sample(npix=512)               # evaluate on 512 x 512 grid
    >>> ap.to_fits('test_circle.fits', npix=1024)  # write to disk as a FITS file with higher sampling


When sampling an `~poppy.AnalyticOpticalElement`, you may choose to obtain various representations of its action on a complex wavefront, including the amplitude transmission; intensity transmission; or phase delay in waves, radians, or meters. 
See the `~poppy.AnalyticOpticalElement` class documentation for detailed arguments to these functions.


`~poppy.OpticalElement` objects have attributes such as `shape` (For a `~poppy.FITSOpticalElement` the array shape in usual Python (Y,X) order; None for a `~poppy.AnalyticOpticalElement`), a descriptive `name` string, and size information such as `pixelscale`. The type of size information present depends on the *plane type*. 

Optical Plane Types
-------------------------


An `~poppy.OpticalSystem` consists of a series of two or more planes, of various types. 
The plane type of a given `OpticalElement` is encoded by its `.planetype` attribute. 
The allowed types of planes are:

 * **Pupil** planes, which have spatial scale measured in meters. For instance
   a telescope could have a diameter of 1 meter and be represented inside an
   array 1024x1024 pixels across with pixel scale 0.002 meters/pixel, so that
   the aperture is a circle filling half the diameter of the array. Pupil planes 
   typically have a `pupil_diam` attribute which, please note, 
   defines the diameter of the *numerical array* (e.g. 2.048 m in this example), 
   rather than whatever subset of that array has nonzero optical transmission.

 * **Image** planes, which have angular sampling measured in arcseconds. The
   default behavior for an image plane in POPPY is to have the sampling
   automatically defined by the natural sampling of a Fourier Transform of the
   previous pupil array. This is generally appropriate for most intermediate
   optical planes in a system. However there are also:

 * **Detector** planes, which are a specialized subset of image plane that has
   a fixed angular sampling (pixel scale).  For instance one could compute the
   PSF of that telescope over a field of view 10 arcseconds square with a
   sampling of 0.01 arcseconds per pixel. 

 * **Rotation** planes, which represent a change of coordinate system rotating
   by some number of degrees around the optical axis. Note that POPPY always
   represents an "unfolded", linear optical system; fold mirrors and/or other
   intermediate powered optics are not represented as such.  Rotations can take
   place after either an image or pupil plane. 

POPPY thus is capable of representing a moderate subset of optical imaging systems, 
though it is not intended as a substitute for a professional optics design package
such as Zemax or Code V for design of full optical systems. 



Defining your own custom optics
----------------------------------

All `~poppy.OpticalElement` classes must have methods
`~poppy.OpticalElement.get_transmission` and `~poppy.OpticalElement.get_opd`
which returns the amplitude transmission and optical path delay representing
that optic, sampled appropriately for a given input `~poppy.Wavefront` and at
the appropriate wavelength. These are combined together to calculate the
complex phasor which is applied to the wavefront's electric field.  To define
your own custom OpticalElements, you can:

1. Subclass `~poppy.AnalyticOpticalElement` and write suitable function(s) to
   describe the properties of your optic, 
2. Combine two or more existing `~poppy.AnalyticOpticalElement` instances as
   part of a `~poppy.CompoundAnalyticOptic`, or
3. Generate suitable transmission and optical path difference arrays
   using some other tool, save them as FITS files with appropriate keywords,
   and instantiate them as an `~poppy.FITSOpticalElement`


FITSOpticalElements have separate attributes for amplitude and phase components, which may be read separately from 2 FITS files:

  * `amplitude`, the electric field amplitude transmission of the optic
  * `opd`, the optical path difference of the optic

Defining functions on a AnalyticOpticalElement subclass allows more flexibility for amplitude transmission or OPDs to vary with wavelength or other properties. 

See :ref:`extending` for more details and examples.
