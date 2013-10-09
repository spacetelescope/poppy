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

To model optical propagation, Poppy implements an object-oriented system for
representing an optical train. There are a variety of :py:class:`OpticalElement` classes
representing both physical elements as apertures, mirrors, and apodizers, and
also implicit operations on wavefronts, such as rotations or tilts. Each
:py:class:`OpticalElement`  may be defined either via analytic functions (e.g. a simple
circular aperture) or by numerical input FITS files (e.g. the complex JWST
aperture with appropriate per-segment WFE). A series of such :py:class:`OpticalElements <OpticalElement>` is
chained together in order in an :py:class:`OpticalSystem` class. That class is capable of generating
:py:class:`Wavefronts <Wavefront>`  (another class) suitable for propagation through the desired elemeents 
(with correct array size and sampling), and then performing the optical propagation onto
the final image plane. 

There is an even higher level class :py:class:`Instrument` which adds support
for selectable instrument mechanisms (such as filter wheels, pupil stops, etc). In particular it adds support for computing via synthetic photometry the
appropriate weights for multiwavelength computations through a spectral bandpass filter, and for PSF blurring due to pointing jitter (neither of which effects are modeled by :py:class:`OpticalSystem`). 
Given a specified instrument configuration, an appropriate :py:class:`OpticalSystem` is generated, the appropriate wavelengths and weights are calculated based on the bandpass filter and target source spectrum, the PSF is calculated, and optionally is then convolved with a blurring kernel due to pointing jitter.  All of the WebbPSF instruments are implemented by subclassing ``poppy.Instrument``.


Poppy presently assumes that optical propagation can be modeled using Fraunhofer diffraction (far-field), such that
the relationship between pupil and image plane optics is given by two-dimensional Fourier transforms. Fresnel propagation is
not currently supported. 

Two different algorithmic flavors of Fourier transforms are used in Poppy. The
familiar FFT algorithm is used for transformations between pupil and image
planes in the general case. This algorithm is relatively fast (*O(N log(N))*)
but imposes strict constraints on the relative sizes and samplings of pupil and
image plane arrays. Obtaining fine sampling in the image plane requires very
large oversized pupil plane arrays and vice versa, and image plane pixel
sampling becomes wavelength dependent. To avoid these constraints, for
transforms onto the final :py:class:`Detector` plane, instead a Matrix Fourier Transform
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
===================================

OpticalElements can be instantiated from FITS files, or created by one of a large number of analytic function definitions implemented as :py:class:`AnalyticOpticalElement` subclasses. 
Typically these classes take some number of arguments to set their properties. 
Once instantiated, any analytic function can be displayed on screen, sampled onto a numerical grid, and/or saved to disk.::

    >>> ap = poppy.CircularAperture(radius=2)
    >>> ap.display(what='both')

    >>> values = ap.sample(npix=512)    # evaluate on 512 x 512 grid
    >>> ap.toFITS('test_circle.fits')   # write to disk as a FITS file. 

See the :py:class:`AnalyticOpticalElement` class documentation for detailed arguments to these functions.

