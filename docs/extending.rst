.. _extending:

Extending POPPY by defining your own optics and instruments
==============================================================

POPPY is designed to make it straightforward to implement your own custom optics classes, which will interoperate with all the built-in classes.  Conceptually all that is needed is defining the getPhasor function for each new class. 

Many examples of this can be found in ``poppy/optics.py``

Defining a custom optic from an analytic function
-------------------------------------------------

AnalyticOpticalElement subclasses must implement a `getPhasor()` functoin that
takes a `Wavefront` as its sole argument besides self, plus whatever `__init__`
arguments are appropriate for defining your optic. 



Example skeleton code::

    class myCustomOptic(poppy.AnalyticOpticalElement):
        def __init__(self, *args, **kwargs):
            """ If your optic has adjustible parameters, then save them as attributes here """

        def getPhasor(self, wave):
            y, x = wave.coordinates() 

            opd = some_function(x,y)
            transmission = other_function(x,y)

            phasor = transmission = np.exp(1.j * 2 * np.pi / wave.wavelength * opd)
            return phasor


Note the use of the `wave.coordinates()` helper function, which returns `y` and
`x` arrays giving the coordinates (in units of meters for pupil plane optics,
in arcseconds ofr image plane optics).  You can use these coordinates to
calculate the transmission and phase retardance appropriate for your optic.  If
your optic has wavelength dependent properties, access the `wave.wavelength`
property to determine the the appropriate wavelength; this will be in units of
meters. 


Defining a custom optic from a FITS file
----------------------------------------

Of course, any arbitrary optic can be represented in discrete form in 2D arrays
and then read into poppy using the FITSOpticalElement class. 

The transmission array should contain floating point values between 0.0 and
1.0.  These represent the local transmission of the electric field amplitude,
not the total intensity. 


The OPD array should contain floating point numbers (positive and negative)
representing a path delay in some physical units.  The unit must be specified
using the `BUNIT` keyword; allowed BUNITs are 'meter', 'micron', 'nanometer' and
their standard metric abbreviations. 

If you are using both an OPD and transmission together to define your optics, 
the arrays must have the same size. 

The spatial or angular scale of these arrays must also be indicated by a FITS
header keyword. By default, poppy checks for the keyword `PIXSCALE` for image
plane pixel scale in arcseconds/pixel or `PUPLSCAL` for pupil plane scale in
meters/pixel. However if your FITS file uses some alternate keyword, you can specify that
keyword name with the `pupilscale=` argument in the call to the `~poppy.FITSOpticalElement` constructor, i.e.::

     myoptic = poppy.FITSOpticalElement(transmission='transfile.fits', opd='opdfile.fits', pupilscale="PIXELSCL")


Lastly if there is no such keyword available, you can specify the numerical scale directly via the same keyword by providing a float instead of a string::

     myoptic = poppy.FITSOpticalElement(transmission='transfile.fits', opd='opdfile.fits', pupilscale=0.020)


Creating a custom instrument
----------------------------

POPPY provides an :py:class:`~poppy.Instrument` class to simplify certain types of calculations. For example, the WebbPSF project uses :py:class:`~poppy.Instrument` subclasses to provide selectable filters, pupil masks, and image masks for the instruments on JWST.

Any calculation you can set up with a bare POPPY :py:class:`~poppy.OpticalSystem` can be wrapped with an :py:class:`~poppy.Instrument` to present a friendlier API to end users. The :py:class:`~poppy.Instrument` will hold the selected instrument configuration and calculation options, passing them to a private method :py:meth:`~poppy.Instrument._getOpticalSystem` which implementors must override to build the :py:class:`~poppy.OpticalSystem` for the PSF calculation.

The general notion of an :py:class:`~poppy.Instrument` is that it consists of both

1. An optical system implemented in the usual fashion, optionally with several configurations such as
   selectable image plane or pupil plane stops or other adjustable properties, and
2. Some defined spectral bandpass(es) such as selectable filters. If the :py:mod:`pysynphot` module is available, it will be used to perform careful synthetic photometry of targets with a given spectrum observed in the given bandpass. If :py:mod:`pysynphot` is not installed, the code will fall back to a much simpler model assuming constant number of counts vs wavelength.  


Configurable options such as optical masks and filters are specified as properties of the instrument instance; an appropriate :py:class:`~poppy.OpticalSystem` will be generated when the :py:meth:`~poppy.Instrument.calcPSF` method is called. 

The :py:class:`~poppy.Instrument` is fairly complex, and has a lot of internal submethods used to modularize the calculation and allow subclassing and customization. For developing your own instrument classes, it may be useful to start with the instrument classes in WebbPSF as worked examples. 


You will at a minimum want to override the following class methods:

  * _getOpticalSystem
  * _getFilterList
  * _getDefaultNLambda
  * _getDefaultFOV
  * _getFITSHeader

For more complicated systems you may also want to override:

  * _validateConfig
  * _getSynphotBandpass
  * _applyJitter

An :py:class:`~poppy.Instrument` will get its configuration from three places:

   (1) The ``__init__`` method of the :py:class:`~poppy.Instrument` subclass

       During ``__init__``, the subclass can set important attributes like ``pixelscale``, add a custom ``pupil`` optic and OPD map, and set a default filter. (n.b. The current implementation may not do what you expect if you are accustomed to calling the superclass' ``__init__`` at the end of your subclass' ``__init__`` method. Look at the implementation in ``poppy/instrument.py`` for guidance.)
   (2) The :py:attr:`~poppy.Instrument.options` dictionary attribute on the :py:class:`~poppy.Instrument` subclass

       The options dictionary allows you to set a subset of options that are loosely considered to be independent of the instrument configuration (e.g. filter wheels) and of the particular calculation. This includes offsetting the source from the center of the FOV, shifting the pupil, applying jitter to the final image, or forcing the parity of the final output array.

       Users are free to introduce new options by documenting an option name and retrieving the value at an appropriate point in their implementation of :py:meth:`~poppy.Instrument._getOpticalSystem` (to which the options dictionary is passed as keyword argument ``options``).
   (3) The :py:meth:`~poppy.Instrument.calcPSF` method of the :py:class:`~poppy.Instrument` subclass

       For interoperability, it's not recommended to change the function signature of :py:meth:`~poppy.Instrument.calcPSF`. However, it is an additional way that users will pass configuration information into the calculation, and a starting point for more involved customization that cannot be achieved by overriding one of the private methods above.

Be warned that the :py:class:`poppy.Instrument` API evolved in tandem with WebbPSF, and certain things are subject to change as we extend it to use cases beyond the requirements of WebbPSF.
