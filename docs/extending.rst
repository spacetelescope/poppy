Extending POPPY by defining your own optics
==============================================

POPPY is designed to make it straightforward to implement your own custom optics classes, which will interoperate with all the built-in classes.  Conceptually all that is needed is defining the getPhasor function for each new class. 

Many examples of this can be found in ``poppy/optics.py``

Defining a custom optic from an analytic function
-------------------------------------------------

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

Defining a custom optic from a FITS file
----------------------------------------

Of course, any arbitrary optic can be represented in discrete form in 2D arrays
and then read into poppy using the FITSOpticalElement class. 

Creating a custom instrument
----------------------------

POPPY provides an :py:class:`~poppy.Instrument` class to simplify certain types of calculations. For example, the WebbPSF project uses :py:class:`~poppy.Instrument` subclasses to provide selectable filters, pupil masks, and image masks for the instruments on JWST.

Any calculation you can set up with a bare POPPY :py:class:`~poppy.OpticalSystem` can be wrapped with an :py:class:`~poppy.Instrument` to present a friendlier API to end users. The :py:class:`~poppy.Instrument` will hold the selected instrument configuration and calculation options, passing them to a private method :py:meth:`~poppy.Instrument._getOpticalSystem` which implementors must override to build the :py:class:`~poppy.OpticalSystem` for the PSF calculation.

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