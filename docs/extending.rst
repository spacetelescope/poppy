Extending POPPY by defining your own optics
==============================================



POPPY is designed to make it straightforward to implement your own custom optics classes, which will
interoperate with all the built-in classes.  Conceptually all that is needed is defining the getPhasor function
for each new class. 

Many examples of this can be found in ``poppy/optics.py``

Defining a custom optic from an analytic function
---------------------------------------------------

Example skeleton code::
    class myCustomOptic(poppy.AnalyticOpticalElement):
        def __init__(self, *args, **kwargs):
            """ If your optic has adjustible parameters, then save them as attributes here """

        def getPhasor(self, wave):
            y, x = wave.coordinates() 

            opd = some_function(x,y)
            transmission = other_function(x,y)

            phasor = transmission = np.exp(1.j * 2* np.pi /wave.wavelength * opd)
            return phasor



Defining a custom optic from a FITS file
---------------------------------------------------

Of course, any arbitrary optic can be represented in discrete form in 2D arrays
and then read into poppy using the FITSOpticalElement class. 


