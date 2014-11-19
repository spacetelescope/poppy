"""

 Classes for wavefront errors in POPPY

 (this is a separate file purely for convienience

 * ZernikeWFE
 * PowerSpectrumWFE
 * KolmogorovWFE


.. warning::

  DEVELOPMENT CODE, NOT YET SUPPORTED



"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from .optics import AnalyticOpticalElement


class WavefrontError(AnalyticOpticalElement):
    def __init__(self, name=None,  **kwargs):
        raise NotImplementedError('Not implemented yet')

    def rms(self):
        """ RMS wavefront error induced by this surface """
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """ Peak-to-valley wavefront error induced by this surface """
        raise NotImplementedError('Not implemented yet')

   

class ZernikeWFE(WavefrontError):
    """ Defines wavefront error over a pupil in terms of Zernike coefficients. 

    Parameters
    ----------
    name : string
        Descriptive name
    size: float
        radius of the circle for which the Zernikes are defined. 
        If not specified, this is attempted to be guessed from the previous pupil plane optic.
    coeffs : iterable of floats
        The Zernike amplitude coefficients for the desired WFE. Defined in the order given in 
        zernike.py. Coefficients must be in units of meters of RMS phase error per term.
    type : str
        'zernike' or 'hexike' to indicate desired polynomial type.

    """

    def __init__(self, name=None,  size=1.0, coeffs=[1], type='zernike', **kwargs):
        if name is None: name = "Zernikes over a circle of radius= %.1f m" % size
        AnalyticOpticalElement.__init__(self,name=name,**kwargs)
        self.size = size
        self.pupil_diam = 2* self.size # for creating default input wavefronts
        self.coeffs = coeffs

    def getPhasor(self,wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        import zernike

        if not isinstance(wave, Wavefront):
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PUPIL)

        y, x = wave.coordinates()

        # compute normalized rho and theta for zernike computation
        rho = np.sqrt( (x/self.size)**2 + (y/self.size)**2)
        theta = np.arctan2( y/self.size, x/self.size)
        del y
        del x

        _log.info("Generating wavefront from Zernike coefficients")
        self.phase = np.empty(wave.shape)
        for i, coeff in enumerate(self.coeffs):
            j = i+1 # zernikes indexing must start with 1
            self.phase += zernike.zernike1( j, theta=theta, rho=r) * coeff


        return self.transmission


        retardance = phase*self.reference_wavelength/wave.wavelength


class StatisticalOpticalElement(WavefrontError):
    """
    A statistical realization of some wavefront error, computed on a fixed grid. 

    This is in a sense like an AnalyticOpticalElement, in that it is in theory computable on any grid,
    but once computed it has some fixed sampling that cannot easily be changed. 

    """
    def __init__(self, name=None,  seed=None, r0=15, L_inner=0.001, L_outer=10,  **kwargs):
        if name is None: name = "Zernikes over a circle of radius= %.1f m" % size
        OpticalElement.__init__(self,name=name,**kwargs)
        raise NotImplementedError('Not implemented yet')
 

class KolmogorovWFE(StatisticalOpticalElement):
    """
    See

    http://www.opticsinfobase.org/view_article.cfm?gotourl=http%3A%2F%2Fwww%2Eopticsinfobase%2Eorg%2FDirectPDFAccess%2F8E2A4176%2DED0A%2D7994%2DFB0AC49CECB235DF%5F142887%2Epdf%3Fda%3D1%26id%3D142887%26seq%3D0%26mobile%3Dno&org=

    http://optics.nuigalway.ie/people/chris/chrispapers/Paper066.pdf

    """
    def __init__(self, name=None,  seed=None, r0=15, L_inner=0.001, L_outer=10,  **kwargs):
        if name is None: name = "Zernikes over a circle of radius= %.1f m" % size
        StatisticalOpticalElement.__init__(self,name=name,**kwargs)
        raise NotImplementedError('Not implemented yet')
 
class PowerSpectralDensityWFE(StatisticalOpticalElement):
    """ Compute WFE from a power spectral density. 

    Inspired by (and loosely derived from) prop_psd_errormap in John Krist's PROPER library.


    For some background on structure functions & why they are useful, see : http://www.optics.arizona.edu/optomech/Spr11/523L/Specifications%20final%20color.pdf

    """
    def __init__(self, name=None,  seed=None, low_freq_amp=1, correlation_length=1.0, powerlaw=1.0,  **kwargs):
        """ 

        Parameters
        -----------
        low_freq_amp : float
            RMS error per spatial frequency at low spatial frequencies. 
        correlation_length : float
            Correlation length parameter in cycles/meter. This indicates where the PSD transitions from
            the low frequency behavior (~ constant amplitude per spatial frequency) to the high
            frequency behavior (~decreasing amplitude per spatial frequency)
        powerlaw : float
            The power law exponent for the falloff in amplitude at high spatial frequencies.
        """
        if name is None: name = "Power Spectral Density WFE map "
        StatisticalOpticalElement.__init__(self,name=name,**kwargs)
        raise NotImplementedError('Not implemented yet')

        # compute X and Y coordinate grids 
        # compute wavenumber K in cycles/meter
        # compute 2D PSD
        # set piston to zero
        # scale RMS error as desired
        # create realization of the PSD using random phases
        # force realized map to have the desired RMS

    def saveto(self, filename):
        raise NotImplementedError('Not implemented yet')
 


