from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage.interpolation
import matplotlib
import astropy.io.fits as fits

from . import utils

import logging

_log = logging.getLogger('poppy')

from poppy import zernike
from .poppy_core import OpticalElement, Wavefront, _PUPIL, _IMAGE, _RADIANStoARCSEC

__all__ = ['AnalyticOpticalElement', 'ScalarTransmission', 'InverseTransmission',
           'BandLimitedCoron', 'IdealFQPM', 'RectangularFieldStop', 'SquareFieldStop',
           'CircularOcculter', 'BarOcculter', 'FQPM_FFT_aligner',  'CircularAperture',
           'HexagonAperture', 'MultiHexagonAperture', 'NgonAperture', 'RectangleAperture',
           'SquareAperture', 'SecondaryObscuration', 'AsymmetricSecondaryObscuration',
           'ThinLens', 'ZernikeAberration', 'ParameterizedAberration', 'CompoundAnalyticOptic']

# ------ Generic Analytic elements -----

class AnalyticOpticalElement(OpticalElement):
    """ Defines an abstract analytic optical element, i.e. one definable by some formula rather than
        by an input OPD or pupil file.

        This class is useless on its own; instead use its various subclasses that implement
        appropriate getPhasor functions. It exists mostly to provide some behaviors &
        initialization common to all analytic optical elements.

        Parameters
        ----------
        name, verbose, oversample, planetype : various
            Same as for OpticalElement
        transmission, opd : string
            These are *not allowed* for Analytic optical elements, and this class will raise an
            error if you try to set one.


    """

    def __init__(self, **kwargs):
        OpticalElement.__init__(self, **kwargs)

        #self.shape = None  # no explicit shape required
        self.pixelscale = None

    @property
    def shape(self):  # Analytic elements don't have shape
        return None

    def __str__(self):
        if self.planetype is _PUPIL:
            return "Pupil plane: %s (Analytic)" % (self.name)
        elif self.planetype is _IMAGE:
            return "Image plane: %s (Analytic)" % (self.name)
        else:
            return "Optic: " + self.name

    def getPhasor(self, wave):
        raise NotImplementedError("getPhasor must be supplied by a derived subclass")

    def sample(self, wavelength=2e-6, npix=512, grid_size=None, what='amplitude',
               return_scale=False, phase_unit='waves'):
        """ Sample the Analytic Optic onto a grid and return the array

        Parameters
        ----------
        wavelength : float
            Wavelength in meters.
        npix : integer
            Number of pixels for sampling the array
        grid_size : float
            Field of view grid size (diameter) for sampling the optic, in meters for
            pupil plane optics and arcseconds for image planes. Default value is
            taken from the optic's properties, if defined. Otherwise defaults to
            6.5 meters or 2 arcseconds depending on plane.
        what : string
            What to return: optic 'amplitude' transmission, 'intensity' transmission, or
            'phase'.  Note that phase with phase_unit = 'meters' should give the optical path
            difference, OPD.
        phase_unit : string
            Unit for returned phase array IF what=='phase'. One of 'radians', 'waves', 'meters'.
        return_scale : float
            if True, will return a tuple containing the desired array and a float giving the
            pixel scale.
        """
        if self.planetype is _PUPIL:
            if grid_size is not None:
                diam = grid_size
            elif hasattr(self, 'pupil_diam'):
                diam = self.pupil_diam
            else:
                diam = 6.5  # meters
            w = Wavefront(wavelength=wavelength, npix=npix, diam=diam)
            pixel_scale = diam / npix

        else:
            #unit="arcsec"

            if grid_size is not None:
                fov = grid_size
            elif hasattr(self, '_default_display_size'):
                fov = self._default_display_size
            else:
                fov = 4
            pixel_scale = fov / npix
            w = Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixel_scale)

        phasor = self.getPhasor(w)
        _log.info("Computing {0} for {1} sampled onto {2} pixel grid".format(what, self.name, npix))
        if what == 'amplitude':
            output_array = np.abs(phasor)
        elif what == 'intensity':
            output_array = np.abs(phasor) ** 2
        elif what == 'phase':
            if phase_unit == 'radians':
                output_array = np.angle(phasor)
            elif phase_unit == 'waves':
                output_array = np.angle(phasor) / (2 * np.pi)
            elif phase_unit == 'meters':
                output_array = np.angle(phasor) / (2 * np.pi) * wavelength
            else:
                raise ValueError('Invalid/unknown phase_unit: {}. Must be one of '
                                 '[radians, waves, meters]'.format(phase_unit))
        elif what == 'complex':
            output_array = phasor
        else:
            raise ValueError('Invalid/unknown what to sample: {}. Must be one of '
                             '[amplitude, intensity, phase, complex]'.format(what))

        if return_scale:
            return output_array, pixel_scale
        else:
            return output_array


    def display(self, nrows=1, row=1, wavelength=2e-6, npix=512, **kwargs):
        """Display an Analytic optic by first computing it onto a grid...

        Parameters
        ----------
        wavelength : float
            Wavelength to evaluate this optic's properties at
        npix : int
            Number of pixels to use when sampling the analytic optical element.

        what : str
            What to display: 'intensity', 'phase', or 'both'
        ax : matplotlib.Axes instance 
            Axes to display into
        nrows, row : integers
            # of rows and row index for subplot display
        crosshairs : bool
            Display crosshairs indicating the center?
        colorbar : bool
            Show colorbar?
        colorbar_orientation : bool
            Desired orientation, horizontal or vertical?
            Default is horizontal if only 1 row of plots, else vertical
        opd_vmax : float
            Max value for OPD image display, in meters.
        title : string
            Plot label


        """

        _log.debug("Displaying " + self.name)
        phasor, pixelscale = self.sample(wavelength=wavelength, npix=npix, what='complex',
                                         return_scale=True)

        # temporarily set attributes appropriately as if this were a regular OpticalElement
        self.amplitude = np.abs(phasor)
        phase = np.angle(phasor) / (2 * np.pi)
        self.opd = phase * wavelength
        self.pixelscale = pixelscale

        #then call parent class display
        OpticalElement.display(self, nrows=nrows, row=row, **kwargs)

        # now un-set all the temporary attributes back, since this is analytic and
        # these are unneeded
        self.pixelscale = None
        self.opd = None
        self.amplitude = None

    def toFITS(self, outname=None, what='amplitude', wavelength=2e-6, npix=512, **kwargs):
        """ Save an analytic optic computed onto a grid to a FITS file 
        
        The FITS file is returned to the calling function, and may optionally be 
        saved directly to disk.

        Parameters
        ------------
        what : string
            What quantity to save. See the sample function of this class
        wavelength : float
            Wavelength in meters. 
        npix : integer
            Number of pixels.
        outname : string, optional
            Filename to write out a FITS file to disk

        See the sample() function for additional optional parameters.

        """

        kwargs['return_scale'] = True

        output_array, pixelscale = self.sample(wavelength=wavelength, npix=npix, what=what,
                                               **kwargs)
        phdu = fits.PrimaryHDU(output_array)
        phdu.header['OPTIC'] = self.name
        phdu.header['SOURCE'] = 'Computed with POPPY'
        phdu.header['CONTENTS'] = what
        phdu.header['PIXSCALE'] = pixelscale

        hdul = fits.HDUList(hdus=[phdu])

        if outname is not None:
            phdu.writeto(outname, clobber=True)
            _log.info("Output written to " + outname)

        return hdul

class ScalarTransmission(AnalyticOpticalElement):
    """ Uniform transmission between 0 and 1.0 in intensity. 
    
    Either a null optic (empty plane) or some perfect ND filter...
    But most commonly this is just used as a null optic placeholder """

    def __init__(self, name=None, transmission=1.0, **kwargs):
        if name is None:
            name = ("-empty-" if transmission == 1.0 else
                    "Scalar Transmission of {0}".format(transmission))
        AnalyticOpticalElement.__init__(self, name=name, **kwargs)
        self.transmission = float(transmission)

    def getPhasor(self, wave):
        res = np.empty(wave.shape)
        res.fill(self.transmission)
        return res


class InverseTransmission(OpticalElement):
    """ Given any arbitrary OpticalElement with transmission T(x,y)
    return the inverse transmission 1 - T(x,y)

    This is a useful ingredient in the SemiAnalyticCoronagraph algorithm.
    """

    def __init__(self, optic=None):
        if optic is None or not hasattr(optic, 'getPhasor'):
            raise ValueError("Need to supply an valid optic to invert!")
        self.uninverted_optic = optic
        self.name = "1 - " + optic.name
        self.planetype = optic.planetype
        #self.shape = optic.shape
        self.pixelscale = optic.pixelscale
        self.oversample = optic.oversample

    @property
    def shape(self): # override parent class shape function
        return self.uninverted_optic.shape

    def getPhasor(self, wave):
        return 1 - self.uninverted_optic.getPhasor(wave)


#------ Analytic Image Plane elements -----

class BandLimitedCoron(AnalyticOpticalElement):
    """ Defines an ideal band limited coronagraph occulting mask.


        Parameters
        ----------
        name : string
            Descriptive name
        kind : string
            Either 'circular' or 'linear'. The linear ones are custom shaped to NIRCAM's design
            with flat bits on either side of the linear tapered bit.
            Also includes options 'nircamcircular' and 'nircamwedge' specialized for the
            JWST NIRCam occulters, including the off-axis ND acq spots and the changing
            width of the wedge occulter.
        sigma : float
            The numerical size parameter, as specified in Krist et al. 2009 SPIE
        wavelength : float
            Wavelength this BLC is optimized for, only for the linear ones.

    """

    def __init__(self, name="unnamed BLC", kind='circular', sigma=1, wavelength=None, **kwargs):
        AnalyticOpticalElement.__init__(self, name=name, planetype=_IMAGE, **kwargs)

        self.kind = kind.lower()  # either circular or linear
        if self.kind not in ['circular', 'linear', 'nircamwedge', 'nircamcircular']:
            raise ValueError("Invalid kind of BLC: " + self.kind)
        self.sigma = float(sigma)  # size parameter. See section 2.1 of Krist et al. SPIE 2007, 2009
        if wavelength is not None:
            self.wavelength = float(wavelength)  # wavelength, for selecting the
                                                 # linear wedge option only
        self._default_display_size = 20.  # default size for onscreen display, sized for NIRCam

    def getPhasor(self, wave):
        """ Compute the amplitude transmission appropriate for a BLC for some given pixel spacing
        corresponding to the supplied Wavefront.

        Based on the Krist et al. SPIE paper on NIRCam coronagraph design

        Note that the equations in Krist et al specify the intensity transmission of the occulter,
        but what we want to return here is the amplitude transmittance. That is the square root
        of the intensity, of course, so the equations as implemented here all differ from those
        written in Krist's SPIE paper by lacking an exponential factor of 2. Thanks to John Krist
        for pointing this out.

        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("BLC getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _IMAGE)

        y, x = wave.coordinates()
        if self.kind == 'circular':
            # larger sigma implies narrower peak? TBD verify if this is correct
            #
            r = np.sqrt(x ** 2 + y ** 2)
            sigmar = self.sigma * r
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs

            self.transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)
        if self.kind == 'nircamcircular':
            # larger sigma implies narrower peak? TBD verify if this is correct
            #
            r = np.sqrt(x ** 2 + y ** 2)
            sigmar = self.sigma * r
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs
            self.transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)

            # add in the ND squares. Note the positions are not exactly the same in the two wedges.
            # See the figures  in Krist et al. of how the 6 ND squares are spaced among the 5
            # corongraph regions
            # Also add in the opaque border of the coronagraph mask holder.
            if self.sigma > 4:
                # MASK210R has one in the corner and one half in the other corner
                wnd = np.where(
                    (y > 5) &
                    (
                        ((x < -5) & (x > -10)) |
                        ((x > 7.5) & (x < 12.5))
                    )
                )
                wborder = np.where((np.abs(y) > 10) | (x < -10))  # left end of mask holder
            else:
                # the others have two halves on in each corner.
                wnd = np.where(
                    (y > 5) &
                    (np.abs(x) > 7.5) &
                    (np.abs(x) < 12.5)
                )
                wborder = np.where(np.abs(y) > 10)

            self.transmission[wnd] = np.sqrt(1e-3)
            self.transmission[wborder] = 0
        elif self.kind == 'linear':
            #raise(NotImplemented("Generic linear not implemented"))
            sigmar = self.sigma * np.abs(y)
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs
            self.transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        elif self.kind == 'nircamwedge':
            # This is hard-coded to the wedge-plus-flat-regions shape for NIRCAM

            # we want a scale factor that goes from 2 to 6 with 1/5th of it as a fixed part on
            # either end
            #scalefact = np.linspace(1,7, x.shape[1]).clip(2,6)

            # the scale fact should depent on X coord in arcsec, scaling across a 20 arcsec FOV.
            # map flat regions to 2.5 arcsec each?
            # map -7.5 to 2, +7.5 to 6. slope is 4/15, offset is +9.5
            scalefact = (2 + (-x + 7.5) * 4 / 15).clip(2, 6)

            #scalefact *= self.sigma / 2 #;2.2513
            #scalefact *= 2.2513
            #scalefact.shape = (1, x.shape[1])
            # This does not work - shape appears to be curved not linear.
            # This is NOT a linear relationship. See calc_blc_wedge in test_poppy.

            if np.abs(self.wavelength - 2.1e-6) < 0.1e-6:
                polyfitcoeffs = np.array([2.01210737e-04, -7.18758337e-03, 1.12381516e-01,
                                          -1.00877701e+00, 5.72538509e+00, -2.12943497e+01,
                                          5.18745152e+01, -7.97815606e+01, 7.02728734e+01])
            elif np.abs(self.wavelength - 4.6e-6) < 0.1e-6:
                polyfitcoeffs = np.array([9.16195583e-05, -3.27354831e-03, 5.11960734e-02,
                                          -4.59674047e-01, 2.60963397e+00, -9.70881273e+00,
                                          2.36585911e+01, -3.63978587e+01, 3.20703511e+01])
            else:
                raise NotImplemented("No defined NIRCam wedge BLC mask for that wavelength?  ")

            sigmas = scipy.poly1d(polyfitcoeffs)(scalefact)

            sigmar = sigmas * np.abs(y)
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs
            self.transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
            # the bar should truncate at +- 10 arcsec:
            woutside = np.where(np.abs(x) > 10)
            self.transmission[woutside] = 1.0

            # add in the ND squares. Note the positions are not exactly the same in the two wedges.
            # See the figures in Krist et al. of how the 6 ND squares are spaced among the 5
            # corongraph regions. Also add in the opaque border of the coronagraph mask holder.
            if np.abs(self.wavelength - 2.1e-6) < 0.1e-6:
                # half ND square on each side
                wnd = np.where(
                    (y > 5) &
                    (
                        ((x < -5) & (x > -10)) |
                        ((x > 7.5) & (x < 12.5))
                    )
                )
                wborder = np.where(np.abs(y) > 10)
            elif np.abs(self.wavelength - 4.6e-6) < 0.1e-6:
                wnd = np.where(
                    (y > 5) &
                    (
                        ((x < -7.5) & (x > -12.5)) |
                        (x > 5)
                    )
                )
                wborder = np.where((np.abs(y) > 10) | (x > 10))  # right end of mask holder

            self.transmission[wnd] = np.sqrt(1e-3)
            self.transmission[wborder] = 0

        if not np.isfinite(self.transmission.sum()):
            #stop()
            _log.warn("There are NaNs in the BLC mask - correcting to zero. (DEBUG LATER?)")
            self.transmission[np.where(np.isfinite(self.transmission) == False)] = 0
        return self.transmission


class IdealFQPM(AnalyticOpticalElement):
    """ Defines an ideal 4-quadrant phase mask coronagraph, with its retardance
    set perfectly to 0.5 waves at one specific wavelength and varying linearly on
    either side of that.  "Ideal" in the sense of ignoring chromatic effects other
    than just the direct scaling of the wavelength.

    Parameters
    ----------
    name : string
        Descriptive name
    wavelength : float
        Wavelength in meters for which the FQPM was designed, and at which there
        is exactly 1/2 a wave of retardance.

    """

    def __init__(self, name="unnamed FQPM ", wavelength=10.65e-6, **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name

        self.central_wavelength = wavelength

    def getPhasor(self, wave):
        """ Compute the amplitude transmission appropriate for a 4QPM for some given pixel spacing
        corresponding to the supplied Wavefront
        """

        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("4QPM getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _IMAGE)

        # TODO this computation could be sped up a lot w/ optimzations
        phase = np.empty(wave.shape)
        n0 = wave.shape[0] / 2
        n0 = int(round(n0))
        phase[:n0, :n0] = 0.5
        phase[n0:, n0:] = 0.5
        phase[n0:, :n0] = 0
        phase[:n0, n0:] = 0

        retardance = phase * self.central_wavelength / wave.wavelength

        #outFITS = fits.HDUList(fits.PrimaryHDU(retardance))
        #outFITS.writeto('retardance_fqpm.fits', clobber=True)
        #_log.info("Retardance is %f waves" % retardance.max())
        FQPM_phasor = np.exp(1.j * 2 * np.pi * retardance)
        return FQPM_phasor


class RectangularFieldStop(AnalyticOpticalElement):
    """ Defines an ideal rectangular field stop

    Parameters
    ----------
    name : string
        Descriptive name
    width, height: float
        Size of the field stop, in arcseconds. Default 0.5 width, height 5.
    angle : float
        Position angle of the field stop sides relative to the detector +Y direction, in degrees.

    """

    def __init__(self, name="unnamed field stop", width=0.5, height=5.0, angle=0, **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name
        self.width = float(width)  # width of square stop in arcseconds.
        self.height = float(height)  # height of square stop in arcseconds.
        self.angle = float(angle)
        self._default_display_size = max(height, width) * 1.2

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the field stop.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("IdealFieldStop getPhasor must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == _IMAGE)

        y, x = wave.coordinates()
        xnew = x * np.cos(np.deg2rad(self.angle)) + y * np.sin(np.deg2rad(self.angle))
        ynew = -x * np.sin(np.deg2rad(self.angle)) + y * np.cos(np.deg2rad(self.angle))
        x, y = xnew, ynew

        w_outside = np.where(
            (abs(y) > (self.height / 2)) |
            (abs(x) > (self.width / 2))
        )
        del x  # for large arrays, cleanup very promptly, before allocating self.transmission
        del y
        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0

        return self.transmission


class SquareFieldStop(RectangularFieldStop):
    """ Defines an ideal square field stop

    Parameters
    ----------
    name : string
        Descriptive name
    size : float
        Size of the field stop, in arcseconds. Default 20.
    angle : float
        Position angle of the field stop sides relative to the detector +Y direction, in degrees.

    """

    def __init__(self, name="unnamed field stop", size=20., angle=0, **kwargs):
        RectangularFieldStop.__init__(self, width=size, height=size, **kwargs)
        self.name = name
        #self.size = size            # size of square stop in arcseconds.
        self.height = self.width
        self.angle = angle
        self._default_display_size = size * 1.2

class CircularOcculter(AnalyticOpticalElement):
    """ Defines an ideal circular occulter (opaque circle)

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the occulting spot, in arcseconds. Default is 1.0

    """

    def __init__(self, name="unnamed occulter", radius=1.0, **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name
        self.radius = radius  # radius of circular occulter in arcseconds.
        self._default_display_size = 10
        #self.pixelscale=0

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _IMAGE)

        y, x = wave.coordinates()
        #y, x = np.indices(wave.shape)
        #y -= wave.shape[0]/2
        #x -= wave.shape[1]/2
        r = np.sqrt(x ** 2 + y ** 2)  #* wave.pixelscale
        w_inside = np.where(r <= self.radius)

        del x
        del y
        del r
        self.transmission = np.ones(wave.shape)
        self.transmission[w_inside] = 0

        return self.transmission


class BarOcculter(AnalyticOpticalElement):
    """ Defines an ideal bar occulter (like in MIRI's Lyot coronagraph)

    Parameters
    ----------
    name : string
        Descriptive name
    width : float
        width of the bar stop, in arcseconds. Default is 1.0
    angle : float
        position angle of the bar, rotated relative to the normal +y direction.

    """

    def __init__(self, name="bar occulter", width=1.0, angle=0, **kwargs):
        AnalyticOpticalElement.__init__(self, planetype=_IMAGE, **kwargs)
        self.name = name
        self.width = width
        self.angle = angle
        #self.pixelscale=0
        self._default_display_size = 10

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _IMAGE)

        y, x = wave.coordinates()

        xnew = x * np.cos(np.deg2rad(self.angle)) + y * np.sin(np.deg2rad(self.angle))
        w_inside = np.where(np.abs(xnew) <= self.width / 2)
        self.transmission = np.ones(wave.shape)
        self.transmission[w_inside] = 0

        return self.transmission


#------ Analytic Pupil Plane elements -----

class FQPM_FFT_aligner(AnalyticOpticalElement):
    """  Helper class for modeling FQPMs accurately

    Adds (or removes) a slight wavelength- and pixel-scale-dependent tilt
    to a pupil wavefront, to ensure the correct alignment of the image plane
    FFT'ed PSF with the desired quad pixel alignment for the FQPM.

    This is purely a computational convenience tool to work around the
    pixel coordinate restrictions imposed by the FFT algorithm,
    not a representation of any physical optic.

    Parameters
    ----------
    direction : string
        'forward' or 'backward'

    """

    def __init__(self, name="FQPM FFT aligner", direction='forward', **kwargs):
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)
        direction = direction.lower()
        if direction != 'forward' and direction != 'backward':
            raise ValueError("Invalid direction %s, must be either"
                             "forward or backward." % direction)
        self.direction = direction
        self._suppress_display = True
        #self.displayable = False

    def getPhasor(self, wave):
        """ Compute the required tilt needed to get the PSF centered on the corner between
        the 4 central pixels, not on the central pixel itself.
        """

        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("FQPM getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)

        fft_im_pixelscale = wave.wavelength / wave.diam / wave.oversample * _RADIANStoARCSEC
        required_offset = -fft_im_pixelscale * 0.5
        if self.direction == 'backward':
            required_offset *= -1
            wave._image_centered = 'pixel'
        else:
            wave._image_centered = 'corner'
        wave.tilt(required_offset, required_offset)

        # gotta return something... so return a value that will not affect the wave any more.
        align_phasor = 1.0
        return align_phasor


class ParityTestAperture(AnalyticOpticalElement):
    """ Defines a circular pupil aperture with boxes cut out.
    This is mostly a test aperture, which has no symmetry and thus can be used to
    test the various Fourier transform algorithms and sign conventions.

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0

    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.

    """

    def __init__(self, name=None, radius=1.0, pad_factor=1.5, **kwargs):
        if name is None: name = "Circle, radius=%.2f m" % radius
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)
        self.radius = radius
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("CircularAperture getPhasor must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == _PUPIL)

        y, x = wave.coordinates()
        r = np.sqrt(x ** 2 + y ** 2)  #* wave.pixelscale

        w_outside = np.where(r > self.radius)
        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0

        w_box1 = np.where(
            (r > (self.radius * 0.5)) &
            (np.abs(x) < self.radius * 0.1) &
            (y < 0)
        )
        w_box2 = np.where(
            (r > (self.radius * 0.75)) &
            (np.abs(y) < self.radius * 0.2) &
            (x < 0)
        )
        self.transmission[w_box1] = 0
        self.transmission[w_box2] = 0

        return self.transmission


class CircularAperture(AnalyticOpticalElement):
    """ Defines an ideal circular pupil aperture

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0

    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.
    """

    def __init__(self, name=None, radius=1.0, pad_factor=1.5, **kwargs):
        try:
            self.radius = float(radius)
        except ValueError:
            raise TypeError("Argument 'radius' must be the radius of the pupil in meters")

        if name is None:
            name = "Circle, radius=%.2f m" % radius
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius


    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("CircularAperture getPhasor must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == _PUPIL)

        y, x = wave.coordinates()
        r = np.sqrt(x ** 2 + y ** 2)
        del x
        del y

        w_outside = np.where(r > self.radius)
        del r
        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0
        return self.transmission


class HexagonAperture(AnalyticOpticalElement):
    """ Defines an ideal hexagonal pupil aperture

    Specify either the side length (= corner radius) or the
    flat-to-flat distance.

    Parameters
    ----------
    name : string
        Descriptive name
    side : float, optional
        side length (and/or radius) of hexagon, in meters. Overrides flattoflat if both are present.
    flattoflat : float, optional
        Distance between sides (flat-to-flat) of the hexagon, in meters. Default is 1.0
    """

    def __init__(self, name=None, flattoflat=None, side=None, **kwargs):
        if flattoflat is None and side is None:
            self.side = 1.0
        elif side is not None:
            self.side = float(side)
        else:
            self.side = float(flattoflat) / np.sqrt(3.)
        self.pupil_diam = 2 * self.side  # for creating input wavefronts
        if name is None:
            name = "Hexagon, side length= %.1f m" % self.side

        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)


    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("HexagonAperture getPhasor must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == _PUPIL)

        y, x = wave.coordinates()
        absy = np.abs(y)

        self.transmission = np.zeros(wave.shape)

        w_rect = np.where(
            (np.abs(x) <= 0.5 * self.side) &
            (absy <= np.sqrt(3) / 2 * self.side)
        )
        w_left_tri = np.where(
            (x <= -0.5 * self.side) &
            (x >= -1 * self.side) &
            (absy <= (x + 1 * self.side) * np.sqrt(3))
        )
        w_right_tri = np.where(
            (x >= 0.5 * self.side) &
            (x <= 1 * self.side) &
            (absy <= (1 * self.side - x) * np.sqrt(3))
        )
        self.transmission[w_rect] = 1
        self.transmission[w_left_tri] = 1
        self.transmission[w_right_tri] = 1

        return self.transmission


class MultiHexagonAperture(AnalyticOpticalElement):
    """ Defines a hexagonally segmented aperture

    Parameters
    ----------
    name : string
        Descriptive name
    rings : integer
        The number of rings of hexagons to include (
        i.e. 2 for a JWST-like aperture, 3 for a Keck-like aperture, and so on)
    side : float, optional
        side length (and/or radius) of hexagon, in meters. Overrides flattoflat if both are present.
    flattoflat : float, optional
        Distance between sides (flat-to-flat) of the hexagon, in meters. Default is 1.0
    gap: float, optional
        Gap between adjacent segments, in meters. Default is 0.01 m = 1 cm
    center : bool, optional
        should the central segment be included? Default is False. 
    segmentlist : list of ints, optional
        This allows one to specify that only a subset of segments are present, for a
        partially populated segmented telescope, non-redundant segment set, etc. 
        Segments are numbered from 0 for the center segment, 1 for the segment immediately
        above it, and then clockwise around each ring. 
        For example, segmentlist=[1,3,5] would make an aperture of 3 segments. 


    Note that this routine becomes a bit slow for nrings >4. For repeated computations on
    the same aperture, it will be faster to create this once, save it to a FITS file using
    the toFITS() method, and then use that. 

    """


    def __init__(self, name="MultiHex", flattoflat=1.0, side=None, gap=0.01, rings=1,
                 segmentlist=None, center=False, **kwargs):
        if flattoflat is None and side is None:
            self.side = 1.0
        elif side is not None:
            self.side = float(side)
        else:
            self.side = float(flattoflat) / np.sqrt(3.)
        self.flattoflat = self.side * np.sqrt(3)
        self.rings = rings
        self.gap = gap
        #self._label_values = True # undocumented feature to draw hex indexes into the array
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)

        self.pupil_diam = (self.flattoflat + self.gap) * (2 * self.rings + 1)

        # make a list of all the segments included in this hex aperture
        if segmentlist is not None:
            self.segmentlist = segmentlist
        else:
            self.segmentlist = range(self._nHexesInsideRing(self.rings + 1))
            if not center: self.segmentlist.remove(0)  # remove center segment 0


    def _nHexesInRing(self, n):
        """ How many hexagons in ring N? """
        return 1 if n == 0 else 6 * n

    def _nHexesInsideRing(self, n):
        """ How many hexagons interior to ring N, not counting N?"""
        return sum([self._nHexesInRing(i) for i in range(n)])

    def _hexInRing(self, hex_index):
        """ What ring is a given hexagon in?"""
        if hex_index == 0:
            return 0
        for i in range(100):
            if self._nHexesInsideRing(i) <= hex_index < self._nHexesInsideRing(i + 1):
                return i
        raise ValueError("Loop exceeded! MultiHexagonAperture is limited to <100 rings of hexagons.")

    def _hexRadius(self, hex_index):
        """ Radius of a given hexagon from the center """
        ring = self._hexInRing(hex_index)
        if ring <= 1:
            return (self.flattoflat + self.gap) * ring

    def _hexCenter(self, hex_index):
        """ Center coordinates of a given hexagon 
        counting clockwise around each ring

        Returns y, x coords

        """
        ring = self._hexInRing(hex_index)

        # now count around from the starting point:
        index_in_ring = hex_index - self._nHexesInsideRing(ring) + 1  # 1-based
        #print("hex %d is %dth in its ring" % (hex_index, index_in_ring))

        angle_per_hex = 2 * np.pi / self._nHexesInRing(ring)  # angle in radians

        # Now figure out what the radius is:
        xpos = None
        if ring <= 1:
            radius = (self.flattoflat + self.gap) * ring
            angle = angle_per_hex * (index_in_ring - 1)
        elif ring == 2:
            if np.mod(index_in_ring, 2) == 1:
                radius = (self.flattoflat + self.gap) * ring  # JWST 'B' segments
            else:
                radius = self.side * 3 + self.gap * np.sqrt(3.) / 2 * 2  # JWST 'C' segments
            angle = angle_per_hex * (index_in_ring - 1)
        elif ring == 3:
            if np.mod(index_in_ring, ring) == 1:
                radius = (self.flattoflat + self.gap) * ring  # JWST 'B' segments
                angle = angle_per_hex * (index_in_ring - 1)
            else:  # C-like segments (in pairs)
                ypos = 2.5 * (self.flattoflat + self.gap)
                xpos = 1.5 * self.side + self.gap * np.sqrt(3) / 4
                radius = np.sqrt(xpos ** 2 + ypos ** 2)
                Cangle = np.arctan2(xpos, ypos)

                if np.mod(index_in_ring, 3) == 2:
                    last_B_angle = ((index_in_ring - 1) // 3) * 3 * angle_per_hex
                    angle = last_B_angle + Cangle * np.mod(index_in_ring - 1, 3)
                else:
                    next_B_angle = (((index_in_ring - 1) // 3) * 3 + 3) * angle_per_hex
                    angle = next_B_angle - Cangle
                xpos = None
        else:  # generalized code!
            # the above are actuall now redundant given that this exists, but
            # I'll leave them alone for now.
            # TODO:jlong: remove redundant code paths?
            whichside = (index_in_ring - 1) // ring  # which of the sides are we on?

            if np.mod(index_in_ring, ring) == 1:
                radius = (self.flattoflat + self.gap) * ring  # JWST 'B' segments
                angle = angle_per_hex * (index_in_ring - 1)
            else:
                # find position of previous 'B' type segment.
                radius0 = (self.flattoflat + self.gap) * ring  # JWST 'B' segments
                last_B_angle = ((index_in_ring - 1) // ring) * ring * angle_per_hex
                #angle0 = angle_per_hex * (index_in_ring-1)
                ypos0 = radius0 * np.cos(last_B_angle)
                xpos0 = radius0 * np.sin(last_B_angle)

                da = (self.flattoflat + self.gap) * np.cos(30 * np.pi / 180)
                db = (self.flattoflat + self.gap) * np.sin(30 * np.pi / 180)

                if whichside == 0:
                    dx, dy = da, -db
                elif whichside == 1:
                    dx, dy = 0, -(self.flattoflat + self.gap)
                elif whichside == 2:
                    dx, dy = -da, -db
                elif whichside == 3:
                    dx, dy = -da, db
                elif whichside == 4:
                    dx, dy = 0, (self.flattoflat + self.gap)
                elif whichside == 5:
                    dx, dy = da, db

                xpos = xpos0 + dx * np.mod(index_in_ring - 1, ring)
                ypos = ypos0 + dy * np.mod(index_in_ring - 1, ring)

        # now clock clockwise around the ring (for rings <=3 only)
        if xpos is None:
            ypos = radius * np.cos(angle)
            xpos = radius * np.sin(angle)

        return ypos, xpos


    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)

        y, x = wave.coordinates()
        absy = np.abs(y)

        self.transmission = np.zeros(wave.shape)

        for i in self.segmentlist:
            self._oneHexagon(wave, i)

        return self.transmission

    def _oneHexagon(self, wave, index):
        """ Draw one hexagon into the self.transmission array """

        y, x = wave.coordinates()

        ceny, cenx = self._hexCenter(index)

        y -= ceny
        x -= cenx
        absy = np.abs(y)

        w_rect = np.where(
            (np.abs(x) <= 0.5 * self.side) &
            (absy <= np.sqrt(3) / 2 * self.side)
        )
        w_left_tri = np.where(
            (x <= -0.5 * self.side) &
            (x >= -1 * self.side) &
            (absy <= (x + 1 * self.side) * np.sqrt(3))
        )
        w_right_tri = np.where(
            (x >= 0.5 * self.side) &
            (x <= 1 * self.side) &
            (absy <= (1 * self.side - x) * np.sqrt(3))
        )

        #val = np.sqrt(float(index)) if self._label_values else 1
        val = 1
        self.transmission[w_rect] = val
        self.transmission[w_left_tri] = val
        self.transmission[w_right_tri] = val


class NgonAperture(AnalyticOpticalElement):
    """ Defines an ideal N-gon pupil aperture. 

    Parameters
    -----------
    name : string
        Descriptive name
    nsides : integer
        Number of sides. Default is 6.
    radius : float
        radius to the vertices, meters. Default is 1. 
    rotation : float
        Rotation angle to first vertex, in degrees counterclockwise from the +X axis. Default is 0.
    """

    def __init__(self, name=None, nsides=6, radius=1, rotation=0., **kwargs):
        self.radius = radius
        self.nsides = nsides
        self.rotation = rotation
        self.pupil_diam = 2 * self.radius  # for creating input wavefronts
        if name is None: name = "%d-gon, radius= %.1f m" % (self.nsides, self.radius)
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)
        y, x = wave.coordinates()

        phase = self.rotation * np.pi / 180
        vertices = np.zeros((self.nsides, 2), dtype=np.float64)
        for i in range(self.nsides):
            vertices[i] = [np.cos(i * 2 * np.pi / self.nsides + phase),
                           np.sin(i * 2 * np.pi / self.nsides + phase)]

        self.transmission = np.zeros(wave.shape)
        for row in range(wave.shape[0]):
            pts = np.asarray(zip(x[row], y[row]))
            #ok = matplotlib.nxutils.points_inside_poly(pts, vertices)
            ok = matplotlib.path.Path(vertices).contains_points(pts)  #, vertices)
            self.transmission[row][ok] = 1.0

        return self.transmission


class RectangleAperture(AnalyticOpticalElement):
    """ Defines an ideal rectangular pupil aperture

    Parameters
    ----------
    name : string
        Descriptive name
    width : float
        width of the rectangle, in meters. Default is 0.5
    height : float
        height of the rectangle, in meters. Default is 1.0
    rotation : float
        Rotation angle for 'width' axis. Default is 0.

    """

    def __init__(self, name=None, width=0.5, height=1.0, rotation=0.0, **kwargs):
        self.width = width
        self.height = height
        self.rotation = rotation
        if name is None:
            name = "Rectangle, size= {s.width:.1f} m wide * {s.height:.1f} m high".format(s=self)
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)
        # for creating input wavefronts:
        self.pupil_diam = np.sqrt(self.height ** 2 + self.width ** 2)

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)

        y, x = wave.coordinates()

        if self.rotation != 0:
            angle = np.deg2rad(self.rotation)
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = -np.sin(angle) * x + np.cos(angle) * y

            x = xp
            y = yp

        w_outside = np.where(
            (abs(y) > (self.height / 2)) |
            (abs(x) > (self.width / 2))
        )
        del y
        del x

        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0
        return self.transmission


class SquareAperture(RectangleAperture):
    """ Defines an ideal square pupil aperture

    Parameters
    ----------
    name : string
        Descriptive name
    size: float
        side length of the square, in meters. Default is 1.0
    rotation : float
        Rotation angle for the square. Default is 0.


    """

    def __init__(self, name=None, size=1.0, **kwargs):
        self._size = size
        if name is None:
            name = "Square, side length= %.1f m" % size * 2
        RectangleAperture.__init__(self, name=name, width=size, height=size, **kwargs)
        self.size = size
        self.pupil_diam = 2 * self.size  # for creating input wavefronts

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
        self.height = value
        self.width = value


class SecondaryObscuration(AnalyticOpticalElement):
    """ Defines the central obscuration of an on-axis telescope including secondary mirror and
    supports

    The number of supports is adjustable but they are always radially symmetric around the center.
    See AsymmetricSecondaryObscuration if you need more flexibility. 

    Parameters
    ----------
    secondary_radius : float
        Radius of the circular secondary obscuration. Default 0.5 m
    n_supports : int
        Number of secondary mirror supports ("spiders"). These will be 
        spaced equally around a circle.  Default is 4.
    support_width : float
        Width of each support, in meters. Default is 0.01 m = 1 cm.
    support_angle_offset : float
        Angular offset, in degrees, of the first secondary support from the X axis.
        
    """

    def __init__(self, name=None, secondary_radius=0.5, n_supports=4, support_width=0.01,
                 support_angle_offset=0.0, **kwargs):
        if name is None:
            name = "Secondary Obscuration with {0} supports".format(n_supports)
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)
        self.secondary_radius = secondary_radius
        self.n_supports = n_supports
        self.support_width = support_width
        self.support_angle_offset = support_angle_offset

        # for creating input wavefronts if this is the first optic in a opticalsystem:
        self.pupil_diam = 4 * self.secondary_radius

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the obscuration
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)

        self.transmission = np.ones(wave.shape)

        y, x = wave.coordinates()
        r = np.sqrt(x ** 2 + y ** 2)  #* wave.pixelscale

        self.transmission[r < self.secondary_radius] = 0

        for i in range(self.n_supports):
            angle = 2 * np.pi / self.n_supports * i + np.deg2rad(self.support_angle_offset)

            # calculate rotated x' and y' coordinates after rotation by that angle.
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = -np.sin(angle) * x + np.cos(angle) * y

            self.transmission[(xp > 0) & (np.abs(yp) < self.support_width / 2)] = 0

            # TODO check here for if there are no pixels marked because the spider is too thin.
            # In that case use a grey scale approximation

        return self.transmission


class AsymmetricSecondaryObscuration(SecondaryObscuration):
    """ Defines a central obscuration with one or more supports which can be oriented at
    arbitrary angles around the primary mirror, a la the three supports of JWST

    Parameters
    ----------
    secondary_radius : float
        Radius of the circular secondary obscuration. Default 0.5 m
    support_angle : ndarray or list of floats
        The angle measured counterclockwise from +Y for each support
    support_width : float, or list of floats
        if scalar, gives the width for all support struts
        if a list, gives separately the width for each support strut independently.
        Widths in meters. Default is 0.01 m = 1 cm.
    """

    def __init__(self, support_angle=(0, 90, 240), support_width=0.01, **kwargs):
        SecondaryObscuration.__init__(self, n_supports=len(support_angle), **kwargs)

        self.support_angle = np.asarray(support_angle)
        if np.isscalar(support_width):
            support_width = np.zeros(len(support_angle)) + support_width
        self.support_width = support_width

    def getPhasor(self, wave):
        """ Compute the transmission inside/outside of the obscuration
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == _PUPIL)

        self.transmission = np.ones(wave.shape)

        y, x = wave.coordinates()
        r = np.sqrt(x ** 2 + y ** 2)  #* wave.pixelscale

        self.transmission[r < self.secondary_radius] = 0

        for angle_deg, width in zip(self.support_angle, self.support_width):
            angle = np.deg2rad(angle_deg + 90)  # 90 deg offset is to start from the +Y direction

            # calculate rotated x' and y' coordinates after rotation by that angle.
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = -np.sin(angle) * x + np.cos(angle) * y

            self.transmission[(xp > 0) & (np.abs(yp) < width / 2)] = 0

            # TODO check here for if there are no pixels marked because the spider is too thin.
            # In that case use a grey scale approximation

        return self.transmission


class ThinLens(CircularAperture):
    """ An idealized thin lens, implemented as a Zernike defocus term.

    Parameters
    -------------
    nwaves : float
        The number of waves of defocus, peak to valley. May be positive or negative.
        This is applied as a normalization over an area defined by the circumscribing circle
        of the input wavefront. That is, there will be nwaves defocus peak-to-valley
        over the region of the pupil that has nonzero input intensity.
    reference_wavelength : float
        Wavelength, in meters, at which that number of waves of defocus is specified. 
    radius : float
        Pupil radius, in meters, over which the Zernike defocus term should be computed
        such that rho = 1 at r = `radius`.
    """

    def __init__(self, name='Thin lens', nwaves=4.0, reference_wavelength=2e-6,
                 radius=None, **kwargs):
        self.reference_wavelength = reference_wavelength
        self.nwaves = nwaves
        self.max_phase_delay = reference_wavelength * nwaves
        CircularAperture.__init__(self, name=name, radius=radius, **kwargs)

    def getPhasor(self, wave):
        y, x = wave.coordinates()
        r = np.sqrt(x ** 2 + y ** 2)
        r_norm = r / self.radius

        # the thin lens, being circular, is implicitly also a circular aperture:
        aperture_intensity = CircularAperture.getPhasor(self, wave)

        # don't forget the factor of 0.5 to make the scaling factor apply as peak-to-valley
        # rather than center-to-peak
        defocus_zernike = ((2 * r_norm ** 2 - 1) *
                           (0.5 * self.nwaves * self.reference_wavelength / wave.wavelength))

        lens_phasor = np.exp(1.j * 2 * np.pi * defocus_zernike * aperture_intensity)

        return lens_phasor


class ParameterizedAberration(AnalyticOpticalElement):
    """
    Define an optical element in terms of its distortion as decomposed into a set or orthonormal
    basis functions (e.g. Zernikes, Hexikes, etc.). Included basis functions are normalized
    such that user-provided coefficients correspond to meters RMS wavefront aberration for that
    basis function.

    Parameters
    ----------
    coefficients : iterable of numbers
        The contribution of each term to the final distortion, in meters RMS wavefront error.
        The coefficients are interpreted as indices in the order of Noll et al. 1976: the first
        term corresponds to j=1, second to j=2, and so on.
    radius : float
        Pupil radius, in meters. Defines the region of the input wavefront array over which
        the distortion terms will be evaluated. For non-circular pupils, this should be the
        circle circumscribing the actual pupil shape.
    basis_factory : callable
        basis_factory will be called with the arguments `nterms`, `rho`, and `theta`.
        `nterms` specifies how many terms to compute, starting with the j=1 term in the
        Noll indexing convention for `nterms` = 1 and counting up. `rho` and `theta` are square
        arrays holding the rho and theta coordinates at each pixel in the pupil plane.

        `rho` is normalized such that `rho` == 1.0 for pixels at `radius` meters from
        the center.
    """
    def __init__(self, name="Parameterized Distortion", coefficients=None, radius=None,
                 basis_factory=None, **kwargs):
        if not callable(basis_factory):
            raise ValueError("'basis_factory' must be a callable that can "
                             "calculate basis functions")
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")
        self.coefficients = coefficients
        self.basis_factory = basis_factory
        AnalyticOpticalElement.__init__(self, name=name, planetype=_PUPIL, **kwargs)

    def getPhasor(self, wave):
        rho, theta = _wave_to_rho_theta(wave, self.radius)
        combined_distortion = np.zeros(rho.shape)

        nterms = len(self.coefficients)
        computed_terms = self.basis_factory(nterms=nterms, rho=rho, theta=theta)

        for idx, coefficient in enumerate(self.coefficients):
            if coefficient == 0.0:
                continue  # save the trouble of a multiply-and-add of zeros
            combined_distortion += coefficient * computed_terms[idx]

        opd_as_phase = 2 * np.pi * combined_distortion / wave.wavelength
        return np.exp(1.0j * opd_as_phase)


def _wave_to_rho_theta(wave, pupil_radius):
    """
    Return wave coordinates in (rho, theta) for a Wavefront object normalized such that
    rho == 1.0 at the pupil radius

    Parameters
    ----------
    wave : Wavefront
        Wavefront object with a `coordinates` method that returns (y, x)
        coordinate arrays in meters in the pupil plane
    pupil_radius : float, optional
        Radius (in meters) of a circle circumscribing the pupil.
        If `None`, this function will attempt to guess the radius from the wave
        intensity array using `_guess_pupil_radius`.
    """
    y, x = wave.coordinates()
    r = np.sqrt(x ** 2 + y ** 2)

    rho = r / pupil_radius
    theta = np.arctan2(y / pupil_radius, x / pupil_radius)

    return rho, theta


class ZernikeAberration(CircularAperture):
    """
    Define an optical element in terms of its Zernike components by providing coefficients
    for each Zernike term modeled by the analytic optical element.

    Parameters
    ----------
    coefficients : iterable of 3-tuples
        Each 3-tuple in coefficients must be of the form (n, m, k), where n and m are the integer
        radial degree and azimuthal frequency indices of the Zernike, and k is the RMS wavefront
        aberration over the pupil in meters for that Zernike component.
    radius : float
        Pupil radius, in meters, over which the Zernike terms should be computed such that
        rho = 1 at r = `radius`.
    """
    def __init__(self, name="Zernike Optic", coefficients=None, radius=None, **kwargs):
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")

        def _validate_coefficients():
            if coefficients is None:
                return False
            for coeff_tuple in coefficients:
                if len(coeff_tuple) != 3:
                    return False
                if not int(coeff_tuple[0]) == coeff_tuple[0] \
                        or not int(coeff_tuple[1]) == coeff_tuple[1]:
                    return False
            return True

        if not _validate_coefficients():
            raise ValueError("Coefficients must be supplied as a sequence of tuples with (n, m, k) "
                             "where n, m are the indices of the Zernike, and k is a leading "
                             "coefficient in meters of wavefront error. "
                             "e.g. coefficients=[(2, 0, 0.214), (2, -2, 0.02)]")
        self.coefficients = coefficients
        CircularAperture.__init__(self, name=name, radius=self.radius, **kwargs)

    def getPhasor(self, wave):
        rho, theta = _wave_to_rho_theta(wave, self.radius)

        # the Zernike optic, being circular, is implicitly also a circular aperture:
        aperture_intensity = CircularAperture.getPhasor(self, wave)

        combined_zernikes = np.zeros(wave.shape, dtype=np.float64)
        for n, m, k in self.coefficients:
            combined_zernikes += k * zernike.zernike(n, m, rho=rho, theta=theta,
                                                     mask_outside=True, outside=0.0)

        combined_zernikes *= aperture_intensity

        opd_as_phase = 2 * np.pi * combined_zernikes / wave.wavelength
        lens_phasor = np.exp(1.j * opd_as_phase)
        return lens_phasor


#------ generic analytic optics ------

class CompoundAnalyticOptic(AnalyticOpticalElement):
    """ Define a compound analytic optical element made up of the combination
    of two or more individual optical elements.

    This is just a convenience routine for semantic organization of optics.
    It can be useful to keep the list of optical planes cleaner, but
    you can certainly just add a whole bunch of planes all in a row without
    using this class to group them.

    All optics should be of the same plane type (pupil or image); propagation between
    different optics contained inside one compound is not supported.

    Parameters
    ----------
    opticslist : list
        A list of AnalyticOpticalElements to be merged together.

    """

    def _validate_only_analytic_optics(self, optics_list):
        for optic in optics_list:
            if isinstance(optic, AnalyticOpticalElement):
                continue  # analytic elements are allowed
            elif isinstance(optic, InverseTransmission):
                if isinstance(optic.uninverted_optic, AnalyticOpticalElement):
                    continue  # inverted elements are allowed, as long as they're analytic elements
                else:
                    return False  # inverted non-analytic elements aren't allowed, skip the rest
            else:
                return False  # no other types allowed, skip the rest of the list
        return True

    def __init__(self, opticslist=None, name="unnamed", verbose=True, **kwargs):
        if opticslist is None:
            raise ValueError("Missing required opticslist argument to CompoundAnalyticOptic")
        AnalyticOpticalElement.__init__(self, name=name, verbose=verbose, **kwargs)

        #self.operation = operation
        self.opticslist = []
        self._default_display_size = 3
        self.planetype = None

        for optic in opticslist:
            if not self._validate_only_analytic_optics(opticslist):
                raise ValueError("Supplied optics list to CompoundAnalyticOptic can "
                                 "only contain AnalyticOptics")
            else:
                # if we are adding the first optic in the list, check what type of optical plane
                # it has
                # for subsequent optics, validate they have the same type
                if len(self.opticslist) == 0:
                    self.planetype = optic.planetype
                elif self.planetype != optic.planetype:
                    raise ValueError("Cannot mix image plane and pupil plane optics in "
                                     "the same CompoundAnalyticOptic")

                self.opticslist.append(optic)
                if hasattr(optic, '_default_display_size'):
                    self._default_display_size = max(self._default_display_size,
                                                     optic._default_display_size)

        if self.planetype == _PUPIL:
            if all([hasattr(o, 'pupil_diam') for o in self.opticslist]):
                self.pupil_diam = np.asarray([o.pupil_diam for o in self.opticslist]).max()

    def getPhasor(self, wave):
        phasor = np.ones(wave.shape, dtype=np.complex)
        for optic in self.opticslist:
            nextphasor = optic.getPhasor(wave)
            phasor *= nextphasor
        return phasor
