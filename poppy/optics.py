import numpy as np
import scipy.special
import scipy.ndimage.interpolation
import matplotlib
import astropy.io.fits as fits
import astropy.units as u
import warnings
import logging
from abc import ABC, abstractmethod

from . import utils
from . import conf
from . import accel_math
from .poppy_core import OpticalElement, Wavefront, BaseWavefront, PlaneType, _RADIANStoARCSEC
from .accel_math import _exp, _r, _float, _complex
from . import geometry

if accel_math._USE_NUMEXPR:
    import numexpr as ne

_log = logging.getLogger('poppy')

__all__ = ['AnalyticOpticalElement', 'ScalarTransmission', 'ScalarOpticalPathDifference', 'InverseTransmission',
           'BandLimitedCoron', 'BandLimitedCoronagraph', 'IdealFQPM', 'CircularPhaseMask', 'RectangularFieldStop', 'SquareFieldStop',
           'AnnularFieldStop', 'HexagonFieldStop',
           'CircularOcculter', 'BarOcculter', 'FQPM_FFT_aligner', 'CircularAperture',
           'HexagonAperture', 'MultiHexagonAperture', 'NgonAperture', 'MultiCircularAperture', 'RectangleAperture',
           'SquareAperture', 'SecondaryObscuration', 'LetterFAperture', 'AsymmetricSecondaryObscuration',
           'ThinLens',  'GaussianAperture', 'KnifeEdge', 'TiltOpticalPathDifference', 'CompoundAnalyticOptic', 'fixed_sampling_optic']

# ------ Generic Analytic elements -----

class AnalyticOpticalElement(OpticalElement):
    """ Defines an abstract analytic optical element, i.e. one definable by
        some formula rather than by an input OPD or pupil file.

        This class is useless on its own; instead use its various subclasses
        that implement appropriate get_opd and/or get_transmission functions.
        It exists mostly to provide some behaviors & initialization common to
        all analytic optical elements.

        Parameters
        ----------
        name, verbose, oversample, planetype : various
            Same as for OpticalElement
        transmission, opd : string
            These are *not allowed* for Analytic optical elements, and this class will raise an
            error if you try to set one.
        shift_x, shift_y : Optional floats
            Translations of this optic, given in meters relative to the optical
            axis for pupil plane elements, or arcseconds relative to the optical axis
            for image plane elements.
        rotation : Optional float
            Rotation of the optic around its center, given in degrees
            counterclockwise.  Note that if you apply both shift and rotation,
            the optic rotates around its own center, rather than the optical
            axis.

    """

    def __init__(self, shift_x=None, shift_y=None, rotation=None,
            inclination_x=None, inclination_y=None,
            **kwargs):
        OpticalElement.__init__(self, **kwargs)

        if shift_x is not None: self.shift_x = shift_x
        if shift_y is not None: self.shift_y = shift_y
        if rotation is not None: self.rotation = rotation
        if inclination_x is not None: self.inclination_x = inclination_x
        if inclination_y is not None: self.inclination_y = inclination_y

        if getattr(self, 'inclination_x', 0) != 0 and getattr(self, 'inclination_y', 0) != 0:
            warnings.warn("It is physically inconsistent to set inclinations on both X and Y at the same time.")
        if np.abs(getattr(self, 'inclination_x', 0)) > 90 or np.abs(getattr(self, 'inclination_y', 0)) > 90:
            warnings.warn("Inclinations should be within the range -90 to 90 degrees")

        # self.shape = None  # no explicit shape required
        self.pixelscale = None

    @property
    def shape(self):  # Analytic elements don't have shape
        return None

    def __str__(self):
        if self.planetype == PlaneType.pupil:
            return "Pupil plane: " + self.name
        elif self.planetype == PlaneType.image:
            return "Image plane: " + self.name
        else:
            return "Optic: " + self.name

    # The following two functions should be replaced by derived subclasses
    # but we provide a default of perfect transmission and zero OPD.
    # Each must return something which is a numpy ndarray.
    def get_opd(self, wave):
        return np.zeros(wave.shape, dtype=_float())

    def get_transmission(self, wave):
        """ Note that this is the **amplitude** transmission, not the
        total intensity transmission. """
        return np.ones(wave.shape, dtype=_float())

    # noinspection PyUnusedLocal
    def get_phasor(self, wave):
        """ Compute a complex phasor from an OPD, given a wavelength.

        The returned value should be the complex phasor array as appropriate for
        multiplying by the wavefront amplitude.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        """
        if isinstance(wave, BaseWavefront):
            wavelength = wave.wavelength
        else:
            wavelength = wave
        scale = 2. * np.pi / wavelength.to(u.meter).value

        if accel_math._USE_NUMEXPR:
            trans = self.get_transmission(wave)
            opd = self.get_opd(wave)
            # we first multiply the two scalars, for a slight performance gain
            scalars = 1.j * scale
            # warning, numexpr exp is crash-prone if fed complex64, so we
            # leave the scalars variable as np.complex128 for reliability
            result = ne.evaluate("trans * exp( opd * scalars)")

            # TODO if single-precision, need to cast the result back to that
            # to work around a bug
            # Not sure why numexpr is casting up to complex128
            # see https://github.com/pydata/numexpr/issues/155
            # (Yes this is inefficient to do math as doubles if in single mode, but
            # numexpr is still a net win)
            if conf.double_precision:
                return result
            else:
                return np.asarray(result, _complex())

        else:
            return self.get_transmission(wave) * np.exp(1.j * self.get_opd(wave) * scale)

    @utils.quantity_input(wavelength=u.meter)
    def sample(self, wavelength=1e-6 * u.meter, npix=512, grid_size=None, what='amplitude',
               return_scale=False, phase_unit='waves'):
        """ Sample the Analytic Optic onto a grid and return the array

        Parameters
        ----------
        wavelength : astropy.units.Quantity or float
            Wavelength (in meters if unit not given explicitly)
        npix : integer
            Number of pixels for sampling the array
        grid_size : float
            Field of view grid size (diameter) for sampling the optic, in meters for
            pupil plane optics and arcseconds for image planes. Default value is
            taken from the optic's properties, if defined. Otherwise defaults to
            6.5 meters or 2 arcseconds depending on plane.
        what : string
            What to return: optic 'amplitude' transmission, 'intensity' transmission,
            'phase', or 'opd'.  Note that optical path difference, OPD, is given in meters.
        phase_unit : string
            Unit for returned phase array IF what=='phase'. One of 'radians', 'waves', 'meters'.
            ('meters' option is deprecated; use what='opd' instead.)
        return_scale : float
            if True, will return a tuple containing the desired array and a float giving the
            pixel scale.
        """
        if self.planetype != PlaneType.image:
            if grid_size is not None:
                diam = grid_size if isinstance(grid_size, u.Quantity) else grid_size * u.meter
            elif hasattr(self, '_default_display_size'):
                diam = self._default_display_size
            elif hasattr(self, 'pupil_diam'):
                diam = self.pupil_diam * 1
            else:
                diam = 1.0 * u.meter
            w = Wavefront(wavelength=wavelength, npix=npix, diam=diam)
            pixel_scale = diam / (npix * u.pixel)

        else:
            if grid_size is not None:
                fov = grid_size if isinstance(grid_size, u.Quantity) else grid_size * u.arcsec
            elif hasattr(self, '_default_display_size'):
                fov = self._default_display_size
            else:
                fov = 4 * u.arcsec
            pixel_scale = fov / (npix * u.pixel)
            w = Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixel_scale)

        _log.info("Computing {0} for {1} sampled onto {2} pixel grid with pixelscale {3}".format(what, self.name, npix, pixel_scale))
        if what == 'amplitude':
            output_array = self.get_transmission(w)
        elif what == 'intensity':
            output_array = self.get_transmission(w) ** 2
        elif what == 'phase':
            if phase_unit == 'radians':
                output_array = np.angle(phasor) * 2 * np.pi / wavelength
            elif phase_unit == 'waves':
                output_array = self.get_opd(w) / wavelength
            elif phase_unit == 'meters':
                warnings.warn("'phase_unit' parameter has been deprecated. Use what='opd' instead.",
                              category=DeprecationWarning)
                output_array = self.get_opd(w)
            else:
                warnings.warn("'phase_unit' parameter has been deprecated. Use what='opd' instead.",
                              category=DeprecationWarning)
                raise ValueError('Invalid/unknown phase_unit: {}. Must be one of '
                                 '[radians, waves, meters]'.format(phase_unit))
        elif what == 'opd':
            output_array = self.get_opd(w)
        elif what == 'complex':
            output_array = self.get_phasor(w)
        else:
            raise ValueError('Invalid/unknown what to sample: {}. Must be one of '
                             '[amplitude, intensity, phase, opd, complex]'.format(what))

        if return_scale:
            return output_array, pixel_scale
        else:
            return output_array

    @utils.quantity_input(wavelength=u.meter)
    def to_fits(self, outname=None, what='amplitude', wavelength=1e-6 * u.meter, npix=512, **kwargs):
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
        try:
            from .version import version
        except ImportError:
            version = ''

        kwargs['return_scale'] = True

        if what == 'complex':
            raise ValueError("FITS cannot handle complex arrays directly. Save the amplitude and opd separately.")

        output_array, pixelscale = self.sample(wavelength=wavelength, npix=npix, what=what,
                                               **kwargs)
        long_contents = {'amplitude': "Electric field amplitude transmission",
                         'intensity': "Electric field intensity transmission",
                         'opd': "Optical path difference",
                         'phase': "Wavefront phase delay"}

        phdu = fits.PrimaryHDU(output_array)
        phdu.header['OPTIC'] = (self.name, "Descriptive name of this optic")
        phdu.header['NAME'] = self.name
        phdu.header['SOURCE'] = 'Computed with POPPY'
        phdu.header['VERSION'] = (version, "software version of POPPY")
        phdu.header['CONTENTS'] = (what, long_contents[what])
        phdu.header['PLANETYP'] = (self.planetype.value, "0=unspecified, 1=pupil, 2=image, 3=detector, 4=rot")
        if self.planetype == PlaneType.image:
            phdu.header['PIXELSCL'] = (pixelscale.to(u.arcsec / u.pixel).value, 'Image plane pixel scale in arcsec/pix')
            outFITS[0].header['PIXUNIT'] = ('arcsecond', "Unit for PIXELSCL")
        else:
            phdu.header['PUPLSCAL'] = (pixelscale.to(u.meter / u.pixel).value, 'Pupil plane pixel scale in meter/pix')
            phdu.header['PIXELSCL'] = (phdu.header['PUPLSCAL'], 'Pupil plane pixel scale in meter/pix')
            phdu.header['PIXUNIT'] = ('meter', "Unit for PIXELSCL")
        if what == 'opd':
            phdu.header['BUNIT'] = ('meter', "Optical Path Difference is given in meters.")

        if hasattr(self, 'shift_x'):
            phdu.header['SHIFTX'] = (self.shift_x, "X axis shift of input optic")
        if hasattr(self, 'shift_y'):
            phdu.header['SHIFTY'] = (self.shift_y, "Y axis shift of input optic")
        if hasattr(self, 'rotation'):
            phdu.header['ROTATION'] = (self.rotation, "Rotation of input optic, in deg")

        hdul = fits.HDUList(hdus=[phdu])

        if outname is not None:
            phdu.writeto(outname, overwrite=True)
            _log.info("Output written to " + outname)

        return hdul

    def get_coordinates(self, wave):
        """Get coordinates of this optic, optionally including shifts

        Method: Calls the supplied wave object's coordinates() method,
        then checks for the existence of the following attributes:
        "shift_x", "shift_y", "rotation", "inclination_x", "inclination_y"
        If any of them are present, then the coordinates are modified accordingly.

        Shifts are given by default implicitly in meters for pupil optics and arcseconds for image
        plane optics. Shifts may optionally also be given with explicit units using Astropy Quantities,
        which in this case must be convertable into meters or arcseconds as appropriate.

        Rotations and inclinations are given implicitly in degrees.

        For multiple transformations, the order of operations is:
            shift, rotate, incline.
        """

        y, x = wave.coordinates()
        if hasattr(self, "shift_x"):
            if isinstance(self.shift_x, u.Quantity):
                desired_unit = u.arcsecond if self.planetype==PlaneType.image else u.meter
                shift_value = self.shift_x.to_value(desired_unit)
                x -= float(shift_value)
            else:
                x -= float(self.shift_x)
        if hasattr(self, "shift_y"):
            if isinstance(self.shift_y, u.Quantity):
                desired_unit = u.arcsecond if self.planetype == PlaneType.image else u.meter
                shift_value = self.shift_y.to_value(desired_unit)
                y -= float(shift_value)
            else:
                y -= float(self.shift_y)
        if hasattr(self, "rotation"):
            angle = np.deg2rad(self.rotation)
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = -np.sin(angle) * x + np.cos(angle) * y
            x = xp
            y = yp
        # inclination around X axis rescales Y, and vice versa:
        if hasattr(self, "inclination_x"):
            y /= np.cos(np.deg2rad(self.inclination_x))
        if hasattr(self, "inclination_y"):
            x /= np.cos(np.deg2rad(self.inclination_y))

        return y, x


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
        self.wavefront_display_hint = 'intensity'

    def get_transmission(self, wave):
        res = np.empty(wave.shape, dtype=_float())
        res.fill(self.transmission)
        return res


class ScalarOpticalPathDifference(AnalyticOpticalElement):
    """Uniform and constant optical path difference

    """
    @utils.quantity_input(opd=u.meter)
    def __init__(self, name=None, opd=1.0*u.micron, **kwargs):
        if name is None:
            name = f'Constant OPD, {opd}'

        super().__init__(name=name, **kwargs)
        self.opd = opd

    def get_opd(self, wave):
        res = np.empty(wave.shape, dtype=_float())
        res.fill(self.opd.to(u.meter).value)
        return res


class InverseTransmission(AnalyticOpticalElement):
    """ Given any arbitrary OpticalElement with transmission T(x,y)
    return the inverse transmission 1 - T(x,y)

    This is a useful ingredient in the SemiAnalyticCoronagraph algorithm.
    """

    def __init__(self, optic=None):
        super(InverseTransmission, self).__init__()
        if optic is None or not hasattr(optic, 'get_transmission'):
            raise ValueError("Need to supply an valid optic to invert!")
        self.uninverted_optic = optic
        self.name = "1 - " + optic.name
        self.planetype = optic.planetype
        self.pixelscale = optic.pixelscale
        self.oversample = optic.oversample
        if hasattr(self.uninverted_optic, '_default_display_size'):
            self._default_display_size = self.uninverted_optic._default_display_size

    @property
    def shape(self):  # override parent class shape function
        return self.uninverted_optic.shape

    def get_transmission(self, wave):
        return 1 - self.uninverted_optic.get_transmission(wave)

    def get_opd(self, wave):
        return self.uninverted_optic.get_opd(wave)

    def display(self, **kwargs):
        if isinstance(self.uninverted_optic, AnalyticOpticalElement):
            AnalyticOpticalElement.display(self, **kwargs)
        else:
            OpticalElement.display(self, **kwargs)


# ------ Analytic Image Plane elements (coordinates in arcsec) -----

class AnalyticImagePlaneElement(AnalyticOpticalElement):
    """ Parent virtual class for AnalyticOptics which are
    dimensioned in angular units such as arcseconds, rather
    than physical length units such as meters.
    """

    def __init__(self, name='Generic image plane optic', *args, **kwargs):
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.image, *args, **kwargs)
        self.wavefront_display_hint = 'intensity'  # preferred display for wavefronts at this plane


class BandLimitedCoronagraph(AnalyticImagePlaneElement):
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
    allowable_kinds = ['circular', 'linear']
    """ Allowable types of BLC supported by this class"""

    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed BLC", kind='circular', sigma=1, wavelength=None, **kwargs):
        AnalyticImagePlaneElement.__init__(self, name=name, **kwargs)

        self.kind = kind.lower()  # either circular or linear
        if self.kind in ['nircamwedge', 'nircamcircular']:
            import warnings
            warnings.warn('JWST NIRCam specific functionality in poppy.BandLimitedCoron is moving to ' +
                          'webbpsf.NIRCam_BandLimitedCoron. The "nircamwedge" and "nircamcircular" options ' +
                          'in poppy will be removed in a future version of poppy.', DeprecationWarning)
        elif self.kind not in self.allowable_kinds:
            raise ValueError("Invalid value for kind of BLC: " + self.kind)
        self.sigma = float(sigma)  # size parameter. See section 2.1 of Krist et al. SPIE 2007, 2009
        if wavelength is not None:
            self.wavelength = float(wavelength)  # wavelength, for selecting the
            # linear wedge option only
        self._default_display_size = 20. * u.arcsec  # default size for onscreen display, sized for NIRCam

    def get_transmission(self, wave):
        """ Compute the amplitude transmission appropriate for a BLC for some given pixel spacing
        corresponding to the supplied Wavefront.

        Based on the Krist et al. SPIE paper on NIRCam coronagraph design

        Note that the equations in Krist et al specify the intensity transmission of the occulter,
        but what we want to return here is the amplitude transmittance. That is the square root
        of the intensity, of course, so the equations as implemented here all differ from those
        written in Krist's SPIE paper by lacking an exponential factor of 2. Thanks to John Krist
        for pointing this out.

        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("BLC get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        if self.kind == 'circular':
            # larger sigma implies narrower peak? TBD verify if this is correct
            #
            r = _r(x, y)
            sigmar = self.sigma * r
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs

            self.transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)
            self.transmission[r == 0] = 0  # special case center point (value based on L'Hopital's rule)
        elif self.kind == 'nircamcircular':
            # larger sigma implies narrower peak? TBD verify if this is correct
            #
            r = _r(x, y)
            sigmar = self.sigma * r
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs
            self.transmission = (1 - (2 * scipy.special.jn(1, sigmar) / sigmar) ** 2)

            # add in the ND squares. Note the positions are not exactly the same in the two wedges.
            # See the figures  in Krist et al. of how the 6 ND squares are spaced among the 5
            # corongraph regions
            # Also add in the opaque border of the coronagraph mask holder.
            if self.sigma > 4:
                # MASK210R has one in the corner and one half in the other corner
                wnd = (
                    (y > 5) &
                    (
                            ((x < -5) & (x > -10)) |
                            ((x > 7.5) & (x < 12.5))
                    )
                )
                wborder = ((np.abs(y) > 10) | (x < -10))  # left end of mask holder
            else:
                # the others have two halves on in each corner.
                wnd = (
                    (y > 5) &
                    (np.abs(x) > 7.5) &
                    (np.abs(x) < 12.5)
                )
                wborder = (np.abs(y) > 10)

            self.transmission[wnd] = np.sqrt(1e-3)
            self.transmission[wborder] = 0
            self.transmission[r == 0] = 0  # special case center point (value based on L'Hopital's rule)
        elif self.kind == 'linear':
            sigmar = self.sigma * np.abs(y)
            sigmar.clip(np.finfo(sigmar.dtype).tiny, out=sigmar)  # avoid divide by zero -> NaNs
            self.transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        elif self.kind == 'nircamwedge':
            # This is hard-coded to the wedge-plus-flat-regions shape for NIRCAM

            # we want a scale factor that goes from 2 to 6 with 1/5th of it as a fixed part on
            # either end
            # scalefact = np.linspace(1,7, x.shape[1]).clip(2,6)

            # the scale fact should depent on X coord in arcsec, scaling across a 20 arcsec FOV.
            # map flat regions to 2.5 arcsec each?
            # map -7.5 to 2, +7.5 to 6. slope is 4/15, offset is +9.5
            scalefact = (2 + (-x + 7.5) * 4 / 15).clip(2, 6)

            # scalefact *= self.sigma / 2 #;2.2513
            # scalefact *= 2.2513
            # scalefact.shape = (1, x.shape[1])
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
            self.transmission[np.abs(x) > 10] = 1.0

            # add in the ND squares. Note the positions are not exactly the same in the two wedges.
            # See the figures in Krist et al. of how the 6 ND squares are spaced among the 5
            # corongraph regions. Also add in the opaque border of the coronagraph mask holder.
            if np.abs(self.wavelength - 2.1e-6) < 0.1e-6:
                # half ND square on each side
                wnd = (
                    (y > 5) &
                    (
                            ((x < -5) & (x > -10)) |
                            ((x > 7.5) & (x < 12.5))
                    )
                )
                wborder = (np.abs(y) > 10)
            elif np.abs(self.wavelength - 4.6e-6) < 0.1e-6:
                wnd = (
                    (y > 5) &
                    (
                            ((x < -7.5) & (x > -12.5)) |
                            (x > 5)
                    )
                )
                wborder = ((np.abs(y) > 10) | (x > 10))  # right end of mask holder

            self.transmission[wnd] = np.sqrt(1e-3)
            self.transmission[wborder] = 0

        if not np.isfinite(self.transmission.sum()):
            _log.warning("There are NaNs in the BLC mask - correcting to zero. (DEBUG LATER?)")
            self.transmission[ np.isnan(self.transmission) ] = 0
        return self.transmission

BandLimitedCoron=BandLimitedCoronagraph # Back compatibility for old name.


class IdealFQPM(AnalyticImagePlaneElement):
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

    @utils.quantity_input(wavelength=u.meter)
    def __init__(self, name="unnamed FQPM ", wavelength=10.65e-6 * u.meter, **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name

        self.central_wavelength = wavelength

    def get_opd(self, wave):
        """ Compute the OPD appropriate for a 4QPM for some given pixel spacing
        corresponding to the supplied Wavefront
        """

        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("4QPM get_opd must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        phase = (1 - np.sign(x) * np.sign(y)) * 0.25

        return phase * self.central_wavelength.to(u.meter).value


class CircularPhaseMask(AnalyticImagePlaneElement):
    """ Circular phase mask coronagraph, with its retardance
    set perfectly at one specific wavelength and varying linearly on
    either side of that.

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the mask
    wavelength : float
        Wavelength in meters for which the phase mask was designed
    retardance : float
        Optical path delay at that wavelength, specified in waves
        relative to the reference wavelengt. Default is 0.5.

    """

    @utils.quantity_input(radius=u.arcsec, wavelength=u.meter)
    def __init__(self, name=None, radius=1*u.arcsec, wavelength=1e-6 * u.meter, retardance=0.5,
            **kwargs):
        if name is None:
            name = "Phase mask r={:.3g}".format(radius)
        AnalyticImagePlaneElement.__init__(self, name=name, **kwargs)
        self.wavefront_display_hint = 'phase'  # preferred display for wavefronts at this plane
        self._default_display_size = 4*radius

        self.central_wavelength = wavelength
        self.radius = radius
        self.retardance = retardance

    def get_opd(self, wave):
        """ Compute the OPD appropriate for that phase mask for some given pixel spacing
        corresponding to the supplied Wavefront
        """

        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_opd must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        r = _r(x, y)

        self.opd= np.zeros(wave.shape, dtype=_float())
        radius = self.radius.to(u.arcsec).value

        self.opd[r <= radius] = self.retardance * self.central_wavelength.to(u.meter).value
        npix = (r<=radius).sum()
        if npix < 50:  # pragma: no cover
            import warnings
            errmsg = "Phase mask is very coarsely sampled: only {} pixels. "\
                     "Improve sampling for better precision!".format(npix)
            warnings.warn(errmsg)
            _log.warn(errmsg)
        return self.opd


class RectangularFieldStop(AnalyticImagePlaneElement):
    """ Defines an ideal rectangular field stop

    Parameters
    ----------
    name : string
        Descriptive name
    width, height: float
        Size of the field stop, in arcseconds. Default 0.5 width, height 5.
    """

    @utils.quantity_input(width=u.arcsec, height=u.arcsec)
    def __init__(self, name="unnamed field stop", width=0.5*u.arcsec, height=5.0*u.arcsec, **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self.width = width    # width of square stop in arcseconds.
        self.height = height  # height of square stop in arcseconds.
        self._default_display_size = max(height, width) * 1.2

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the field stop.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("IdealFieldStop get_transmission must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)

        w_inside = (
            (abs(y) <= (self.height.to(u.arcsec).value / 2)) &
            (abs(x) <= (self.width.to(u.arcsec).value / 2))
        )
        self.transmission = w_inside.astype(_float())

        return self.transmission


class SquareFieldStop(RectangularFieldStop):
    """ Defines an ideal square field stop

    Parameters
    ----------
    name : string
        Descriptive name
    size : float
        Size of the field stop, in arcseconds. Default 20.
    """

    @utils.quantity_input(size=u.arcsec)
    def __init__(self, name="unnamed field stop", size=20.*u.arcsec, **kwargs):
        RectangularFieldStop.__init__(self, width=size, height=size, **kwargs)
        self.name = name
        self.height = self.width
        self._default_display_size = size * 1.2


class HexagonFieldStop(AnalyticImagePlaneElement):
    """ Defines an ideal hexagonal field stop

    Specify either the side length (= corner radius) or the
    flat-to-flat distance, or the point-to-point diameter, in
    angular units

    Parameters
    ----------
    name : string
        Descriptive name
    side : float, optional
        side length (and/or radius) of hexagon, in arcsec. Overrides flattoflat if both are present.
    flattoflat : float, optional
        Distance between sides (flat-to-flat) of the hexagon, in arcsec. Default is 1.0
    diameter : float, optional
        point-to-point diameter of hexagon. Twice the side length. Overrides flattoflat, but is overridden by side.


    Note you can also specify the standard parameter "rotation" to rotate the hexagon by some amount.

    """

    @utils.quantity_input(side=u.arcsec, diameter=u.arcsec, flattoflat=u.arcsec)
    def __init__(self, name=None, side=None, diameter=None, flattoflat=None, **kwargs):
        if flattoflat is None and side is None and diameter is None:
            self.side = 1.0 * u.arcsec
        elif side is not None:
            self.side = side
        elif diameter is not None:
            self.side = diameter / 2
        else:
            self.side = flattoflat / np.sqrt(3.)

        if name is None:
            name = "Hexagon, side length= {}".format(self.side)

        AnalyticImagePlaneElement.__init__(self, name=name, **kwargs)

    @property
    def diameter(self):
        return self.side * 2

    @property
    def flat_to_flat(self):
        return self.side * np.sqrt(3.)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("HexagonFieldStop get_transmission must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        side = self.side.to(u.arcsec).value
        absy = np.abs(y)

        self.transmission = np.zeros(wave.shape, dtype=_float())

        w_rect = (
            (np.abs(x) <= 0.5 * side) &
            (absy <= np.sqrt(3) / 2 * side)
        )
        w_left_tri = (
            (x <= -0.5 * side) &
            (x >= -1 * side) &
            (absy <= (x + 1 * side) * np.sqrt(3))
        )
        w_right_tri = (
            (x >= 0.5 * side) &
            (x <= 1 * side) &
            (absy <= (1 * side - x) * np.sqrt(3))
        )
        self.transmission[w_rect] = 1
        self.transmission[w_left_tri] = 1
        self.transmission[w_right_tri] = 1

        return self.transmission


class AnnularFieldStop(AnalyticImagePlaneElement):
    """ Defines a circular field stop with an (optional) opaque circular center region

    Parameters
    ------------
    name : string
        Descriptive name
    radius_inner : float
        Radius of the central opaque region, in arcseconds. Default is 0.0 (no central opaque spot)
    radius_outer : float
        Radius of the circular field stop outer edge. Default is 10. Set to 0.0 for no outer edge.
    """

    @utils.quantity_input(radius_inner=u.arcsec, radius_outer=u.arcsec)
    def __init__(self, name="unnamed annular field stop", radius_inner=0.0, radius_outer=1.0, **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer
        self._default_display_size = 2* max(radius_outer, radius_inner)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the field stop.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        r = _r(x, y)

        radius_inner = self.radius_inner.to(u.arcsec).value
        radius_outer = self.radius_outer.to(u.arcsec).value

        pxscl = wave.pixelscale.to(u.arcsec/u.pixel).value
        ypix=y/pxscl  # The filled_circle_aa code and in particular pxwt doesn't seem reliable with pixel scale <1
        xpix=x/pxscl

        if self.radius_outer > 0:
            self.transmission = geometry.filled_circle_aa(wave.shape, 0,0, radius_outer/pxscl, xarray=xpix, yarray=ypix)
        else:
            self.transmission = np.ones(wave.shape, dtype=_float())

        if self.radius_inner > 0:
            self.transmission -= geometry.filled_circle_aa(wave.shape, 0,0, radius_inner/pxscl, xarray=xpix, yarray=ypix)

        return self.transmission


class CircularOcculter(AnnularFieldStop):
    """ Defines an ideal circular occulter (opaque circle)

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the occulting spot, in arcseconds. Default is 1.0

    """

    @utils.quantity_input(radius=u.arcsec)
    def __init__(self, name="unnamed occulter", radius=1.0, **kwargs):
        super(CircularOcculter, self).__init__(name=name, radius_inner=radius, radius_outer=0.0, **kwargs)
        self._default_display_size = 10 * u.arcsec


class BarOcculter(AnalyticImagePlaneElement):
    """ Defines an ideal bar occulter (like in MIRI's Lyot coronagraph)

    Parameters
    ----------
    name : string
        Descriptive name
    width : float
        width of the bar stop, in arcseconds. Default is 1.0
    height: float
        heightof the bar stop, in arcseconds. Default is 10.0

    """

    @utils.quantity_input(width=u.arcsec, height=u.arcsec)
    def __init__(self, name="bar occulter", width=1.0*u.arcsec, height=10.0*u.arcsec, **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self.width = width
        self.height= height
        self._default_display_size = max(height, width) * 1.2

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)

        w_inside = ( (np.abs(x) <= self.width.to(u.arcsec).value / 2) &
                     (np.abs(y) <= self.height.to(u.arcsec).value / 2) )

        self.transmission = np.ones(wave.shape, dtype=_float())
        self.transmission[w_inside] = 0

        return self.transmission


# ------ Analytic Pupil or Intermediate Plane elements (coordinates in meters) -----

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
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)
        direction = direction.lower()
        if direction != 'forward' and direction != 'backward':
            raise ValueError("Invalid direction %s, must be either"
                             "forward or backward." % direction)
        self.direction = direction
        self._suppress_display = True
        self.wavefront_display_hint = 'phase'  # preferred display for wavefronts at this plane

    def get_opd(self, wave):
        """ Compute the required tilt needed to get the PSF centered on the corner between
        the 4 central pixels, not on the central pixel itself.
        """

        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("FQPM get_opd must be called with a Wavefront to define the spacing")
        assert wave.planetype != PlaneType.image, "This optic does not work on image planes"

        fft_im_pixelscale = wave.wavelength / wave.diam / wave.oversample * u.radian
        required_offset = fft_im_pixelscale * 0.5
        if self.direction == 'backward':
            wave._image_centered = 'pixel'
        else:
            required_offset *= -1  # changed in poppy 1.0 for sign convention update
            wave._image_centered = 'corner'
        wave.tilt(required_offset, required_offset)

        # gotta return something... so return a value that will not affect the wave any more.
        return 0  # null OPD


class ParityTestAperture(AnalyticOpticalElement):
    """ Defines a circular pupil aperture with boxes cut out.
    This is mostly a test aperture, which has no symmetry and thus can be used to
    test the various Fourier transform algorithms and sign conventions.

    See also LetterFAperture

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

    @utils.quantity_input(radius=u.meter)
    def __init__(self, name=None, radius=1.0 * u.meter, pad_factor=1.0, **kwargs):
        if name is None: name = "Asymmetric Parity Test Aperture, radius={}".format(radius)
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)
        self.radius = radius
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius
        self.wavefront_display_hint = 'intensity'  # preferred display for wavefronts at this plane

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("CircularAperture get_opd must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype != PlaneType.image)

        radius = self.radius.to(u.meter).value
        y, x = self.get_coordinates(wave)
        r = _r(x, y)

        w_outside = (r > radius)
        self.transmission = np.ones(wave.shape, dtype=_float())
        self.transmission[w_outside] = 0

        w_box1 = (
            (r > (radius * 0.5)) &
            (np.abs(x) < radius * 0.1) &
            (y < 0)
        )
        w_box2 = (
            (r > (radius * 0.75)) &
            (np.abs(y) < radius * 0.2) &
            (x < 0)
        )
        self.transmission[w_box1] = 0
        self.transmission[w_box2] = 0

        return self.transmission


class LetterFAperture(AnalyticOpticalElement):
    """ Define a capital letter F aperture. This is sometimes useful for
    disambiguating pupil orientations. See also AsymmetricParityTestAperture
    and LetterFOpticalPathDifference.
    """

    def __init__(self, name=None, radius=1.0 * u.meter, pad_factor=1.5, **kwargs):
        if name is None: name = f"Letter F Parity Test Aperture, radius={radius}"
        super().__init__(name=name, planetype=PlaneType.pupil, **kwargs)
        self.radius = radius
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius
        self.wavefront_display_hint = 'intensity'  # preferred display for wavefronts at this plane

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the aperture.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("LetterFAperture get_opd must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype != PlaneType.image)

        radius = self.radius.to(u.meter).value
        y, x = self.get_coordinates(wave)
        yr = y / radius
        xr = x / radius

        self.transmission = np.zeros(wave.shape, dtype=float)
        self.transmission[(xr <  0) & (xr > -0.5) & (np.abs(yr) < 1)] = 1
        self.transmission[(xr >= 0) & (xr < 0.75) & (np.abs(yr - 0.75) < 0.25)] = 1
        self.transmission[(xr >= 0) & (xr < 0.5) & (np.abs(yr) < 0.25)] = 1
        return self.transmission

class LetterFOpticalPathDifference(AnalyticOpticalElement):
    """ Define a capital letter F in OPD. This is sometimes useful for
    disambiguating pupil orientations. See also LetterFAperture.
    """

    def __init__(self, name=None, radius=1.0 * u.meter, pad_factor=1.5, opd=1*u.micron, **kwargs):
        if name is None: name = f"Letter F Parity Test Aperture, radius={radius}"
        super().__init__(name=name, planetype=PlaneType.pupil, **kwargs)
        self.radius = radius
        self._opd_amount = opd.to_value(u.m)
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius
        self.wavefront_display_hint = 'intensity'  # preferred display for wavefronts at this plane

    def get_opd(self, wave):
        """ Compute the OPD inside/outside of the aperture.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("LetterFAperture get_opd must be called with a Wavefront "
                             "to define the spacing")

        radius = self.radius.to(u.meter).value
        y, x = self.get_coordinates(wave)
        yr = y / radius
        xr = x / radius

        self.opd = np.zeros(wave.shape, dtype=float)
        self.opd[(xr <  0) & (xr > -0.5) & (np.abs(yr) < 1)] = self._opd_amount
        self.opd[(xr >= 0) & (xr < 0.75) & (np.abs(yr - 0.75) < 0.25)] = self._opd_amount
        self.opd[(xr >= 0) & (xr < 0.5) & (np.abs(yr) < 0.25)] = self._opd_amount
        return self.opd

class CircularAperture(AnalyticOpticalElement):
    """ Defines an ideal circular pupil aperture

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0
    gray_pixel : bool
        Apply gray pixel approximation to return fractional transmission for
        edge pixels that are only partially within this aperture?
    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.

    """

    @utils.quantity_input(radius=u.meter)
    def __init__(self, name=None, radius=1.0 * u.meter, pad_factor=1.0, planetype=PlaneType.unspecified,
            gray_pixel=True, **kwargs):

        if name is None:
            name = "Circle, radius={}".format(radius)
        super(CircularAperture, self).__init__(name=name, planetype=planetype, **kwargs)
        if radius <= 0*u.meter:
            raise ValueError("radius must be a positive nonzero number.")
        self.radius = radius
        # for creating input wavefronts - let's pad a bit:
        self.pupil_diam = pad_factor * 2 * self.radius
        self._default_display_size = 3 * self.radius
        self._use_gray_pixel = bool(gray_pixel)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the aperture.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("CircularAperture get_transmission must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype != PlaneType.image)

        y, x = self.get_coordinates(wave)
        radius = self.radius.to(u.meter).value
        if self._use_gray_pixel:
            pixscale = wave.pixelscale.to(u.meter/u.pixel).value
            self.transmission = geometry.filled_circle_aa(wave.shape, 0, 0, radius/pixscale, x/pixscale, y/pixscale)
        else:
            r = _r(x, y)
            self.transmission = (r<=radius).astype(_float())
        return self.transmission


class HexagonAperture(AnalyticOpticalElement):
    """ Defines an ideal hexagonal pupil aperture

    Specify either the side length (= corner radius) or the
    flat-to-flat distance, or the point-to-point diameter.

    Parameters
    ----------
    name : string
        Descriptive name
    side : float, optional
        side length (and/or radius) of hexagon, in meters. Overrides flattoflat if both are present.
    flattoflat : float, optional
        Distance between sides (flat-to-flat) of the hexagon, in meters. Default is 1.0
    diameter : float, optional
        point-to-point diameter of hexagon. Twice the side length. Overrides flattoflat, but is overridden by side.

    """

    @utils.quantity_input(side=u.meter, diameter=u.meter, flattoflat=u.meter)
    def __init__(self, name=None, side=None, diameter=None, flattoflat=None, **kwargs):
        if flattoflat is None and side is None and diameter is None:
            self.side = 1.0 * u.meter
        elif side is not None:
            self.side = side
        elif diameter is not None:
            self.side = diameter / 2
        else:
            self.side = flattoflat / np.sqrt(3.)

        self.pupil_diam = 2 * self.side  # for creating input wavefronts
        self._default_display_size = 3 * self.side
        if name is None:
            name = "Hexagon, side length= {}".format(self.side)

        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)

    @property
    def diameter(self):
        return self.side * 2

    @property
    def flat_to_flat(self):
        return self.side * np.sqrt(3.)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("HexagonAperture get_transmission must be called with a Wavefront "
                             "to define the spacing")
        assert (wave.planetype != PlaneType.image)

        y, x = self.get_coordinates(wave)
        side = self.side.to(u.meter).value
        absy = np.abs(y)

        self.transmission = np.zeros(wave.shape, dtype=_float())

        w_rect = (
            (np.abs(x) <= 0.5 * side) &
            (absy <= np.sqrt(3) / 2 * side)
        )
        w_left_tri = (
            (x <= -0.5 * side) &
            (x >= -1 * side) &
            (absy <= (x + 1 * side) * np.sqrt(3))
        )
        w_right_tri = (
            (x >= 0.5 * side) &
            (x <= 1 * side) &
            (absy <= (1 * side - x) * np.sqrt(3))
        )
        self.transmission[w_rect] = 1
        self.transmission[w_left_tri] = 1
        self.transmission[w_right_tri] = 1

        return self.transmission

class MultiSegmentAperture(AnalyticOpticalElement, ABC):
    """Abstract base class for an aperture made of sub-apertures
    This is subclassed to hexagons and circles below.
    """

    @utils.quantity_input(segment_size=u.meter, gap=u.meter)
    def __init__(self, name="MultiSegment", segment_size=1, gap=0.01, rings=1,
                 segmentlist=None, center=False, **kwargs):
        self.rings = rings
        self.gap = gap
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)

        # spacing between segment centers
        self._segment_spacing = (segment_size + gap).to_value(u.meter)

        self.pupil_diam = (self._segment_spacing) * (2 * self.rings + 1)

        # make a list of all the segments included in this hex aperture
        if segmentlist is not None:
            self.segmentlist = segmentlist
        else:
            self.segmentlist = list(range(self._n_aper_inside_ring(self.rings + 1)))
            if not center:
                self.segmentlist.remove(0)  # remove center segment 0

    def _n_aper_in_ring(self, n):
        """ How many hexagons or circles in ring N? """
        return 1 if n == 0 else 6 * n

    def _n_aper_inside_ring(self, n):
        """ How many hexagons or circles interior to ring N, not counting N?"""
        return sum([self._n_aper_in_ring(i) for i in range(n)])

    def _aper_in_ring(self, hex_index):
        """ What ring is a given hexagon or circle in?"""
        if hex_index == 0:
            return 0
        for i in range(100):
            if self._n_aper_inside_ring(i) <= hex_index < self._n_aper_inside_ring(i + 1):
                return i
        raise ValueError("Loop exceeded! MultiSegmentAperture is limited to <100 rings of segments.")

    def _aper_radius(self, hex_index):
        """ Radius of a given hexagon from the center """
        ring = self._aper_in_ring(hex_index)
        if ring <= 1:
            return (self._segment_spacing) * ring

    def _aper_center(self, aper_index):
        """ Center coordinates of a given hexagon
        counting clockwise around each ring

        Returns y, x coords

        """
        ring = self._aper_in_ring(aper_index)

        # handle degenerate case of center segment
        # to avoid div by 0 in the main code below
        if ring == 0:
            return 0, 0

        # now count around from the starting point:
        index_in_ring = aper_index - self._n_aper_inside_ring(ring) + 1  # 1-based
        angle_per_hex = 2 * np.pi / self._n_aper_in_ring(ring)  # angle in radians

        radius = (self._segment_spacing) * ring  # like JWST 'B' segments, aka corners for a hexagon
        if np.mod(index_in_ring, ring) == 1:
            angle = angle_per_hex * (index_in_ring - 1)
            ypos = radius * np.cos(angle)
            xpos = radius * np.sin(angle)
        else:
            # find position of previous 'B' type segment.
            last_B_angle = ((index_in_ring - 1) // ring) * ring * angle_per_hex
            ypos0 = radius * np.cos(last_B_angle)
            xpos0 = radius * np.sin(last_B_angle)

            # count around from that corner
            da = (self._segment_spacing) * np.cos(30 * np.pi / 180)
            db = (self._segment_spacing) * np.sin(30 * np.pi / 180)

            whichside = (index_in_ring - 1) // ring  # which of the sides are we on?
            if whichside == 0:
                dx, dy = da, -db
            elif whichside == 1:
                dx, dy = 0, -self._segment_spacing
            elif whichside == 2:
                dx, dy = -da, -db
            elif whichside == 3:
                dx, dy = -da, db
            elif whichside == 4:
                dx, dy = 0, self._segment_spacing
            elif whichside == 5:
                dx, dy = da, db

            xpos = xpos0 + dx * np.mod(index_in_ring - 1, ring)
            ypos = ypos0 + dy * np.mod(index_in_ring - 1, ring)

        return ypos, xpos

    @abstractmethod
    def _one_aperture(self, wave, index, value=1):
        """Implement how to draw one aperture"""
        pass

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype != PlaneType.image)

        self.transmission = np.zeros(wave.shape, dtype=_float())

        for i in self.segmentlist:
            self._one_aperture(wave, i)

        return self.transmission


class MultiHexagonAperture(MultiSegmentAperture):
    """ Defines a hexagonally segmented aperture

    Parameters
    ----------
    name : string
        Descriptive name
    rings : integer
        The number of rings of hexagons to include, not counting the central segment
        (i.e. 2 for a JWST-like aperture, 3 for a Keck-like aperture, and so on)
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
    the same aperture, avoid repeated evaluations of this function. It will be faster to create
    this aperture, evalute it once, and save the result onto a discrete array, via either
       (1) saving it to a FITS file using the to_fits() method, and then use that in a
       FITSOpticalElement, or
       (2) Use the fixed_sampling_optic function to create an ArrayOpticalElement with
       a sampled version of this.

    """

    @utils.quantity_input(side=u.meter, flattoflat=u.meter, gap=u.meter)
    def __init__(self, name="MultiHex", flattoflat=1.0, side=None, gap=0.01, rings=1,
                 segmentlist=None, center=False, **kwargs):
        if flattoflat is None and side is None:
            self.side = 1.0 * u.meter
        elif side is not None:
            self.side = side
        else:
            self.side = flattoflat / np.sqrt(3.)
        self.flattoflat = self.side * np.sqrt(3)

        super().__init__(name=name, segment_size=self.flattoflat,
                         gap=gap, rings=rings, segmentlist=segmentlist, center=center, **kwargs)


    def _one_aperture(self, wave, index, value=1):
        """ Draw one hexagon into the self.transmission array """

        y, x = self.get_coordinates(wave)
        side = self.side.to(u.meter).value

        ceny, cenx = self._aper_center(index)

        y -= ceny
        x -= cenx
        absy = np.abs(y)

        w_rect = (
            (np.abs(x) <= 0.5 * side) &
            (absy <= np.sqrt(3) / 2 * side)
        )
        w_left_tri = (
            (x <= -0.5 * side) &
            (x >= -1 * side) &
            (absy <= (x + 1 * side) * np.sqrt(3))
        )
        w_right_tri = (
            (x >= 0.5 * side) &
            (x <= 1 * side) &
            (absy <= (1 * side - x) * np.sqrt(3))
        )

        self.transmission[w_rect] = value
        self.transmission[w_left_tri] = value
        self.transmission[w_right_tri] = value


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

    @utils.quantity_input(radius=u.meter)
    def __init__(self, name=None, nsides=6, radius=1 * u.meter, rotation=0., **kwargs):
        self.radius = radius
        self.nsides = nsides
        self.pupil_diam = 2 * self.radius  # for creating input wavefronts
        if name is None:
            name = "{}-gon, radius= {}".format(self.nsides, self.radius)
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, rotation=rotation, **kwargs)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype != PlaneType.image)
        y, x = self.get_coordinates(wave)

        phase = self.rotation * np.pi / 180
        vertices = np.zeros((self.nsides, 2), dtype=_float())
        for i in range(self.nsides):
            vertices[i] = [np.cos(i * 2 * np.pi / self.nsides + phase),
                           np.sin(i * 2 * np.pi / self.nsides + phase)]
        vertices *= self.radius.to(u.meter).value

        self.transmission = np.zeros(wave.shape, dtype=_float())
        for row in range(wave.shape[0]):
            pts = np.asarray(list(zip(x[row], y[row])))
            ok = matplotlib.path.Path(vertices).contains_points(pts)
            self.transmission[row][ok] = 1.0

        return self.transmission

class MultiCircularAperture(MultiSegmentAperture):
    """ Defines a circularly segmented aperture in close compact configuration
    
    Parameters
    ----------
    name : string
        descriptive name
    rings : integer
         The number of rings of hexagons to include, not counting the central segment
    segment_radius : float, optional
        radius of the circular sub-apertures in meters, default is 1 meters
    gap: float, otional
        Gap between adjacent segments, in meters. Default is 0.01 m = 1 cm
    center : bool, optional
        should the central segment be included? Default is True.
    segmentlist : list of ints, optional
        This allows one to specify that only a subset of segments are present, for a
        partially populated segmented telescope, non-redundant segment set, etc.
        Segments are numbered from 0 for the center segment, 1 for the segment immediately
        above it, and then clockwise around each ring.
        For example, segmentlist=[1,3,5] would make an aperture of 3 segments.
    gray_pixel : bool, optional
        Apply gray pixel approximation to return fractional transmission for
        edge pixels that are only partially within this aperture? default : True
    
    """
    
    @utils.quantity_input(segment_radius=u.meter, gap=u.meter)
    def __init__(self, name = "multiCirc",rings = 1, segment_radius = 1.0, gap = 0.01,
                 segmentlist = None, center = True, gray_pixel = True, **kwargs):
        self.segment_radius = segment_radius
        segment_diameter = 2*segment_radius

        super().__init__(name=name, segment_size=segment_diameter,
                         gap=gap, rings=rings, segmentlist=segmentlist, center=center, **kwargs)
        self.pupil_diam = (segment_diameter) * (2 * self.rings + 1)+ gap * (2*rings)
        
        self._use_gray_pixel = bool(gray_pixel)
        
    def _one_aperture(self, wave, index, value=1):
        """ Draw one circular aperture into the self.transmission array """

        y, x = self.get_coordinates(wave)
        segRadius = self.segment_radius.to(u.meter).value

        ceny, cenx = self._aper_center(index)

        y -= ceny
        x -= cenx
        
        if self._use_gray_pixel:
            pixscale = wave.pixelscale.to(u.meter/u.pixel).value
            tmpTransmission = geometry.filled_circle_aa(wave.shape, 0, 0, segRadius/pixscale, x/pixscale, y/pixscale)
            self.transmission += tmpTransmission 
        else:
            r = _r(x, y)
            del x
            del y

            w_inside = np.where(r < segRadius)
            del r
            self.transmission[w_inside] = value

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

    @utils.quantity_input(width=u.meter, height=u.meter)
    def __init__(self, name=None, width=0.5 * u.meter, height=1.0 * u.meter, rotation=0.0, **kwargs):
        self.width = width
        self.height = height
        if name is None:
            name = "Rectangle, size= {s.width:.1f} wide * {s.height:.1f} high".format(s=self)
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, rotation=rotation, **kwargs)
        # for creating input wavefronts:
        self.pupil_diam = np.sqrt(self.height ** 2 + self.width ** 2)

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype != PlaneType.image)

        y, x = self.get_coordinates(wave)

        w_inside = (
            (abs(y) <= (self.height.to(u.meter).value / 2)) &
            (abs(x) <= (self.width.to(u.meter).value / 2))
        )

        self.transmission = w_inside.astype(dtype=_float())
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

    @utils.quantity_input(size=u.meter)
    def __init__(self, name=None, size=1.0 * u.meter, **kwargs):
        self._size = size
        if name is None:
            name = "Square, side length= {}".format(size)
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
    secondary_radius : float or astropy Quantity length
        Radius of the circular secondary obscuration, in meters or other unit.
        Default 0.5 m
    n_supports : int
        Number of secondary mirror supports ("spiders"). These will be
        spaced equally around a circle.  Default is 4.
    support_width : float or astropy Quantity length
        Width of each support, in meters or other unit. Default is 0.01 m = 1 cm.
    support_angle_offset : float
        Angular offset, in degrees, of the first secondary support from the X axis.

    """

    @utils.quantity_input(secondary_radius=u.meter, support_width=u.meter)
    def __init__(self, name=None, secondary_radius=0.5 * u.meter, n_supports=4, support_width=0.01 * u.meter,
                 support_angle_offset=0.0, **kwargs):
        if name is None:
            name = "Secondary Obscuration with {0} supports".format(n_supports)
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)
        self.secondary_radius = secondary_radius
        self.n_supports = n_supports
        self.support_width = support_width
        self.support_angle_offset = support_angle_offset

        # for creating input wavefronts if this is the first optic in a opticalsystem:
        self.pupil_diam = 4 * self.secondary_radius

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the obscuration
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype != PlaneType.image)

        self.transmission = np.ones(wave.shape, dtype=_float())

        y, x = self.get_coordinates(wave)
        r = np.sqrt(x ** 2 + y ** 2)  # * wave.pixelscale

        self.transmission[r < self.secondary_radius.to(u.meter).value] = 0

        for i in range(self.n_supports):
            angle = 2 * np.pi / self.n_supports * i + np.deg2rad(self.support_angle_offset)

            # calculate rotated x' and y' coordinates after rotation by that angle.
            xp = np.cos(angle) * x + np.sin(angle) * y
            yp = -np.sin(angle) * x + np.cos(angle) * y

            self.transmission[(xp > 0) & (np.abs(yp) < self.support_width.to(u.meter).value / 2)] = 0

            # TODO check here for if there are no pixels marked because the spider is too thin.
            # In that case use a grey scale approximation

        return self.transmission


class AsymmetricSecondaryObscuration(SecondaryObscuration):
    """ Defines a central obscuration with one or more supports which can be oriented at
    arbitrary angles around the primary mirror, a la the three supports of JWST

    This also allows for secondary supports that do not intersect with
    the primary mirror center; use the support_offset_x and support_offset_y parameters
    to apply offsets relative to the center for the origin of each strut.

    Parameters
    ----------
    secondary_radius : float
        Radius of the circular secondary obscuration. Default 0.5 m
    support_angle : ndarray or list of floats
        The angle measured counterclockwise from +Y for each support
    support_width : float or astropy Quantity of type length, or list of those
        if scalar, gives the width for all support struts
        if a list, gives separately the width for each support strut independently.
        Widths in meters or other unit if specified. Default is 0.01 m = 1 cm.
    support_offset_x : float, or list of floats.
        Offset in the X direction of the start point for each support.
        if scalar, applies to all supports; if a list, gives a separate offset for each.
    support_offset_y : float, or list of floats.
        Offset in the Y direction of the start point for each support.
        if scalar, applies to all supports; if a list, gives a separate offset for each.
    """

    @utils.quantity_input(support_width=u.meter)
    def __init__(self, support_angle=(0, 90, 240), support_width=0.01 * u.meter,
                 support_offset_x=0.0, support_offset_y=0.0, **kwargs):
        SecondaryObscuration.__init__(self, n_supports=len(support_angle), **kwargs)

        self.support_angle = np.asarray(support_angle)

        if np.isscalar(support_width.value):
            support_width = np.zeros(len(support_angle)) + support_width
        self.support_width = support_width

        if np.isscalar(support_offset_x):
            support_offset_x = np.zeros(len(support_angle)) + support_offset_x
        self.support_offset_x = support_offset_x

        if np.isscalar(support_offset_y):
            support_offset_y = np.zeros(len(support_angle)) + support_offset_y
        self.support_offset_y = support_offset_y

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the obscuration
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype != PlaneType.image)

        self.transmission = np.ones(wave.shape, dtype=_float())

        y, x = self.get_coordinates(wave)
        r = np.sqrt(x ** 2 + y ** 2)

        self.transmission[r < self.secondary_radius.to(u.meter).value] = 0

        for angle_deg, width, offset_x, offset_y in zip(self.support_angle,
                                                        self.support_width,
                                                        self.support_offset_x,
                                                        self.support_offset_y):
            angle = np.deg2rad(angle_deg + 90)  # 90 deg offset is to start from the +Y direction

            # calculate rotated x' and y' coordinates after rotation by that angle.
            # and application of offset
            xp = np.cos(angle) * (x - offset_x) + np.sin(angle) * (y - offset_y)
            yp = -np.sin(angle) * (x - offset_x) + np.cos(angle) * (y - offset_y)

            self.transmission[(xp > 0) & (np.abs(yp) < width.to(u.meter).value / 2)] = 0

            # TODO check here for if there are no pixels marked because the spider is too thin.
            # In that case use a grey scale approximation

        return self.transmission


class ThinLens(CircularAperture):
    """ An idealized thin lens, implemented as a Zernike defocus term.

    The sign convention adopted is the usual for lenses: a "positive" lens
    is converging (i.e. convex), a "negative" lens is diverging (i.e. concave).

    Recall the sign convention choice that the OPD is positive if the aberrated wavefront
    leads the ideal unaberrated wavefront; a converging wavefront leads at its outer edges
    relative to a flat wavefront.

    In other words, a positive number of waves of defocus indicates a
    lens with more positive OPD at the edges than at the center.

    NOTE - this sign convention was different in prior versions of poppy < 1.0.

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

    @utils.quantity_input(reference_wavelength=u.meter)
    def __init__(self, name='Thin lens', nwaves=4.0, reference_wavelength=1e-6 * u.meter,
                 radius=1.0*u.meter, **kwargs):
        self.reference_wavelength = reference_wavelength
        self.nwaves = nwaves
        self.max_phase_delay = reference_wavelength * nwaves
        CircularAperture.__init__(self, name=name, radius=radius, **kwargs)
        self.wavefront_display_hint = 'phase'  # preferred display for wavefronts at this plane

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)
        r = np.sqrt(x ** 2 + y ** 2)
        r_norm = r / self.radius.to(u.meter).value


        # don't forget the factor of 0.5 to make the scaling factor apply as peak-to-valley
        # rather than center-to-peak
        defocus_zernike = ((2 * r_norm ** 2 - 1) *
                           (0.5 * self.nwaves * self.reference_wavelength.to(u.meter).value))
        opd = defocus_zernike

        # the thin lens is explicitly also a circular aperture:
        # we use the aperture intensity here to mask the OPD we return, in
        # order to avoid bogus values outside the aperture
        aperture_intensity = CircularAperture.get_transmission(self, wave)
        opd[aperture_intensity==0] = 0

        return opd


class GaussianAperture(AnalyticOpticalElement):
    """ Defines an ideal Gaussian apodized pupil aperture,
    or at least as much of one as can be fit into a finite-sized
    array

    The Gaussian's width must be set with either the fwhm or w parameters.

    Note that this makes an optic whose electric *field amplitude*
    transmission is the specified Gaussian; thus the intensity
    transmission will be the square of that Gaussian.


    Parameters
    ----------
    name : string
        Descriptive name
    fwhm : float, optional.
        Full width at half maximum for the Gaussian, in meters.
    w : float, optional
        Beam width parameter, equal to fwhm/(2*sqrt(ln(2))).
    pupil_diam : float, optional
        default pupil diameter for cases when it is not otherwise
        specified (e.g. displaying the optic by itself.) Default
        value is 3x the FWHM.

    """

    @utils.quantity_input(fwhm=u.meter, w=u.meter, pupil_diam=u.meter)
    def __init__(self, name=None, fwhm=None, w=None, pupil_diam=None, **kwargs):
        if fwhm is None and w is None:
            raise ValueError("Either the fwhm or w parameter must be set.")
        elif w is not None:
            self.w = w
        elif fwhm is not None:
            self.w = fwhm / (2 * np.sqrt(np.log(2)))

        if pupil_diam is None:
            pupil_diam = 3 * self.fwhm  # for creating input wavefronts
        self.pupil_diam = pupil_diam
        if name is None:
            name = "Gaussian aperture with fwhm ={0:.2f}".format(self.fwhm)
        AnalyticOpticalElement.__init__(self, name=name, planetype=PlaneType.pupil, **kwargs)

    @property
    def fwhm(self):
        return self.w * (2 * np.sqrt(np.log(2)))

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the aperture.
        """
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        y, x = self.get_coordinates(wave)

        r = np.sqrt(x ** 2 + y ** 2)

        transmission = np.exp((- (r / self.w.to(u.meter).value) ** 2))

        return transmission


class TiltOpticalPathDifference(AnalyticOpticalElement):
    """A simple tilt in OPD.

    With the sign convention used in poppy, a wavefront that is positively increasing
    in the +X direction will deflect the beam in the -X direction, and so on.

    Parameters
    ----------
    tilt_angle : angle, as an astropy unit
        Angle of the tilt
    rotation : float
        Position angle, in degrees, for the direction in which the beam should be tilted

    use the rotation parameter (available for any AnalyticOpticalElement)
    to adjust the position angle of the tilt

    """
    def __init__(self, name='Tilt', tilt_angle=0.1 * u.arcsec, rotation=0, **kwargs):
        self.tilt_angle=tilt_angle
        super().__init__(name=name, rotation=0, **kwargs)

    def get_opd(self, wave):
        # Get local coordinates for this wave; note this will implicitly include any
        # rotation.
        y, x = self.get_coordinates(wave)

        # SIGN CONVENTION: an OPD or wavefront increasing in the -y direction will deflect the beam
        # into the +y direction, and vice versa.
        angle = self.tilt_angle.to_value(u.radian)
        opd = y * -angle

        return opd



# ------ generic analytic optics ------

class KnifeEdge(AnalyticOpticalElement):
    """ A half-infinite opaque plane, with a perfectly sharp edge
    through the origin.

    Use the 'rotation', 'shift_x', and 'shift_y' parameters to adjust
    location and orientation.

    Rotation=0 yields a knife edge oriented vertically (edge parallel to +y)
    with the opaque side to the right.

    """
    def __init__(self, name=None, rotation=0, **kwargs):
        if name is None:
            name = "Knife edge at {} deg".format(rotation)
        AnalyticOpticalElement.__init__(self, name=name, rotation=rotation, **kwargs)

    def get_transmission(self, wave):
        if not isinstance(wave, BaseWavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        y, x = self.get_coordinates(wave)
        return x < 0


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
    mergemode : string, default = 'and'
        Method for merging transmissions:
            'and' : resulting transmission is product of constituents. (E.g
                    trans = trans1*trans2)
            'or'  : resulting transmission is sum of constituents, with overlap
                    subtracted.  (E.g. trans = trans1 + trans2 - trans1*trans2)
        In both methods, the resulting OPD is the sum of the constituents' OPDs.

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

    def __init__(self, opticslist=None, name="unnamed", mergemode="and", verbose=True, **kwargs):
        if opticslist is None:
            raise ValueError("Missing required opticslist argument to CompoundAnalyticOptic")
        AnalyticOpticalElement.__init__(self, name=name, verbose=verbose, **kwargs)

        self.opticslist = []
        self.planetype = None

        # check for valid mergemode
        if mergemode == "and":
            self.mergemode = "and"
        elif mergemode == "or":
            self.mergemode = "or"
        else:
            raise ValueError("mergemode must be either 'and' or 'or'.")

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
                elif (self.planetype != optic.planetype and self.planetype != PlaneType.unspecified and
                      optic.planetype != PlaneType.unspecified):
                    raise ValueError("Cannot mix image plane and pupil plane optics in "
                                     "the same CompoundAnalyticOptic")

                self.opticslist.append(optic)
                if hasattr(optic, '_default_display_size'):
                    if hasattr(self, '_default_display_size'):
                        self._default_display_size = max(self._default_display_size,
                                                         optic._default_display_size)
                    else:
                        self._default_display_size = optic._default_display_size
                if hasattr(optic, 'pupil_diam'):
                    if not hasattr(self, 'pupil_diam'):
                        self.pupil_diam = optic.pupil_diam
                    else:
                        self.pupil_diam = max(self.pupil_diam, optic.pupil_diam)

        if self.planetype == PlaneType.pupil:
            if all([hasattr(o, 'pupil_diam') for o in self.opticslist]):
                self.pupil_diam = np.asarray([o.pupil_diam.to(u.meter).value for o in self.opticslist]).max() * u.meter

    def get_transmission(self, wave):
        if self.mergemode == "and":
            trans = np.ones(wave.shape, dtype=_float())
            for optic in self.opticslist:
                trans *= optic.get_transmission(wave)
        elif self.mergemode == "or":
            trans = np.zeros(wave.shape, dtype=_float())
            for optic in self.opticslist:
                trans = trans + optic.get_transmission(wave) - trans * optic.get_transmission(wave)
        else:
            raise ValueError("mergemode must be either 'and' or 'or'.")
        self.transmission = trans
        return self.transmission

    def get_opd(self, wave):
        opd = np.zeros(wave.shape, dtype=_float())

        if self.mergemode == 'and':
            for optic in self.opticslist:
                opd += optic.get_opd(wave)
        elif self.mergemode == "or":
            for optic in self.opticslist:
                # spatially disjoint optics; each only contributes OPD to where it has nonzero transmission.
                trans = optic.get_transmission(wave)
                opd[trans != 0] += optic.get_opd(wave)[trans != 0]
        else:
            raise ValueError("mergemode must be either 'and' or 'or'.")

        self.opd = opd
        return self.opd

# ------ convert analytic optics to array optics ------

def fixed_sampling_optic(optic, wavefront, oversample=2):
    """Convert a variable-sampling AnalyticOpticalElement to a fixed-sampling ArrayOpticalElement

    For a given input optic this produces an equivalent output optic stored in simple arrays rather
    than created each time via function calls.

    If you know a priori the desired sampling will remain constant for some
    application, and don't need any of the other functionality of the
    AnalyticOpticalElement machinery with get_opd and get_transmission functions,
    you can save time by setting the sampling to a fixed value and saving arrays
    computed on that sampling.

    Also, you can use this to evaluate any optic on a finer sampling scale and then bin the
    results to the desired scale, using the so-called gray-pixel approximation. (i.e. the
    value for each output pixel is computed as the average of N*N finer pixels in an
    intermediate array.)

    Parameters
    ----------
    optic : poppy.AnalyticOpticalElement
        Some optical element
    wave : poppy.Wavefront
        A wavefront to define the desired sampling pixel size and number.
    oversample : int
        Subpixel sampling factor for "gray pixel" approximation: the optic will be
        evaluated on a finer pixel scale and then binned down to the desired sampling.

    Returns
    -------
    new_array_optic : poppy.ArrayOpticalElement
        A version ofthe input optic with fixed arrays for OPD and transmission.

    """
    from .poppy_core import ArrayOpticalElement
    npix = wavefront.shape[0]
    grid_size = npix*u.pixel*wavefront.pixelscale
    _log.debug("Converting {} to fixed sampling with grid_size={}, npix={}, oversample={}".format(
        optic.name, grid_size, npix, oversample))

    if oversample>1:
        _log.debug("retrieving oversampled opd and transmission arrays")
        sampled_opd = optic.sample(what='opd', npix=npix*oversample, grid_size=grid_size)
        sampled_trans = optic.sample(what='amplitude', npix=npix*oversample, grid_size=grid_size)

        _log.debug("binning down opd and transmission arrays")
        sampled_opd = utils.krebin(sampled_opd, wavefront.shape)/oversample**2
        sampled_trans = utils.krebin(sampled_trans, wavefront.shape)/oversample**2
    else:
        sampled_opd = optic.sample(what='opd', npix=npix, grid_size=grid_size)
        sampled_trans = optic.sample(what='amplitude', npix=npix, grid_size=grid_size)

    return ArrayOpticalElement(opd=sampled_opd,
                               transmission=sampled_trans,
                               pixelscale=wavefront.pixelscale,
                               name=optic.name)
