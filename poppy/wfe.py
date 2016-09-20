"""
Analytic optical element classes to introduce a specified wavefront
error in an OpticalSystem

 * ZernikeWFE
 * ParameterizedWFE (for use with hexike or zernike basis functions)
 * SineWaveWFE
 * TODO: MultiSineWaveWFE ?
 * TODO: PowerSpectrumWFE
 * TODO: KolmogorovWFE

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
from functools import wraps
import numpy as np
import astropy.units as u

from .optics import AnalyticOpticalElement, CircularAperture
from .poppy_core import Wavefront, _PUPIL
from . import zernike
from . import utils

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE', 'SineWaveWFE']

def _accept_wavefront_or_meters(f):
    """Decorator that ensures the first positional method argument
    is a poppy.Wavefront or a floating point number of meters
    for a wavelength
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Wavefront):
            wave = Wavefront(wavelength=args[1])
            new_args = (args[0],) + (wave,) + (args[2:])
            return f(*new_args, **kwargs)
        else:
            return f(*args, **kwargs)
    return wrapper

class WavefrontError(AnalyticOpticalElement):
    """A base class for different sources of wavefront error

    Analytic optical elements that represent wavefront error should
    derive from this class and override methods appropriately.
    Defined to be a pupil-plane optic.
    """
    def __init__(self, **kwargs):
        if 'planetype' not in kwargs:
            kwargs['planetype'] = _PUPIL
        super(WavefrontError, self).__init__(**kwargs)
        # in general we will want to see phase rather than intensity at this plane
        self.wavefront_display_hint='phase'

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """Construct the optical path difference array for a wavefront error source
        as evaluated across the pupil for an input wavefront `wave`

        Parameters
        ----------
        wave : Wavefront
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane
        units : 'meters' or 'waves'
            The units of optical path difference (Default: meters)
        """
        if not isinstance(wave, Wavefront):
            wave = Wavefront(wavelength=wave)

    @_accept_wavefront_or_meters
    def getPhasor(self, wave):
        """Construct the phasor array for an input wavefront `wave`
        that will apply the wavefront error model based on an OPD
        map from the `get_opd` method.

        Parameters
        ----------
        wave : Wavefront or float
            Wavefront object with a `coordinates` method that returns (y, x)
            coordinate arrays in meters in the pupil plane, or float with
            a wavefront wavelength in meters
        """
        opd_map = self.get_opd(wave, units='meters')
        opd_as_phase = 2 * np.pi * opd_map / (wave.wavelength.to(u.meter).value)
        wfe_phasor = np.exp(1.j * opd_as_phase)
        return wfe_phasor

    def rms(self):
        """RMS wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """Peak-to-valley wavefront error induced by this surface"""
        raise NotImplementedError('Not implemented yet')

def _wave_to_rho_theta(wave, pupil_radius):
    """
    Return wave coordinates in (rho, theta) for a Wavefront object
    normalized such that rho == 1.0 at the pupil radius

    Parameters
    ----------
    wave : Wavefront
        Wavefront object with a `coordinates` method that returns (y, x)
        coordinate arrays in meters in the pupil plane
    pupil_radius : float
        Radius (in meters) of a circle circumscribing the pupil.
    """
    y, x = wave.coordinates()
    r = np.sqrt(x ** 2 + y ** 2)

    rho = r / pupil_radius
    theta = np.arctan2(y / pupil_radius, x / pupil_radius)

    return rho, theta

class ParameterizedWFE(WavefrontError):
    """
    Define an optical element in terms of its distortion as decomposed
    into a set of orthonormal basis functions (e.g. Zernikes,
    Hexikes, etc.). Included basis functions are normalized such that
    user-provided coefficients correspond to meters RMS wavefront
    aberration for that basis function.

    Parameters
    ----------
    coefficients : iterable of numbers
        The contribution of each term to the final distortion, in meters
        RMS wavefront error. The coefficients are interpreted as indices
        in the order of Noll et al. 1976: the first term corresponds to
        j=1, second to j=2, and so on.
    radius : float
        Pupil radius, in meters. Defines the region of the input
        wavefront array over which the distortion terms will be
        evaluated. For non-circular pupils, this should be the circle
        circumscribing the actual pupil shape.
    basis_factory : callable
        basis_factory will be called with the arguments `nterms`, `rho`,
        `theta`, and `outside`.

        `nterms` specifies how many terms to compute, starting with the
        j=1 term in the Noll indexing convention for `nterms` = 1 and
        counting up.

        `rho` and `theta` are square arrays holding the rho and theta
        coordinates at each pixel in the pupil plane. `rho` is
        normalized such that `rho` == 1.0 for pixels at `radius` meters
        from the center.

        `outside` contains the value to assign pixels outside the
        radius `rho` == 1.0. (Always 0.0, but provided for
        compatibility with `zernike.zernike_basis` and
        `zernike.hexike_basis`.)
    """
    def __init__(self, name="Parameterized Distortion", coefficients=None, radius=None,
                 basis_factory=None, **kwargs):
        if not isinstance(basis_factory, collections.Callable):
            raise ValueError("'basis_factory' must be a callable that can "
                             "calculate basis functions")
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")
        self.coefficients = coefficients
        self.basis_factory = basis_factory
        super(ParameterizedWFE, self).__init__(name=name, **kwargs)

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        rho, theta = _wave_to_rho_theta(wave, self.radius)
        combined_distortion = np.zeros(rho.shape)

        nterms = len(self.coefficients)
        computed_terms = self.basis_factory(nterms=nterms, rho=rho, theta=theta, outside=0.0)

        for idx, coefficient in enumerate(self.coefficients):
            if coefficient == 0.0:
                continue  # save the trouble of a multiply-and-add of zeros
            combined_distortion += coefficient * computed_terms[idx]
        if units == 'meters':
            return combined_distortion
        elif units == 'waves':
            return combined_distortion / wave.wavelength
        else:
            raise ValueError("'units' argument must be 'meters' or 'waves'")

class ZernikeWFE(WavefrontError):
    """
    Define an optical element in terms of its Zernike components by
    providing coefficients for each Zernike term contributing to the
    analytic optical element.

    Parameters
    ----------
    coefficients : iterable of floats
        Specifies the coefficients for the Zernike terms, ordered
        according to the convention of Noll et al. JOSA 1976. The
        coefficient is in meters of optical path difference (not waves).
    radius : float
        Pupil radius, in meters, over which the Zernike terms should be
        computed such that rho = 1 at r = `radius`.
    """
    def __init__(self, name="Zernike WFE", coefficients=None, radius=None, **kwargs):
        try:
            self.radius = float(radius)
        except TypeError:
            raise ValueError("'radius' must be the radius of a circular aperture in meters"
                             "(optionally circumscribing a pupil of another shape)")

        self.coefficients = coefficients
        self.circular_aperture = CircularAperture(radius=self.radius, **kwargs)
        kwargs.update({'name': name})
        super(ZernikeWFE, self).__init__(**kwargs)

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        units : 'meters' or 'waves'
            Coefficients are supplied in `ZernikeWFE.coefficients` as
            meters of OPD, but the resulting OPD can be converted to
            waves based on the `Wavefront` wavelength or a supplied
            wavelength value.
        """
        rho, theta = _wave_to_rho_theta(wave, self.radius)

        # the Zernike optic, being normalized on a circle, is
        # implicitly also a circular aperture:
        aperture_intensity = self.circular_aperture.get_transmission(wave)

        pixelscale_m = wave.pixelscale.to(u.meter/u.pixel).value

        combined_zernikes = np.zeros(wave.shape, dtype=np.float64)
        for j, k in enumerate(self.coefficients, start=1):
            combined_zernikes += k * zernike.cached_zernike1(
                j,
                wave.shape,
                pixelscale_m,
                self.radius,
                outside=0.0,
                noll_normalize=True
            )

        combined_zernikes *= aperture_intensity
        if units == 'waves':
            combined_zernikes /= wave.wavelength
        return combined_zernikes

class SineWaveWFE(WavefrontError):
    """ A single sine wave ripple across the optic

    Specified as a a spatial frequency in cycles per meter, an optional phase offset in cycles,
    and an amplitude.

    By default the wave is oriented in the X direction.
    Like any AnalyticOpticalElement class, you can also specify a rotation parameter to
    rotate the direction of the sine wave.


    (N.b. we intentionally avoid letting users specify this in terms of a spatial wavelength
    because that would risk potential ambiguity with the wavelength of light.)
    """
    @utils.quantity_input(spatialfreq=1./u.meter, amplitude=u.meter)
    def  __init__(self,  name='Sine WFE', spatialfreq=1.0, amplitude=1e-6, phaseoffset=0, **kwargs):
        super(WavefrontError, self).__init__(name=name, **kwargs)

        self.sine_spatial_freq = spatialfreq
        self.sine_phase_offset = phaseoffset
        # note, can't call this next one 'amplitude' since that's already a property
        self.sine_amplitude = amplitude

    @_accept_wavefront_or_meters
    def get_opd(self, wave, units='meters'):
        """
        Parameters
        ----------
        wave : poppy.Wavefront (or float)
            Incoming Wavefront before this optic to set wavelength and
            scale, or a float giving the wavelength in meters
            for a temporary Wavefront used to compute the OPD.
        units : 'meters' or 'waves'
            Coefficients are supplied as meters of OPD, but the
            resulting OPD can be converted to
            waves based on the `Wavefront` wavelength or a supplied
            wavelength value.
        """

        y, x = self.get_coordinates(wave)  # in meters

        opd = self.sine_amplitude.to(u.meter).value * \
                np.sin( 2*np.pi * (x * self.sine_spatial_freq.to(1/u.meter).value + self.sine_phase_offset))

        if units == 'waves':
            opd /= wave.wavelength.to(u.meter).value
        return opd
