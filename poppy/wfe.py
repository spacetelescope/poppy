"""
Analytic optical element classes to introduce a specified wavefront
error in an OpticalSystem

 * ZernikeWFE
 * ParameterizedWFE (for use with hexike or zernike basis functions)
 * TODO: PowerSpectrumWFE
 * TODO: KolmogorovWFE

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import collections
import numpy as np

from .optics import AnalyticOpticalElement, CircularAperture
from .poppy_core import Wavefront, _PUPIL
from . import zernike

__all__ = ['WavefrontError', 'ParameterizedWFE', 'ZernikeWFE']

class WavefrontError(AnalyticOpticalElement):
    def __init__(self, **kwargs):
        super(WavefrontError, self).__init__(planetype=_PUPIL, **kwargs)

    def rms(self):
        """ RMS wavefront error induced by this surface """
        raise NotImplementedError('Not implemented yet')

    def peaktovalley(self):
        """ Peak-to-valley wavefront error induced by this surface """
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
        and `theta`. `nterms` specifies how many terms to compute,
        starting with the j=1 term in the Noll indexing convention for
        `nterms` = 1 and counting up. `rho` and `theta` are square
        arrays holding the rho and theta coordinates at each pixel in
        the pupil plane.

        `rho` is normalized such that `rho` == 1.0 for pixels at
        `radius` meters from the center.
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

    def getPhasor(self, wave):
        # getPhasor specified to accept wave as float wavelength or
        # Wavefront instance:
        if not isinstance(wave, Wavefront):
            wave = Wavefront(wavelength=wave)

        rho, theta = _wave_to_rho_theta(wave, self.radius)

        # the Zernike optic, being normalized on a circle, is
        # implicitly also a circular aperture:
        aperture_intensity = self.circular_aperture.getPhasor(wave)

        combined_zernikes = np.zeros(wave.shape, dtype=np.float64)
        for j, k in enumerate(self.coefficients, start=1):
            combined_zernikes += k * zernike.zernike1(
                j,
                rho=rho,
                theta=theta,
                mask_outside=False,
                outside=0.0
            )

        combined_zernikes *= aperture_intensity

        opd_as_phase = 2 * np.pi * combined_zernikes / wave.wavelength
        zernike_wfe_phasor = np.exp(1.j * opd_as_phase)
        return zernike_wfe_phasor
