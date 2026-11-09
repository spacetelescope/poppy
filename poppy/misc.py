############################################################################
#
#  Airy functions for comparison's sake
#
############################################################################

import math

import matplotlib.pyplot as plt

from poppy import _scipy_compat, accel_math
from poppy.accel_math import _float, xp

_RADtoARCSEC = 180. * 60 * 60 / math.pi  # ~ 206265
_ARCSECtoRAD = math.pi / (180. * 60 * 60)


def _sinc(x):
    """Normalized sinc, sin(pi x)/(pi x), built from standard functions (sinc is not in the array API)"""
    px = math.pi * x
    safe = xp.where(px == 0, 1.0, px)
    return xp.where(px == 0, 1.0, xp.sin(safe) / safe)


def airy_1d(diameter=1.0, wavelength=1e-6, length=512, pixelscale=0.010,
            obscuration=0.0, plot=False):
    """ 1-dimensional Airy function PSF calculator

    Parameters
    ----------
    diameter : float
        aperture diameter in meters
    wavelength : float
        wavelength in meters
    length : int
        array size
    pixelscale : float
        arcseconds
    obscuration: float, optional
        ratio of secondary obscuration (between 0 and 1)
    plot : bool
        Display a plot automatically

    Returns
    -------
    r : array
        radius array in arcsec
    airy : array
        Array with the Airy function values, normalized to 1 at peak
    """

    r = xp.arange(length, dtype=_float()) * pixelscale

    v = math.pi * (r * _ARCSECtoRAD) * diameter / wavelength
    e = obscuration

    # pedantically avoid divide by 0 by setting 0s to minimum nonzero number
    v = xp.where(v == 0, xp.finfo(v.dtype).eps, v)

    airy = 1./(1-e**2)**2 * ((2*_scipy_compat.j1(v) - e*2*_scipy_compat.j1(e*v))/v)**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248

    if plot:
        plt.semilogy(r, airy)
        plt.xlabel("radius [arcsec]")
        plt.ylabel("PSF intensity")
    return r, airy


def airy_2d(diameter=1.0, wavelength=1e-6, shape=(512, 512), pixelscale=0.010,
            obscuration=0.0, center=None):
    """ 2-dimensional Airy function PSF calculator

    Parameters
    ----------
    diameter: float
        aperture diameter in meters
    wavelength : float
        Wavelength in meters
    shape : tuple
        array shape
    pixelscale :
        arcseconds
    obscuration: float, optional
        Diameter of secondary obscuration
    center: tuple, optional
        Offset coordinates for center of output array
    """

    if center is None:
        center = ((shape[0] - 1.) / 2, (shape[1] - 1.) / 2)
    y, x = accel_math.indices(shape, dtype=_float())
    y = (y - float(center[0])) * pixelscale
    x = (x - float(center[1])) * pixelscale
    r = xp.sqrt(x ** 2 + y ** 2)

    radius = float(diameter)/2.0

    k = 2 * math.pi / wavelength  # wavenumber
    v = k * radius * r * _ARCSECtoRAD
    e = obscuration

    # pedantically avoid divide by 0 by setting 0s to minimum nonzero number
    v = xp.where(v == 0, xp.finfo(v.dtype).eps, v)

    airy = 1./(1-e**2)**2 * ((2*_scipy_compat.j1(v) - e*2*_scipy_compat.j1(e*v))/v)**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248
    return airy


def sinc2_2d(width=1.0, height=None, wavelength=1e-6, shape=(512, 512), pixelscale=0.010,
             center=None):
    """
    Create a 2D sinc function PSF, representing the PSF of a square or rectangular aperture

    Parameters
    ----------
    width : float
        Width in meters of the aperture.
    height : float, optional
        height in meters of the aperture. If not specified, the aperture is assumed
        to be a square so height=width
    wavelength : float
        wavelength in meters
    shape : tuple with 2 elements
        shape of array to create
    pixelscale : float
        pixel scale in arcseconds per pixel
    center : tuple with 2 elements, optional
        Center coordinates of the PSF. Defaults to center of array.

    """

    if height is None:
        height = width
    halfwidth = float(width)/2
    halfheight = float(height)/2

    if center is None:
        center = ((shape[0] - 1.) / 2, (shape[1] - 1.) / 2)
    y, x = accel_math.indices(shape, dtype=_float())
    y = (y - float(center[0])) * pixelscale
    x = (x - float(center[1])) * pixelscale

    k = 2 * math.pi / wavelength  # wavenumber
    alpha = k * x * halfwidth * _ARCSECtoRAD
    beta = k * y * halfheight * _ARCSECtoRAD

    psf = (_sinc(alpha)) ** 2 * (_sinc(beta)) ** 2

    return psf
################################################################################


