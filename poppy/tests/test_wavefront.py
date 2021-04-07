
import poppy
from .. import poppy_core
from .. import optics
import numpy as np
import astropy.io.fits as fits
from .test_core import check_wavefront
import astropy.units as u



wavelength=1e-6*u.m


def test_wavefront_in_pixels():

    # basic creation, pixel coords
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)
    assert wave.shape[0] == 100
    assert wave.shape[1] == 100
    assert wave.planetype == poppy_core._PUPIL
    assert wave.wavelength == wavelength

def test_wavefront_in_arcsec():
    # basic creation, spatial coords
    wave = poppy_core.Wavefront(npix=50, pixelscale=0.2, wavelength=wavelength)
    assert wave.shape[0] == 50
    assert wave.shape[1] == 50
    assert wave.planetype == poppy_core._IMAGE

def test_wavefront_coordinates():
    wave = poppy_core.Wavefront(npix=50, pixelscale=0.2, wavelength=wavelength)
    assert wave.coordinates()[0].shape[0]== 50
    assert wave.coordinates()[0][0,0] == -4.9
    # check we get the right values w/in close to machine precision
    assert np.abs(wave.coordinates()[0][0,0] +4.9) < np.finfo(float).eps*5
    assert np.abs(wave.coordinates()[0][13,33] +2.3) < np.finfo(float).eps*5
    assert np.abs(wave.coordinates()[0][-13,-33] -2.5) < np.finfo(float).eps*5


def test_wavefront_str():
    # test __str__
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)
    string_representation = str(wave)
    assert string_representation =="""Wavefront:
        wavelength = 1.0 micron
        shape = (100, 100)
        sampling = 0.08 m / pix"""

def test_wavefront_copy():
    # test copy
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)
    wave2 = wave.copy()
    assert str(wave) == str(wave2)
    assert wave is not wave2


def test_wavefront_rotation():
    # test rotation doesn't do anything in imul, but instead does stuff in propatage
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)
    rot = poppy_core.Rotation(10)
    wave0 = wave
    wave *= rot
    assert np.allclose(wave0.wavefront, wave.wavefront)

def test_wavefront_rot90_vs_ndimagerotate_consistency(plot=False):
    """Test that rotating a Wavefront via either of the two
    methods yields consistent results. This compares an exact
    90 degree rotation and an interpolating not-quite-90-deg rotation.
    Both methods should rotate counterclockwise and consistently.
    """
    letterf = poppy.optics.LetterFAperture()
    wave = poppy.Wavefront(diam=3 * u.m, npix=128)
    wave *= letterf
    wave2 = wave.copy()

    wave.rotate(90)
    wave2.rotate(89.99999)

    assert np.allclose(wave2.intensity, wave.intensity, atol=1e-5), "Inconsistent results from the two rotation methods"

    from poppy.tests.test_sign_conventions import brighter_top_half, brighter_left_half

    assert brighter_left_half(wave.intensity), "Rotated wavefront orientation not as expected"
    assert not brighter_top_half(wave.intensity), "Rotated wavefront orientation not as expected"

    if plot:
        fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
        wave.display(ax=axes[0])
        wave2.display(ax=axes[1])
        axes[0].set_title("Rot90")
        axes[1].set_title("ndimage rotate(89.9999)")

def test_wavefront_inversion():
    npix=100
    wave = poppy_core.Wavefront(npix=npix, wavelength=1e-6)
    wave *= optics.ParityTestAperture()


    # test the coord transform doesn't do anything in imul (but instead does stuff in propagate)
    inv = poppy_core.CoordinateInversion('both')
    wave0 = wave
    wave *= inv
    assert np.allclose(wave0.wavefront, wave.wavefront)

    # test the inversion
    wave_inv = wave.copy()
    wave_inv.invert()
    assert np.allclose(wave0.wavefront[npix//2], wave_inv.wavefront[npix//2, ::-1])
    assert np.allclose(wave0.wavefront[:, npix//2], wave_inv.wavefront[::-1, npix//2])


def test_wavefront_as_fits():
    """ Test casting wavefronts to FITS arrays, in all possible combinations
    """
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)

    intens = wave.as_fits(what='intensity')
    assert isinstance(intens,fits.HDUList)
    assert intens[0].data.shape == wave.shape
    phase = wave.as_fits(what='phase')
    assert phase[0].data.shape == wave.shape
    assert np.all(np.isreal(phase[0].data))
    allparts = wave.as_fits(what='all')
    assert allparts[0].data.shape == (3, wave.shape[0],wave.shape[1])
    parts = wave.as_fits(what='parts')
    assert parts[0].data.shape == (2, wave.shape[0],wave.shape[1])
    #complexwave = wave.as_fits(what='complex')
    #assert complexwave[0].data.shape == wave.shape
    #assert np.all(np.iscomplex(phase[0].data))

