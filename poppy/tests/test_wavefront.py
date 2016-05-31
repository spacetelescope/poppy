
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
    assert np.allclose(wave0.wavefront[npix/2], wave_inv.wavefront[npix/2, ::-1])
    assert np.allclose(wave0.wavefront[:, npix/2], wave_inv.wavefront[::-1, npix/2])


def test_wavefront_asFITS():
    """ Test casting wavefronts to FITS arrays, in all possible combinations
    """
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)

    intens = wave.asFITS(what='intensity')
    assert isinstance(intens,fits.HDUList)
    assert intens[0].data.shape == wave.shape
    phase = wave.asFITS(what='phase')
    assert phase[0].data.shape == wave.shape
    assert np.all(np.isreal(phase[0].data))
    allparts = wave.asFITS(what='all')
    assert allparts[0].data.shape == (3, wave.shape[0],wave.shape[1])
    parts = wave.asFITS(what='parts')
    assert parts[0].data.shape == (2, wave.shape[0],wave.shape[1])
    #complexwave = wave.asFITS(what='complex')
    #assert complexwave[0].data.shape == wave.shape
    #assert np.all(np.iscomplex(phase[0].data))

