#Tests for individual Optic classes

from .. import poppy_core 
import numpy as np
import astropy.io.fits as fits
from .test_core import check_wavefront



wavelength=1e-6


def test_wavefront():

    # basic creation, pixel coords
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)
    assert wave.shape[0] == 100
    assert wave.shape[1] == 100
    assert wave.planetype == poppy_core._PUPIL
    assert wave.wavelength == wavelength

    # basic creation, spatial coords
    wave = poppy_core.Wavefront(npix=50, pixelscale=0.2, wavelength=wavelength)
    assert wave.shape[0] == 50
    assert wave.shape[1] == 50
    assert wave.planetype == poppy_core._IMAGE

    assert wave.coordinates()[0].shape[0]== 50
    assert wave.coordinates()[0][0,0] == -4.9
    # check we get the right values w/in close to machine precision
    assert np.abs(wave.coordinates()[0][0,0] +4.9) < np.finfo(float).eps*5
    assert np.abs(wave.coordinates()[0][13,33] +2.3) < np.finfo(float).eps*5
    assert np.abs(wave.coordinates()[0][-13,-33] -2.5) < np.finfo(float).eps*5


    # test __str__
    wave = poppy_core.Wavefront(npix=100, wavelength=1e-6)
    string_representation = str(wave)
    assert string_representation =="""Wavefront:
        wavelength = 1.000000 microns
        shape = (100,100)
        sampling = 0.080000 meters/pixel"""

    # test copy
    wave2 = wave.copy()
    assert str(wave) == str(wave2)
    assert wave is not wave2


    # test rotation doesn't do anything in imul, but instead does stuff in propatage
    rot = poppy_core.Rotation(10)
    wave0 = wave
    wave *= rot
    assert wave is wave0

def test_scalar_transmission():
    """ Verify this adjusts the wavefront intensity appropriately """
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    for transmission in [1.0, 1.0e-3, 0.0]:

        optic = poppy_core.ScalarTransmission()
        assert( optic.getPhasor(wave) == 1.0)



def test_ParityTestAperture():
    """ Verify that this aperture is not symmetric in either direction"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    array = poppy_core.ParityTestAperture().getPhasor(wave)

    assert np.any(array[::-1,:] != array)
    assert np.any(array[:,::-1] != array)



