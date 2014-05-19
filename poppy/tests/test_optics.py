#Tests for individual Optic classes

from .. import poppy_core 
import numpy as np
import astropy.io.fits as fits
from .test_core import check_wavefront



wavelength=1e-6



#def test_OpticalElement():
#    pass


#def test_FITSOpticalElement():
#    pass

#def test_Rotation():
#    pass

#def test_InverseTransmission():
#    pass

#------ Generic Analytic elements -----

def test_scalar_transmission():
    """ Verify this adjusts the wavefront intensity appropriately """
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    for transmission in [1.0, 1.0e-3, 0.0]:

        optic = poppy_core.ScalarTransmission()
        assert( optic.getPhasor(wave) == 1.0)


#------ Analytic Image Plane elements -----

def test_IdealRectangularFieldStop():
    fs = poppy_core.IdealRectangularFieldStop(width=1, height=10)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= fs
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1000 # 1/10 of the 1e4 element array


def test_IdealBarOcculter():
    optic= poppy_core.IdealBarOcculter(width=1, angle=0)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 9000 # 9/10 of the 1e4 element array

#def test_rotations_IdealRectangularFieldStop():
#
#    # First let's do a rotation of the wavefront itself by 90^0 after an optic
#
#    # now try a 90^0 rotation for the field stop at that optic. Assuming perfect system w/ no aberrations when comparing rsults. ? 
#    fs = poppy_core.IdealRectangularFieldStop(width=1, height=10, ang;le=90)
#    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square
#
#    wave*= fs
#    assert wave.shape[0] == 100
#    assert fs.intensity.sum() == 1000 # 1/10 of the 1e4 element array
#
#


#------ Analytic Pupil Plane elements -----

def test_ParityTestAperture():
    """ Verify that this aperture is not symmetric in either direction"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    array = poppy_core.ParityTestAperture().getPhasor(wave)

    assert np.any(array[::-1,:] != array)
    assert np.any(array[:,::-1] != array)


def test_RectangleAperture():
    optic= poppy_core.RectangleAperture(width=5, height=3)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1500 # 50*30 pixels of the 1e4 element array

    optic= poppy_core.RectangleAperture(width=2, height=7)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1400 # 50*30 pixels of the 1e4 element array


def test_HexagonAperture():
    optic= poppy_core.RectangleAperture(width=5, height=3)
    # should make hexagon PSF and compare to analytic expression
    pass


def test_MultiHexagonAperture():
    # should make multihexagon PSF and compare to analytic expression
    pass


