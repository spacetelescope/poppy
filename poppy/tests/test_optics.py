#Tests for individual Optic classes

from .. import poppy_core as poppy
import numpy as np
import astropy.io.fits as fits
from .test_core import check_wavefront



wavelength=1e-6


def test_scalar_transmission():

    wave = poppy.Wavefront(npix=100, wavelength=wavelength)
    nulloptic = poppy.ScalarTransmission()

    assert( nulloptic.getPhasor(wave) == 1.0)

    NDoptic = poppy.ScalarTransmission(transmission=1e-3)
    assert( NDoptic.getPhasor(wave) == 1.0e-3)


