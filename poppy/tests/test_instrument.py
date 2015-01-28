from __future__ import (absolute_import, division, print_function, unicode_literals)

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import pytest

try:
    import pysynphot
    _HAS_PYSYNPHOT = True
except ImportError:
    pysynphot = None
    _HAS_PYSYNPHOT = False

from .. import poppy_core
from .. import instrument
from .. import optics
from .. import zernike

WEIGHTS_DICT = {'wavelengths': [2.0e-6, 2.1e-6, 2.2e-6], 'weights': [0.3, 0.5, 0.2]}
WAVELENGTHS_ARRAY = np.array(WEIGHTS_DICT['wavelengths'])
WEIGHTS_ARRAY = np.array(WEIGHTS_DICT['weights'])
FOV_PIXELS = 100

def test_instrument_source_weight_dict(weights_dict=WEIGHTS_DICT):
    """
    Tests the ability to provide a source spectrum in the form of wavelengths and weights
    """
    inst = instrument.Instrument()
    psf = inst.calcPSF(source=weights_dict, fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(weights_dict['wavelengths']), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    osys = inst._getOpticalSystem(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    output = np.zeros((2 * FOV_PIXELS, 2 * FOV_PIXELS))
    for wavelength, weight in zip(weights_dict['wavelengths'], weights_dict['weights']):
        output += weight * osys.calcPSF(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf

def test_instrument_source_weight_array(wavelengths=WAVELENGTHS_ARRAY, weights=WEIGHTS_ARRAY):
    """
    Tests the ability to provide a source spectrum as a (wavelengths, weights) tuple of arrays
    """
    inst = instrument.Instrument()
    psf = inst.calcPSF(source=(wavelengths, weights), fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    osys = inst._getOpticalSystem(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    output = np.zeros((2 * FOV_PIXELS, 2 * FOV_PIXELS))
    for wavelength, weight in zip(wavelengths, weights):
        output += weight * osys.calcPSF(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf

def test_instrument_wavelengths_and_weights(wavelengths=WAVELENGTHS_ARRAY, weights=WEIGHTS_ARRAY):
    """
    Tests the ability to just provide wavelengths, weights directly
    """
    inst = instrument.Instrument()
    psf = inst.calcPSF(wavelength=wavelengths, weight=weights, fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    osys = inst._getOpticalSystem(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    output = np.zeros((2 * FOV_PIXELS, 2 * FOV_PIXELS))
    for wavelength, weight in zip(wavelengths, weights):
        output += weight * osys.calcPSF(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf

@pytest.mark.skipif(not _HAS_PYSYNPHOT, reason="PySynphot dependency not met")
def test_instrument_source_pysynphot():
    """
    Tests the ability to provide a source as a pysynphot.Spectrum object
    """

    # 5700 K blackbody + Johnson B filter
    wavelengths = np.array([3.94000000e-07, 4.22000000e-07, 4.50000000e-07,
                            4.78000000e-07, 5.06000000e-07])
    weights = np.array([0.18187533, 0.29036168, 0.26205719, 0.1775512, 0.0881546])

    inst = instrument.Instrument()
    inst.filter = 'B'
    psf_weights_explicit = inst.calcPSF(source=(wavelengths, weights), fov_pixels=FOV_PIXELS,
                                        detector_oversample=2, fft_oversample=2, nlambda=5)
    psf_weights_pysynphot = inst.calcPSF(source=pysynphot.BlackBody(5700), fov_pixels=FOV_PIXELS,
                                         detector_oversample=2, fft_oversample=2, nlambda=5)
    assert psf_weights_pysynphot[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    assert np.allclose(psf_weights_explicit[0].data, psf_weights_pysynphot[0].data), (
        "PySynphot multiwavelength PSF does not match the weights and wavelengths pre-computed for "
        "a 5500 K blackbody in Johnson B (has pysynphot changed?)"
    )
    return psf_weights_pysynphot
