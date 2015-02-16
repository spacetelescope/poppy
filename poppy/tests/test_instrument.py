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

from poppy import poppy_core, instrument, optics, utils

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

def test_instrument_gaussian_jitter():
    """
    Tests that jitter is applied by convolving with the specified Gaussian and
    the resulting PSF FWHM is wider than the PSF without jitter.
    Also test that the resulting PSF FWHM is close to the expected value based
    on the root sum of squares of the initial FWHM and the jitter kernel.
    """


    inst = instrument.Instrument()
    inst.pixelscale=0.010
    inst.options['jitter'] = None
    oversample = 1 # oversample
    psf_no_jitter = inst.calcPSF(monochromatic=1e-6, fov_arcsec=2, oversample=oversample)


    jitter_sigmas = [ 0.005,  0.020,  0.080, 0.16, 0.5]  # arcseconds

    fwhm_to_sigma = 2*np.sqrt(2*np.log(2))
    # Scale factor from Gaussian FWHM to Gaussian sigma

    tolerance = 0.05
    # how close the expected and measured sigma values should be, post jitter
    # Measured PSF sigma must be within 5% of expected value


    for JITTER_SIGMA in jitter_sigmas:

        inst.options['jitter'] = 'gaussian'
        inst.options['jitter_sigma'] = JITTER_SIGMA
        psf_jitter = inst.calcPSF(monochromatic=1e-6, fov_arcsec=2, oversample=oversample)

        fwhm_no_jitter = utils.measure_fwhm(psf_no_jitter)
        fwhm_with_jitter = utils.measure_fwhm(psf_jitter)

        assert fwhm_with_jitter > fwhm_no_jitter, ("Applying jitter didn't increase the "
            "FWHM of the PSF")

        assert psf_jitter[0].header['JITRTYPE'] == 'Gaussian convolution'
        assert psf_jitter[0].header['JITRSIGM'] == JITTER_SIGMA
        assert psf_jitter[0].header['JITRSTRL'] < 1.0

        expected_post_sigma = np.sqrt((fwhm_no_jitter/fwhm_to_sigma)**2 + JITTER_SIGMA**2)
        pre_sigma = fwhm_no_jitter/fwhm_to_sigma
        post_sigma = fwhm_with_jitter/fwhm_to_sigma
        poppy_core._log.info("TEST: Jitter sigma={0:.4f}.   PSF sigma pre: {1:.4f}    post: {2:.4f}    expected: {3:.4f}".format(JITTER_SIGMA, pre_sigma, post_sigma, expected_post_sigma))

        reldiff = np.abs(post_sigma-expected_post_sigma)/post_sigma
        assert reldiff < tolerance, "Post-jitter PSF width is too different from expected width"




