
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import pytest
from astropy import units as u

from poppy import poppy_core, instrument, optics, utils
from poppy.instrument import _HAS_SYNPHOT

WEIGHTS_DICT = {'wavelengths': [2.0e-6, 2.1e-6, 2.2e-6], 'weights': [0.3, 0.5, 0.2]}
WAVELENGTHS_ARRAY = np.array(WEIGHTS_DICT['wavelengths'])
WEIGHTS_ARRAY = np.array(WEIGHTS_DICT['weights'])
FOV_PIXELS = 100

def test_instrument_source_weight_dict(weights_dict=WEIGHTS_DICT):
    """
    Tests the ability to provide a source spectrum in the form of wavelengths and weights
    """
    inst = instrument.Instrument()
    psf = inst.calc_psf(source=weights_dict, fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(weights_dict['wavelengths']), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    osys = inst.get_optical_system(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    output = np.zeros((2 * FOV_PIXELS, 2 * FOV_PIXELS))
    for wavelength, weight in zip(weights_dict['wavelengths'], weights_dict['weights']):
        output += weight * osys.calc_psf(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf

def test_instrument_source_weight_array(wavelengths=WAVELENGTHS_ARRAY, weights=WEIGHTS_ARRAY):
    """
    Tests the ability to provide a source spectrum as a (wavelengths, weights) tuple of arrays
    """
    inst = instrument.Instrument()
    psf = inst.calc_psf(source=(wavelengths, weights), fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    osys = inst.get_optical_system(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    output = np.zeros((2 * FOV_PIXELS, 2 * FOV_PIXELS))
    for wavelength, weight in zip(wavelengths, weights):
        output += weight * osys.calc_psf(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf

@pytest.mark.skipif(not _HAS_SYNPHOT, reason="synphot dependency not met")
def test_instrument_source_synphot():
    """
    Tests the ability to provide a source as a SourceSpectrum object
    """
    from synphot import SourceSpectrum
    from synphot.models import BlackBodyNorm1D

    # 5700 K blackbody + Johnson B filter
    wavelengths = np.array([3.94000000e-07, 4.22000000e-07, 4.50000000e-07,
                            4.78000000e-07, 5.06000000e-07])
    weights = np.array([0.18187533, 0.29036168, 0.26205719, 0.1775512, 0.0881546])

    inst = instrument.Instrument()
    inst.filter = 'B'
    psf_weights_explicit = inst.calc_psf(source=(wavelengths, weights), fov_pixels=FOV_PIXELS,
                                        detector_oversample=2, fft_oversample=2, nlambda=5)
    bb = SourceSpectrum(BlackBodyNorm1D, temperature=5700 * u.K)
    psf_weights_synphot = inst.calc_psf(source=bb, fov_pixels=FOV_PIXELS,
                                         detector_oversample=2, fft_oversample=2, nlambda=5)
    assert psf_weights_synphot[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    assert np.allclose(psf_weights_explicit[0].data, psf_weights_synphot[0].data,
            rtol=1e-4), ( # Slightly larger tolerance to accomodate minor changes w/ synphot versions
        "synphot multiwavelength PSF does not match the weights and wavelengths pre-computed for "
        "a 5500 K blackbody in Johnson B (has synphot changed?)"
    )
    return psf_weights_synphot

@pytest.mark.skipif(not _HAS_SYNPHOT, reason="synphot dependency not met")
def test_synphot_spectra_cache():
    """
    The result of the synphot calculation is cached. This ensures the appropriate
    key appears in the cache after one calculation, and that subsequent calculations
    proceed without errors (exercising the cache lookup code).
    """
    from synphot import SourceSpectrum
    from synphot.models import BlackBodyNorm1D

    source = SourceSpectrum(BlackBodyNorm1D, temperature=5700 * u.K)
    nlambda = 2
    ins = instrument.Instrument()
    cache_key = ins._get_spec_cache_key(source, nlambda)
    assert cache_key not in ins._spectra_cache, "How is the cache populated already?"
    psf = ins.calc_psf(nlambda=2, source=source, fov_pixels=2)
    assert cache_key in ins._spectra_cache, "Cache was not populated"
    psf2 = ins.calc_psf(nlambda=2, source=source, fov_pixels=2)
    maxdiff = np.abs(psf[0].data - psf2[0].data).max()

    assert(maxdiff < 1e-7), "PSF using cached spectrum differs from first PSF calculated"


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
    psf_no_jitter = inst.calc_psf(monochromatic=1e-6, fov_arcsec=3, oversample=oversample)


    jitter_sigmas = [ 0.005,  0.020,  0.080, 0.16, 0.5]  # arcseconds

    fwhm_to_sigma = 2*np.sqrt(2*np.log(2))
    # Scale factor from Gaussian FWHM to Gaussian sigma

    tolerance = 0.05
    # how close the expected and measured sigma values should be, post jitter
    # Measured PSF sigma must be within 5% of expected value


    for JITTER_SIGMA in jitter_sigmas:

        inst.options['jitter'] = 'gaussian'
        inst.options['jitter_sigma'] = JITTER_SIGMA
        psf_jitter = inst.calc_psf(monochromatic=1e-6, fov_arcsec=3, oversample=oversample)

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
        assert reldiff < tolerance, "Post-jitter PSF width is too different from expected width: {:.4f}, {:.4f} arcsec".format(post_sigma, expected_post_sigma)


def test_instrument_calc_datacube():
    """ Tests ability to make a datacube"""

    inst = instrument.Instrument()
    psf = inst.calc_datacube(WAVELENGTHS_ARRAY, fov_pixels=FOV_PIXELS,
                       detector_oversample=2, fft_oversample=2)
    assert psf[0].header['NWAVES'] == len(WAVELENGTHS_ARRAY), \
        "Number of wavelengths in PSF header does not match number requested"
    assert len(psf[0].data.shape) == 3, "Incorrect dimensions for output cube"
    assert psf[0].data.shape[0] ==  len(WAVELENGTHS_ARRAY), \
                    "Spectral axis of datacube does not match number requested"

    # Check individual planes
    osys = inst.get_optical_system(fov_pixels=FOV_PIXELS,
                                  detector_oversample=2, fft_oversample=2)
    for i, wavelength in enumerate(WAVELENGTHS_ARRAY):

        monopsf = osys.calc_psf(wavelength=wavelength)

        assert np.allclose(psf[0].data[i], monopsf[0].data), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    return psf
