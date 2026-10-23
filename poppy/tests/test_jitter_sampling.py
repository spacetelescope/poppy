"""Jitter must broaden a sampled PSF even below one detector sample."""
import numpy as np
import pytest
from astropy.io import fits

from poppy.instrument import Instrument


def apply_jitter(image, sigma, pixelscale=1):
    result = fits.HDUList([fits.PrimaryHDU(image.copy())])
    result[0].header['PIXELSCL'] = pixelscale
    Instrument()._apply_jitter(result, {'jitter': 'gaussian', 'jitter_sigma': sigma})
    return result


@pytest.mark.parametrize('sigma', [0, 0.05, 0.22, 0.45, 1.4])
def test_jitter_gaussian_variance(sigma):
    # A well-sampled Gaussian has a known continuous convolution: variances add.
    x = np.arange(65) - 32
    image = np.exp(-(x[:, None]**2 + x[None, :]**2) / (2 * 2.0**2))
    image /= image.sum()
    result = apply_jitter(image, sigma)
    actual_variance = np.sum(result[0].data * x[None, :]**2)
    np.testing.assert_allclose(actual_variance, 2.0**2 + sigma**2, rtol=1e-8)
    np.testing.assert_allclose(result[0].data.sum(), image.sum(), rtol=1e-14)
    assert result[0].header['JITRSIGM'] == sigma
    np.testing.assert_allclose(result[0].header['JITRSTRL'], result[0].data.max() / image.max())


@pytest.mark.parametrize('shape', [(17, 19), (18, 20)])
@pytest.mark.parametrize('sigma', [0.22, 1.4])
def test_jitter_gaussian_transfer_function(shape, sigma):
    # A cosine mode is an eigenfunction of Gaussian convolution. Reflecting
    # about the outer pixel edges continues this mode without a discontinuity.
    ny, nx = shape
    ky, kx = 3 * np.pi / ny, 4 * np.pi / nx
    mode = np.cos(ky * (np.arange(ny)[:, None] + 0.5)) * np.cos(kx * (np.arange(nx) + 0.5))
    image = 1 + 0.1 * mode
    result = apply_jitter(image, sigma * 0.027, pixelscale=0.027)
    expected = 1 + 0.1 * mode * np.exp(-0.5 * sigma**2 * (ky**2 + kx**2))
    np.testing.assert_allclose(result[0].data, expected, atol=1e-14, rtol=1e-14)


def test_jitter_preserves_constant_image_at_boundaries():
    image = np.full((19, 24), 3.0)
    np.testing.assert_allclose(apply_jitter(image, 2.4)[0].data, image, atol=1e-14)


@pytest.mark.parametrize('sigma', [0, -0.1])
def test_nonpositive_width_preserves_exact_image(sigma):
    image = np.random.default_rng(12).random((17, 19))
    np.testing.assert_array_equal(apply_jitter(image, sigma)[0].data, image)
