# Test accelerated math functions
import numpy as np
import pytest

from .. import accel_math, matrixDFT, optics


@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
@pytest.mark.skipif(accel_math._NUMEXPR_AVAILABLE is False, reason="numexpr not available")
def test_MFT_MFTwithnumexpr_equivalence(display=False, displaycrop=None):
    """ Test that the basic MFT transform is numerically equivalent to the
    version accelerated with NUMEXPR, and both match the FFT if calculated on the correct sampling. """

    centering='FFTSTYLE' # needed if you want near-exact agreement of MFT and FFT!

    default_use_numexpr = accel_math._USE_NUMEXPR

    for useflag in [True,False]:
        accel_math._USE_NUMEXPR = useflag

        imgin = optics.ParityTestAperture().sample(wavelength=1e-6, npix=256)

        npix = imgin.shape
        nlamD = np.asarray(imgin.shape)
        mft = matrixDFT.MatrixFourierTransform(centering=centering)
        mftout = mft.perform(imgin, nlamD, npix)

        # SIGN CONVENTION: with our adopted sign conventions, forward propagation requires an inverse fft
        # This differs from behavior in versions of poppy prior to 1.0.
        # Further, note that the numpy normalization convention includes 1/n for the inverse transform and 1 for
        # the forward transform, while we want to more symmetrically apply 1/sqrt(n) in both directions.
        fftout = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(imgin))) * np.sqrt(imgin.shape[0] * imgin.shape[1])

        norm_factor = abs(mftout).sum()

        absdiff = abs(mftout-fftout) / norm_factor

        assert(np.all(absdiff < 1e-10))


    accel_math._USE_NUMEXPR = default_use_numexpr



@pytest.mark.skipif(accel_math._NUMEXPR_AVAILABLE is False, reason="numexpr not available")
def test_r():
    """ Test that calculating the radius gives equivalent results via
    plain numpy and numexpr"""
    y, x = np.indices((10,20))

    default_use_numexpr = accel_math._USE_NUMEXPR

    accel_math._USE_NUMEXPR = True
    r1 = accel_math._r(x,y)

    accel_math._USE_NUMEXPR = False
    r2 = accel_math._r(x,y)

    np.testing.assert_almost_equal(r1,r2)

    accel_math._USE_NUMEXPR = default_use_numexpr

@pytest.mark.skipif(accel_math._NUMEXPR_AVAILABLE is False, reason="numexpr not available")
@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
def test_exp_numexpr():
    """ Test that calculating the exponential gives equivalent results via
    plain numpy and numexpr"""
    x = np.linspace(-3,3,13)

    default_use_numexpr = accel_math._USE_NUMEXPR

    accel_math._USE_NUMEXPR = True
    r1 = accel_math._exp(x)

    accel_math._USE_NUMEXPR = False
    r2 = accel_math._exp(x)

    np.testing.assert_almost_equal(r1,r2)

    accel_math._USE_NUMEXPR = default_use_numexpr

@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
def test_benchmark_fft():
    # minimalist case for speed, but at least it tests the function:
    accel_math.benchmark_fft(npix=512, iterations=2)


def test_backend_selection():
    """The backend is resolved at import, and is one of the supported array-API namespaces"""
    assert accel_math._BACKEND in accel_math._SUPPORTED_BACKENDS
    x = accel_math.xp.zeros((4, 4), dtype=accel_math._complex())
    assert accel_math.to_numpy(x).dtype == (np.complex128 if poppy_conf_double() else np.complex64)


def poppy_conf_double():
    from .. import conf
    return conf.double_precision


def test_unsupported_backend_raises():
    from .. import conf
    default = conf.array_backend
    try:
        conf.array_backend = 'jax'  # not yet supported
        with pytest.raises(NotImplementedError):
            accel_math._resolve_backend()
    finally:
        conf.array_backend = default


def test_to_numpy_passthrough():
    arr = np.arange(3.)
    assert accel_math.to_numpy(arr) is arr
    assert accel_math.to_numpy(2.5) == 2.5
    assert not accel_math.is_on_gpu(arr)


def test_fft_2d_does_not_require_inplace():
    """fft_2d must round trip, and return normalized results, using only returned values"""
    rng = np.random.default_rng(1)
    arr = accel_math.xp.asarray(rng.random((32, 32)) + 0j)
    back = accel_math.fft_2d(accel_math.fft_2d(accel_math.xp.asarray(arr.copy()), forward=True), forward=False)
    np.testing.assert_allclose(accel_math.to_numpy(back), accel_math.to_numpy(arr), atol=1e-12)
