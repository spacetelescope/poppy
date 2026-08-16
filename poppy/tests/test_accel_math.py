# Test accelerated math functions
import numpy as np
import pytest

from .. import accel_math, conf, matrixDFT, optics


@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
@pytest.mark.skipif(accel_math._USE_MLX, reason="Test not relevant if using MLX")
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

    rtol = 1e-8 if conf.double_precision else 1e-6
    default_use_numexpr = accel_math._USE_NUMEXPR

    accel_math._USE_NUMEXPR = True
    r1 = accel_math._r(x,y)

    accel_math._USE_NUMEXPR = False
    r2 = accel_math._r(x,y)

    np.testing.assert_allclose(r1,r2, rtol=rtol)

    accel_math._USE_NUMEXPR = default_use_numexpr

@pytest.mark.skipif(accel_math._NUMEXPR_AVAILABLE is False, reason="numexpr not available")
@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
def test_exp_numexpr():
    """ Test that calculating the exponential gives equivalent results via
    plain numpy and numexpr"""
    x = np.linspace(-3,3,13)

    default_use_numexpr = accel_math._USE_NUMEXPR
    rtol = 1e-8 if conf.double_precision else 1e-6

    accel_math._USE_NUMEXPR = True
    r1 = accel_math._exp(x)

    accel_math._USE_NUMEXPR = False
    r2 = accel_math._exp(x)

    np.testing.assert_allclose(r1,r2, rtol=rtol)

    accel_math._USE_NUMEXPR = default_use_numexpr

@pytest.mark.skipif(accel_math._USE_CUPY, reason="Test not relevant if using CuPy")
def test_benchmark_fft():
    # minimalist case for speed, but at least it tests the function:
    accel_math.benchmark_fft(npix=512, iterations=2)

@pytest.mark.skipif(accel_math._MLX_AVAILABLE is False, reason="MLX not available")
def test_ensure_numpy_array_mlx_returns_numpy():
    import mlx.core as mx

    defaults = conf.use_mlx, conf.double_precision
    try:
        conf.use_mlx = True
        accel_math.update_math_settings()

        arr = mx.arange(16).reshape(4, 4)
        arr_np = accel_math.ensure_numpy_array(arr)
        assert isinstance(arr_np, np.ndarray)
        np.testing.assert_allclose(arr_np, np.arange(16).reshape(4, 4))
    finally:
        conf.use_mlx, conf.double_precision = defaults
        accel_math.update_math_settings()


def test_numexpr_disabled_when_mlx_enabled():
    defaults = conf.use_numexpr, conf.use_cupy, conf.use_mlx, conf.double_precision
    try:
        conf.use_numexpr = True
        conf.use_cupy = False
        conf.use_mlx = True
        accel_math.update_math_settings()

        assert accel_math._USE_NUMEXPR == (accel_math._NUMEXPR_AVAILABLE and not accel_math._USE_MLX)
    finally:
        conf.use_numexpr, conf.use_cupy, conf.use_mlx, conf.double_precision = defaults
        accel_math.update_math_settings()


def test_indices():
    """Test accel_math.indices() returns correct index arrays for 1D, 2D, and 3D shapes.

    Verifies:
    - Shape of each element matches the requested array shape
    - Values match np.indices as the reference
    - dtype=None gives integer arrays (matching np.indices default)
    - Explicit dtype via _float() gives floating-point arrays
    - MLX backend produces values matching NumPy reference
    """

    def _assert_indices_correct(shape, label=""):
        """Check that accel_math.indices matches np.indices for the given shape."""
        result = accel_math.indices(shape)
        ref = np.indices(shape)
        assert len(result) == len(shape), (
            f"{label}Expected {len(shape)} index arrays for shape {shape}, got {len(result)}"
        )
        for i in range(len(shape)):
            arr = np.asarray(accel_math.ensure_numpy_array(result[i]))
            assert arr.shape == shape, (
                f"{label}Dimension {i}: shape {arr.shape} != expected {shape}"
            )
            np.testing.assert_array_equal(
                arr, ref[i],
                err_msg=f"{label}Dimension {i} values do not match np.indices for shape {shape}",
            )

    # --- NumPy backend (default or whatever is currently active) ---
    _assert_indices_correct((7,), label="1D: ")            # 1D
    _assert_indices_correct((4, 5), label="2D: ")          # 2D
    _assert_indices_correct((2, 3, 4), label="3D: ")       # 3D

    # dtype=None should give integer arrays (matching np.indices default)
    result_int = accel_math.indices((3, 5))
    for arr in result_int:
        assert np.issubdtype(np.asarray(accel_math.ensure_numpy_array(arr)).dtype, np.integer), (
            "Default dtype should be integer"
        )

    # Explicit float dtype via _float() should give floating-point arrays
    result_float = accel_math.indices((3, 5), dtype=accel_math._float())
    assert len(result_float) == 2
    for arr in result_float:
        arr_np = np.asarray(accel_math.ensure_numpy_array(arr))
        assert arr_np.shape == (3, 5)
        assert np.issubdtype(arr_np.dtype, np.floating), "dtype=_float() should give float arrays"

    # --- MLX backend (if available) ---
    if accel_math._MLX_AVAILABLE:
        defaults = conf.use_mlx, conf.double_precision
        try:
            conf.use_mlx = True
            accel_math.update_math_settings()

            _assert_indices_correct((7,), label="MLX 1D: ")
            _assert_indices_correct((4, 5), label="MLX 2D: ")
            _assert_indices_correct((2, 3, 4), label="MLX 3D: ")

            # dtype=None on MLX should also give integer arrays
            result_mlx_int = accel_math.indices((3, 5))
            assert result_mlx_int.shape == (2, 3, 5), "MLX: stacked result should have shape (ndim, *shape)"
            for i in range(2):
                arr_np = np.asarray(accel_math.ensure_numpy_array(result_mlx_int[i]))
                assert np.issubdtype(arr_np.dtype, np.integer), (
                    "MLX: default dtype should be integer"
                )

            # Passing a NumPy dtype to the MLX backend should raise ValueError
            with pytest.raises(ValueError, match="MLX backend requires dtype"):
                accel_math.indices((4, 4), dtype=np.float32)
        finally:
            conf.use_mlx, conf.double_precision = defaults
            accel_math.update_math_settings()
