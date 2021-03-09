# Test accelerated math functions
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as fits

import pytest

from .. import matrixDFT
from .. import accel_math
from .. import optics

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
def test_exp():
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

def test_benchmark_fft():
    # minimalist case for speed, but at least it tests the function:
    accel_math.benchmark_fft(npix=512, iterations=2)
