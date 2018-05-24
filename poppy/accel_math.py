# accel_math.py
#
# Various functions related to accelerated computations using FFTW, CUDA, numexpr, and related.
#
import numpy as np
from . import conf

# Setup infrastructure for FFTW
_FFTW_INIT = {}  # dict of array sizes for which we have already performed the required FFTW planning step
_FFTW_FLAGS = ['measure']

try:
    # try to import FFTW to see if it is available
    import pyfftw

    _FFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    _FFTW_AVAILABLE = False

try:
    # try to import accelerate package to see if it is available
    import accelerate

    _ACCELERATE_AVAILABLE = True
except ImportError:
    accelerate = None
    _ACCELERATE_AVAILABLE = False

try:
    # try to import numexpr package to see if it is available
    import numexpr as ne

    _NUMEXPR_AVAILABLE = True
except ImportError:
    ne = None
    _NUMEXPR_AVAILABLE = False

_USE_CUDA = (conf.use_cuda and _ACCELERATE_AVAILABLE)
_USE_NUMEXPR = (conf.use_numexpr and _NUMEXPR_AVAILABLE)

if _USE_NUMEXPR:
    import numexpr as ne

if _USE_CUDA:
    from numba import cuda


def _float():
    """ Returns numpy data type for desired precision based on configuration """
    # How many bits per float to use?
    return np.float64 if conf.double_precision else np.float32


def _complex():
    """ Returns numpy data type for desired precision based on configuration """
    # How many bits per complex float to use?
    return np.complex128 if conf.double_precision else np.complex64


def _r(x, y):
    """ Function to return the radius given x and y """
    if _USE_NUMEXPR:
        return ne.evaluate("sqrt(x**2+y**2)")
    else:
        return np.sqrt(x ** 2 + y ** 2)


def _exp(x):
    """
    Function to speed up taking exponential of an array if NumExpr is available.
    Otherwise defaults to np.exp()

    """
    if _USE_NUMEXPR:
        return ne.evaluate("exp(x)", optimization='moderate', )
    else:
        return np.exp(x)


def _fftshift(x):
    N = x.shape[0]
    if _USE_CUDA & (N == x.shape[1]):
        blockdim = (32, 32)  # threads per block
        numBlocks = (int(N / blockdim[0]), int(N / blockdim[1]))
        cufftShift_2D_kernel[numBlocks, blockdim](x.ravel(), N)
        return x
    else:
        return np.fft.fftshift(x)


if _USE_CUDA:
    @cuda.jit()
    def cufftShift_2D_kernel(data, N):
        """
        adopted CUDA FFT shift code from:
        https://github.com/marwan-abdellah/cufftShift
        (GNU Lesser Public License)
        """

        # // 2D Slice & 1D Line
        sLine = N
        sSlice = N * N
        # // Transformations Equations
        sEq1 = int((sSlice + sLine) / 2)
        sEq2 = int((sSlice - sLine) / 2)
        x, y = cuda.grid(2)
        # // Thread Index Converted into 1D Index
        index = (y * N) + x

        if x < N / 2:
            if y < N / 2:
                # // First Quad
                temp = data[index]
                data[index] = data[index + sEq1]
                # // Third Quad
                data[index + sEq1] = temp
        else:
            if y < N / 2:
                # // Second Quad
                temp = data[index]
                data[index] = data[index + sEq2]
                data[index + sEq2] = temp
