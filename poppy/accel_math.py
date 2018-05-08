# accel_math.py
#
# Various functions related to accelerated computations using FFTW, CUDA, numexpr, and related.
#
import numpy as np
import multiprocessing
from . import conf

import logging
_log = logging.getLogger('poppy')



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


try:
    # try to import pyopencl and gpyfft to see if OpenCL FFT is available
    import pyopencl
    import pyopencl.array
    import gpyfft
    _OPENCL_AVAILABLE = True
    _OPENCL_STATE = dict()
except ImportError:
    _OPENCL_AVAILABLE = False

_USE_CUDA = (conf.use_cuda and _ACCELERATE_AVAILABLE)
_USE_OPENCL = (conf.use_numexpr and _OPENCL_AVAILABLE)
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


def _r(x,y):
    """ Function to speed up computing the radius given x and y, using Numexpr if available
    Otherwise defaults to numpy. """
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
        return  ne.evaluate("exp(x)", optimization='moderate',)
    else:
        return np.exp(x)

def _fftshift(x):
    """ FFT shifts of array contents, using CUDA if available.
    Otherwise defaults to numpy.

    Note - TODO write an OpenCL version
    """

    N=x.shape[0]
    if (_USE_CUDA) & (N==x.shape[1]):
        blockdim = (32, 32) # threads per block
        numBlocks = (int(N/blockdim[0]),int(N/blockdim[1]))
        cufftShift_2D_kernel[numBlocks, blockdim](x.ravel(),N)
        return x
    else:
        return np.fft.fftshift(x)



def _fft_2d(wavefront, FFT_direction, normalization):
    """ main entry point for FFTs, used in Wavefront._propagate_fft

    This function handles ONLY the core numerics itself, as fast as possible,
    (and some minor related logging) .
    All the interaction with object state for Wavefront arrays should happen elsewhere.

    """
    # To use FFTW, it must both be enabled and the library itself has to be present
    _USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE)
    _USE_OPENCL = (conf.use_numexpr and _OPENCL_AVAILABLE)

    # OpenCL only can FFT certain array sizes. 
    # This check is more stringent that necessary - opencl can handle powers of a few small integers
    # but this simple version helps during development
    if _USE_OPENCL and not ispowerof2(wavefront.shape[0]):
        _log.debug("Wavefront size {} not supported by OpenCL, therefore disabling USE_OPENCL for this calculation.".format(wavefront.shape))
        _USE_OPENCL = False

    # Setup for FFT
    if _USE_OPENCL:
        method = 'pyopencl'
    elif _USE_FFTW:
        method = 'pyfftw'
        do_fft = pyfftw.interfaces.numpy_fft.fft2 if FFT_direction=='forward' else pyfftw.interfaces.numpy_fft.ifft2
    else:
        method = 'numpy'
        do_fft =  np.fft.fft2 if FFT_direction=='forward' else np.fft.ifft2
    _log.debug("using {2} FFT of {0} array, FFT_direction={1}".format(str(wavefront.shape), FFT_direction, method))

    if FFT_direction =='backward': wavefront = np.fft.ifftshift(wavefront)

    if _USE_OPENCL:
        context, queue = get_opencl_context()
        wf_on_gpu = pyopencl.array.to_device(queue, wavefront)
        transform = gpyfft.fft.FFT(context, queue, wf_on_gpu, axes=(0,1))
        event, = transform.enqueue()
        event.wait()
        wavefront = wf_on_gpu.get()

        if FFT_direction =='backward':
            # I can't figure out how to tell gpyfft to do a backward FFT.  ?!?!
            # so instead we have to mess with the normalization manually here. 
            normalization = 1./normalization #

            # and we seem to need to swap in order to get the parity right? 
            wavefront = wavefront[::-1, ::-1]
        _log.debug("Normalization: {}".format(normalization))

    elif _USE_FFTW:
        if (wavefront.shape, FFT_direction) not in _FFTW_INIT:
            # The first time you run FFTW to transform a given size, it does a speed test to
            # determine optimal algorithm that is destructive to your chosen array.
            # So only do that test on a copy, not the real array:
            _log.info("Measuring pyfftw optimal plan for %s, direction=%s" % (
                str(wavefront.shape), FFT_direction))

            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30)

            test_array = np.zeros(wavefront.shape)
            test_array = do_fft(test_array, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                threads=multiprocessing.cpu_count())

            _FFTW_INIT[(wavefront.shape, FFT_direction)] = True

        wavefront = do_fft(wavefront, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                threads=multiprocessing.cpu_count())
    else:
        wavefront = do_fft(wavefront)

    if FFT_direction == 'forward':
        wavefront = np.fft.fftshift(wavefront)

    wavefront *= normalization

    return wavefront



def ispowerof2(num):
    """ Is this number a power of 2?"""
    # see http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    return (num & (num-1) == 0)

if _OPENCL_AVAILABLE:
    def get_opencl_context():
        """ Create, save, and retrieve OpenCL handles to the GPU """
        if len(_OPENCL_STATE) == 0:
            platforms = pyopencl.get_platforms()
            if len(platforms) == 1:
                _OPENCL_STATE['platform'] = platforms[0]
            else:
                raise RuntimeError("OpenCL code needs update for multiple platforms")
            gpus = _OPENCL_STATE['platform'].get_devices(device_type=pyopencl.device_type.GPU)
            if len(gpus) == 1:
                device = gpus[0]
                _OPENCL_STATE['device'] = device
            else:
                raise RuntimeError("OpenCL code could not uniquely identify which device to use as GPU")
            context = pyopencl.Context(devices=[device])
            queue = pyopencl.CommandQueue(context)

            _OPENCL_STATE['context'] = context
            _OPENCL_STATE['queue'] = queue
        return (_OPENCL_STATE['context'], _OPENCL_STATE['queue'])





if  _USE_CUDA:
    @cuda.jit()
    def cufftShift_2D_kernel(data, N):
        '''
        adopted CUDA FFT shift code from:
        https://github.com/marwan-abdellah/cufftShift
        (GNU Lesser Public License)
        '''

        #// 2D Slice & 1D Line
        sLine = N
        sSlice = N * N
        #// Transformations Equations
        sEq1 = int((sSlice + sLine) / 2)
        sEq2 = int((sSlice - sLine) / 2)
        x, y = cuda.grid(2)
        #// Thread Index Converted into 1D Index
        index = (y * N) + x

        if (x < N / 2):
            if (y < N / 2):
                #// First Quad
                temp = data[index]
                data[index] = data[index + sEq1]
            #// Third Quad
                data[index + sEq1] = temp
        else:
            if (y < N / 2):
                #// Second Quad
                temp=data[index]
                data[index] = data[index + sEq2]
                data[index + sEq2] = temp


