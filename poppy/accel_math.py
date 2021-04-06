# accel_math.py
#
# Various functions related to accelerated computations using FFTW, CUDA, numexpr, and related.
#
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from . import conf

import time
import logging
_log = logging.getLogger('poppy')


try:
    # try to import FFTW to see if it is available
    import pyfftw
    # Setup infrastructure for FFTW
    _FFTW_INIT = {}  # dict of array sizes for which we have already performed the required FFTW planning step
    _FFTW_FLAGS = ['measure']
    _FFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    _FFTW_AVAILABLE = False

try:
    # try to import numexpr package to see if it is available
    import numexpr as ne

    _NUMEXPR_AVAILABLE = True
except ImportError:
    ne = None
    _NUMEXPR_AVAILABLE = False


try:
    import mkl_fft
    _MKLFFT_AVAILABLE = True
except ImportError:
    _MKLFFT_AVAILABLE = False


try:
    # try to import CUDA packages to see if they are available
    import pyculib
    from numba import cuda
    _CUDA_PLANS = {} # plans for various array sizes already prepared
    _CUDA_AVAILABLE = True
except ImportError:
    pyculib = None
    _CUDA_AVAILABLE = False

try:
    # try to import OpenCL packages to see if they are is available
    import pyopencl
    import pyopencl.array
    import gpyfft
    _OPENCL_AVAILABLE = True
    _OPENCL_STATE = dict()
except ImportError:
    _OPENCL_AVAILABLE = False

_USE_CUDA = (conf.use_cuda and _CUDA_AVAILABLE)
_USE_OPENCL = (conf.use_opencl and _OPENCL_AVAILABLE)
_USE_NUMEXPR = (conf.use_numexpr and _NUMEXPR_AVAILABLE)
_USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE)
_USE_MKL = (conf.use_mkl and _MKLFFT_AVAILABLE)


def update_math_settings():
    """ Update the module-level math flags, based on user settings
    """
    global _USE_CUDA, _USE_OPENCL, _USE_NUMEXPR, _USE_FFTW, _USE_MKL
    _USE_CUDA = (conf.use_cuda and _CUDA_AVAILABLE)
    _USE_OPENCL = (conf.use_opencl and _OPENCL_AVAILABLE)
    _USE_NUMEXPR = (conf.use_numexpr and _NUMEXPR_AVAILABLE)
    _USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE)
    _USE_MKL = (conf.use_mkl and _MKLFFT_AVAILABLE)


def _float():
    """ Returns numpy data type for desired precision based on configuration """
    # How many bits per float to use?
    return np.float64 if conf.double_precision else np.float32


def _complex():
    """ Returns numpy data type for desired precision based on configuration """
    # How many bits per complex float to use?
    return np.complex128 if conf.double_precision else np.complex64


def _r(x, y):
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
        return ne.evaluate("exp(x)", optimization='moderate', )
    else:
        return np.exp(x)

def _fftshift(x):
    """ FFT shifts of array contents, using CUDA if available.
    Otherwise defaults to numpy.

    Note - TODO write an OpenCL version

    See also ifftshift
    """

    N=x.shape[0]
    # the CUDA fftshift is set up to work on blocks of 32, so
    # N must be a multiple of 32. We check this rapidly using a bit mask:
    #    (x & 31)==0  is a ~20x faster equivalent of (np.mod(x,32)==0)
    if (_USE_CUDA) & (N==x.shape[1]) & ((N & 31)==0):
        blockdim = (32, 32) # threads per block
        numBlocks = (int(N/blockdim[0]),int(N/blockdim[1]))
        cufftShift_2D_kernel[numBlocks, blockdim](x.ravel(),N)
        return x
    else:
        return np.fft.fftshift(x)

def _ifftshift(x):
    """ Inverse FFT shifts of array contents, using CUDA if available.
    Otherwise defaults to numpy.
    Note, ifftshift and fftshift are identical for even-length x,
    the functions differ by one sample for odd-length x. This function
    implictly assumes that if using CUDA, the array size must be even,
    so we can use the same algorithm as fftshift.

    Note - TODO write an OpenCL version

    See also fftshift
    """

    N=x.shape[0]
    # the CUDA fftshift is set up to work on blocks of 32, so
    # N must be a multiple of 32. We check this rapidly using a bit mask:
    #   not (x & 31)  is a ~20x faster equivalent of (np.mod(x,32)==0)
    if (_USE_CUDA) & (N==x.shape[1]) & ((N & 31)==0):
        blockdim = (32, 32) # threads per block
        numBlocks = (int(N/blockdim[0]),int(N/blockdim[1]))
        cufftShift_2D_kernel[numBlocks, blockdim](x.ravel(),N)
        return x
    else:
        return np.fft.ifftshift(x)




def fft_2d(wavefront, forward=True, normalization=None, fftshift=True):
    """ main entry point for FFTs, used in Wavefront._propagate_fft and
    elsewhere. This will invoke one of the following, depending on availability:
        - CUDA on NVidia GPU
        - OpenCL on AMD GPU
        - FFTW on CPU
        - numpy on CPU

    This function handles ONLY the core numerics itself, as fast as possible,
    (and some minor related logging) .
    All the interaction with object state for Wavefront arrays should happen elsewhere.

    TODO: this should execute an IN PLACE FFT, so we don't have to pass around arrays to return
    anything.

    Parameters
    -----------
    forward : bool
        set to True for forward FFT, False for inverse fft
        SIGN CONVENTION: Note, forward optical propagation requires an inverse FFT, and
        vice versa. This is to have a positive sign in the exponent for forward propagations.
        See sign convention documentation.
    normalization : float, optional
        Normalization factor. Defaults to 1./wavefront.shape[0] for forward,
        and wavefront.shape[0] for inverse. Use this only if you need a non-default
        behavior.
    fftshift : bool
        apply FFT shift after forwards propagation or before inverse propagation?
        SIGN CONVENTION: Note, that means after an inverse FFT or before a forward
        FFT. This differs from the behavior of poppy versions < 1.0.

    """
    ## To use a fast FFT, it must both be enabled and the library itself has to be present
    global _USE_OPENCL, _USE_CUDA # need to declare global in case we need to change it, below
    t0 = time.time()

    # OpenCL cfFFT only can FFT certain array sizes.
    if _USE_OPENCL and not isproductofsmallprimes(wavefront.shape[0]):
        _log.debug(("Wavefront size {} not supported by OpenCL, therefore disabling "+
            "USE_OPENCL for this calculation.").format(wavefront.shape))
        _USE_OPENCL = False

    # This annoyingly complicated if/elif is just for the debug print statement
    if _USE_CUDA:
        method = 'pyculib (CUDA GPU)'
    elif _USE_OPENCL:
        method = 'pyopencl (OpenCL GPU)'
    elif _USE_MKL:
        method = 'mkl_fft'
    elif _USE_FFTW:
        method = 'pyfftw'
    else:
        method = 'numpy'
    _log.debug("using {2} FFT of {0} array, FFT_direction={1}".format(
        str(wavefront.shape), 'forward' if forward else 'backward', method))

    if (forward) and fftshift: # shift before backwards propagations (using forward FFT)
        # This needs to be an ifftshift, for machine-precision equivalence to MFT
        # as tested by test_MFT_FFT_equivalence_in_OpticalSystem; ifftshift and fftshift are
        # precisely equivalent for even-sized arrays but differ by 1 pixel offset for odd-sized
        # arrays. The use of ifftshift here ensures the PSF is correctly centered in the center pixel
        # for odd-sized arrays.
        wavefront = _ifftshift(wavefront)

    t1 = time.time()
    if _USE_CUDA:
        if normalization is None:
            normalization = 1./wavefront.shape[0]  # regardless of direction, for CUDA

        # We need a CUDA FFT plan for each size and shape of FFT.
        # The plans can be cached for reuse, since they cost some
        # 10s of milliseconds to create
        params = (wavefront.shape, wavefront.dtype, wavefront.dtype)
        try:
            cufftplan = _CUDA_PLANS[params]
        except KeyError:
            cufftplan = pyculib.fft.FFTPlan(*params)
            _CUDA_PLANS[params] = cufftplan

        # perform FFT on GPU, and return results in place to same array.
        if forward:
            cufftplan.forward(wavefront, out=wavefront)
        else:
            cufftplan.inverse(wavefront, out=wavefront)

    elif _USE_OPENCL:
        if normalization is None:
            normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]

        context, queue = get_opencl_context()
        wf_on_gpu = pyopencl.array.to_device(queue, wavefront)
        transform = gpyfft.fft.FFT(context, queue, wf_on_gpu, axes=(0,1))
        event, = transform.enqueue(forward=forward)
        event.wait()
        wavefront[:] = wf_on_gpu.get()
        del wf_on_gpu

    elif _USE_MKL:
        # Intel MKL is a drop-in replacement for numpy fft but much faster
        do_fft = mkl_fft.fft2 if forward else mkl_fft.ifft2
        if normalization is None:
            normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]
        wavefront = do_fft(wavefront)

    elif _USE_FFTW:
        FFT_direction = 'forward' if forward else 'backward' # back compatible for use in _FFTW_INIT
        do_fft = pyfftw.interfaces.numpy_fft.fft2 if forward else pyfftw.interfaces.numpy_fft.ifft2
        if normalization is None:
            normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]


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
    else: # Basic numpy FFT
        do_fft =  np.fft.fft2 if forward else np.fft.ifft2
        if normalization is None:
            normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]
        wavefront = do_fft(wavefront)
    t2 = time.time()

    if (not forward) and fftshift:  # shift after forwards propagations (using inverse FFT)
        # This needs to be a fftshift here, since we use ifftshift above
        # See comment above in this function.
        wavefront = _fftshift(wavefront)

    wavefront *= normalization
    t3 = time.time()
    _log.debug("    FFT_2D: FFT in {:3f} s, full function  in {:.3f} s".format(t2-t1, t3-t0))

    return wavefront



def ispowerof2(num):
    """ Is this number a power of 2?"""
    # see http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    return (num & (num-1) == 0)


def isproductofsmallprimes(num):
    """ CLFFT only supports array sizes which are products of primes <= 13;
    Is this integer a product of primes no greater than that?
    """
    tmp = num*1 # make copy
    small_primes = [2,3,5,7,9,11,13]
    for p in small_primes:
        quotient, remainder = divmod(tmp, p)
        while not remainder:
            tmp = quotient
            quotient, remainder = divmod(tmp, p)
        if tmp==1: return True # if we get to 1 we're done
    return False # if we don't get to 1 then it's got higher factors

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
                #raise RuntimeError("OpenCL code could not uniquely identify which device to use as GPU")
                device = gpus[1]
                _log.warning('Caution - hard coded use of gpu #1 if > 1 GPUs present')
            context = pyopencl.Context(devices=[device])
            queue = pyopencl.CommandQueue(context)

            _OPENCL_STATE['context'] = context
            _OPENCL_STATE['queue'] = queue
        return (_OPENCL_STATE['context'], _OPENCL_STATE['queue'])





if  _CUDA_AVAILABLE:
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

# ##################################################################
#
#     Performance benchmarks


def benchmark_fft(npix=2048, iterations=20, double_precision=True):
    """ Performance benchmark function for standard imaging """
    #import poppy
    import timeit
    import poppy


    complextype = 'complex128' if double_precision else 'complex64'

    timer = timeit.Timer("tmp2 = poppy.accel_math.fft_2d(tmp, fftshift=False)",
            setup="""
import poppy
import numpy as np
tmp = np.asarray(np.random.rand({npix},{npix}), np.{complextype})
            """.format(npix=npix, complextype=complextype))
    print("Timing performance of FFT for {npix} x {npix}, {complextype}, with {iterations} iterations".format(
        npix=npix, iterations=iterations, complextype=complextype))

    defaults = (poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr, poppy.conf.use_cuda,
            poppy.conf.use_opencl, poppy.conf.double_precision)
    poppy.conf.double_precision = double_precision

    # Time baseline performance in numpy
    print("Timing performance in plain numpy:")

    poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr, poppy.conf.use_cuda, poppy.conf.use_opencl = (False, False, False, False, False)
    update_math_settings()
    time_numpy = timer.timeit(number=iterations) / iterations
    print("  {:.3f} s".format(time_numpy))

    if poppy.accel_math._FFTW_AVAILABLE:
        print("Timing performance with FFTW:")
        poppy.conf.use_fftw = True
        update_math_settings()
        time_fftw = timer.timeit(number=iterations) / iterations
        print("  {:.3f} s".format(time_fftw))
    else:
        time_fftw = np.NaN

    if poppy.accel_math._NUMEXPR_AVAILABLE:
        print("Timing performance with Numexpr + FFTW:")
        poppy.conf.use_numexpr = True
        update_math_settings()
        time_numexpr = timer.timeit(number=iterations) / iterations
        print("  {:.3f} s".format(time_numexpr))
    else:
        time_numexpr = np.NaN

    if poppy.accel_math._MKLFFT_AVAILABLE:
        print("Timing performance with Numexpr + MKL:")
        poppy.conf.use_numexpr = True
        poppy.conf.use_fftw = False
        poppy.conf.use_mkl = True
        update_math_settings()
        time_mkl = timer.timeit(number=iterations) / iterations
        print("  {:.3f} s".format(time_mkl))
    else:
        time_mkl = np.NaN

    if poppy.accel_math._CUDA_AVAILABLE:
        print("Timing performance with CUDA:")
        poppy.conf.use_cuda = True
        poppy.conf.use_opencl = False
        update_math_settings()
        time_cuda = timer.timeit(number=iterations) / iterations
        print("  {:.3f} s".format(time_cuda))
    else:
        time_cuda = np.NaN

    if poppy.accel_math._OPENCL_AVAILABLE:
        print("Timing performance with OpenCL:")
        poppy.conf.use_opencl = True
        poppy.conf.use_cuda = False
        update_math_settings()
        time_opencl = timer.timeit(number=iterations) / iterations
        print("  {:.3f} s".format(time_opencl))
    else:
        time_opencl = np.NaN


    poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr, poppy.conf.use_cuda,\
            poppy.conf.use_opencl, poppy.conf.double_precision = defaults

    return {'numpy': time_numpy,
            'fftw': time_fftw,
            'numexpr': time_numexpr,
            'mkl': time_mkl,
            'cuda': time_cuda,
            'opencl': time_opencl}


def get_processor_name():
    """Return processor hardware name, in a cross-platform manner

    With thanks to https://stackoverflow.com/a/20161999
    """
    import os, platform, subprocess, re

    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string".split()
        return (subprocess.check_output(command).strip()).decode()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def get_physical_cpu_count():
    """Get physical number of CPU cores

    You might think multiprocessing.cpu_count() would give this, but no.
    Hyperthreading makes that number harder to interpret.

    """
    import os, platform, subprocess, re

    if platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n hw.physicalcpu".split()
        return subprocess.check_output(command).strip().decode()
    else:
        raise NotImplementedError("make this cross platform")

def benchmark_2d_ffts(mode='poppy', max_pow=13, verbose=False, savefig=False):
    """Benchmark FFT runtime vs array size, for multiple transform libraries

    Parameters
    ----------
    mode : string, 'poppy' or 'base'
        What to test, either the full poppy usage of the 2D transform, including
        surrounding setup and normalization code, or the basic transform on its own.
    max_pow : int
        Maximum power of 2 array size to test up to
    verbose : bool
        output numpy config info
    savefig : bool
        save plot result to a PDF?

    """

    # modified based on https://github.com/numpy/numpy/issues/17839#issuecomment-733543850


    try:
        from simple_benchmark import benchmark
    except ImportError:
        raise RuntimeError("You must have simple_benchmark installed to run this function")

    threads = multiprocessing.cpu_count()

    if mode == 'base':
        # Functions for benchmarking the low-level FFT functions in each library
        def pocketfft(it):
            np.fft.fft2(it)

        def scipyfft(it):
            scipy.fft.fft2(it)

        def fftw_1thread(it):
            # note, FFTW defaults to 1 thread, unless you override the config,
            # but that's not a fair comparison
            pyfftw.interfaces.numpy_fft.fft2(it, thread=1)

        def fftw(it):
            # explictly try multithreaded here
            pyfftw.interfaces.numpy_fft.fft2(it, threads=threads)

        def mklfft(it):
            mkl_fft.fft2(it)

        funcs_to_test = [pocketfft, fftw, mklfft, scipyfft, fftw_1thread, ]
        function_aliases = {pocketfft: 'numpy.fft', fftw: 'pyfftw.fft, multithreaded', scipyfft: 'scipy.fft',
                            mklfft: "MKL FFT", fftw_1thread: 'pyfftw.fft, single-thread'}
        title = "Basic 2D FFT only"

    elif mode == 'poppy':
        # Functions for benchmarking poppy transforms using each library

        def poppy_numpyfft(it):
            global _USE_FFTW, _USE_MKL
            _USE_FFTW = False
            _USE_MKL = False

            fft_2d(it, fftshift=False)

        def poppy_fftw(it):
            global _USE_FFTW, _USE_MKL
            _USE_FFTW = True
            _USE_MKL = False

            fft_2d(it, fftshift=False)

        def poppy_mklfft(it):
            global _USE_FFTW, _USE_MKL
            _USE_FFTW = False
            _USE_MKL = True

            fft_2d(it, fftshift=False)

        funcs_to_test = [poppy_numpyfft, poppy_fftw, poppy_mklfft]
        function_aliases = {poppy_numpyfft: 'poppy using numpy.fft', poppy_fftw: 'poppy using pyfftw.fft',
                            poppy_mklfft: "poppy using MKL FFT"}
        title = 'full poppy.accel_math.fft_2d'
    else:
        raise ValueError(f"Unknown/invalid value for 'base' parameter: {base}")

    def shp(len):
        return (len, len)

    if verbose:
        print(np.version)
        np.show_config()

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
    b_array = benchmark(
        funcs_to_test,
        arguments={2 ** i: np.random.uniform(size=shp(2 ** i)) + 1j * np.random.uniform(size=shp(2 ** i)) for i in
                   range(2, max_pow)},
        argument_name='array size',
        function_aliases=function_aliases
    )
    plt.figure(figsize=(12, 8))
    b_array.plot()
    plt.grid(which='both', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.xlim(1, 2e4)
    plt.ylim(1e-6, 1e1)

    cpu_label = get_processor_name() + f", {get_physical_cpu_count()} cores"
    plt.title(f"Time for 2D complex FFTs: {title}\n{cpu_label}", fontweight='bold')
    if savefig:
        plt.savefig(f"bench_ffts_{mode}.png")


def benchmark_2d_mfts(max_pow=13, savefig=False):
    """Benchmark MFT runtime vs array size, for multiple transform libraries

    Parameters
    ----------
    mode : string, 'poppy' or 'base'
        What to test, either the full poppy usage of the 2D transform, including
        surrounding setup and normalization code, or the basic transform on its own.
    max_pow : int
        Maximum power of 2 array size to test up to
    savefig : bool
        save plot result to a PDF?

    """

    # modified based on https://github.com/numpy/numpy/issues/17839#issuecomment-733543850
    from simple_benchmark import benchmark

    import functools
    import poppy

    def shp(len):
        return (len, len)

    def test_mft_numpy(array, npix=64):
        global _USE_NUMEXPR

        _USE_NUMEXPR = False
        poppy.matrixDFT.matrix_dft(array, 16, npix)

    def test_mft_numexpr(array, npix=64):
        global _USE_NUMEXPR
        _USE_NUMEXPR = True

        poppy.matrixDFT.matrix_dft(array, 16, npix)

    test_mft_numpy_512 = functools.partial(test_mft_numpy, npix=512)
    test_mft_numexpr_512 = functools.partial(test_mft_numexpr, npix=512)

    b_array = benchmark(
        [test_mft_numpy, test_mft_numexpr, test_mft_numpy_512, test_mft_numexpr_512],
        arguments={2 ** i: np.random.uniform(size=shp(2 ** i)) + 1j * np.random.uniform(size=shp(2 ** i)) for i in
                   range(2, max_pow)},
        argument_name='pupil array size, npupil',
        function_aliases={test_mft_numpy: "MFT with numpy, npix=64", test_mft_numexpr: "MFT with numexpr, npix=64",
                          test_mft_numpy_512: "MFT with numpy, npix=512",
                          test_mft_numexpr_512: "MFT with numexpr, npix=512"}
    )
    plt.figure(figsize=(12, 8))
    b_array.plot()
    plt.grid(which='both', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.xlim(1, 2e4)
    plt.ylim(1e-6, 1e1)

    cpu_label = get_processor_name() + f", {get_physical_cpu_count()} cores"
    plt.title(f"Matrix Fourier Transform timings\n{cpu_label}", fontweight='bold')

    if savefig:
        plt.savefig(f"bench_mfts.png")