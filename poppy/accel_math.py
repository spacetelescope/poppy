# accel_math.py
#
# Array backend selection (numpy or cupy, via the Python array API standard) and related
# accelerated computations using FFTW, MKL, numexpr, and related.
#
# Conventions for array code throughout poppy:
#   * All computation on wavefront/optic arrays uses ``xp`` (the array-API namespace for the backend
#     selected at import time), ``_float()``/``_complex()`` for dtypes, and no in-place mutation
#     where a functional expression (e.g. ``xp.where``, ``xp.stack``) will do.
#   * Host-side work (FITS I/O, matplotlib, astropy Quantities, scalar and shape math, scipy routines
#     without array-API support) uses numpy. Convert at that boundary with ``to_numpy()``, and bring
#     external data in with ``xp.asarray()``. Never test for a specific backend to do this.
#   * Pass dtypes to ``xp`` constructors by keyword (``xp.asarray(x, dtype=...)``), as the standard requires.
#
import logging
import multiprocessing
import time

import array_api_compat
import array_api_compat.numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import conf

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
    # try to import cupy packages to see if they are is available
    # and check if GPU hardware is available
    import cupy as cp
    import cupyx.scipy.ndimage
    import cupyx.scipy.signal
    import cupyx.scipy.special
    cp.cuda.Device() # checks if a GPU exists
    _CUPY_PLANS = {}
    _CUPY_AVAILABLE = True
except:
    cp = None
    _CUPY_AVAILABLE = False    

_SUPPORTED_BACKENDS = ('numpy', 'cupy')


def _resolve_backend():
    """Determine which array backend to use, from ``conf.array_backend``.

    The backend is chosen once, at import time. 'auto' preserves the historical behavior:
    use CuPy if ``conf.use_cupy`` is set and a GPU is available, otherwise numpy.
    """
    requested = str(conf.array_backend).lower()
    if requested == 'auto':
        return 'cupy' if (conf.use_cupy and _CUPY_AVAILABLE) else 'numpy'
    if requested not in _SUPPORTED_BACKENDS:
        raise NotImplementedError(f"conf.array_backend={requested!r} is not supported; "
                                  f"choose one of {_SUPPORTED_BACKENDS + ('auto',)}")
    if requested == 'cupy' and not _CUPY_AVAILABLE:
        raise ImportError("conf.array_backend='cupy' was requested but CuPy (with a GPU) is not available")
    return requested


_BACKEND = _resolve_backend()
_USE_CUPY = (_BACKEND == 'cupy')

# Array-API compatible namespace for the selected backend. Fixed for the life of the process.
if _USE_CUPY:
    import array_api_compat.cupy  # requires cupy, so only import if selected
    xp = array_api_compat.cupy
    _scipy = cupyx.scipy
else:
    xp = array_api_compat.numpy
    _scipy = scipy


def update_math_settings():
    """Update the numpy-backend accelerator flags (numexpr, FFTW, MKL) from the current ``conf`` values.

    The array backend itself (``xp``) is selected once at import time from ``conf.array_backend``
    and cannot be swapped at runtime; a warning is logged if the configuration has changed since import.
    """
    global _USE_NUMEXPR, _USE_FFTW, _USE_MKL

    if _resolve_backend() != _BACKEND:
        _log.warning(f"conf.array_backend changed after import; poppy continues to use '{_BACKEND}'. "
                     "Set the backend in the config file before importing poppy.")

    # These accelerate the numpy path only; they provide no benefit on other array backends
    _USE_NUMEXPR = (conf.use_numexpr and _NUMEXPR_AVAILABLE and not _USE_CUPY)
    _USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE and not _USE_CUPY)
    _USE_MKL = (conf.use_mkl and _MKLFFT_AVAILABLE and not _USE_CUPY)


# Call update_math_settings once on initial import
# This ensures an initial setup is done prior to any code execution
update_math_settings()


def _float():
    """ Returns array data type for desired precision based on configuration """
    return xp.float64 if conf.double_precision else xp.float32


def _complex():
    """ Returns array data type for desired precision based on configuration """
    return xp.complex128 if conf.double_precision else xp.complex64


def _r(x, y):
    """ Function to speed up computing the radius given x and y, using Numexpr if available
    Otherwise defaults to numpy. """
    if _USE_NUMEXPR:
        return ne.evaluate("sqrt(x**2+y**2)")
    else:
        return xp.sqrt(x ** 2 + y ** 2)


def _exp(x):
    """
    Function to speed up taking exponential of an array if NumExpr is available.
    Otherwise defaults to xp.exp()

    """
    if _USE_NUMEXPR:
        return ne.evaluate("exp(x)", optimization='moderate', )
    else:
        return xp.exp(x)

def _fftshift(x):
    """ FFT shifts of array contents, using CUDA/CuPY if available.
    Otherwise defaults to numpy.

    Note - This used to be an interesting function with a clever CUDA implementation
             of fftshift, which has since been obsoleted by cupy's implementation.
             This wrapper function could be refactored out in the future

    See also ifftshift
    """
    return xp.fft.fftshift(x)

def _ifftshift(x):
    """ Inverse FFT shifts of array contents, using CUDA if available.
    Otherwise defaults to numpy.
    Note, ifftshift and fftshift are identical for even-length x,
    the functions differ by one sample for odd-length x. This function
    implicitly assumes that if using CUDA, the array size must be even,
    so we can use the same algorithm as fftshift.

    Note - This used to be an interesting function with a clever CUDA implementation
             of fftshift, which has since been obsoleted by cupy's implementation.
             This wrapper function could be refactored out in the future

    See also fftshift
    """

    return xp.fft.ifftshift(x)




def fft_2d(wavefront, forward=True, normalization=None, fftshift=True):
    """ main entry point for FFTs, used in Wavefront._propagate_fft and
    elsewhere. This will invoke one of the following, depending on the array backend and availability:
        - CuPy FFT on NVidia GPU (if the cupy backend is selected)
        - MKL or FFTW on CPU (numpy backend only, if enabled and installed)
        - the array backend's standard ``fft`` module (e.g. numpy.fft)

    This function handles ONLY the core numerics itself, as fast as possible,
    (and some minor related logging) .
    All the interaction with object state for Wavefront arrays should happen elsewhere.

    Always use the returned array; the input array may be overwritten by some FFT implementations.

    Parameters
    ----------
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
    t0 = time.time()

    if normalization is None:
        normalization = 1./wavefront.shape[0] if forward else wavefront.shape[0]

    # This annoyingly complicated if/elif is just for the debug print statement
    if _USE_CUPY:
        method = 'cupy (GPU)'
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
    if _USE_MKL:
        # Intel MKL is a drop-in replacement for numpy fft but much faster
        do_fft = mkl_fft.fft2 if forward else mkl_fft.ifft2
        wavefront = do_fft(wavefront)

    elif _USE_FFTW:
        FFT_direction = 'forward' if forward else 'backward' # back compatible for use in _FFTW_INIT
        do_fft = pyfftw.interfaces.numpy_fft.fft2 if forward else pyfftw.interfaces.numpy_fft.ifft2

        if (wavefront.shape, FFT_direction) not in _FFTW_INIT:
            # The first time you run FFTW to transform a given size, it does a speed test to
            # determine optimal algorithm that is destructive to your chosen array.
            # So only do that test on a copy, not the real array:
            _log.info(f"Measuring pyfftw optimal plan for {str(wavefront.shape)}, direction={FFT_direction}")

            pyfftw.interfaces.cache.enable()
            pyfftw.interfaces.cache.set_keepalive_time(30)

            test_array = np.zeros(wavefront.shape)
            test_array = do_fft(test_array, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                threads=multiprocessing.cpu_count())

            _FFTW_INIT[(wavefront.shape, FFT_direction)] = True

        wavefront = do_fft(wavefront, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                threads=multiprocessing.cpu_count())
    else:  # Standard array-API FFT, for numpy, cupy, or any other backend
        do_fft = xp.fft.fft2 if forward else xp.fft.ifft2
        wavefront = do_fft(wavefront)
    t2 = time.time()

    if (not forward) and fftshift:  # shift after forwards propagations (using inverse FFT)
        # This needs to be a fftshift here, since we use ifftshift above
        # See comment above in this function.
        wavefront = _fftshift(wavefront)

    wavefront = wavefront * normalization
    t3 = time.time()
    _log.debug(f"    FFT_2D: FFT in {t2-t1:3f} s, full function  in {t3-t0:.3f} s")

    return wavefront



def ispowerof2(num):
    """ Is this number a power of 2?"""
    # see http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
    return (num & (num-1) == 0)



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
            setup=f"""
import poppy
import numpy as np
tmp = np.asarray(np.random.rand({npix},{npix}), np.{complextype})
            """)
    print(f"Timing performance of FFT for {npix} x {npix}, {complextype}, with {iterations} iterations")

    defaults = (poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr, poppy.conf.double_precision)
    poppy.conf.double_precision = double_precision

    # Time baseline performance in numpy
    print("Timing performance in plain numpy:")

    poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr = (False, False, False)
    update_math_settings()
    time_numpy = timer.timeit(number=iterations) / iterations
    print(f"  {time_numpy:.3f} s")

    if poppy.accel_math._FFTW_AVAILABLE:
        print("Timing performance with FFTW:")
        poppy.conf.use_fftw = True
        update_math_settings()
        time_fftw = timer.timeit(number=iterations) / iterations
        print(f"  {time_fftw:.3f} s")
    else:
        time_fftw = np.nan

    if poppy.accel_math._NUMEXPR_AVAILABLE:
        print("Timing performance with Numexpr + FFTW:")
        poppy.conf.use_numexpr = True
        update_math_settings()
        time_numexpr = timer.timeit(number=iterations) / iterations
        print(f"  {time_numexpr:.3f} s")
    else:
        time_numexpr = np.nan

    if poppy.accel_math._MKLFFT_AVAILABLE:
        print("Timing performance with Numexpr + MKL:")
        poppy.conf.use_numexpr = True
        poppy.conf.use_fftw = False
        poppy.conf.use_mkl = True
        update_math_settings()
        time_mkl = timer.timeit(number=iterations) / iterations
        print(f"  {time_mkl:.3f} s")
    else:
        time_mkl = np.nan

    update_math_settings()  # restore flags to the configured values below
    poppy.conf.use_mkl, poppy.conf.use_fftw, poppy.conf.use_numexpr, poppy.conf.double_precision = defaults
    update_math_settings()

    return {'numpy': time_numpy,
            'fftw': time_fftw,
            'numexpr': time_numexpr,
            'mkl': time_mkl}


def get_processor_name():
    """Return processor hardware name, in a cross-platform manner

    With thanks to https://stackoverflow.com/a/20161999
    """
    import os
    import platform
    import re
    import subprocess

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
    import os
    import platform
    import subprocess

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
            # explicitly try multithreaded here
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
        raise ValueError(f"Unknown/invalid value for 'mode' parameter: {mode}")

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
    import functools

    from simple_benchmark import benchmark

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
        plt.savefig("bench_mfts.png")


def is_on_gpu(array):
    """Simple utility function to check if an array is on the GPU
    (only possible if using CuPY currently)
    """
    return bool(_USE_CUPY and array_api_compat.is_cupy_array(array))


def to_numpy(array):
    """Convert an array from the active array backend to a plain numpy array, for use in
    host-side code such as FITS output, matplotlib, and astropy.

    Numpy arrays, scalars, and other non-array objects are returned unchanged. This is the
    sanctioned boundary between backend arrays (``xp``) and numpy.
    """
    if array_api_compat.is_numpy_array(array):
        return array
    if array_api_compat.is_cupy_array(array):
        return array.get()
    if array_api_compat.is_array_api_obj(array):
        return np.asarray(array)
    return array
