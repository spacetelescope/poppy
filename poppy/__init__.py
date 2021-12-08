# Licensed under a 3-clause BSD style license - see LICENSE.md

"""Physical Optics Propagation in PYthon (POPPY)

POPPY is a Python package that simulates physical optical propagation including diffraction.
It implements a flexible framework for modeling Fraunhofer (far-field) diffraction
and point spread function formation, particularly in the context of astronomical telescopes.
POPPY was developed as part of a simulation package for JWST, but is more broadly applicable to many kinds of
imaging simulations.

Developed by Marshall Perrin and colleagues at STScI, for use simulating the James Webb Space Telescope
and other NASA missions.

Documentation can be found online at https://poppy-optics.readthedocs.io/
"""
# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys
from astropy import config as _config

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''

__minimum_python_version__ = "3.7"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("poppy does not support Python < {}".format(__minimum_python_version__))


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `poppy`.
    """

    use_multiprocessing = _config.ConfigItem(False,
                                             'Should PSF calculations run in parallel using multiple processors'
                                             'using the Python multiprocessing framework (if True; faster but '
                                             'does not allow display of each wavelength) or run serially in a '
                                             'single process (if False; slower but shows the calculation in '
                                             'progress. Also a bit more robust.)')

    # Caution: Do not make this next too large on high-CPU-count machines
    # because this is a memory-intensive calculation and you will
    # just end up thrashing IO and swapping out a ton, so everything
    # becomes super slow.
    n_processes = _config.ConfigItem(4, 'Maximum number of additional ' +
                                     'worker processes to spawn, if multiprocessing is enabled. ' +
                                     'Set to 0 for autoselect. Note, PSF calculations are likely RAM ' +
                                     'limited more than CPU limited for higher N on modern machines.')

    use_fftw = _config.ConfigItem(True, 'Use FFTW for FFTs (assuming it' +
                                  'is available)?  Set to False to force numpy.fft always, True to' +
                                  'try importing and using FFTW via PyFFTW.')
    autosave_fftw_wisdom = _config.ConfigItem(True, 'Should POPPY ' +
                                              'automatically save and reload FFTW ' +
                                              '"wisdom" for improved speed?')
    use_mkl = _config.ConfigItem(True, "Use Intel MKL for FFTs (assuming it is available). "
                                       "This has highest priority for CPU-based FFT over other FFT options, if multiple are set True.")

    use_cuda = _config.ConfigItem(True, 'Use cuda for FFTs on GPU (assuming it' +
            'is available)?')
    use_opencl = _config.ConfigItem(True, 'Use OpenCL for FFTs on GPU (assuming it' +
            'is available)?')
    use_numexpr = _config.ConfigItem(True, 'Use NumExpr to accelarate array math (assuming it' +
            'is available)?')

    double_precision = _config.ConfigItem(True, 'Floating point values use float64 and complex128 if True,' +
            'otherwise float32 and complex64.')

    default_image_display_fov = _config.ConfigItem(5.0, 'Default image' +
                                                   'display field of view, in arcseconds. Adjust this to display ' +
                                                   'only a subregion of a larger output array.')

    default_logging_level = _config.ConfigItem('INFO', 'Logging ' +
                                               'verbosity: one of {DEBUG, INFO, WARN, ERROR, or CRITICAL}')

    enable_speed_tests = _config.ConfigItem(False, 'Enable additional ' +
                                            'verbose printout of computation times. Useful for benchmarking.')
    enable_flux_tests = _config.ConfigItem(False, 'Enable additional ' +
                                           'verbose printout of fluxes and flux conservation during ' +
                                           'calculations. Useful for testing.')
    cmap_sequential = _config.ConfigItem(
        'gist_heat',
        'Select a default colormap to represent sequential data (e.g. intensity)'
    )
    cmap_diverging = _config.ConfigItem(
        'RdBu_r',
        'Select a default colormap to represent diverging data (e.g. OPD)'
    )
    cmap_pupil_intensity = _config.ConfigItem(
        'gray',
        'Select a default colormap to represent intensity at pupils or aperture masks'
    )


conf = Conf()

from . import poppy_core
from . import utils
from . import optics
from . import misc
from . import fresnel
from . import physical_wavefront
from . import wfe
from . import dms
from . import active_optics

from .poppy_core import *
from .utils import *
from .optics import *
from .wfe import *
from .fresnel import *
from .physical_wavefront import *
from .special_prop import *
from .dms import *
from .active_optics import *

from .instrument import Instrument

# if we might have autosaved, then auto reload as well
#if accel_math._FFTW_AVAILABLE:
#    utils.fftw_load_wisdom()

__all__ = ['conf', 'Instrument', '__version__'] + utils.__all__ + poppy_core.__all__ + optics.__all__ + \
          fresnel.__all__ + wfe.__all__ + dms.__all__ + active_optics.__all__
