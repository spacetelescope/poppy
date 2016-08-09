# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Physical Optics Propagation in PYthon (POPPY)


POPPY is a Python package that simulates physical optical propagation including diffraction.
It implements a flexible framework for modeling Fraunhofer (far-field) diffraction
and point spread function formation, particularly in the context of astronomical telescopes.
POPPY was developed as part of a simulation package for JWST, but is more broadly applicable to many kinds of
imaging simulations.

Developed by Marshall Perrin at STScI, 2010-2014, for use simulating the James Webb Space Telescope.

Documentation can be found online at https://pythonhosted.org/poppy/

This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
# make use of astropy affiliate framework to set __version__, __githash__, and
# add the test() helper function
from ._astropy_init import *
# ----------------------------------------------------------------------------


import astropy as _astropy
if _astropy.version.major + _astropy.version.minor*0.1 < 0.4:  # pragma: no cover
    raise ImportError("astropy >= 0.4 is required for this version of poppy.")

from astropy import config as _config


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
    autosave_fftw_wisdom= _config.ConfigItem(True, 'Should POPPY ' +
            'automatically save and reload FFTW ' +
            '"wisdom" for improved speed?')

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
from . import wfe

from .poppy_core import *
from .utils import *
from .optics import *
from .wfe import *
from .fresnel import *

from .instrument import Instrument

# if we might have autosaved, then auto reload as well
if conf.use_fftw and conf.autosave_fftw_wisdom:
    try:
        import pyfftw
        utils.fftw_load_wisdom()
    except ImportError:
        pyfftw = None

__all__ = ['conf', 'Instrument'] + utils.__all__ + poppy_core.__all__ + optics.__all__ + fresnel.__all__ + wfe.__all__
