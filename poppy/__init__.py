# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Physical Optics Propagation in PYthon (POPPY)


POPPY is a Python package that simulates physical optical propagation including diffraction. 
It implements a flexible framework for modeling Fraunhofer (far-field) diffraction
and point spread function formation, particularly in the context of astronomical telescopes.
POPPY was developed as part of a simulation package for JWST, but is more broadly applicable to many kinds of 
imaging simulations. 

Developed by Marshall Perrin at STScI, 2010-2014, for use simulating the James Webb Space Telescope. 

Documentation can be found online at http://www.stsci.edu/~mperrin/software/poppy/


"""

# ----------------------------------------------------------------------------
# make use of astropy affiliate framework to set __version__, __githash__, and 
# add the test() helper function
from ._astropy_init import *
# ----------------------------------------------------------------------------

import astropy as _astropy
if _astropy.version.major + _astropy.version.minor*0.1 < 0.4:
    raise ImportError("astropy >= 0.4 is required for this version of poppy.")

from . import poppy_core
from . import utils
from . import optics

from .poppy_core import *
from .utils import * 
from .optics import *

 #(display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
 #   measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl, measure_anisotropy,
 #   specFromSpectralType, rebin_array)

from .instrument import Instrument

from .config import conf

# Not yet implemented:
#from .wfe import ZernikeWFE, PowerSpectralDensityWFE, KolmogorovWFE



if conf.autosave_fftw_wisdom: # if we might have autosaved, then auto reload as well
   # the following will just return if FFTW is not present
   utils.fftw_load_wisdom()


__all__ = ['conf','Instrument'] +  utils.__all__ + poppy_core.__all__


