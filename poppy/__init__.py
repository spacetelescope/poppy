# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Physical Optics Propagation in PYthon (POPPY)

This package implements an object-oriented system for modeling physical optics
propagation with diffraction, particularly for telescopic and coronagraphic
imaging. 

POPPY makes use of python's ``logging`` facility for log messages, using
the logger name "poppy".


"""

try:
    from .version import version as __version__
except ImportError:
    # TODO: Issue a warning using the logging framework
    __version__ = ''
try:
    from .version import githash as __githash__
except ImportError:
    # TODO: Issue a warning using the logging framework
    __githash__ = ''


from .poppy_core import (Wavefront, OpticalElement, FITSOpticalElement, Rotation, AnalyticOpticalElement, 
	ScalarTransmission, InverseTransmission, ThinLens, BandLimitedCoron, 
	FQPM_FFT_aligner, IdealFQPM, IdealFieldStop, IdealRectangularFieldStop, IdealCircularOcculter, 
	IdealBarOcculter, ParityTestAperture, CircularAperture, HexagonAperture, 
	MultiHexagonAperture, NgonAperture, SquareAperture, RectangleAperture, SecondaryObscuration, 
    AsymmetricSecondaryObscuration, CompoundAnalyticOptic, 
	Detector, OpticalSystem, SemiAnalyticCoronagraph)

from .utils import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
    measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl,
    specFromSpectralType, rebin_array)

from .instrument import Instrument
import conf

# Not yet implemented:
#from .wfe import ZernikeWFE, PowerSpectralDensityWFE, KolmogorovWFE



if conf.autosave_fftw_wisdom(): # if we have autosaved, then auto reload as well
   # the following will just return if FFTW is not present
   utils.fftw_load_wisdom()


# Possible in astropy 0.3, but about to be deprecated in 0.4:
def save_config():
    """ Save package configuration variables using the Astropy.config system """
    astropy.config.save_config('poppy')


def test( verbose=False ) :
    #
    import os, pytest

    # find the directory where the test package lives
    from . import tests
    dir = os.path.dirname( tests.__file__ )

    # assemble the py.test args
    args = [ dir ]

    # run py.test
    try :
        return pytest.main( args )
    except SystemExit as e :
        return e.code

