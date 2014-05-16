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


from .poppy_core import (Wavefront, OpticalElement, FITSOpticalElement, Rotation, AnalyticOpticalElement, 
	ScalarTransmission, InverseTransmission, ThinLens, BandLimitedCoron, 
	FQPM_FFT_aligner, IdealFQPM, IdealFieldStop, IdealRectangularFieldStop, IdealCircularOcculter, 
	IdealBarOcculter, ParityTestAperture, CircularAperture, HexagonAperture, 
	MultiHexagonAperture, NgonAperture, SquareAperture, RectangleAperture, SecondaryObscuration, 
    AsymmetricSecondaryObscuration, CompoundAnalyticOptic, 
	Detector, OpticalSystem, SemiAnalyticCoronagraph)

from .utils import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
    measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl, measure_anisotropy,
    specFromSpectralType, rebin_array)

from .instrument import Instrument

try:
    # if we have astropy >=0.4
    from config import conf
except:
    # if we have astropy 0.3
    import conf

# Not yet implemented:
#from .wfe import ZernikeWFE, PowerSpectralDensityWFE, KolmogorovWFE



if conf.autosave_fftw_wisdom(): # if we might have autosaved, then auto reload as well
   # the following will just return if FFTW is not present
   utils.fftw_load_wisdom()


#def test( verbose=False ) :
#    #
#    import os, pytest
#
#    # find the directory where the test package lives
#    from . import tests
#    dir = os.path.dirname( tests.__file__ )
#
#    # assemble the py.test args
#    args = [ dir ]
#
#    # run py.test
#    try :
#        return pytest.main( args )
#    except SystemExit as e :
#        return e.code
#
