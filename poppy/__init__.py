from .poppy import (Wavefront, OpticalElement, FITSOpticalElement, Rotation, AnalyticOpticalElement, 
	ScalarTransmission, InverseTransmission, ThinLens, BandLimitedCoron, 
	FQPM_FFT_aligner, IdealFQPM, IdealFieldStop, IdealCircularOcculter, 
	IdealBarOcculter, ParityTestAperture, CircularAperture, HexagonAperture, 
	MultiHexagonAperture, NgonAperture, SquareAperture, CompoundAnalyticOptic, 
	Detector, OpticalSystem, SemiAnalyticCoronagraph)

from .poppy import (_USE_FFTW3, _USE_MULTIPROC, _MULTIPROC_NPROCESS, _TIMETESTS, _FLUXCHECK, _IMAGECROP)

from .utils import (display_PSF, display_PSF_difference, display_EE, display_profiles, radial_profile,
        measure_EE, measure_radial, measure_fwhm, measure_sharpness, measure_centroid, measure_strehl,
        measure_anisotropy, specFromSpectralType, rebin_array)



from .instrument import Instrument
from .wfe import ZernikeWFE, PowerSpectralDensityWFE, KolmogorovWFE

from ._version import __version__

__doc__ = """
=============================================
Physical Optics Propagation in PYthon (POPPY)
=============================================


This package implements an object-oriented system for modeling physical optics
propagation with diffraction, particularly for telescopic and coronagraphic
imaging. Right now only Fraunhoffer diffraction is supported, providing image and pupil planes; 
intermediate planes are a future goal.

Classes:
--------
 * Wavefront
 * OpticalElement
   * AnalyticOpticalElement
     * BandLimitedCoron
     * IdealMonoFQPM
     * IdealFieldStop
     * IdealCircularOcculter
   * Detector
 * OpticalSystem


POPPY makes use of python's ``logging`` facility for log messages, using
the logger name "poppy".


    Code by Marshall Perrin <mperrin@stsci.edu>


Module-level configuration constants
------------------------------------

_USE_FFTW3 : bool
    Should the FFTW3 library be used? Set automatically to True if fftw3 is importable, else False.
_USE_MULTIPROC : bool
    Should we use python multiprocessing to spawn multiple simultaneous processes to speed calculations?
    Note that FFTW3 automatically makes use of multiple cores so this should not be used at the same time as
    _USE_FFTW3. Default is False.
_MULTIPROC_NPROCESS : int
    If the above is set, how many processes should be used? Default is 4. Note that you probably don't want to 
    set this too large due to memory limitations; i.e. don't just blindly set this to 16 on a 16-core machine unless
    you have many, many GB of RAM...

_TIMETESTS : bool
    Print out simple benchmarking of elapsed time to screen. Default False
_FLUXCHECK : bool
    Print out total flux after each step of a propagation. Useful for debugging, mostly.

_IMAGECROP : float
    Default zoom in region for image displays. Default is 5.0


"""


