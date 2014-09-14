import poppy
#---- begin top of poppy_core.py
from __future__ import division
import multiprocessing
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage.interpolation
import matplotlib
import time


from matplotlib.colors import LogNorm  # for log scaling of images, with automatic colorbar support

import astropy.io.fits as fits



#from .utils import imshow_with_mouseover, estimate_optimal_nprocesses, fftw_load_wisdom, fftw_save_
from .matrixDFT import MatrixFourierTransform
from . import utils
from . import settings


import logging
_log = logging.getLogger('poppy')

try:
    from IPython.core.debugger import Tracer; stop = Tracer()
except:
    pass

# Setup infrastructure for FFTW
_FFTW_INIT = {}  # dict of array sizes for which we have already performed the required FFTW planning step
_FFTW_FLAGS = ['measure']
if settings.use_fftw():
    try:
        # try to import FFTW and use it
        import pyfftw
    except:
        # we tried but failed to import it. 
        settings.use_fftw.set(False)

# internal constants for types of plane
_PUPIL = 1
_IMAGE = 2
_DETECTOR = 3 # specialized type of image plane.
_ROTATION = 4 # not a real optic, just a coordinate transform
_typestrs = ['', 'Pupil plane', 'Image plane', 'Detector', 'Rotation']


#conversions
_RADIANStoARCSEC = 180.*60*60 / np.pi

#---end top of poppy_core.py

class Fresnel_Wavefront(poppy.wavefront):
    def __init__(self):
        self.z0=
        self.w0=
    '''
    A wavefront class that automatically selects the correct diffraction method from one plane to the next.
    '''
    def propogate(self,dz,optic):

        
