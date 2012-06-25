#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywcs, pyfits
#from IPython.Debugger import Tracer; stop = Tracer()
import logging
_log = logging.getLogger('t')

def airy_1d( aperture=6.5, wavelength=2e-6, length = 512, pixelscale=0.010, 
        obscuration=0.0, center=None, plot_=False):
    """ 1-dimensional Airy function PSF calculator 
    
    Parameters
    ----------
    aperture, wavelength : float
        aperture diam and wavelength in meters
    size : tuple
        array size
    pixelscale : 
        arcseconds

    Returns 
    --------
    r : array
        radius array in arcsec
    airy : array
        Array with the Airy function values, normalized to 1 at peak
    """

    center = (length-1)/2.
    r = np.arange(length)*pixelscale

    RADtoARCSEC = 360.*60*60/np.pi # ~ 206265
    v = np.pi * (r/ RADtoARCSEC) * aperture/wavelength
    e = obscuration
    
    airy =  1./(1-e**2)**2* ((2*scipy.special.jn(1,v) - e*2*scipy.special.jn(1,e*v))/v )**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248

    if plot_:
        plt.semilogy(r, airy)
        plt.xlabel("radius [arcsec]")
        plt.ylabel("PSF intensity")
    return r, airy



def airy_2d( aperture=6.5, wavelength=2e-6, shape=(512,512), pixelscale=0.010, 
        obscuration=0.0, center=None):
    """ 2-dimensional Airy function PSF calculator 
    
    Parameters
    ----------
    aperture, wavelength : float
        aperture diam and wavelength in meters
    size : tuple
        array size
    pixelscale : 
        arcseconds
    """

    if center is None:
        center = (np.asarray(shape)-1.)/2
    y, x = np.indices(shape)
    y -= center[0]
    x -= center[1]
    y *= pixelscale
    x *= pixelscale
    r = np.sqrt(x**2 + y**2)

    RADtoARCSEC = 360.*60*60/np.pi # ~ 206265
    v = np.pi * (r/ RADtoARCSEC) * aperture/wavelength
    e = obscuration

    
    airy =  1./(1-e**2)**2* ((2*scipy.special.jn(1,v) - e*2*beselj(1,e*v))/v )**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248





if __name__ == "__main__":
        
    logging.basicConfig(level=logging.INFO, format='%(name)-12s: %(levelname)-8s %(message)s',)

    pass



