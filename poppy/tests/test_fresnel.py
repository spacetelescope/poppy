from .. import poppy_core  as poppy
from .. import optics 

from .. import fresnel
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

def test_GaussianBeamParams():
    """Confirm that gaussian beam parameters agree with expectations"""
    gw=fresnel.Wavefront(100*u.um,wavelength=830e-9)
    _PUPIL = 1
    gl=fresnel.GaussianLens(50*u.mm,planetype=_PUPIL)
    gw.apply_optic(gl,78.0*u.mm)
    assert(np.round(gw.w_0,9) == np.round(0.0001061989749146441,9))
    assert(np.round(gw.z_w0,9) == np.round(0.15957902236417937,9))
    assert(np.round(gw.z_R,9) == np.round(0.042688650889351865,9))





def test_CircularAperturePropagation(display=False):
    """Confirm that magnitude of central spike from diffraction due to a circular aperture agrees with expectation"""
    npix = 1024#~512 is minimum to accurately recover the central diffraction spike
    gw = fresnel.Wavefront(0.5*u.m,wavelength=2200e-9,npix=npix,oversample=2)
    gw *= optics.CircularAperture(radius=0.5,oversample=gw.oversample)
    
    if display:
        plt.figure()
        gw.display('both',colorbar=True)
        
    gw.propagate_fresnel(5e3*u.m)
    if display:
        
        plt.figure()
        gw.display('both',colorbar=True)
        plt.figure()
        plt.plot(np.arange(gw.intensity.shape[0]), gw.intensity[gw.intensity.shape[1]/2,:])
        plt.title("z=%2i m, compare to Anderson and Enmark fig.6.15"%z)
    assert(np.round(3.3633280006866424,9) == np.round(np.max(gw.intensity)),9)

def test_spherical_lens(display=False):
    """Make sure that spherical lens operator is working"""
    # not yet implemented.
