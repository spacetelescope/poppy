import numpy as np
import astropy.units as u

from .. import poppy_core
from .. import physical_wavefront
from .. import fresnel
from .. import optics

wavelength=1e-6

def test_GaussianBeamParams():
    """Confirm that gaussian beam parameters agree with expectations.
    
    This test case has been copied from test_fresnel.py and adopted
    appropriately.
    """
    
    gw = physical_wavefront.PhysicalFresnelWavefront(100*u.um,
                                                     wavelength=830e-9,
                                                     n0=1.0)
    gw.propagate_fresnel(50*u.mm)
    gl = fresnel.QuadraticLens(50*u.mm)
    gw.propagate_fresnel(28*u.mm)
    gw.apply_lens_power(gl, ignore_wavefront=True)
    
    assert(np.round(gw.w_0.value, 9) == np.round(0.0001061989749146441, 9))
    assert(np.round(gw.z_w0.value, 9) == np.round(0.15957902236417937, 9))
    assert(np.round(gw.z_r.value, 9) == np.round(0.042688650889351865, 9))
    # FIXME MP: where do the above values come from?

def test_power():
    """Confirm that the power is scaled correctly."""
    
    wf = physical_wavefront.PhysicalFresnelWavefront(100*u.um, wavelength=830e-9)
    wf *= optics.CircularAperture(radius=50*u.um)
    wf *= fresnel.QuadraticLens(50*u.mm)
    P0 = 10000.0
    wf.scale_power(P0) # Scale its power to 10kW
    assert(np.round(wf.power, 9) == np.round(P0, 9))

def test_radius():
    w0 = 10e-2              # beam radius (m)
    w_extend = 6            # weight of the spatial extend
    wavelength = 1064e-9    # wavelength in vacuum (m)
    npix = 512              # spatial resolution
    M2 = 3.827              # beam quality factor
    z = 150.0               # propagation distance (m)
    R0 = 250.0              # initial radius of curvature (m)
    
    k = 2*np.pi/wavelength # wave number (1/m)
    rad_ana = np.sqrt((2*z*M2/k/w0)**2 + (w0*(1-z/R0))**2) # analytical value for beam radius at z
    
    wf = physical_wavefront.PhysicalFresnelWavefront(
            beam_radius=w_extend*w0*u.m,
            wavelength=wavelength,
            npix=npix,
            oversample=2,
            M2=M2,
            n0=1.0)
    wf *= optics.GaussianAperture(w=w0*u.m)
    wf *= fresnel.QuadraticLens(f_lens=R0*u.m)
    rad_1 = wf.radius[0]
    
    wf.propagate_fresnel(z*u.m)
    rad_2 = wf.radius[0]
    
    assert(np.round(rad_1, 7) == np.round(w0, 7))
    assert(np.round(rad_2, 7) == np.round(rad_ana, 7))

def test_M2():
    """Confirm that the beam quality factor is computed correctly."""
    
    w0 = 10e-2              # beam radius (m)
    P0 = 10e3               # beam power (W)
    w_extend = 6            # weight of the spatial extend
    wavelength = 1064e-9    # wavelength in vacuum (m)
    npix = 256              # spatial resolution
    M2 = 3.827              # beam quality factor
    
    wf = physical_wavefront.PhysicalFresnelWavefront(
            beam_radius=w_extend*w0*u.m,
            wavelength=wavelength,
            npix=npix,
            oversample=2,
            M2=M2)
    
    wf *= optics.GaussianAperture(w=w0*u.m)
    wf.scale_power(P0)
    
    M2_, _, _, _, _, _ = wf.M2()
    
    assert(np.round(M2, 3) == np.round(M2_, 3))
