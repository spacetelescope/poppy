from .. import poppy_core  as poppy
from .. import optics
from poppy.poppy_core import _log


from .. import fresnel
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

def test_GaussianBeamParams():
    """Confirm that gaussian beam parameters agree with expectations"""
    gw=fresnel.Wavefront(100*u.um,wavelength=830e-9)
    gw.propagate_fresnel(50*u.mm)
    gl=fresnel.GaussianLens(50*u.mm,planetype=fresnel._INTERMED)
    gw.apply_optic(gl,78.0*u.mm,ignore_wavefront=True)
    assert(np.round(gw.w_0.value,9) == np.round(0.0001061989749146441,9))
    assert(np.round(gw.z_w0.value,9) == np.round(0.15957902236417937,9))
    assert(np.round(gw.z_R.value,9) == np.round(0.042688650889351865,9))
    # FIXME MP: where do the above values come from?




def test_Gaussian_Beam_curvature_near_waist(npoints=5, plot=False):
    """Verify the beam curvature and spreading near the waist
    are as expected from simple analytic theory forg
    Gaussian beams
    """
    # setup an initial Gaussian beam in an aperture.g
    ap = optics.CircularAperture()
    wf0 = fresnel.Wavefront(2*u.m, wavelength=1e-6)

    # use that to scale theg
    z_rayleigh = wf0.z_R
    z = z_rayleigh * np.logspace(-1,1,num=npoints)
    zdzr = z/z_rayleigh

    calc_rz = []
    calc_wz = []
    for zi in z:
        #setup entrance wave and propagate to z
        wf = fresnel.Wavefront(2*u.m, wavelength=1e-6)
        wf.propagate_fresnel(zi)

        # calculate the beam radius and curvature at z
        calc_rz.append( (wf.R_c()/z_rayleigh).value)
        calc_wz.append( (wf.spot_radius()/wf.w_0).value)

    # Calculate analytic solution for Gaussian beam propagation
    # compare to the results from Fresnel prop.
    rz = (z**2 + z_rayleigh**2)/z
    wz = wf0.w_0*np.sqrt(1+zdzr**2)

    if plot:
        plt.plot(zdzr, rz/z_rayleigh, label="$R(z)/z_R$ (analytical)", color='blue')
        plt.plot(zdzr, calc_rz, ls='dashed', linewidth=3, color='purple', label="$R(z)/z_R$ (calc.)")

        plt.plot(zdzr, wz/wf.w_0, label="$w(z)/w_0$ (analytical)", color='orange')
        plt.plot(zdzr, calc_wz, ls='dashed', linewidth=3, color='red', label="$w(z)/w_0$ (calc.)")

        plt.xlabel("$z/z_R$")
        plt.legend(loc='lower right', frameon=False)

    assert np.allclose(rz/z_rayleigh, calc_rz)
    assert np.allclose(wz/wf.w_0, calc_wz)


def test_CircularAperturePropagation(display=False):
    """Confirm that magnitude of central spike from diffraction
    due to a circular aperture agrees with expectation"""
    npix = 1024#~512 is minimum to accurately recover the central diffraction spike
    gw = fresnel.Wavefront(0.5*u.m,wavelength=2200e-9,npix=npix,oversample=2)
    gw *= optics.CircularAperture(radius=0.5,oversample=gw.oversample)

    if display:
        plt.figure()
        gw.display('both',colorbar=True)

  #  z = 5e3*u.m

    gw.propagate_fresnel(5e3*u.m)
    if display:

        plt.figure()
        gw.display('both',colorbar=True)
        plt.figure(figsize=(12,6))

        plt.plot(np.arange(gw.intensity.shape[0]), gw.intensity[gw.intensity.shape[1]/2,:])
        plt.title("z={:0.2e} , compare to Anderson and Enmark fig.6.15".format(z))
        plt.text(1300,2, "Max value: {0:.4f}".format(np.max(gw.intensity)))
        plt.set_xlim(0,2048)

    assert(np.round(3.3633280006866424,9) == np.round(np.max(gw.intensity),9))


    # also let's test that the output is centered on the array as expected.
    # If so, we can flip in either X or Y without changing the value appreciably
    inten = gw.intensity
    assert(np.allclose(inten, inten[:, ::-1]))
    assert(np.allclose(inten, inten[::-1, :]))


def test_spherical_lens(display=False):
    """Make sure that spherical lens operator is working"""
    # not yet implemented.
