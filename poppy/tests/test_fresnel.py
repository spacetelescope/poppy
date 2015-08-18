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


def test_Circular_Aperture_PTP(display=False, npix=512, display_proper=False):
    """Confirm that magnitude of central spike from diffraction
    due to a circular aperture agrees with expectation.

    The comparison is against a worked example presented as
    Figure 6.15 in Anderson and Enmark, Integrated Modeling of Telescopes.
    Values are also compared against a simulation of the same
    case using PROPER.

    Note this tests only the Plane-to-Plane propagation method,
    since the propagation distance z < z_Rayleigh ~ 360 km

    See https://github.com/mperrin/poppy/issues/106 for further
    discussion of this test case.


    """
    # for npix, n.b. ~512 is minimum to accurately recover the central diffraction spike

    z = 5e3*u.m

    # the following points were traced/digitized using GraphClick from
    # the published figure 6.15 by Anderson and Enmark. Beware of limited precision!
    ref_x = [-0.505, -0.412, -0.368, -0.323, -0.302, -0.027, 0]
    ref_y = [ 0.233,  1.446,  0.757,  1.264,  0.784,  1.520, 3.134]

    proper_x = [0]
    proper_y = [3.30790] # from high res sim: 16384**2 pix, beam ratio=0.125


    # The test condition here is subtle, as the exact peak value expected
    # here depends on the sampling.  A high resolution simulation with
    # PROPER yields peak=3.30790 for this case but we will only get roughly
    # close to that with lower sampling.  we do not want to add too high a
    # computation load to the unit test system, so we have to tolerate some
    # imprecision here.
    # 
    # Therefore tune the expected tolerance based on empirical past results.
    # N.B.  this is only approximate and the test may still fail depending
    # on particular choice of npix, without indicating anything
    # fundamentally wrong rather than just the inherent limits of discrete
    # transformations of samplings of continuous functions.

    if npix< 512:
        raise ValueError('npix too low for reasonable accuracy')
    elif npix>=512 and npix < 1024:
        tolerance=0.02
    elif npix>=1024 and npix<2048:
        # for some strange reason, 1024 pix does relatively worse than 512, assuming oversample=4
        tolerance = 0.035
    else:
        tolerance=0.03

    # Note there is a slight error in the text of Anderson and Enmark; the
    # paragraph just before Fig 6.15 says "diameter D=0.5 m", but the 
    # figure actually depicts a case with radius r=0.5 m, as is immediately
    # and obviously the case based on the x axis of the figure. 

    gw = fresnel.Wavefront(beam_radius=0.5*u.m,wavelength=2200e-9,npix=npix,oversample=4)
    gw *= optics.CircularAperture(radius=0.5,oversample=gw.oversample)

    gw.propagate_fresnel(z)
    inten = gw.intensity

    y, x = gw.coordinates()

    if display:

        plt.figure()
        gw.display('both',colorbar=True)
        plt.figure(figsize=(12,6))

        plt.plot(x[0,:], inten[inten.shape[1]/2,:], label='POPPY')
        plt.title("z={:0.2e} , compare to Anderson and Enmark fig.6.15".format(z))
        plt.gca().set_xlim(-1, 1)
        plt.text(0.1,2, "Max value: {0:.4f}\nExpected:   {1:.4f}".format(np.max(inten), max(proper_y)))


        if display_proper:
            # This needs a data file output from PROPER, which we don't
            # bother with including for automated unit testing.
            proper_result = fits.getdata('proper_circle_5e3_cut.fits')
            plt.plot(proper_result[0], proper_result[1], linestyle='-', color='red', alpha=0.5, label="PROPER")

        plt.plot( ref_x, ref_y, 'o', label="Anderson & Enmark")
        plt.legend(loc='upper right', frameon=False)


    _log.debug("test_Circular_Aperture_PTP peak flux comparison: "+str(np.abs(max(proper_y) - np.max(inten))))
    _log.debug("  allowed tolerance for {0} is {1}".format(npix, tolerance))
    #assert(np.round(3.3633280006866424,9) == np.round(np.max(gw.intensity),9))
    assert (np.abs(max(proper_y) - np.max(inten)) < tolerance)

    # also let's test that the output is centered on the array as expected.
    # the peak pixel should be at the coordinates (0,0)
    assert inten[np.where((y==0) & (x==0))] == inten.max()

    # and the image should be symmetric if you flip in X or Y
    # (approximately but not perfectly to machine precision
    # due to the minor asymmetry from having the FFT center pixel
    # not precisely centered in the array.)

    cen = inten.shape[0]/2
    cutsize=10
    center_cut_x = inten[cen-cutsize:cen+cutsize+1, cen]
    assert(np.all((center_cut_x- center_cut_x[::-1])/center_cut_x < 0.001))

    center_cut_y = inten[cen, cen-cutsize:cen+cutsize+1]
    assert(np.all((center_cut_y- center_cut_y[::-1])/center_cut_y < 0.001))



def test_spherical_lens(display=False):
    """Make sure that spherical lens operator is working"""
    # not yet implemented.
