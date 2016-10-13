from .. import poppy_core
from .. import optics
from .. import misc
from .. import fresnel
from .. import utils
from poppy.poppy_core import _log, PlaneType

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import numpy as np
from .. import fwcentroid
from scipy.ndimage import zoom,shift

def test_GaussianBeamParams():
    """Confirm that gaussian beam parameters agree with expectations"""
    gw=fresnel.FresnelWavefront(100*u.um,wavelength=830e-9)
    gw.propagate_fresnel(50*u.mm)
    gl=fresnel.QuadraticLens(50*u.mm)
    gw.propagate_fresnel(28*u.mm)
    gw.apply_lens_power(gl,ignore_wavefront=True)
    assert(np.round(gw.w_0.value,9) == np.round(0.0001061989749146441,9))
    assert(np.round(gw.z_w0.value,9) == np.round(0.15957902236417937,9))
    assert(np.round(gw.z_r.value,9) == np.round(0.042688650889351865,9))
    # FIXME MP: where do the above values come from?



def test_Gaussian_Beam_curvature_near_waist(npoints=5, plot=False):
    """Verify the beam curvature and spreading near the waist
    are as expected from simple analytic theory forg
    Gaussian beams
    """
    # setup an initial Gaussian beam in an aperture.
    ap = optics.CircularAperture()
    wf0 = fresnel.FresnelWavefront(2*u.m, wavelength=1e-6)

    # use that to scale the
    z_rayleigh = wf0.z_r
    z = z_rayleigh * np.logspace(-1,1,num=npoints)
    zdzr = z/z_rayleigh

    calc_rz = []
    calc_wz = []
    for zi in z:
        #setup entrance wave and propagate to z
        wf = fresnel.FresnelWavefront(2*u.m, wavelength=1e-6)
        wf.propagate_fresnel(zi)

        # calculate the beam radius and curvature at z
        calc_rz.append( (wf.r_c()/z_rayleigh).value)
        calc_wz.append( (wf.spot_radius()/wf.w_0).value)

    # Calculate analytic solution for Gaussian beam propagation
    # compare to the results from Fresnel prop.
    rz = (z**2 + z_rayleigh**2)/z
    wz = wf0.w_0*np.sqrt(1+zdzr**2)

    if plot:
        plt.plot(zdzr, rz/z_rayleigh, label="$R(z)/z_r$ (analytical)", color='blue')
        plt.plot(zdzr, calc_rz, ls='dashed', linewidth=3, color='purple', label="$R(z)/z_r$ (calc.)")

        plt.plot(zdzr, wz/wf.w_0, label="$w(z)/w_0$ (analytical)", color='orange')
        plt.plot(zdzr, calc_wz, ls='dashed', linewidth=3, color='red', label="$w(z)/w_0$ (calc.)")

        plt.xlabel("$z/z_r$")
        plt.legend(loc='lower right', frameon=False)

    assert np.allclose(rz/z_rayleigh, calc_rz)
    assert np.allclose(wz/wf.w_0, calc_wz)


def test_Circular_Aperture_PTP_long(display=False, npix=512, display_proper=False):
    """ Tests plane-to-plane propagation at large distances.

    Confirm that magnitude of central spike from diffraction
    due to a circular aperture agrees with expectation.

    The comparison is against a worked example presented as
    Figure 6.15 in Anderson and Enmark, Integrated Modeling of Telescopes.
    Values are also compared against a simulation of the same
    case using PROPER.

    Note this tests only the Plane-to-Plane propagation method,
    since the propagation distance z < z_rayleigh ~ 360 km

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

    gw = fresnel.FresnelWavefront(beam_radius=0.5*u.m,wavelength=2200e-9,npix=npix,oversample=4)
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


def test_Circular_Aperture_PTP_short(display=False, npix=512, display_proper=False):
    """ Tests plane-to-plane propagation at short distances, by comparison
    of the results from propagate_ptp and propagate_direct calculations

    """
    #test short distance propagation, as discussed in issue #194 (https://github.com/mperrin/poppy/issues/194)
    wf = fresnel.FresnelWavefront(
        2 * u.um,
        wavelength=10e-9*u.m,
        npix=npix,
        oversample=4)
    wf *= optics.CircularAperture(radius=800 * 1e-9*u.m)
    wf_2 = wf.copy()
    z = 12. * u.um

    # Calculate same result using 2 different algorithms:
    wf.propagate_direct(z)
    wf_2.propagate_fresnel(z)

    # The results have different pixel scale so we need to resize
    # in order to compare them
    zoomed=(zoom(wf.intensity,(wf.pixelscale/wf_2.pixelscale).decompose().value))
    n = zoomed.shape[0]

    crop_2=wf_2.intensity[int(1023-n/2):int(1023+n/2), int(1023-n/2):int(1023+n/2)]
    #zooming shifted the centroids, find new centers
    cent=fwcentroid.fwcentroid(zoomed,halfwidth=8)
    cent2=fwcentroid.fwcentroid(crop_2,halfwidth=8)
    shifted=shift(crop_2,[cent[1]-cent2[1],cent[0]-cent2[0]])
    diff=shifted/shifted.max()-zoomed/zoomed.max()
    assert(diff.max() < 1e-3)

def test_spherical_lens(display=False):
    """Make sure that spherical lens operator is working.

    Comparison values were taken from a simple PROPER model, implemented in IDL:
        sampling=1
        wavel=500e-9
        prop_begin,wavefront,.5,wavelength,gridsize
        d = 0.025d
        fl = 0.70d0
        prop_begin, wavefront, d, wavel, 256, 1.0
        prop_lens, wavefront, fl
        print, mean(prop_get_phase(wavefront))
        print, max(prop_get_phase(wavefront))
    """
    proper_wavefront_mean =  0.00047535727 #IDL> print,mean(prop_get_phase(wavefront))
    proper_phase_max = 0.0014260283 #IDL> print,max(prop_get_phase(wavefront))
    #build a simple system without apertures.
    beam_diameter =  .025*u.m
    fl = u.m*0.7
    wavefront = fresnel.FresnelWavefront(beam_radius=beam_diameter/2., 
                                   wavelength=500.0e-9, npix=256,
                                   oversample=1)
    diam = beam_diameter
    lens = fresnel.QuadraticLens(fl, name='M1')
    wavefront.apply_lens_power(lens)
    #IDL/PROPER results should be within 10^(-11).
    assert  1e-11 > abs(np.mean(wavefront.phase)-proper_wavefront_mean)
    assert  1e-11 > abs(np.max(wavefront.phase)-proper_phase_max)

def test_fresnel_optical_system_Hubble(display=False):
    """ Test the FresnelOpticalSystem infrastructure
    This is a fairly comprehensive test case using as its
    example a simplified version of the Hubble Space Telescope.

    The specific numeric values used for Hubble are taken from
    the HST example case in the PROPER manual by John Krist,
    version 2.0a available from http://proper-library.sourceforge.net
    This is not intended as a high-fidelity model of Hubble, and this
    test case neglects the primary aperture obscuration as well as the
    specific conic constants of the optics including the infamous
    spherical aberration.

    This function tests the FresnelOpticalSystem functionality including
    assembly of the optical system and propagation of wavefronts,
    intermediate beam sizes through the optical system,
    intermediate and final system focal lengths,
    toggling between different types of optical planes,
    and the properties of the output PSF including FWHM and
    comparison to the Airy function.

    """

    # HST example - Following example in PROPER Manual V2.0 page 49.
    # This is an idealized case and does not correspond precisely to the real telescope
    # Define system with units
    diam = 2.4 * u.m
    fl_pri = 5.52085 * u.m
    d_pri_sec = 4.907028205 * u.m  # This is what's used in the PROPER example
    #d_pri_sec = 4.9069 * u.m      # however Lallo 2012 gives this value, which differs slightly
                                   # from what is used in the PROPER example case.
    fl_sec = -0.6790325 * u.m
    d_sec_to_focus = 6.3916645 * u.m # place focal plane right at the beam waist after the SM

    osamp = 2 #oversampling factor

    hst = fresnel.FresnelOpticalSystem(pupil_diameter=2.4*u.m, beam_ratio=0.25)
    g1 = fresnel.QuadraticLens(fl_pri, name='Primary', planetype=poppy_core.PlaneType.pupil)
    g2 = fresnel.QuadraticLens(fl_sec, name='Secondary')

    hst.add_optic(optics.CircularAperture(radius=diam.value/2))
    hst.add_optic(g1)
    hst.add_optic(g2, distance=d_pri_sec)
    hst.add_optic(optics.ScalarTransmission(planetype=poppy_core.PlaneType.image), distance=d_sec_to_focus)

    # Create a PSF
    psf, waves = hst.calc_psf(wavelength=0.5e-6, display_intermediates=display, return_intermediates=True)


    ### check the beam size is as expected at primary and secondary mirror
    assert(np.allclose(waves[1].spot_radius().value, 1.2))
    # can't find a definitive reference for the beam diam at the SM, but
    # the secondary mirror's radius is 14.05 cm
    # We find that the beam is indeed slightly smaller than that.
    assert(waves[2].spot_radius() > 13*u.cm )
    assert(waves[2].spot_radius() < 14*u.cm )

    ### check the focal length of the overall system is as expected
    expected_system_focal_length = 1./(1./fl_pri + 1./fl_sec - (d_pri_sec)/(fl_pri*fl_sec))
    # n.b. the value calculated here, 57.48 m, is a bit less than the 
    # generally stated focal length of Hubble, 57.6 meters. Adjusting the
    # primary-to-secondary spacing by about 100 microns can resolve this
    # discrepancy. We here opt to stick with the values used in the PROPER
    # example, to facilitate cross-checking the two codes. 

    assert(not np.isfinite(waves[0].focal_length))  # plane wave after circular aperture
    assert(waves[1].focal_length==fl_pri)           # focal len after primary
    # NB. using astropy.Quantities with np.allclose() doesn't work that well
    # so pull out the values here:
    assert(np.allclose(waves[2].focal_length.to(u.m).value,
           expected_system_focal_length.to(u.m).value)) # focal len after secondary

    ### check the FWHM of the PSF is as expected
    measured_fwhm = utils.measure_fwhm(psf)
    expected_fwhm = 1.028*0.5e-6/2.4*206265
    # we only require this to have < 5% accuracy with respect to the theoretical value
    # given discrete pixelization etc.
    assert(np.abs((measured_fwhm-expected_fwhm)/expected_fwhm) < 0.05)

    ### check the various plane types are as expected, including toggling into angular coordinates
    assert_message = ("Expected FresnelWavefront at plane #{} to have {} == {}, but got {}")
    system_planetypes = [PlaneType.pupil, PlaneType.pupil, PlaneType.intermediate, PlaneType.image]
    for idx, (wavefront, planetype) in enumerate(zip(waves, system_planetypes)):
        assert wavefront.planetype == planetype, assert_message.format(
            idx, "planetype", plane_type, wavefront.planetype
        )

    angular_coordinates_flags = [False, False, False, True]
    for idx, (wavefront, angular_coordinates) in enumerate(zip(waves, angular_coordinates_flags)):
        assert wavefront.angular_coordinates == angular_coordinates, assert_message.format(
            idx, "angular_coordinates", angular_coordinates, wavefront.angular_coordinates
        )

    spherical_flags = [False, True, True, False]
    for idx, (wavefront, spherical) in enumerate(zip(waves, spherical_flags)):
        assert wavefront.spherical == spherical, assert_message.format(
            idx, "spherical", spherical, wavefront.spherical
        )

    ### and check that the resulting function is a 2D Airy function
    #create an airy function matching the center part of this array
    airy = misc.airy_2d(diameter=diam.value, wavelength=0.5e-6,
                              shape=(128,128), pixelscale=psf[0].header['PIXELSCL'],
                             center=(64,64))

    centerpix = hst.npix / hst.beam_ratio / 2
    cutout = psf[0].data[centerpix-64:centerpix+64, centerpix-64:centerpix+64] / psf[0].data[centerpix,centerpix]
    assert( np.abs(cutout-airy).max() < 1e-4 )



