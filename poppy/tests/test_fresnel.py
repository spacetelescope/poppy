import pytest

import poppy
from .. import poppy_core
from .. import optics
from .. import misc
from .. import fresnel
from .. import utils
from poppy.poppy_core import _log, PlaneType

import poppy
import os

import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
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
    gw *= optics.CircularAperture(radius=0.5,oversample=gw.oversample, gray_pixel=False)

    gw.propagate_fresnel(z)
    inten = gw.intensity

    y, x = gw.coordinates()

    if display:

        plt.figure()
        gw.display('both',colorbar=True)
        plt.figure(figsize=(12,6))

        plt.plot(x[0,:], inten[inten.shape[1]//2,:], label='POPPY')
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
    assert inten[((y==0) & (x==0))] == inten.max()

    # and the image should be symmetric if you flip in X or Y
    # (approximately but not perfectly to machine precision
    # due to the minor asymmetry from having the FFT center pixel
    # not precisely centered in the array.)

    cen = inten.shape[0]//2
    cutsize=10
    center_cut_x = inten[cen-cutsize:cen+cutsize+1, cen]
    assert(np.all((center_cut_x- center_cut_x[::-1])/center_cut_x < 0.001))

    center_cut_y = inten[cen, cen-cutsize:cen+cutsize+1]
    assert(np.all((center_cut_y- center_cut_y[::-1])/center_cut_y < 0.001))


try:
    from skimage.registration import phase_cross_correlation
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

@pytest.mark.skipif(not HAS_SKIMAGE, reason='This test requires having scikit-image installed.')
def test_Circular_Aperture_PTP_short(display=False, npix=512, oversample=4, include_wfe=True, display_proper=False):
    """ Tests plane-to-plane propagation at short distances, by comparison
    of the results from propagate_ptp and propagate_direct calculations

    This test also now include wavefront error, so as to demonstrate consistent sign conventions for phase aberrations
    in both methods.

    """
    #test short distance propagation, as discussed in issue #194 (https://github.com/mperrin/poppy/issues/194)
    wf_direct = fresnel.FresnelWavefront(
        2 * u.um,
        wavelength=10e-9*u.m,
        npix=npix,
        oversample=oversample)
    wf_direct *= optics.CircularAperture(radius=800 * 1e-9*u.m)

    if include_wfe:
        wf_direct *= poppy.wfe.ZernikeWFE(radius=800 * 1e-9*u.m,
                     coefficients=(0,0,0,0,0, 0,1e-9))

    wf_fresnel = wf_direct.copy()
    z = 12. * u.um

    # Calculate same result using 2 different algorithms:
    wf_direct.propagate_direct(z)
    wf_fresnel.propagate_fresnel(z)

    # The results have different pixel scales so we need to resize
    # in order to compare them
    scalefactor = (wf_direct.pixelscale/wf_fresnel.pixelscale).decompose().value
    zoomed_direct=zoom(wf_direct.intensity,scalefactor)
    print(f"Rescaling by {scalefactor} to match pixel scales")
    n = zoomed_direct.shape[0]
    center_npix = npix*oversample/2
    print(f"center npix: {center_npix}")
    cropped_fresnel = wf_fresnel.intensity[int(center_npix - n / 2):int(center_npix + n / 2), int(center_npix - n / 2):int(center_npix + n / 2)]

    # Zooming also shifts the centroids, so we have to re-align before we can compare pixel values.
    # In theory we could figure out this offset directly from the pixel scales and how zoom works, which would be
    # more elegant. However for current purposes it is sufficient to brute force it by registering the images together
    if not include_wfe and False:
        # For in-focus images, we can just measure the centroids empirically to align
        #zooming shifted the centroids, find new centers
        cent=fwcentroid.fwcentroid(zoomed_direct,halfwidth=8)
        cent2=fwcentroid.fwcentroid(cropped_fresnel,halfwidth=8)

        center_offset = np.asarray(cent)-np.asarray(cent2)
        print(f"After rescaling, found center offset = {center_offset}")
    else:
        # For defocused images, after rescaling we can register via FFT correlation
        import skimage
        center_offset, error, diffphase = skimage.registration.phase_cross_correlation(zoomed_direct, cropped_fresnel, upsample_factor=50)
        # upsample_factor of 50 or more is required to get sufficiently good alignment to pass the test criterion below
        print(f"Offset from skimage: {center_offset}")

    shifted_fresnel=shift(cropped_fresnel, (center_offset[1], center_offset[0]))

    normalization = zoomed_direct.max() / shifted_fresnel.max() # work around different normalizations
                                            # In some sense this is a bug in propagate_direct that they are not consistent
    zoomed_direct /= normalization
    print(f"Making consistent normalization with scale factor {normalization}")
    diff=shifted_fresnel-zoomed_direct

    if display:
        boxhalfsize = npix//4

        zoomed_crop = zoomed_direct[n//2-boxhalfsize:n//2+boxhalfsize, n//2-boxhalfsize:n//2+boxhalfsize]
        shifted_crop = shifted_fresnel[n//2-boxhalfsize:n//2+boxhalfsize, n//2-boxhalfsize:n//2+boxhalfsize]
        diff_crop = diff[n//2-boxhalfsize:n//2+boxhalfsize, n//2-boxhalfsize:n//2+boxhalfsize]

        plt.imshow(zoomed_crop)
        plt.colorbar()
        plt.title("From propagate_direct\nrescaled to match scale of propagate_fresnel")
        plt.figure()
        plt.imshow(shifted_crop)
        plt.colorbar()
        plt.title("From propagate_fresnel\nshifted to align to propagate_direct")
        plt.figure()
        plt.imshow(diff_crop)
        plt.colorbar()
        plt.title("Difference of those two, after normalization")

    maxreldiff = diff.max() / shifted_fresnel.max()
    assert maxreldiff < 1e-3 , f"Pixel values different more than expected; max relative difference is {maxreldiff}"


def test_fresnel_conservation_of_intensity(display=True, npix=256):
    """Test that conservation of energy is maintained

    """
    aperture_diam = 10 * u.micron
    distances = [10 * u.um, 100 * u.um, 200 * u.um, 1 * u.cm, 1 * u.m]

    wf = poppy.fresnel.FresnelWavefront(
        aperture_diam * 1.5,  # beam radius
        wavelength=100 * u.nm,
        npix=npix,
        oversample=2)
    wf *= poppy.optics.CircularAperture(radius=aperture_diam / 2)

    ti0 = wf.total_intensity

    for d in distances:
        wf.propagate_fresnel(d)
        assert np.allclose(ti0, wf.total_intensity), "Propagation appears to violate conservation of energy"
        if display:
            plt.figure()
            wf.display(title=f'After {d}', scale='linear', showpadding=False)


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
    lens = fresnel.QuadraticLens(fl, name='M1')
    wavefront.apply_lens_power(lens)
    #IDL/PROPER results should be within 10^(-11).
    assert 1e-11 > abs(np.mean(wavefront.phase)-proper_wavefront_mean)
    assert 1e-11 > abs(np.max(wavefront.phase)-proper_phase_max)

def test_fresnel_optical_system_Hubble(display=False, sampling=2):
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

    hst = fresnel.FresnelOpticalSystem(pupil_diameter=2.4*u.m, beam_ratio=1./sampling)
    g1 = fresnel.QuadraticLens(fl_pri, name='Primary', planetype=poppy_core.PlaneType.pupil)
    g2 = fresnel.QuadraticLens(fl_sec, name='Secondary')

    hst.add_optic(optics.CircularAperture(radius=diam.value/2))
    hst.add_optic(g1)
    hst.add_optic(g2, distance=d_pri_sec)
    hst.add_optic(optics.ScalarTransmission(planetype=poppy_core.PlaneType.image), distance=d_sec_to_focus)

    # Create a PSF
    psf, waves = hst.calc_psf(wavelength=0.5e-6, display_intermediates=display, return_intermediates=True)

    if len(waves)>1:

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
        cen = utils.measure_centroid(psf)
        measured_fwhm = utils.measure_fwhm(psf, center=cen)
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

    centerpix = int(hst.npix / hst.beam_ratio / 2)
    cutout = psf[0].data[centerpix-64:centerpix+64, centerpix-64:centerpix+64] / psf[0].data[centerpix,centerpix]
    assert( np.abs(cutout-airy).max() < 1e-4 )

    if display:
        plt.figure()
        utils.display_psf(psf, imagecrop=1)


def test_fresnel_FITS_Optical_element(tmpdir, display=False, verbose=False):
    """ Test that Fresnel works with FITS optical elements.

    Incidentally serves as a test of the fix for the FITS endian issue
    in recent scipy builds. See https://github.com/mperrin/poppy/issues/213

    Incidentally also serves as a test that we can round-trip an
    AnalyticOpticalElement into a FITS file and then back into
    a FITSOpticalElement. See #49
    import tempfile

    Parameters
    ----------
    tmpdir : string
        temporary directory for output FITS file. To be provided by py.test's
        tmpdir test fixture.
    display : bool
        Show plots?
    verbose : bool
        Print some remarks when running?

    """
    import os.path
    import astropy.io.fits as fits
    from poppy import wfe

    # parameters for calculation test case:
    radius = 1.0 * u.m
    npix = 128

    conv_lens = fresnel.QuadraticLens(1.0 * u.m)
    circular_aperture = optics.CircularAperture(radius=radius, gray_pixel=False)

    # To test two different versions of the FITS handling, we will repeat this test twice:
    # once with the FITS element crafted to precisely match the pixel scale of the
    # wavefront in the Fresnel propagation, and once with a mismatch in the pixel scale
    # so that it has to be interpolated. We should get the same result both ways within the
    # tolerances.

    # Below we will create a FITSOpticalElement and show that it works in the FresnelOpticalSystem.
    # To make the test interesting we put some astigmatism in that FITS file, but this is arbitrary.
    # We make the OPD by creating a Zernike WFE object, sampling it as desired, writing it
    # out as a temp file, then reading back in to a FITSOpticalElement

    for matchscale in [True, False]:

        # Create the FITS element to test
        m1_zernike = wfe.ZernikeWFE(radius=radius,
                                    coefficients=[0, 0, 0, 0, 0, 1e-7])
        if matchscale:
            fits_zern = m1_zernike.to_fits(what='opd', grid_size=2 * radius, npix=npix)
        else:
            fits_zern = m1_zernike.to_fits(what='opd', npix=376)

        filename = os.path.join(str(tmpdir), "astigmatism.fits")
        fits_zern.writeto(filename, overwrite=True)
        astig_surf = poppy_core.FITSOpticalElement(opd=filename,
                                                   planetype=poppy_core._INTERMED,
                                                   oversample=1)

        if verbose:
            print("Astigmatism surface from FITS has pixelscale {}, npix={}".format(astig_surf.pixelscale,
                                                                                    astig_surf.shape[0]))

        # Now we put that FITSOpticalElement into a Fresnel optical system.
        fosys = fresnel.FresnelOpticalSystem(pupil_diameter=radius * 2, beam_ratio=0.25, npix=npix)
        fosys.add_optic(circular_aperture)
        fosys.add_optic(astig_surf)
        fosys.add_optic(conv_lens)
        fosys.add_optic(optics.ScalarTransmission(name='focus'), distance=1 * u.m)

        # perform the calculation, then check results are as expected
        psf_with_astigmatism, wfronts = fosys.calc_psf(display_intermediates=display, return_intermediates=True)

        cx, cy = utils.measure_centroid(psf_with_astigmatism)
        expected_cx = expected_cy = psf_with_astigmatism[0].data.shape[0] // 2
        assert np.abs(cx - expected_cx) < 0.02, "PSF centroid is not as expected in X"
        assert np.abs(cy - expected_cy) < 0.02, "PSF centroid is not as expected in Y"
        assert psf_with_astigmatism[0].data.sum() > 0.99, "PSF total flux is not as expected."
        assert np.abs(psf_with_astigmatism[0].data.max() - 0.033212) < 2e-5, "PSF peak pixel is not as expected"

        if verbose:
            print("Tests of FITSOpticalElement in FresnelOpticalSystem pass.")


def test_fresnel_propagate_direct_forward_and_back():
    """ Test the propagate_direct FFT algorithm, applied forward and back, is a null operation"""
    npix = 1024
    wavelen = 2200 * u.nm
    wf = fresnel.FresnelWavefront(
        0.5 * u.m, wavelength=wavelen, npix=npix, oversample=4
    )
    wf *= optics.CircularAperture(radius=0.5)
    z = ((wf.pixelscale * u.pix) ** 2 * wf.n / (2200 * u.nm)).to(u.m)
    start = wf.wavefront.copy()
    wf.propagate_direct(z)
    wf.propagate_direct(-z)
    np.testing.assert_almost_equal(wf.wavefront, start)


def test_fresnel_propagate_direct_back_and_forward():
    """ Test the propagate_fresnel FFT algorithm, applied forward and back, is a null operation"""
    npix = 1024
    wavelen = 2200 * u.nm
    wf = fresnel.FresnelWavefront(
        0.5 * u.m, wavelength=wavelen, npix=npix, oversample=4
    )
    wf *= optics.CircularAperture(radius=0.5)
    z = ((wf.pixelscale * u.pix) ** 2 * wf.n / (2200 * u.nm)).to(u.m)
    start = wf.wavefront.copy()
    wf.propagate_direct(-z)
    wf.propagate_direct(z)
    np.testing.assert_almost_equal(wf.wavefront, start)


def test_fresnel_propagate_direct_2forward_and_back():
    """ Test that propagate_direct forward twice and back once is the same as forward once

    (This seems redundant, and I can no longer remember why this test was implemented...)
    """
    npix = 1024
    wavelen = 2200 * u.nm
    wf = fresnel.FresnelWavefront(
        0.5 * u.m, wavelength=wavelen, npix=npix, oversample=4
    )
    wf *= optics.CircularAperture(radius=0.5)
    z = ((wf.pixelscale * u.pix) ** 2 * wf.n / (2200 * u.nm)).to(u.m)

    wf.propagate_direct(z)
    start = wf.wavefront.copy()
    wf.propagate_direct(z)
    wf.propagate_direct(-z)
    np.testing.assert_almost_equal(wf.wavefront, start)

def test_fresnel_return_complex():
    """Test that we can return a complex wavefront from a Fresnel propagation, and
    that complex wavefront is consistent with the usual PSF in real intensity units
    """
    # physical radius values
    M1_radius = 3. * u.m
    fl_M1 = M1_radius/2.0
    # intermediary distances

    tel = fresnel.FresnelOpticalSystem(pupil_diameter=2.4*u.m)
    gl=fresnel.QuadraticLens(500*u.cm)

    tel.add_optic(gl)
    tel.add_optic(optics.CircularAperture(radius=M1_radius,name="M1 aperture"))
    tel.add_optic(optics.ScalarTransmission( name="primary mirror focal plane"), distance=fl_M1)

    psf=tel.calc_psf(return_final=True)

    assert len(psf[1])==1
    assert np.allclose(psf[1][0].intensity,psf[0][0].data)


def test_detector_in_fresnel_system(npix=256):
    """ Show that we can put a detector in a FresnelOpticalSystem
    and it will resample the wavefront to the desired sampling and size.

    Also checks conservation of intensity through the resampling operation.
    """

    output_npix = 400
    out_pixscale = 210

    # Setup Fresnel system, with a detector that changes the sampling
    # note - ensure pupil array diameter is at least a bit larger than the actual aperture
    osys = fresnel.FresnelOpticalSystem(pupil_diameter=0.051*u.m, npix=npix, beam_ratio=0.25)
    osys.add_optic(optics.CircularAperture(radius=0.025))
    osys.add_optic(optics.ScalarTransmission(), distance=10*u.m)
    osys.add_detector(pixelscale=out_pixscale*u.micron/u.pixel, fov_pixels=output_npix)

    # Calculate a PSF
    psf, waves = osys.calc_psf(wavelength=1e-6, return_intermediates=True)

    # Check the output pixel scale is as desired
    np.testing.assert_almost_equal(psf[0].header['PIXELSCL'],  out_pixscale/1e6)

    # Check the wavefront gets cropped to the right size of pixels, from something different
    assert waves[0].shape == (1024, 1024)
    assert waves[1].shape == (1024, 1024)
    assert waves[2].shape == (output_npix, output_npix)
    assert psf[0].data.shape == (output_npix, output_npix)

    assert psf[0].header['NAXIS1'] == output_npix

    # Check the PSF is centered and not offset
    psfdata = psf[0].data
    ny, nx = psfdata.shape
    assert psfdata[ny//2, nx//2] == psfdata.max(), "Peak (spot of Arago) is not in the center"
    assert np.allclose(psfdata[output_npix//2],
                       np.roll(psfdata[output_npix//2][::-1], 1), atol=3e-8), "PSF is unexpectedly asymmetric"
    y, x = np.indices(psfdata.shape)
    x -= nx//2
    y -= ny//2
    assert (psf[0].data*x).sum() / \
        psf[0].data.sum() < 0.001, "PSF is surprisingly offset from centered in X"
    assert (psf[0].data*y).sum() / \
        psf[0].data.sum() < 0.001, "PSF is surprisingly offset from centered in Y"

    # Check flux conservation
    # Note this test relies on the detector covering a sufficiently large area of the PSF that the
    # encircled energy is nearly total
    assert np.abs(waves[2].total_intensity -
                  1) < 0.001, "PSF total flux is surprisingly different from 1"

def test_wavefront_conversions():
    """ Test conversions between Wavefront and FresnelWavefront
    in both directions.
    """
    import poppy

    props = lambda wf: (wf.shape, wf.ispadded, wf.oversample, wf.pixelscale)

    optic = poppy.CircularAperture()
    w = poppy.Wavefront(diam=4*u.m)
    w*= optic

    fw = poppy.FresnelWavefront(beam_radius=2*u.m)
    fw*= optic

    # test convert from Fraunhofer to Fresnel
    fw2 = poppy.FresnelWavefront.from_wavefront(w)
    assert props(fw)==props(fw2)
    #np.testing.assert_allclose(fw.wavefront, fw2.wavefront)

    # test convert from Fresnel to Fraunhofer
    w2 = poppy.Wavefront.from_fresnel_wavefront(fw)
    assert props(w)==props(w2)


def test_CompoundOpticalSystem_fresnel(npix=128, display=False):
    """ Test that the CompoundOpticalSystem container works for Fresnel systems

    Parameters
    ----------
    npix : int
        Number of pixels for the pupil sampling. Kept small by default to
        reduce test run time.
    """

    import poppy

    opt1 = poppy.SquareAperture()
    opt2 = poppy.CircularAperture(radius=0.55)

    # a single optical system
    osys = poppy.FresnelOpticalSystem(beam_ratio=0.25, npix=npix)
    osys.add_optic(opt1)
    osys.add_optic(opt2, distance=10*u.cm)
    osys.add_optic(poppy.QuadraticLens(1.0*u.m))
    osys.add_optic(poppy.Detector(pixelscale=0.25*u.micron/u.pixel, fov_pixels=512), distance=1*u.m)

    psf = osys.calc_psf(display_intermediates=display)

    if display:
        plt.figure()

    # a Compound Fresnel optical system
    osys1 = poppy.FresnelOpticalSystem(beam_ratio=0.25, npix=npix)
    osys1.add_optic(opt1)
    osys2 = poppy.FresnelOpticalSystem(beam_ratio=0.25)
    osys2.add_optic(opt2, distance=10*u.cm)
    osys2.add_optic(poppy.QuadraticLens(1.0*u.m))
    osys2.add_optic(poppy.Detector(pixelscale=0.25*u.micron/u.pixel, fov_pixels=512), distance=1*u.m)

    cosys = poppy.CompoundOpticalSystem([osys1, osys2])

    psf2 = cosys.calc_psf(display_intermediates=display)

    assert np.allclose(psf[0].data, psf2[0].data), "Results from simple and compound Fresnel systems differ unexpectedly."

    return psf, psf2

def test_CompoundOpticalSystem_hybrid(npix=128):
    """ Test that the CompoundOpticalSystem container works for hybrid Fresnel+Fraunhofer systems

    Defining "works correctly" here is a bit arbitrary given the different physical assumptions.
    For the purpose of this test we consider a VERY simple case, mostly a Fresnel system. We split
    out the first optic and put that in a Fraunhofer system. We then test that a compound hybrid
    system yields the same results as the original fully-Fresnel system.

    Parameters
    ----------
    npix : int
        Number of pixels for the pupil sampling. Kept small by default to
        reduce test run time.
    """

    import poppy

    opt1 = poppy.SquareAperture()
    opt2 = poppy.CircularAperture(radius=0.55)

    ###### Simple test case to exercise the conversion functions, with only trivial propagation
    osys1 = poppy.OpticalSystem()
    osys1.add_pupil(opt1)
    osys2 = poppy.FresnelOpticalSystem()
    osys2.add_optic(poppy.ScalarTransmission())
    osys3 = poppy.OpticalSystem()
    osys3.add_pupil(poppy.ScalarTransmission())
    osys3.add_detector(fov_pixels=64, pixelscale=0.01)
    cosys = poppy.CompoundOpticalSystem([osys1, osys2, osys3])
    psf, ints = cosys.calc_psf( return_intermediates=True)
    assert len(ints) == 4, "Unexpected number of intermediate  wavefronts"
    assert isinstance(ints[0], poppy.Wavefront), "Unexpected output type"
    assert isinstance(ints[1], poppy.FresnelWavefront), "Unexpected output type"
    assert isinstance(ints[2], poppy.Wavefront), "Unexpected output type"

    ###### Harder case involving more complex actual propagations

    #===== a single Fresnel optical system =====
    osys = poppy.FresnelOpticalSystem(beam_ratio=0.25, npix=128, pupil_diameter=2*u.m)
    osys.add_optic(opt1)
    osys.add_optic(opt2, distance=10*u.cm)
    osys.add_optic(poppy.QuadraticLens(1.0*u.m))
    osys.add_optic(poppy.Detector(pixelscale=0.125*u.micron/u.pixel, fov_pixels=512), distance=1*u.m)

    #===== two systems, joined into a CompoundOpticalSystem =====
    # First part is Fraunhofer then second is Fresnel
    osys1 = poppy.OpticalSystem(npix=128, oversample=4, name="FIRST PART, FRAUNHOFER")
    # Note for strict consistency we need to apply a half pixel shift to optics in the Fraunhofer part;
    # this accomodates the differences between different types of image centering.
    pixscl = osys.input_wavefront().pixelscale
    halfpixshift = (pixscl*0.5*u.pixel).to(u.m).value
    opt1shifted = poppy.SquareAperture(shift_x = halfpixshift, shift_y = halfpixshift)
    osys1.add_pupil(opt1shifted)

    osys2 = poppy.FresnelOpticalSystem(name='SECOND PART, FRESNEL')
    osys2.add_optic(opt2, distance=10*u.cm)
    osys2.add_optic(poppy.QuadraticLens(1.0*u.m))
    osys2.add_optic(poppy.Detector(pixelscale=0.125*u.micron/u.pixel, fov_pixels=512), distance=1*u.m)

    cosys = poppy.CompoundOpticalSystem([osys1, osys2])

    #===== PSF calculations =====
    psf_simple = osys.calc_psf(return_intermediates=False)
    poppy.poppy_core._log.info("******=========calculation divider============******")
    psf_compound = cosys.calc_psf(return_intermediates=False)

    np.testing.assert_allclose(psf_simple[0].data, psf_compound[0].data,
                               err_msg="PSFs do not match between equivalent simple and compound/hybrid optical systems")


def test_inwave_fresnel(plot=False):
    '''Verify basic functionality of the inwave kwarg for a basic FresnelOpticalSystem()'''
    npix = 128
    oversample = 2
    # HST example - Following example in PROPER Manual V2.0 page 49.
    lambda_m = 0.5e-6 * u.m
    diam = 2.4 * u.m
    fl_pri = 5.52085 * u.m
    d_pri_sec = 4.907028205 * u.m
    fl_sec = -0.6790325 * u.m
    d_sec_to_focus = 6.3919974 * u.m

    m1 = poppy.QuadraticLens(fl_pri, name='Primary')
    m2 = poppy.QuadraticLens(fl_sec, name='Secondary')

    hst = poppy.FresnelOpticalSystem(pupil_diameter=diam, npix=npix, beam_ratio=1 / oversample)
    hst.add_optic(poppy.CircularAperture(radius=diam.value / 2))
    hst.add_optic(poppy.SecondaryObscuration(secondary_radius=0.396,
                                             support_width=0.0264,
                                             support_angle_offset=45.0))
    hst.add_optic(m1)
    hst.add_optic(m2, distance=d_pri_sec)
    hst.add_optic(poppy.ScalarTransmission(planetype=poppy_core.PlaneType.image, name='focus'), distance=d_sec_to_focus)

    if plot:
        plt.figure(figsize=(12, 8))
    psf1, wfs1 = hst.calc_psf(wavelength=lambda_m, display_intermediates=plot, return_intermediates=True)

    # now test the system by inputting a wavefront first
    wfin = poppy.FresnelWavefront(beam_radius=diam / 2, wavelength=lambda_m,
                                  npix=npix, oversample=oversample)
    if plot:
        plt.figure(figsize=(12, 8))
    psf2, wfs2 = hst.calc_psf(wavelength=lambda_m, display_intermediates=plot, return_intermediates=True,
                              inwave=wfin)

    wf = wfs1[-1].wavefront
    wf_no_in = wfs2[-1].wavefront

    assert np.allclose(wf,
                       wf_no_in), 'Results differ unexpectedly when using inwave argument for FresnelOpticalSystem().'



def test_FixedSamplingImagePlaneElement(display=False):
    poppy_tests_fpath = os.path.dirname(os.path.abspath(poppy.__file__))+'/tests/'
    
    # HST example - Following example in PROPER Manual V2.0 page 49.
    # This is an idealized case and does not correspond precisely to the real telescope
    # Define system with units
    lambda_m = 0.5e-6*u.m
    diam = 2.4 * u.m
    fl_pri = 5.52085 * u.m
    d_pri_sec = 4.907028205 * u.m  # This is what's used in the PROPER example
    #d_pri_sec = 4.9069 * u.m      # however Lallo 2012 gives this value, which differs slightly
                                   # from what is used in the PROPER example case.
    fl_sec = -0.6790325 * u.m
    d_sec_to_focus = 6.3916645 * u.m # place focal plane right at the beam waist after the SM
    fl_oap = 0.5*u.m

    sampling=2
    hst = poppy.FresnelOpticalSystem(npix=128, pupil_diameter=2.4*u.m, beam_ratio=1./sampling)
    g1 = poppy.QuadraticLens(fl_pri, name='Primary', planetype=poppy.poppy_core.PlaneType.pupil)
    g2 = poppy.QuadraticLens(fl_sec, name='Secondary')
    fpm = poppy.FixedSamplingImagePlaneElement('BOWTIE FPM', poppy_tests_fpath+'bowtie_fpm_0.05lamD.fits')
    oap = poppy.QuadraticLens(fl_oap, name='OAP')

    hst.add_optic(poppy.CircularAperture(radius=diam.value/2))
    hst.add_optic(g1)
    hst.add_optic(g2, distance=d_pri_sec)
    hst.add_optic(fpm, distance=d_sec_to_focus)
    hst.add_optic(oap, distance=fl_oap)
    hst.add_optic(oap, distance=fl_oap)
    hst.add_optic(poppy.ScalarTransmission(planetype=poppy.poppy_core.PlaneType.intermediate, name='Image'), distance=fl_oap)

    # Create a PSF
    if display: fig=plt.figure(figsize=(10,5))
    psf, waves = hst.calc_psf(wavelength=lambda_m, display_intermediates=display, return_intermediates=True)

    # still have to do comparison of arrays
    psf_result = fits.open(poppy_tests_fpath+'FITSFPMElement_test_result.fits')
    psf_result_data = psf_result[0].data
    psf_result_pxscl = psf_result[0].header['PIXELSCL']
    psf_result.close()
    
    np.testing.assert_allclose(psf[0].data, psf_result_data, rtol=1e-6,
                               err_msg="PSF of this test does not match the saved result.", verbose=True)
    np.testing.assert_allclose(waves[-1].pixelscale.value, psf_result_pxscl,
                               err_msg="PSF pixelscale of this test does not match the saved result.", verbose=True)
    

def test_fresnel_noninteger_oversampling(display_intermediates=False):
    '''Test for noninteger oversampling for basic FresnelOpticalSystem() using HST example system'''
    lambda_m = 0.5e-6 * u.m
    # lambda_m = np.linspace(0.475e-6, 0.525e-6, 3) * u.m
    diam = 2.4 * u.m
    fl_pri = 5.52085 * u.m
    d_pri_sec = 4.907028205 * u.m
    fl_sec = -0.6790325 * u.m
    d_sec_to_focus = 6.3919974 * u.m

    m1 = poppy.QuadraticLens(fl_pri, name='Primary')
    m2 = poppy.QuadraticLens(fl_sec, name='Secondary')
    image_plane = poppy.ScalarTransmission(planetype=poppy_core.PlaneType.image, name='focus')

    npix = 128

    oversample1 = 2
    hst1 = poppy.FresnelOpticalSystem(pupil_diameter=diam, npix=npix, beam_ratio=1 / oversample1)
    hst1.add_optic(poppy.CircularAperture(radius=diam.value / 2))
    hst1.add_optic(poppy.SecondaryObscuration(secondary_radius=0.396,
                                             support_width=0.0264,
                                             support_angle_offset=45.0))
    hst1.add_optic(m1)
    hst1.add_optic(m2, distance=d_pri_sec)
    hst1.add_optic(image_plane, distance=d_sec_to_focus)

    if display_intermediates: plt.figure(figsize=(12, 8))
    psf1 = hst1.calc_psf(wavelength=lambda_m, display_intermediates=display_intermediates)

    # now test the second system which has a different oversampling factor
    oversample2 = 2.0
    hst2 = poppy.FresnelOpticalSystem(pupil_diameter=diam, npix=npix, beam_ratio=1 / oversample2)
    hst2.add_optic(poppy.CircularAperture(radius=diam.value / 2))
    hst2.add_optic(poppy.SecondaryObscuration(secondary_radius=0.396,
                                             support_width=0.0264,
                                             support_angle_offset=45.0))
    hst2.add_optic(m1)
    hst2.add_optic(m2, distance=d_pri_sec)
    hst2.add_optic(image_plane, distance=d_sec_to_focus)
    
    if display_intermediates: plt.figure(figsize=(12, 8))
    psf2 = hst2.calc_psf(wavelength=lambda_m, display_intermediates=display_intermediates)

    # Now test a 3rd HST system with oversample of 2.5 and compare to hardcoded result
    oversample3=2.5
    hst3 = poppy.FresnelOpticalSystem(pupil_diameter=diam, npix=npix, beam_ratio=1 / oversample3)
    hst3.add_optic(poppy.CircularAperture(radius=diam.value / 2))
    hst3.add_optic(poppy.SecondaryObscuration(secondary_radius=0.396,
                                             support_width=0.0264,
                                             support_angle_offset=45.0))
    hst3.add_optic(m1)
    hst3.add_optic(m2, distance=d_pri_sec)
    hst3.add_optic(image_plane, distance=d_sec_to_focus)

    if display_intermediates: plt.figure(figsize=(12, 8))
    psf3 = hst3.calc_psf(wavelength=lambda_m, display_intermediates=display_intermediates)

    assert np.allclose(psf1[0].data, psf2[0].data), 'PSFs with oversampling 2 and 2.0 are surprisingly different.'
    np.testing.assert_almost_equal(psf3[0].header['PIXELSCL'], 0.017188733797782272, decimal=7, 
                                   err_msg='pixelscale for the PSF with oversample of 2.5 is surprisingly different from expected result.')

