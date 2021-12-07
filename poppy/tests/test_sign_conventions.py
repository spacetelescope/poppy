import numpy as np
import astropy.units as u
import poppy


####################################
# Test sign conventions
####################################

# Test: positive OPD in an optic produces positive WFE in a wave after interacting with that optic
#       (in the absence of phase wrapping)

def test_wfe_and_opd_have_consistent_signs():
    """ Verify that the sign and amplitude of wavefront error matches that of an optic's OPD,
    for cases that do NOT encounter phase wrapping.

    """

    for opd_amount in (100*u.nm, -0.25*u.micron, 1e-8*u.m):
        constant_opd = poppy.ScalarOpticalPathDifference(opd=opd_amount)
        wave = poppy.Wavefront(wavelength=1*u.micron, npix=64)

        wave *= constant_opd

        assert np.allclose(constant_opd.opd.to_value(u.m),
                           wave.wfe.to_value(u.m)), "optic OPD and wavefront WFE should have consistent signs"


# Test: Tilt with increasing WFE towards the +X direction moves the PSF in the -X direction

def test_wavefront_tilt_sign_and_direction(plot=False, npix=128):
    """ Test that tilt with increasing WFE towards the +X direction moves the PSF in the -X direction
    Fraunhofer propagation version

    See also test_core.test_source_offsets_in_OpticalSystem
    """
    # Create a wavefront and apply a tilt
    wave = poppy.Wavefront(diam=1*u.m, npix=npix)
    wave *= poppy.CircularAperture(radius=0.5*u.m)

    tilt_angle = -0.2    # must be a negative number (for -X direction shift), and within the FOV

    wave.tilt(Xangle=tilt_angle)  # for this function, the input is the desired direction for the image to tilt.
                                  # A shift to -X is implemented by creating an OPD that increases toward +X
    n = wave.shape[0]
    assert wave.wfe[n//2, n//2-5] <  wave.wfe[n//2, n//2+5], "Wavefront error should increase to +X"

    if plot:
        plt.suptitle("Wavefront tilt sign test (Fraunhofer propagation)", fontweight='bold')
        wave.display(what='both')

    wave.propagate_to(poppy.Detector(pixelscale=0.05, fov_pixels=128))

    if plot:
        plt.figure()
        wave.display(what='both', crosshairs=True, imagecrop=2)

    n = wave.shape[0]
    cen = poppy.measure_centroid(wave.as_fits())
    assert np.allclose(cen[0], (n-1)/2), "Tilt in X should not displace the PSF in Y"
    assert cen[1] < (n-1)/2, "WFE tilt increasing to +X should displace the PSF to -X"
    assert np.allclose(((cen[1]-(n-1)/2)*u.pixel*wave.pixelscale).to_value(u.arcsec), tilt_angle), "PSF offset did not match expected amount"

def test_wavefront_tilt_sign_and_direction_fresnel(plot=False, npix=128):
    """ Test that tilt with increasing WFE towards the +X direction moves the PSF in the -X direction
    Fresnel propagation version

    See also test_core.test_source_offsets_in_OpticalSystem
    """
    # Create a wavefront and apply a tilt
    wave = poppy.FresnelWavefront(beam_radius=0.5 * u.m, npix=npix, oversample=8)
    wave *= poppy.CircularAperture(radius=0.5 * u.m)

    # tilt in arcseconds
    tilt_angle = -0.2  # must be a negative number (for -X direction shift), and within the FOV

    wave.tilt(Xangle=tilt_angle)  # for this function, the input is the desired direction for the image to tilt.
    # A shift to -X is implemented by creating an OPD that increases toward +X
    n = wave.shape[0]
    assert wave.wfe[n // 2, n // 2 - 5] < wave.wfe[n // 2, n // 2 + 5], "Wavefront error should increase to +X"

    if plot:
        plt.suptitle("Wavefront tilt sign test (Fresnel propagation)", fontweight='bold')
        wave.display(what='both')

    focal_length = 1 * u.m
    wave *= poppy.QuadraticLens(f_lens=focal_length)

    wave.propagate_fresnel(focal_length)

    if plot:
        plt.figure()
        wave.display(what='both', crosshairs=True, imagecrop=0.00001, scale='log')

    n = wave.shape[0]
    nominal_cen = n // 2  # In Fresnel mode, PSFs are centered on a pixel by default
    # (different from in Frauhofer mode by half a pixel)

    cen = poppy.measure_centroid(wave.as_fits())
    assert np.allclose(cen[0], nominal_cen), "Tilt in X should not displace the PSF in Y"
    assert cen[1] < nominal_cen, "WFE tilt increasing to +X should displace the PSF to -X"
    assert np.allclose(((cen[1] - nominal_cen) * u.pixel * wave.pixelscale).to_value(u.m),
                       ((tilt_angle * u.arcsec).to_value( u.radian) * focal_length).to_value(u.m)), "PSF offset distance did not match expected amount"


# Test: A positive lens has positive WFE
def test_lens_wfe_sign(plot=False):
    """ Test that a positive lens has a positive WFE, and that
    the WFE profile matches the expectations for the parabola increasing toward the edges of the lens
    """

    # Setup lens
    nwaves = 1
    wavelength = 1*u.micron
    lens = poppy.ThinLens(name = "Positive lens, converging", nwaves=nwaves, radius=1)

    #Sample
    npix =33  # use odd number so it passes exactly through the origin
    lens_opd = lens.sample(what='opd', npix=npix, grid_size=2, wavelength=wavelength)

    # Compute expected WFE for comparison
    cen = (npix-1)//2
    x = (np.arange(npix)-cen) / (npix/2)
    y = nwaves*wavelength.to_value(u.m) * x**2
    y += lens_opd.min()

    assert np.allclose(lens_opd[lens_opd.shape[0]//2], y), "Lens WFE did not match expectations for a positive lens"

    if plot:
        plt.figure(figsize=(8,6))
        lens.display(what='both')

        plt.figure()
        plt.imshow(lens_opd, cmap=poppy.conf.cmap_diverging)
        plt.colorbar()
        plt.title("Lens WFE, sampled coarsely")
        plt.figure()
        plt.plot(lens_opd[lens_opd.shape[0]//2])

        plt.plot(y, ls='--')
        plt.axvline(cen, ls=":")
        plt.xlabel("Pixels")
        plt.ylabel("WFE [m]")
        plt.title("Cut along X axis")



# Test: a negative weak lens produces images (before focus) that have consistent orientation with the exit pupil
#       a positive weak lens produces images (after focus) that have the opposite orientation as the exit pupil

# First, define some utility functions for checking which way each of the letter F's is oriented.
def brighter_top_half(image):
    s = image.shape
    top = image[s[0]//2:].sum()
    bot = image[:s[0]//2].sum()
    return top > bot

def brighter_left_half(image):
    return not brighter_top_half(image.transpose())

def test_pupil_orientations_before_and_after_focus(plot=False, npix_pupil=128, npix_fov=128):
    """ Verify pupil orientations before and after focus, and signs of thin lens defocus

    1. A negative weak lens produces images (before focus) that have consistent orientation with the exit pupil
    2. A positive weak lens produces images (after focus) that have the opposite orientation as the exit pupil
    3. Images with the same magnitude but opposite signs of defocus should be 180 degree rotations of one another.

    """

    wave0 = poppy.Wavefront(diam=3*u.m, npix=npix_pupil)
    wave0 *= poppy.LetterFAperture()
    wave1 = wave0.copy()
    wave2 = wave0.copy()

    wave1 *= poppy.ThinLens(nwaves=-5)
    wave1.propagate_to(poppy.Detector(fov_pixels=npix_fov, pixelscale=0.03*u.arcsec/u.pixel))

    wave2 *= poppy.ThinLens(nwaves=+5)
    wave2.propagate_to(poppy.Detector(fov_pixels=npix_fov, pixelscale=0.03*u.arcsec/u.pixel))

    if plot:
        fig, axes = plt.subplots(figsize=(15, 6), ncols=3)
        plt.suptitle("Before and After Focus sign test (Fresnel propagation)", fontweight='bold')

        wave0.display(ax=axes[0])
        wave1.display(imagecrop=fov, title='Intensity at plane before focus', scale='log', ax=axes[1])
        wave2.display(imagecrop=fov, title='Intensity at plane after focus', scale='log', ax=axes[2])

    # check entrance pupil orientation
    assert brighter_top_half(wave0.intensity) and brighter_left_half(wave0.intensity), "Letter F should be brighter at top and left"
    # check orientation of what should be an image before focus
    assert brighter_top_half(wave1.intensity) and brighter_left_half(wave1.intensity), "Image with negative lens (before focus) should have same orientation as the pupil "
    # check orientation of what should be an image after focus
    assert (not brighter_top_half(wave2.intensity)) and (not brighter_left_half(wave2.intensity)), "Image with positive lens (after focus) should have opposite orientation as the pupil "
    # check consistency of defocus diffraction pattern on either side of focus, just with opposite signs (for this no-WFE case)
    assert np.allclose(wave1.intensity, np.rot90(wave2.intensity, 2)), "Positive and negative weak lenses should be 180 degree rotation of one another"

def test_pupil_orientations_before_and_after_focus_fresnel(plot=False, npix_pupil=128, npix_fov=128):
    """ Verify pupil orientations before and after focus, and signs of thin lens defocus

    1. A negative weak lens produces images (before focus) that have consistent orientation with the exit pupil
    2. A positive weak lens produces images (after focus) that have the opposite orientation as the exit pupil
    3. Images with the same magnitude but opposite signs of defocus should be 180 degree rotations of one another.

    """


    wave0 = poppy.FresnelWavefront(beam_radius=1.5*u.m, oversample=2)
    wave0 *= poppy.LetterFAperture(radius = 1/np.sqrt(2)*u.m)

    focal_length = 1.0*u.m
    lens = poppy.QuadraticLens(f_lens=focal_length, name="Converging lens")
    wave0 *= lens

    wave1 = wave0.copy()
    wave2 = wave0.copy()

    fov = 0.00003

    wave1.propagate_fresnel(0.99999*focal_length)

    wave2.propagate_fresnel(1.00001*focal_length)

    if plot:
        fig, axes = plt.subplots(figsize=(15, 6), ncols=3)
        plt.suptitle("Before and After Focus sign test (Fresnel propagation)", fontweight='bold')

        wave0.display(ax=axes[0])
        wave1.display(imagecrop=fov, title='Intensity at plane before focus', scale='log', ax=axes[1])
        wave2.display(imagecrop=fov, title='Intensity at plane after focus', scale='log', ax=axes[2])

    assert brighter_top_half(wave0.intensity) and brighter_left_half(wave0.intensity), "Letter F should be brighter at top and left at pupil"
    assert brighter_top_half(wave1.intensity) and brighter_left_half(wave1.intensity), "Image with negative lens (before focus) should have same orientation as the pupil "
    assert (not brighter_top_half(wave2.intensity)) and (not brighter_left_half(wave2.intensity)), "Image with positive lens (after focus) should have opposite orientation as the pupil "

# Test: Tilts of segments in a segmented DM have the expected effects
def test_segment_tilt_sign_and_direction(display=False):
    hexdm = poppy.HexSegmentedDeformableMirror(flattoflat=0.5 * u.m, rings=1)

    osys2 = poppy.OpticalSystem(pupil_diameter=2 * u.m, npix=128, oversample=1)
    osys2.add_pupil(poppy.MultiHexagonAperture(flattoflat=0.5 * u.m, rings=1, center=True))
    osys2.add_pupil(hexdm)
    osys2.add_detector(0.10, fov_arcsec=10)

    psf_ref = osys2.calc_psf()  # reference, with no tilts

    hexdm.set_actuator(0, 0.2 * u.micron, 0, 0)  # piston
    hexdm.set_actuator(1, 0, 2 * u.arcsec, 0)  # tip
    hexdm.set_actuator(2, 0, 0, 1 * u.arcsec)  # tilt

    if display:
        import matplotlib.pyplot as plt
        hexdm.display(what='opd', colorbar_orientation='vertical', opd_vmax=2e-6)
        plt.figure(figsize=(14, 5))
        plt.suptitle("Segment tilt sign test (Fraunhofer propagation)", fontweight='bold')

    psf2 = osys2.calc_psf(display_intermediates=display)

    diff_psf = psf2[0].data - psf_ref[0].data

    if display:
        plt.figure()
        poppy.display_psf_difference(psf2, psf_ref)

    assert brighter_left_half(diff_psf), 'Tilting a segment with +X tilt WFE should move its PSF to -X'
    assert not brighter_top_half(diff_psf), 'Tilting a segment with +Y tilt WFE should move its PSF to -Y'
