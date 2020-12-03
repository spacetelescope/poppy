import numpy as np
import astropy.units as u
import poppy


####################################
# Test sign conventions
####################################

# Test: positive OPD in an optic produces positive WFE in a wave after interacting with that optic
#       (in the absence of phase wrapping)

def test_wfe_opd_consistency():
    """ Verify that the sign and amplitude of wavefront error matches that of an optic's OPD,
    for cases that do NOT encounter phase wrapping.

    """

    for opd_amount in (100*u.nm, -0.25*u.micron, 1e-8*u.m):
        constant_opd = poppy.ScalarOpticalPathDifference(opd=opd_amount)
        wave = poppy.Wavefront(wavelength=1*u.micron, npix=64)

        wave *= constant_opd

        assert np.allclose(constant_opd.opd, wave.wfe), "optic OPD and wavefront WFE should have consistent signs"


# Test: Tilt with increasing WFE towards the +X direction moves the PSF in the -X direction

def test_wavefront_tilt_sign_and_direction(display=False, npix=128):
    """ Test that tilt with increasing WFE towards the +X direction moves the PSF in the -X direction

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

    if display:
        wave.display(what='both')

    wave.propagate_to(poppy.Detector(pixelscale=0.05, fov_pixels=128))

    if display:
        plt.figure()
        wave.display(what='both', crosshairs=True, imagecrop=2)

    n = wave.shape[0]
    cen = poppy.measure_centroid(wave.as_fits())
    assert np.allclose(cen[0], (n-1)/2), "Tilt in X should not displace the PSF in Y"
    assert cen[1] < (n-1)/2, "WFE tilt increasing to +X should displace the PSF to -X"
    assert np.allclose(((cen[1]-(n-1)/2)*u.pixel*wave.pixelscale).to_value(u.arcsec), tilt_angle), "PSF offset did not match expected amount"



# Test: A positive lens has positive WFE
def test_lens_wfe_sign(display=False):
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

    if display:
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


# Test: Tilts of segments in a segmented DM have the expected effects
#       Implemented in the Sign Conventions notebook, could be moved here but not yet done.

# Test: a negative weak lens produces images (before focus) that have consistent orientation with the exit pupil
#       a positive weak lens produces images (after focus) that have the opposite orientation as the exit pupil
def test_pupil_orientations_before_and_after_focus(display=False, npix_pupil=128, npix_fov=128):
    """ Verify pupil orientations before and after focus, and signs of thin lens defocus

    1. A negative weak lens produces images (before focus) that have consistent orientation with the exit pupil
    2. A positive weak lens produces images (after focus) that have the opposite orientation as the exit pupil
    3. Images with the same magnitude but opposite signs of defocus should be 180 degree rotations of one another.

    """

    wave = poppy.Wavefront(diam=3*u.m, npix=npix_pupil)
    wave *= poppy.LetterFAperture()
    wave0 = wave.copy()
    wave2 = wave.copy()

    wave *= poppy.ThinLens(nwaves=-5)
    wave.propagate_to(poppy.Detector(fov_pixels=npix_fov, pixelscale=0.03*u.arcsec/u.pixel))

    wave2 *= poppy.ThinLens(nwaves=+5)
    wave2.propagate_to(poppy.Detector(fov_pixels=npix_fov, pixelscale=0.03*u.arcsec/u.pixel))

    if display:
        wave0.display()
        plt.figure()
        wave.display(imagecrop=30, title='Intensity at detector plane with\nnegative thin lens at pupil\n(before focus)')
        plt.figure()
        wave2.display(imagecrop=30, title='Intensity at detector plane with\npositive thin lens at pupil\n(after focus)')

    # Define some utility functions for checking which way each of the letter F's is oriented.
    def brighter_top_half(image):
        s = image.shape
        top = image[s[0]//2:].sum()
        bot = image[:s[0]//2].sum()
        return top > bot

    def brighter_left_half(image):
        return not brighter_top_half(image.transpose())

    # check entrance pupil orientation
    assert brighter_top_half(wave0.intensity) and brighter_left_half(wave0.intensity), "Letter F should be brighter at top and left"
    # check orientation of what should be an image before focus
    assert brighter_top_half(wave.intensity) and brighter_left_half(wave.intensity), "Image with negative lens (before focus) should have same orientation as the pupil "
    # check orientation of what should be an image after focus
    assert (not brighter_top_half(wave2.intensity)) and (not brighter_left_half(wave2.intensity)), "Image with positive lens (after focus) should have opposite orientation as the pupil "
    # check consistency of defocus diffraction pattern on either side of focus, just with opposite signs (for this no-WFE case)
    assert np.allclose(wave.intensity, np.rot90(wave2.intensity, 2)), "Positive and negative weak lenses should be 180 degree rotation of one another"



