# Tests for individual Optic classes

import matplotlib.pyplot as pl
import numpy as np
import astropy.io.fits as fits
import astropy.units as u

from .. import poppy_core
from .. import optics
from .. import zernike
from .test_core import check_wavefront

wavelength=1e-6



#def test_OpticalElement():
#    pass


#def test_FITSOpticalElement():
#    pass

#def test_Rotation():
#    pass

def test_InverseTransmission():
    """ Verify this inverts the optic throughput appropriately"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    # vary uniform scalar transmission
    for transmission in np.arange(10, dtype=float)/10:

        optic = optics.ScalarTransmission(transmission=transmission)
        inverted = optics.InverseTransmission(optic)
        assert( np.all(  np.abs(optic.get_phasor(wave) - (1-inverted.get_phasor(wave))) < 1e-10 ))

    # vary 2d shape
    for radius in np.arange(10, dtype=float)/10:

        optic = optics.CircularAperture(radius=radius)
        inverted = optics.InverseTransmission(optic)
        assert( np.all(  np.abs(optic.get_phasor(wave) - (1-inverted.get_phasor(wave))) < 1e-10 ))

        assert optic.shape==inverted.shape


#------ Generic Analytic elements -----

def test_scalar_transmission():
    """ Verify this adjusts the wavefront intensity appropriately """
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    for transmission in [1.0, 1.0e-3, 0.0]:

        optic = optics.ScalarTransmission(transmission=transmission)
        assert( np.all(optic.get_phasor(wave) == transmission))



def test_roundtrip_through_FITS():
    """ Verify we can make an analytic element, turn it into a FITS file and back,
    and get the same thing
    """
    optic = optics.ParityTestAperture()
    array = optic.sample(npix=512)

    fitsfile = optic.to_fits(npix=512)
    optic2 = poppy_core.FITSOpticalElement(transmission=fitsfile)

    assert np.all(optic2.amplitude == array), "Arrays before/after casting to FITS file didn't match"


def test_shifting_optics( npix=30,  grid_size = 3, display=False):
    """Test shifting (translation) of Analytic and FITS Optical elements.
    Does shifting work as expected? Is it consistent between the two classes?

    Tests the fix for #247
    """
    import poppy
    pixsize =grid_size/npix
    shift_size = np.round(0.2/pixsize)*pixsize  # by construction, an integer # of pixels

    # Create a reference array
    circ = poppy.CircularAperture()
    circ_samp = circ.sample(npix=npix, grid_size=grid_size)

    # Create a shifted version, and make sure it's different
    circ_shift = poppy.CircularAperture( shift_x=shift_size)
    circ_shift_samp = circ_shift.sample(npix=npix, grid_size=grid_size)

    if display:
        plt.imshow(circ_samp-circ_shift_samp)
    assert np.allclose(circ_samp, circ_shift_samp) is False, "Shift didn't change array"

    # Make a FITS element.
    circ_fits = circ.to_fits(npix=npix, grid_size=grid_size)

    # Show we can shift that and get the same result as shifting the analytic element
    fits_shifted = poppy.FITSOpticalElement(transmission=circ_fits, shift_x=shift_size)
    np.testing.assert_allclose(fits_shifted.amplitude, circ_shift_samp, atol=1e-9,
                                       err_msg="Shifting Analytic and FITS versions are not consistent (v1, via shift_x)")

    # FITSOpticalElement also lets you specify shifts via fraction of the array. Let's
    # show that is  consistent.  This is older syntax that is discouraged, and may be
    # deprecated and removed eventually. But while available it should be correct.
    array_frac = shift_size/grid_size
    fits_shifted_v2 = poppy.FITSOpticalElement(transmission=circ_fits, shift=(array_frac, 0))
    np.testing.assert_allclose(fits_shifted.amplitude, fits_shifted_v2.amplitude, atol=1e-9,
                                       err_msg="Shifting FITS via shift/shift_x are not consistent")
    np.testing.assert_allclose(fits_shifted.amplitude, circ_shift_samp, atol=1e-9,
                                       err_msg="Shifting Analytic and FITS versions are not consistent (v2, via shift)")


    # Check in a 1D cut that the amount of shift is as expected -
    # this is implicitly also checked above via the match of Analytic and FITS
    # which use totally different methods to perform the shift.
    shift_in_pixels = int(shift_size/pixsize)
    assert np.allclose(np.roll(circ_samp[npix//2], shift_in_pixels),
                               circ_shift_samp[npix//2])


def test_shift_rotation_consistency(npix=30, grid_size = 1.5, angle=35, display=False):
    """Test shifting & rotation together for FITS and Analytic optics
    Do we get consistent behavior from each? Are the signs and
    order of operations consistent?

    Tests the fix for issue #275.
    """
    import poppy
    if npix < 30:
        raise ValueError("Need npix>=30 for enough resolution for this test")

    # Create rectangle, rotated
    rect = poppy.RectangleAperture(shift_x=0.25, rotation=angle)
    rect_samp = rect.sample(grid_size=grid_size, npix=npix)

    # Create rectangle, turn into FITS, then rotate
    rect_fits = poppy.RectangleAperture().to_fits(grid_size=grid_size, npix=npix)
    rect2 = poppy.FITSOpticalElement(transmission=rect_fits, shift_x=0.25, rotation=angle)

    # Compare that they are consistent enough, meaning
    # no more than 1% pixel difference. That tolerance allows for the
    # imprecision of rotating low-res binary masks.
    diff = np.round(rect2.amplitude)-rect_samp
    assert np.abs(diff).sum() <= 0.01*rect_samp.sum(), "Shift and rotations differ unexpectedly"

    if display:
        plt.figure()
        plt.subplot(131)
        plt.imshow(rect_samp)
        plt.subplot(132)
        plt.imshow(np.round(rect2.amplitude))
        plt.subplot(133)
        plt.imshow(diff)


#------ Analytic Image Plane elements -----

def test_RectangularFieldStop():
    optic= optics.RectangularFieldStop(width=1, height=10)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1000 # 1/10 of the 1e4 element array


def test_SquareFieldStop():
    optic= optics.SquareFieldStop(size=2)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 400 # 1/10 of the 1e4 element array


def test_CircularPhaseMask():
    import poppy
    optic= optics.CircularPhaseMask(radius=1, retardance=0.25, wavelength=3e-6)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.05, wavelength=3e-6)

    wave*= optic
    assert wave.phase[50,0]==0
    assert wave.phase[50,29]==0
    np.testing.assert_almost_equal(wave.phase[50,30], np.pi/2)
    np.testing.assert_almost_equal(wave.phase[50,50], np.pi/2)
    np.testing.assert_almost_equal(wave.phase[50,69], np.pi/2)
    assert wave.phase[50,70]==0
    assert wave.phase[50,80]==0


def test_BarOcculter():
    optic= optics.BarOcculter(width=1, rotation=0)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 9000 # 9/10 of the 1e4 element array


def test_AnnularFieldStop():
    optic= optics.AnnularFieldStop(radius_inner=1.0, radius_outer=2.0)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    # Just check a handful of points that it goes from 0 to 1 back to 0
    assert wave.intensity[50,50] == 0
    assert wave.intensity[55,50] == 0
    assert wave.intensity[60,50] == 1
    assert wave.intensity[69,50] == 1
    assert wave.intensity[75,50] == 0
    assert wave.intensity[95,50] == 0
    # and check the area is approximately right
    expected_area = np.pi*(optic.radius_outer**2 - optic.radius_inner**2) * 100
    expected_area = expected_area.to(u.arcsec**2).value
    area = wave.intensity.sum()
    assert np.abs(expected_area-area) < 0.01*expected_area


def test_BandLimitedOcculter(halfsize = 5) :
    # For now, just tests the center pixel value.
    # See https://github.com/mperrin/poppy/issues/137

    mask = optics.BandLimitedCoron( kind = 'circular',  sigma = 1.)

    # odd number of pixels; center pixel should be 0
    sample = mask.sample(npix = 2*halfsize+1, grid_size = 10, what = 'amplitude')
    assert sample[halfsize, halfsize] == 0
    assert sample[halfsize, halfsize] != sample[halfsize-1, halfsize]
    assert sample[halfsize+1, halfsize] == sample[halfsize-1, halfsize]

    # even number of pixels; center 4 should be equal
    sample2 = mask.sample(npix = 2*halfsize, grid_size = 10, what = 'amplitude')
    assert sample2[halfsize, halfsize] != 0
    assert sample2[halfsize-1, halfsize-1] == sample2[halfsize, halfsize]
    assert sample2[halfsize-1, halfsize] == sample2[halfsize, halfsize]
    assert sample2[halfsize, halfsize-1] == sample2[halfsize, halfsize]



def test_rotations():
    # Some simple tests of the rotation code on AnalyticOpticalElements. Incomplete!

    # rotating a square by +45 and -45 should give the same result
    ar1 = optics.SquareAperture(rotation=45, size=np.sqrt(2)).sample(npix=256, grid_size=2)
    ar2 = optics.SquareAperture(rotation=-45, size=np.sqrt(2)).sample(npix=256, grid_size=2)
    assert np.allclose(ar1,ar2)

    # rotating a rectangle with flipped side lengths by 90 degrees should give the same result
    fs1 = optics.RectangularFieldStop(width=1, height=10).sample(npix=256, grid_size=10)
    fs2 = optics.RectangularFieldStop(width=10, height=1, rotation=90).sample(npix=256, grid_size=10)
    assert np.allclose(fs1,fs2)

    # check some pixel values for a 45-deg rotated rectangle
    fs3 = optics.RectangularFieldStop(width=10, height=1, rotation=45).sample(npix=200, grid_size=10)
    for i in [50, 100, 150]:
        assert fs3[i, i]==1
        assert fs3[i, i+20]!=1
        assert fs3[i, i-20]!=1

#def test_rotations_RectangularFieldStop():
#
#    # First let's do a rotation of the wavefront itself by 90^0 after an optic
#
#    # now try a 90^0 rotation for the field stop at that optic. Assuming perfect system w/ no aberrations when comparing rsults. ?
#    fs = poppy_core.RectangularFieldStop(width=1, height=10, ang;le=90)
#    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square
#
#    wave*= fs
#    assert wave.shape[0] == 100
#    assert fs.intensity.sum() == 1000 # 1/10 of the 1e4 element array
#
#


#------ Analytic Pupil Plane elements -----

def test_ParityTestAperture():
    """ Verify that this aperture is not symmetric in either direction"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    array = optics.ParityTestAperture().get_phasor(wave)

    assert np.any(array[::-1,:] != array)
    assert np.any(array[:,::-1] != array)


def test_RectangleAperture():
    """ Test rectangular aperture
    based on areas of 2 different rectangles,
    and also that the rotation works to swap the axes
    """
    optic= optics.RectangleAperture(width=5, height=3)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1500 # 50*30 pixels of the 1e4 element array

    optic= optics.RectangleAperture(width=2, height=7)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1400 # 50*30 pixels of the 1e4 element array


    optic1= optics.RectangleAperture(width=2, height=7, rotation=90)
    optic2= optics.RectangleAperture(width=7, height=2)
    wave1 = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave2= poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave1*= optic1
    wave2*= optic2

    assert wave1.shape[0] == 100
    assert np.all(np.abs(wave1.intensity - wave2.intensity) < 1e-6)




def test_HexagonAperture(display=False):
    """ Tests creating hexagonal aperture """

    # should make hexagon PSF and compare to analytic expression
    optic= optics.HexagonAperture(side=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: optic.display()

def test_MultiHexagonAperture(display=False):
    # should make multihexagon PSF and compare to analytic expression
    optic= optics.MultiHexagonAperture(side=1, rings=2)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: optic.display()


def test_NgonAperture(display=False):
    """ Test n-gon aperture

    Note we could better test this if we impemented symmetry checks using the rotation argument?
    """
    # should make n-gon PSF for n=4, 6 and compare to square and hex apertures
    optic= optics.NgonAperture(nsides=4, radius=1, rotation=45)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display:
        pl.subplot(131)
        optic.display()

    optic= optics.NgonAperture(nsides=5, radius=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display:
        pl.subplot(132)
        optic.display()



    optic= optics.NgonAperture(nsides=6, radius=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display:
        pl.subplot(133)
        optic.display()



def test_ObscuredCircularAperture_Airy(display=False):
    """ Compare analytic 2d Airy function with the results of a POPPY
    numerical calculation of the PSF for a circular aperture.

    Note that we expect very close but not precisely perfect agreement due to
    the quantization of the POPPY PSF relative to a perfect geometric circle.
    """

    from ..misc import airy_2d

    pri_diam = 1
    sec_diam = 0.4
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=pri_diam, obscuration=sec_diam/pri_diam)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    osys.add_pupil(
            optics.CompoundAnalyticOptic( [optics.CircularAperture(radius=pri_diam/2) ,
                                           optics.SecondaryObscuration(secondary_radius=sec_diam/2, n_supports=0) ]) )
    osys.add_detector(pixelscale=0.010,fov_pixels=512, oversample=1)
    numeric = osys.calc_psf(wavelength=1.0e-6, display=False)

    # Comparison
    difference = numeric[0].data-analytic
    #assert np.all(np.abs(difference) < 3e-5)


    if display:
        from .. import utils
        #comparison of the two
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=1e-6, vmax=1e-2)

        pl.figure(figsize=(15,5))
        pl.subplot(141)
        ax1=pl.imshow(analytic, norm=norm)
        pl.title("Analytic")
        pl.subplot(142)
        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_PSF(numeric, vmin=1e-6, vmax=1e-2, colorbar=False)
        pl.title("Numeric")
        pl.subplot(143)
        ax2=pl.imshow(numeric[0].data-analytic, norm=norm)
        pl.title("Difference N-A")
        pl.subplot(144)
        ax2=pl.imshow(np.abs(numeric[0].data-analytic) < 1e-4)
        pl.title("Difference <1e-4")


def test_CompoundAnalyticOptic(display=False):
    wavelen = 2e-6
    nwaves = 2
    r = 3

    # First test the "and" mergemode

    osys_compound = poppy_core.OpticalSystem()
    osys_compound.add_pupil(
        optics.CompoundAnalyticOptic([
            optics.CircularAperture(radius=r),
            optics.ThinLens(nwaves=nwaves, reference_wavelength=wavelen,
                            radius=r)
        ]
        , mergemode='and')
    )
    osys_compound.add_detector(pixelscale=0.010, fov_pixels=512, oversample=1)
    psf_compound = osys_compound.calc_psf(wavelength=wavelen, display=False)

    osys_separate = poppy_core.OpticalSystem()
    osys_separate.add_pupil(optics.CircularAperture(radius=r))    # pupil radius in meters
    osys_separate.add_pupil(optics.ThinLens(nwaves=nwaves, reference_wavelength=wavelen,
                                           radius=r))
    osys_separate.add_detector(pixelscale=0.010, fov_pixels=512, oversample=1)
    psf_separate = osys_separate.calc_psf(wavelength=wavelen, display=False)

    if display:
        from matplotlib import pyplot as plt
        from poppy import utils
        plt.figure()
        plt.subplot(1, 2, 1)
        utils.display_PSF(psf_separate, title='From Separate Optics (and)')
        plt.subplot(1, 2, 2)
        utils.display_PSF(psf_compound, title='From Compound Optics (and)')

    difference = psf_compound[0].data - psf_separate[0].data

    assert np.all(np.abs(difference) < 1e-3)

    # Next test the 'or' mergemode
    # This creates two overlapping RectangleAperture with different
    # heights and check that the result equals the larger

    #TODO this fails.  Looks like the resulting aperture is too small when doing calc_psf.

    w = 1.0
    h1=2.0 
    h2=0.5

    osys_compound = poppy_core.OpticalSystem()
    osys_c_pupil = optics.CompoundAnalyticOptic([
            optics.RectangleAperture(width=w, height=h1),
            optics.RectangleAperture(width=w, height=h2)
        ]
        , mergemode='or')
    osys_compound.add_pupil(
        osys_c_pupil
    )
    osys_compound.add_detector(pixelscale=0.010, fov_pixels=512, oversample=1)
    psf_compound, ints_compound = osys_compound.calc_psf(wavelength=wavelen, display=False, return_intermediates=True)

    osys_separate = poppy_core.OpticalSystem()
    osys_s_pupil = optics.RectangleAperture(width=w, height=max(h1, h2))
    osys_separate.add_pupil(osys_s_pupil)
    osys_separate.add_detector(pixelscale=0.010, fov_pixels=512, oversample=1)
    psf_separate, ints_separate = osys_separate.calc_psf(wavelength=wavelen, display=False, return_intermediates=True)
    if display: 
        #from matplotlib import pyplot as plt
        #from poppy import utils
        plt.figure()
        osys_s_pupil.display(title='Separate pupil (or)')
        plt.figure()
        osys_c_pupil.display(title='Compound pupil (or)')
        plt.figure()
        ints_separate[0].display(title='Separate wave[0] (or)')
        plt.figure()
        ints_compound[0].display(title='Compound wave[0] (or)')
        plt.figure()
        utils.display_PSF(psf_separate, title='From Separate Optics (or)')
        plt.figure()
        utils.display_PSF(psf_compound, title='From Compound Optics (or)')

    #check transmission of OpticalElement objects
    # PASSES commit 1e4709b
    testwave = poppy_core.Wavefront(wavelength=wavelen,npix=1024)
    diff_trans = osys_c_pupil.get_transmission(testwave) - osys_s_pupil.get_transmission(testwave)
    
    assert np.all(np.abs(diff_trans) < 1e-3)
    
    #check pupil amplitudes
    # FAILS commit 1e4709b
    diff_amp = ints_compound[0].amplitude - ints_separate[0].amplitude
    
    assert np.all(np.abs(diff_amp) < 1e-3)
    
    #check psf
    # FAILS commit 1e4709b
    difference = psf_compound[0].data - psf_separate[0].data

    assert np.all(np.abs(difference) < 1e-3)

def test_AsymmetricObscuredAperture(display=False):
    """  Test that we can run the code with asymmetric spiders
    """

    from ..misc import airy_2d

    pri_diam = 1
    sec_diam = 0.4
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=pri_diam, obscuration=sec_diam/pri_diam)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    osys.add_pupil(
            optics.CompoundAnalyticOptic( [optics.CircularAperture(radius=pri_diam/2) ,
                                           optics.AsymmetricSecondaryObscuration(secondary_radius=sec_diam/2, support_angle=[0,150,210], support_width=0.1) ]) )
    osys.add_detector(pixelscale=0.030,fov_pixels=512, oversample=1)
    if display: osys.display()
    numeric = osys.calc_psf(wavelength=1.0e-6, display=False)

    # Comparison
    difference = numeric[0].data-analytic
    #assert np.all(np.abs(difference) < 3e-5)


    if display:
        from .. import utils
        #from matplotlib.colors import LogNorm
        #norm = LogNorm(vmin=1e-6, vmax=1e-2)

        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_PSF(numeric, vmin=1e-8, vmax=1e-2, colorbar=False)
        #pl.title("Numeric")

def test_GaussianAperture(display=False):
    """ Test the Gaussian aperture """

    ga = optics.GaussianAperture(fwhm=1)
    w = poppy_core.Wavefront(npix=101) # enforce odd npix so there is a pixel at the exact center

    w *= ga

    assert(ga.w == ga.fwhm/(2*np.sqrt(np.log(2))))

    assert(w.intensity.max() ==1)


    # now mock up a wavefront with very specific coordinate values
    # namely the origin, one HWHM away, and one w or sigma away.
    class mock_wavefront(poppy_core.Wavefront):
        def __init__(self, *args, **kwargs):
            #super(poppy.Wavefront, self).__init__(*args, **kwargs) # super does not work for some reason?
            poppy_core.Wavefront.__init__(self, *args, **kwargs)

            self.wavefront = np.ones(5, dtype=np.complex128)
            self.planetype=poppy_core.PlaneType.pupil
            self.pixelscale = 0.5
        def coordinates(self):
            w = ga.w.to(u.meter).value
            return (np.asarray([0,0.5, 0.0, w, 0.0]), np.asarray([0, 0, 0.5, 0, -w ]))

    trickwave = mock_wavefront()
    trickwave *= ga
    assert(trickwave.amplitude[0]==1)
    assert(np.allclose(trickwave.amplitude[1:3], 0.5))
    assert(np.allclose(trickwave.amplitude[3:5], np.exp(-1)))


def test_ThinLens(display=False):
    pupil_radius = 1

    pupil = optics.CircularAperture(radius=pupil_radius)
    # let's add < 1 wave here so we don't have to worry about wrapping
    lens = optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6, radius=pupil_radius)
    # n.b. npix is 99 so that there are an integer number of pixels per meter (hence multiple of 3)
    # and there is a central pixel at 0,0 (hence odd npix)
    # Otherwise the strict test against half a wave min max doesn't work
    # because we're missing some (tiny but nonzero) part of the aperture
    wave = poppy_core.Wavefront(npix=99, diam=3.0, wavelength=1e-6)
    wave *= pupil
    wave *= lens

    assert np.allclose(wave.phase.max(),  np.pi/2)
    assert np.allclose(wave.phase.min(), -np.pi/2)

    # regression test to ensure null optical elements don't change ThinLens behavior
    # see https://github.com/mperrin/poppy/issues/14
    osys = poppy_core.OpticalSystem()
    osys.add_pupil(optics.CircularAperture(radius=1))
    for i in range(3):
        osys.add_image()
        osys.add_pupil()

    osys.add_pupil(optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6,
                                  radius=pupil_radius))
    osys.add_detector(pixelscale=0.01, fov_arcsec=3.0)
    psf = osys.calc_psf(wavelength=1e-6)

    osys2 = poppy_core.OpticalSystem()
    osys2.add_pupil(optics.CircularAperture(radius=1))
    osys2.add_pupil(optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6,
                                   radius=pupil_radius))
    osys2.add_detector(pixelscale=0.01, fov_arcsec=3.0)
    psf2 = osys2.calc_psf()


    if display:
        import poppy
        poppy.display_PSF(psf)
        poppy.display_PSF(psf2)

    assert np.allclose(psf[0].data,psf2[0].data), (
        "ThinLens shouldn't be affected by null optical elements! Introducing extra image planes "
        "made the output PSFs differ beyond numerical tolerances."
    )

def test_fixed_sampling_optic():
    optic= optics.HexagonAperture(side=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square

    array_optic= optics.fixed_sampling_optic(optic, wave)

    assert np.allclose(array_optic.amplitude, optic.get_transmission(wave)), 'mismatch between original and fixed sampling version'
