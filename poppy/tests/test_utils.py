import warnings
import numpy as np
import astropy.io.fits as fits
import pytest

try:
    import pyfftw
except ImportError:
    pyfftw = None

from .. import utils
from .. import poppy_core
import poppy
import scipy

def test_pad_to_size():

    for starting_shape in [(20,20), (21,21), (300,300), (128,256)]:

        square = np.ones(starting_shape)

        for desiredshape in [ (500, 500), (400,632), (2048, 312)]:
            newshape = utils.pad_to_size(square, desiredshape).shape
            for i in [0,1]:
                assert newshape[i] == desiredshape[i], "Error padding from {} to {}".format(starting_shape, desired_shape)




# Utility function used in test_measure_FWHM
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    From https://gist.github.com/andrewgiessel/4635563
    As variously modified by marshall
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = (size-1)*.5
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def test_radial_profile(plot=False):
    """ Test radial profile calculation, including circular and square apertures,
    and including with the pa_range option.
    """

    ### Tests on a circular aperture

    o = poppy_core.OpticalSystem()
    o.add_pupil(poppy.CircularAperture(radius=1.0))
    o.add_detector(0.010, fov_pixels=512)
    psf = o.calc_psf()

    rad, prof = poppy.radial_profile(psf)
    rad2, prof2 = poppy.radial_profile(psf, pa_range=[-20,20])
    rad3, prof3 = poppy.radial_profile(psf, pa_range=[-20+90, 20+90])


    # Compute analytical Airy function, on exact same radial sampling as that profile.
    v = np.pi*  rad*poppy.misc._ARCSECtoRAD * 2.0/1e-06
    airy = ((2*scipy.special.jn(1, v))/v)**2
    r0 = 33 # 0.33 arcsec ~ first airy ring in this case.
    airy_peak_envelope = airy[r0]*prof.max() / (rad/rad[r0])**3

    absdiff =  np.abs(prof - airy*prof.max())

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        poppy.display_psf(psf, colorbar_orientation='horizontal', title='Circular Aperture, d=2 m')

        plt.subplot(1,2,2)
        plt.semilogy(rad,prof)
        plt.semilogy(rad2,prof2, ls='--', color='red')
        plt.semilogy(rad3,prof3, ls=':', color='cyan')
        plt.semilogy(rad, airy_peak_envelope, color='gray')
        plt.semilogy(rad, airy_peak_envelope/50, color='gray', alpha=0.5)


        plt.semilogy(rad, absdiff, color='purple')

    # Test the radial profile is close to the analytical Airy function.
    # It's hard to define relative fractional closeness for comparisons to
    # a function with many zero crossings; we can't just take (f1-f2)/(f1+f2)
    # This is a bit of a hack but let's test that the difference between
    # numerical and analytical is always less than 1/50th of the peaks of the
    # Airy function (fit based on the 1/r^3 power law fall off)

    assert np.all( absdiff[0:300] < airy_peak_envelope[0:300]/50)

    # Test that the partial radial profiles agree with the full one. This test is
    # a little tricky since the sampling in r may not agree exactly.
    # TODO write test comparison here

    # Let's also test that the partial radial profiles on 90 degrees agree with each other.
    # These should match to machine precision.
    assert np.allclose(prof2, prof3)

    ### Next test is on a square aperture
    o = poppy.OpticalSystem()
    o.add_pupil(poppy.SquareAperture())
    o.add_detector(0.010, fov_pixels=512)
    psf = o.calc_psf()
    rad, prof = poppy.radial_profile(psf)
    rad2, prof2 = poppy.radial_profile(psf, pa_range=[-20,20])
    rad3, prof3 = poppy.radial_profile(psf, pa_range=[-20+90, 20+90])


    if plot:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        poppy.display_psf(psf, colorbar_orientation='horizontal', title='Square Aperture, size=1 m')

        plt.subplot(1,2,2)
        plt.semilogy(rad,prof)
        plt.semilogy(rad2,prof2, ls='--', color='red')
        plt.semilogy(rad3,prof3, ls=':', color='cyan')

    assert np.allclose(prof2, prof3)
    # TODO compare those to be near a sinc profile as expected?

def test_radial_profile_of_offset_source():
    """Test that we can compute the radial profile for a source slightly outside the FOV,
    compare that to a calculation for a centered source, and check we get consistent results
    for the overlapping range of the radius parameter space.

    Also, make a plot showing the consistency.
    """
    import matplotlib.pyplot as plt

    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture(radius=1.0))
    osys.add_detector(pixelscale=0.01, fov_pixels=128)

    # compute centered PSF
    psf0 = osys.calc_psf()

    # Compute a PSF with the source offset
    osys.source_offset_r = 1.0 # outside of FOV
    psf1 = osys.calc_psf()

    # Calculate the radial profiles of those two PSFs
    r0,p0 = poppy.radial_profile(psf0)
    # For the offset PSF, compute apparent coordinates of the offset source in that image
    # (this will be a 'virtual' pixel value outside of the FOV)
    halfsize = psf1[0].header['NAXIS1']//2
    offset_ypos_in_pixels = osys.source_offset_r / psf1[0].header['PIXELSCL'] + halfsize
    offset_target_center_pixels = (halfsize, offset_ypos_in_pixels)
    r1,p1 = poppy.radial_profile(psf1, center=offset_target_center_pixels)

    fig, axes = plt.subplots(figsize=(16,5), ncols=3)
    poppy.display_psf(psf0, ax=axes[0], title='Centered', colorbar_orientation='horizontal')
    poppy.display_psf(psf1, ax=axes[1], title='Offset', colorbar_orientation='horizontal')
    axes[2].semilogy(r0,p0)
    axes[2].semilogy(r1,p1)

    # Measure radial profiles as interpolator objects, so we can evaluate them at identical radii
    prof0 = poppy.measure_radial(psf0)
    prof1 = poppy.measure_radial(psf1, center=offset_target_center_pixels)

    # Test consistency of those two radial profiles at various radii within the overlap region
    test_radii = np.linspace(0.4, 0.8, 7)
    for rad in test_radii:
        print(prof0(rad), prof1(rad))
        axes[2].axvline(rad, ls=':', color='black')

        # Check PSF agreement within 10%;
        # also add an absolute tolerance since relative error can be higher for radii right on the dark Airy nuls
        assert np.allclose(prof0(rad), prof1(rad), rtol=0.1, atol=5e-8), "Disagreement between centered and offset radial profiles"


def test_measure_FWHM(display=False, verbose=False):
    """ Test the utils.measure_FWHM function

    Current implementation can be off by a
    couple percent for small FWHMs that are only
    marginally well sampled by the array pixels, so
    the allowed tolerance for measured_fwhm = input_fwhm
    is that it's allowed to be off by a couple percent.

    """

    # Test the basic output on simple Gaussian arrays
    desired = (3, 4.5, 5, 8, 12)
    tolerance= 0.01

    for des in desired:


        desired_fwhm = des #4.0 # pixels
        pxscl = 0.010

        center=(24.5,26.25)
        ar = makeGaussian(50, fwhm=desired_fwhm, center=center)

        testfits = fits.HDUList(fits.PrimaryHDU(ar))
        testfits[0].header['PIXELSCL'] = pxscl

        meas_fwhm = utils.measure_fwhm(testfits, center=center)
        if verbose:
            print("Measured FWHM: {0:.4f} arcsec, {1:.4f} pixels ".format(meas_fwhm, meas_fwhm/pxscl))

        reldiff =  np.abs((meas_fwhm/pxscl) - desired_fwhm ) / desired_fwhm
        result = "Measured: {3:.4f} pixels; Desired: {0:.4f} pixels. Relative difference: {1:.4f}    Tolerance: {2:.4f}".format(desired_fwhm, reldiff, tolerance, meas_fwhm/pxscl)
        if verbose:
            print(result)
        assert reldiff < tolerance, result

    # Test on Poppy outputs too
    # We test both well sampled and barely sampled cases.
    # In this test case the FWHM is 0.206265 arcsec, so pixel scale up to 0.2 arcsec.
    pixscales = [0.01, 0.1, 0.2]
    # We allow slightly worse accurance for less well sampled data
    tolerances= [0.01, 0.015, 0.04]

    for pixscale, tolerance in zip(pixscales, tolerances):

        import astropy.units as u
        o = poppy.OpticalSystem()
        o.add_pupil(poppy.CircularAperture(radius=0.5*u.m))
        o.add_detector(pixscale, fov_pixels=128)
        psf = o.calc_psf(wavelength=1*u.micron)

        meas_fwhm = poppy.measure_fwhm(psf)
        expected_fwhm = ((1*u.micron/(1*u.m)).decompose().value*u.radian).to(u.arcsec).value

        reldiff =  np.abs((meas_fwhm - expected_fwhm ) / expected_fwhm)

        result = "Measured: {3:.4f} arcsec; Desired: {0:.4f} arcsec. Relative difference: {1:.4f}    Tolerance: {2:.4f}".format(expected_fwhm, reldiff, tolerance, meas_fwhm)

        assert reldiff < tolerance, result


def test_measure_radius_at_ee():
    """ Test the function measure_radius_at_ee in poppy/utils.py which measures the encircled
    energy vs radius and return as an interpolator.
    """

    # Tests on a circular aperture
    o = poppy.OpticalSystem()
    o.add_pupil(poppy.CircularAperture())
    o.add_detector(0.010, fov_pixels=512)
    psf = o.calc_psf()

    # Create outputs of the 2 inverse functions
    rad = utils.measure_radius_at_ee(psf)
    ee = utils.measure_ee(psf)

    # The ee and rad functions should undo each other and yield the input value
    for i in [0.1, 0.5, 0.8]:
        np.testing.assert_almost_equal(i, ee(rad(i)), decimal=3, err_msg="Error: Values not equal")


    # Repeat test with normalization to psf sum=1.
    # This time we can go right up to 1.0, or at least arbitrarilyclose to it.
    rad = utils.measure_radius_at_ee(psf, normalize='total')
    ee = utils.measure_ee(psf, normalize='total')
    for i in [0.1, 0.5,  0.9999]:
        np.testing.assert_almost_equal(i, ee(rad(i)), decimal=3, err_msg="Error: Values not equal")




@pytest.mark.skipif(pyfftw is None, reason="pyFFTW not found")
def test_load_save_fftw_wisdom(tmpdir):
    with tmpdir.as_cwd():
        utils.fftw_save_wisdom('./wisdom.json')
    assert tmpdir.join('wisdom.json').exists()
    with tmpdir.as_cwd():
        utils.fftw_load_wisdom('./wisdom.json')
    assert utils._loaded_fftw_wisdom is True

@pytest.mark.skipif(pyfftw is None, reason="pyFFTW not found")
def test_load_corrupt_fftw_wisdom(tmpdir):
    utils._loaded_fftw_wisdom = False
    with tmpdir.as_cwd():
        with open('./wisdom.json', 'w') as f:
            f.write('''{"longdouble": "(fftw-3.3.4 fftwl_wisdom #x0821b5c7 #xa4c07d5a #x21b58211 #xebe513ab\\n)\\n", "single": "(fftw-3.3.4 fftwf_wisdom #xa84d9475 #xdb220970 #x4aa6f1c4 #xf3163254\\n)\\n", "_FFTW_INIT":''')
        assert tmpdir.join('wisdom.json').exists()
        with warnings.catch_warnings(record=True) as w:
            utils.fftw_load_wisdom('./wisdom.json')
            assert len(w) == 1
            assert issubclass(w[-1].category, utils.FFTWWisdomWarning)
    assert utils._loaded_fftw_wisdom is False
