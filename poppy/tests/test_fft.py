# Tests for FFT based propagation

import numpy as np
import astropy.io.fits as fits

import pytest

from .. import poppy_core
from .. import optics
from .. import accel_math
from .test_core import check_wavefront


# For some reason, the following block of code in poppy_core is not sufficient
# to handle the "no pyfftw installed" case when running in test mode, even though it works
# just fine when actually running poppy. Empirically this check has to be repeated
# here in order to let the tests work when pyfftw is not present.
from .. import conf
if conf.use_fftw:
    try:
        # try to import FFTW and use it
        import pyfftw
    except:
        # we tried but failed to import it.
        conf.use_fftw = False


wavelen = 1e-6
radius = 6.5/2


def test_fft_normalization():
    """ Test the PSF normalization for FFTs"""

    poppy_core._log.info('TEST: test_fft_normalization')

    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=radius)
    osys.add_pupil(pupil)
    osys.add_image() # null plane to force FFT
    osys.add_pupil() # null plane to force FFT
    osys.add_detector(pixelscale=0.01, fov_arcsec=10.0) # use a large FOV so we grab essentially all the ligh

    poppy_core._log.info('TEST: wavelen = {0}, radius = {1}'.format(wavelen, radius))

    psf, waves = osys.calc_psf(wavelength=2.0e-6, normalize='first', return_intermediates=True)

    for i in range(3):
        assert np.isclose(waves[i].total_intensity, 1.0), f'intensity was not conserved for plane {i}'

    # Expected value here is 0.9977, limited by FOV size as the aperture
    poppy_core._log.info('TEST: Computed PSF of circular aperture')
    poppy_core._log.info('TEST: PSF total intensity sum is {0}'.format(psf[0].data.sum()))
    poppy_core._log.info('TEST:  Expected value is 0.9977 ')

    assert abs(psf[0].data.sum() - 0.9977) < 0.001


def test_fft_blc_coronagraph():
    """ Test that a simple band limited coronagraph blocks most of the light """

    lyot_radius = 6.5/2.5
    osys = poppy_core.OpticalSystem("test", oversample=2)
    osys.add_pupil( optics.CircularAperture(radius=radius, pad_factor=1.5) )
    osys.add_image()
    osys.add_image( optics.BandLimitedCoron( kind='circular', sigma=5.0))
    osys.add_pupil()
    osys.add_pupil( optics.CircularAperture(radius=lyot_radius) )
    osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)

    psf, int_wfs = osys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=True)


    # after the Lyot plane, the wavefront should be all real.
    lyot_wf = int_wfs[-2]
    lyot_wf_fits = lyot_wf.as_fits(what='all') # need to save this for the multiwave comparison in test_3_multiwave()
    assert check_wavefront(lyot_wf_fits, test='is_real', comment='(Lyot Plane)')

    # and the flux should be low.
    assert psf[0].data.sum() <  0.005 #1e-4
                                      # MDP note: sheepishly I must admit I have lost track of why I set the
                                      # expected value here to 1e-4 in some previous version. That does not
                                      # appear to be the correct value as of 2014 August and so I am updating
                                      # this, but leave this note as a TODO that this needs some attention/validation
                                      # at some future point.


def test_fft_fqpm(): #oversample=2, verbose=True, wavelength=2e-6):
    """ Test FQPM plus field mask together. The check is that there should be very low flux in the final image plane
    Perfect circular case  with FQPM with fieldMask
    Test  ideal FQPM, with field mask. Verify proper behavior in Lyot plane"""


    oversamp=2
    osys = poppy_core.OpticalSystem("test", oversample=oversamp)
    osys.add_pupil(optics.CircularAperture(radius=radius))
    osys.add_pupil(optics.FQPM_FFT_aligner())
    osys.add_image(optics.IdealFQPM(wavelength=wavelen))  # perfect FQPM for this wavelength
    osys.add_image(optics.RectangularFieldStop(width=6.0))
    osys.add_pupil(optics.FQPM_FFT_aligner(direction='backward'))
    osys.add_pupil(optics.CircularAperture(radius=radius))
    osys.add_detector(pixelscale=0.01, fov_arcsec=10.0)

    psf = osys.calc_psf(wavelength=wavelen)
    assert psf[0].data.sum() < 0.002


def test_parity_FFT_forward_inverse(display=False):
    """ Test that transforming from a pupil, to an image, and back to the pupil
    leaves you with the same pupil as you had in the first place.

    In other words it doesn't flip left/right or up/down etc.

    See https://github.com/mperrin/webbpsf/issues/35
    That was for the MFT, but for thoroughness let's test both FFT and MFT
    to demonstrate proper behavior

    **  See also: test_matrixDFT.test_parity_MFT_forward_inverse() for a  **
    **  parallel function to this.                                        **

    """

    # set up optical system with 2 pupil planes and 2 image planes
    sys = poppy_core.OpticalSystem(oversample=1)
    sys.add_pupil(optics.ParityTestAperture())
    sys.add_image()
    sys.add_pupil()
    sys.add_detector(pixelscale=0.010, fov_arcsec=1)

    psf, planes = sys.calc_psf(display=display, return_intermediates=True)

    # the wavefronts are padded by 0s. With the current API the most convenient
    # way to ensure we get unpadded versions is via the as_fits function.
    p0 = planes[0].as_fits(what='intensity', includepadding=False)
    p2 = planes[2].as_fits(what='intensity', includepadding=False)

    # To confirm the parity is consistent,
    # Let's check the difference is smaller than if it were flipped around
    assert (np.abs(p0[0].data[512] - p2[0].data[512]).sum()  <
            np.abs(p0[0].data[512] - p2[0].data[512][::-1]).sum() ), ("Difference "+
            "appears worse than if the parity were flipped")


    # for checking the overall parity it's sufficient to check the intensity.
    # we can have arbitrarily large differences in phase for regions with
    # intensity =0, so don't check the complex field or phase here.


    thresh = 1e-10

    absdiff = (np.abs(p0[0].data - p2[0].data))
    maxabsdiff = np.max(absdiff)
    assert (maxabsdiff < thresh)


    if display:
        import matplotlib.pyplot as plt
        nplanes = len(planes)
        for i, plane in enumerate(planes):
            ax = plt.subplot(2,nplanes,i+1)
            plane.display(ax = ax)
            plt.title("Plane {0}".format(i))
        plt.subplot(2,nplanes,nplanes+1)
        plt.imshow(absdiff)
        plt.title("Abs(Pupil0-Pupil2)")
        plt.colorbar()
        print("Max abs(difference) = {}".format(maxabsdiff))
    return (p0,p2)


############################################################################
# Test different FFT algorithms for consistency:
#    - numpy
#    - fftw
#    - CUDA / pyculib
#    - OpenCL / clfft / gpyfft
#
# The test method is the same for all: calculate PSFs for an
# optical system with both forward and backward FFTs, then compare
# results for consistency (within a factor of 5 of machine precision)
# applying the check to both the final PSFs and the intermediate WFs.
############################################################################

def setup_test_osys():
    """ Create test case optical system for the FFT routines
    used in the below tests."""
    ap = optics.ParityTestAperture()
    sys = poppy_core.OpticalSystem()
    sys.add_pupil(ap)
    sys.add_image()
    sys.add_pupil(ap)
    sys.add_detector(0.02, fov_pixels=512)  # fairly arbitrary, but big enough to get most of the flux
    return sys


@pytest.mark.skipif(accel_math._FFTW_AVAILABLE is False, reason="FFTW not available")
def test_pyfftw_vs_numpyfft(verbose=False):
    """ Create an optical system with 2 parity test apertures,
    propagate light through it, and compare that we get the same results from both numpy and pyfftw"""

    defaults = conf.use_fftw, conf.use_cuda, conf.use_opencl

    conf.use_cuda = False
    conf.use_opencl = False

    sys = setup_test_osys()

    conf.use_fftw = False
    psf_numpy, intermediates_numpy = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    conf.use_fftw = True
    psf_fftw, intermediates_fftw = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    # check the final PSFs are consistent
    assert np.abs(psf_fftw[0].data-psf_numpy[0].data).max() < 1e-6

    # Check flux conservation for the intermediate arrays behaves the same for both
    intermediates = intermediates_fftw
    epsilon = np.finfo(intermediates[0].wavefront.dtype).eps
    total_int_input = intermediates_numpy[0].total_intensity
    for i in [1,2]:
        assert np.abs(intermediates[i].total_intensity - total_int_input) < 5*epsilon

    # Check flux in output array is about 0.5% less than input array (due to finite FOV)
    expected = 0.004949550538272617927759
    assert np.abs(intermediates[3].total_intensity - total_int_input) - expected < 5*epsilon

    if verbose:
        print ("PSF difference: ", np.abs(psf_fftw[0].data-psf_numpy[0].data).max())
        for i in [1,2]:
            print(" Int. WF {} intensity diff: {}".format(i, np.abs(intermediates[i].total_intensity-total_int_input)) )
        print(" Final PSF intensity diff:", np.abs(intermediates[3].total_intensity-total_int_input) - expected)

    conf.use_fftw, conf.use_cuda, conf.use_opencl = defaults


@pytest.mark.skipif(accel_math._CUDA_AVAILABLE is False, reason="CUDA not available")
def test_cuda_vs_numpyfft(verbose=False):
    """ Create an optical system with 2 parity test apertures,
    propagate light through it, and compare that we get the same results from both numpy and CUDA"""

    defaults = conf.use_fftw, conf.use_cuda, conf.use_opencl

    conf.use_fftw = False
    conf.use_opencl = False

    sys = setup_test_osys()

    conf.use_cuda = False
    psf_numpy, intermediates_numpy = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    conf.use_cuda = True
    psf_cuda, intermediates_cuda = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    # check the final PSFs are consistent
    assert np.abs(psf_cuda[0].data-psf_numpy[0].data).max() < 1e-6

    # Check flux conservation for the intermediate arrays behaves properly
    intermediates = intermediates_cuda
    epsilon = np.finfo(intermediates[0].wavefront.dtype).eps
    total_int_input = intermediates_numpy[0].total_intensity
    for i in [1,2]:
        assert np.abs(intermediates[i].total_intensity - total_int_input) < 5*epsilon

    # Check flux in output array is about 0.5% less than input array (due to finite FOV)
    expected = 0.004949550538272617927759
    assert np.abs(intermediates[3].total_intensity - total_int_input) - expected < 5*epsilon

    if verbose:
        print ("PSF difference: ", np.abs(psf_cuda[0].data-psf_numpy[0].data).max())
        for i in [1,2]:
            print(" Int. WF {} intensity diff: {}".format(i, np.abs(intermediates[i].total_intensity-total_int_input)) )
        print(" Final PSF intensity diff:", np.abs(intermediates[3].total_intensity-total_int_input) - expected)

    conf.use_fftw, conf.use_cuda, conf.use_opencl = defaults

@pytest.mark.skipif(accel_math._OPENCL_AVAILABLE is False, reason="OPENCL not available")
def test_opencl_vs_numpyfft(verbose=False):
    """ Create an optical system with 2 parity test apertures,
    propagate light through it, and compare that we get the same results from both numpy and CUDA"""

    defaults = conf.use_fftw, conf.use_cuda, conf.use_opencl

    conf.use_fftw = False
    conf.use_cudal = False

    sys = setup_test_osys()

    conf.use_opencl = False
    psf_numpy, intermediates_numpy = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    conf.use_opencl = True
    psf_opencl, intermediates_opencl = sys.calc_psf(wavelength=1e-6, return_intermediates=True)

    # check the final PSFs are consistent
    assert np.abs(psf_opencl[0].data-psf_numpy[0].data).max() < 1e-6

    # Check flux conservation for the intermediate arrays behaves the same for both
    intermediates = intermediates_opencl
    epsilon = np.finfo(intermediates[0].wavefront.dtype).eps
    total_int_input = intermediates_numpy[0].total_intensity
    for i in [1,2]:
        assert np.abs(intermediates[i].total_intensity - total_int_input) < 5*epsilon

    # Check flux in output array is about 0.5% less than input array (due to finite FOV)
    expected = 0.004949550538272617927759
    assert np.abs(intermediates[3].total_intensity - total_int_input) - expected < 5*epsilon

    if verbose:
        print ("PSF difference: ", np.abs(psf_opencl[0].data-psf_numpy[0].data).max())
        for i in [1,2]:
            print(" Int. WF {} intensity diff: {}".format(i, np.abs(intermediates[i].total_intensity-total_int_input)) )
        print(" Final PSF intensity diff:", np.abs(intermediates[3].total_intensity-total_int_input) - expected)

    conf.use_fftw, conf.use_cuda, conf.use_opencl = defaults


# TODO: Add a function that uses both the DFT and MFT for the exact same calc, and compare the results
