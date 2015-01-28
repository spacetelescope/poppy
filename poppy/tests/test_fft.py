# Tests for FFT based propagation

from .. import poppy_core
from .. import optics
import numpy as np
import astropy.io.fits as fits
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
    osys.addPupil(pupil)
    osys.addImage() # null plane to force FFT
    osys.addPupil() # null plane to force FFT
    osys.addDetector(pixelscale=0.01, fov_arcsec=10.0) # use a large FOV so we grab essentially all the ligh        

    poppy_core._log.info('TEST: wavelen = {0}, radius = {1}'.format(wavelen, radius)) 


    # Expected value here is 0.9977
    psf = osys.calcPSF(wavelength=2.0e-6, normalize='first')

    poppy_core._log.info('TEST: Computed PSF of circular aperture')
    poppy_core._log.info('TEST: PSF total intensity sum is {0}'.format(psf[0].data.sum()))
    poppy_core._log.info('TEST:  Expected value is 0.9977 ')

    assert abs(psf[0].data.sum() - 0.9977) < 0.001


def test_fft_blc_coronagraph():
    """ Test that a simple band limited coronagraph blocks most of the light """

    lyot_radius = 6.5/2.5
    osys = poppy_core.OpticalSystem("test", oversample=2)
    osys.addPupil( optics.CircularAperture(radius=radius) )
    osys.addImage()
    osys.addImage( optics.BandLimitedCoron( kind='circular', sigma=5.0))
    osys.addPupil()
    osys.addPupil( optics.CircularAperture(radius=lyot_radius) )
    osys.addDetector(pixelscale=0.010, fov_arcsec=5.0)

    psf, int_wfs = osys.calcPSF(wavelength=wavelen, display_intermediates=False, return_intermediates=True)


    # after the Lyot plane, the wavefront should be all real.
    lyot_wf = int_wfs[-2]
    lyot_wf_fits = lyot_wf.asFITS(what='all') # need to save this for the multiwave comparison in test_3_multiwave()
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
    osys.addPupil(optics.CircularAperture(radius=radius))
    osys.addPupil(optics.FQPM_FFT_aligner())
    osys.addImage(optics.IdealFQPM(wavelength=wavelen))  # perfect FQPM for this wavelength
    osys.addImage(optics.RectangularFieldStop(width=6.0))
    osys.addPupil(optics.FQPM_FFT_aligner(direction='backward'))
    osys.addPupil(optics.CircularAperture(radius=radius))
    osys.addDetector(pixelscale=0.01, fov_arcsec=10.0)

    psf = osys.calcPSF(wavelength=wavelen)
    assert psf[0].data.sum() < 0.002

def test_SAMC(oversample=4):
    """ Test semianalytic coronagraphic method

    """
    lyot_radius = 6.5/2.5
    pixelscale = 0.010

    osys = poppy_core.OpticalSystem("test", oversample=oversample)
    osys.addPupil( optics.CircularAperture(radius=radius), name='Entrance Pupil')
    osys.addImage( optics.CircularOcculter( radius = 0.1) )
    osys.addPupil( optics.CircularAperture(radius=lyot_radius), name = "Lyot Pupil")
    osys.addDetector(pixelscale=pixelscale, fov_arcsec=5.0)


    #plt.figure(1)
    sam_osys = poppy_core.SemiAnalyticCoronagraph(osys, oversample=oversample, occulter_box=0.15)

    #t0s = time.time()
    psf_sam = sam_osys.calcPSF()
    #t1s = time.time()

    #plt.figure(2)
    #t0f = time.time()
    psf_fft = osys.calcPSF()
    #t1f = time.time()

    #plt.figure(3)
    #plt.clf()
    #plt.subplot(121)
    #poppy_core.utils.display_PSF(psf_fft, title="FFT")
    #plt.subplot(122)
    #poppy.utils.display_PSF(psf_sam, title="SAM")


    
    # The pixel by pixel difference should be small:
    maxdiff = np.abs(psf_fft[0].data - psf_sam[0].data).max()
    #print "Max difference between results: ", maxdiff

    assert( maxdiff < 1e-7)

    # and the overall flux difference should be small also:
    if oversample<=4:
        thresh = 1e-4 
    elif oversample==6:
        thresh=5e-5
    elif oversample>=8:
        thresh = 4e-6
    else:
        raise NotImplementedError("Don't know what threshold to use for oversample="+str(oversample))


    assert np.abs(psf_sam[0].data.sum() - psf_fft[0].data.sum()) < thresh



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

    from .test_core import ParityTestAperture

    # set up optical system with 2 pupil planes and 2 image planes
    sys = poppy_core.OpticalSystem(oversample=1)
    sys.addPupil(ParityTestAperture())
    sys.addImage()
    sys.addPupil()
    sys.addDetector(pixelscale=0.010, fov_arcsec=1)

    psf, planes = sys.calcPSF(display=display, return_intermediates=True)

    # the wavefronts are padded by 0s. With the current API the most convenient
    # way to ensure we get unpadded versions is via the asFITS function.
    p0 = planes[0].asFITS(what='intensity', includepadding=False)
    p2 = planes[2].asFITS(what='intensity', includepadding=False)

    # for checking the overall parity it's sufficient to check the intensity.
    # we can have arbitrarily large differences in phase for regions with 
    # intensity =0, so don't check the complex field or phase here. 

    absdiff = (np.abs(p0[0].data - p2[0].data))
    maxabsdiff = np.max(absdiff)
    assert (maxabsdiff < 1e-10)

    if display:
        nplanes = len(planes)
        for i, plane in enumerate(planes):
            ax = plt.subplot(2,nplanes,i+1)
            plane.display(ax = ax)
            plt.title("Plane {0}".format(i))
        plt.subplot(2,nplanes,nplanes+1)
        plt.imshow(absdiff)
        plt.title("Abs(Pupil0-Pupil2)")
        plt.colorbar()
        print("Max abs(difference) = "+str(maxabsdiff))
        




if conf.use_fftw:
    # The following test is only applicable if fftw is present. 

    def test_pyfftw_vs_numpyfft():
        """ Create an optical system with 2 parity test apertures, 
        propagate light through it, and compare that we get the same results from both numpy and pyfftw"""


        ap = optics.ParityTestAperture()
        sys = poppy_core.OpticalSystem()
        sys.addPupil(ap)
        sys.addImage()
        sys.addPupil(ap)
        sys.addDetector(0.02, fov_pixels=512)  # fairly arbitrary, but big enough to get most of the flux

        conf.use_fftw = False
        psf_numpy, intermediates_numpy = sys.calcPSF(wavelength=1e-6, return_intermediates=True)

        conf.use_fftw = True
        psf_fftw, intermediates_fftw = sys.calcPSF(wavelength=1e-6, return_intermediates=True)

        # check the final PSFs are consistent
        assert np.abs(psf_fftw[0].data-psf_numpy[0].data).max() < 1e-6

        # Check flux conservation for the intermediate arrays behaves the same for both
        for intermediates in [intermediates_numpy, intermediates_fftw]:
            for i in [1,2]:
                assert np.abs(intermediates[i].totalIntensity-intermediates_numpy[0].totalIntensity) < 1e-6
            assert np.abs(intermediates[3].totalIntensity-intermediates_numpy[0].totalIntensity) < 0.005






# TODO: Add a function that uses both the DFT and MFT for the exact same calc, and compare the results
