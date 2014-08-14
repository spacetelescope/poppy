# Tests for FFT based propagation

from .. import poppy_core
from .. import optics
import numpy as np
import astropy.io.fits as fits
from .test_core import check_wavefront


# For some reason, the following block of code in poppy_core is not sufficient
# to handle the "no pyfftw" case when running in test mode, even though it works
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
    osys.addPupil( optics.CircularAperture(radius=radius)   )
    osys.addPupil( optics.FQPM_FFT_aligner()  ) #'FQPM_FFT_aligner')
    osys.addImage( optics.IdealFQPM( wavelength=wavelen) )  # perfect FQPM for this wavelength
    osys.addImage( optics.RectangularFieldStop( width=6.0))
    osys.addPupil( optics.FQPM_FFT_aligner(direction='backward'))
    osys.addPupil( optics.CircularAperture(radius=radius))
    osys.addDetector(pixelscale=0.01, fov_arcsec=10.0)

    psf = osys.calcPSF(wavelength=wavelen, oversample=oversamp)
    assert psf[0].data.sum() <  0.002
    #_log.info("post-FQPM flux is appropriately low.")



def test_SAMC():
    """ Test semianalytic coronagraphic method

    """
    lyot_radius = 6.5/2.5
    pixelscale = 0.010

    osys = poppy_core.OpticalSystem("test", oversample=4)
    osys.addPupil( optics.CircularAperture(radius=radius), name='Entrance Pupil')
    osys.addImage( optics.CircularOcculter( radius = 0.1) )
    osys.addPupil( optics.CircularAperture(radius=lyot_radius), name = "Lyot Pupil")
    osys.addDetector(pixelscale=pixelscale, fov_arcsec=5.0)


    #plt.figure(1)
    sam_osys = poppy_core.SemiAnalyticCoronagraph(osys, oversample=8, occulter_box=0.15)

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


    
    maxdiff = np.abs(psf_fft[0].data - psf_sam[0].data).max()
    #print "Max difference between results: ", maxdiff

    assert( maxdiff < 1e-7)



# TODO: Add a function that uses both the DFT and MFT for the exact same calc, and compare the results
