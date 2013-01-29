import os,sys

import numpy as np
import unittest
import matplotlib.pyplot as plt
import matplotlib
import time
import scipy

try:
    import astropy.io.fits as fits
except:
    import pyfits as fits



import poppy
from poppy.fwcentroid import fwcentroid

import logging
_log = logging.getLogger('poppy-tester')
_log.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')

""" Testing code for POPPY

    This is a suite of test cases for the various Poppy optical propagation tools, 
    right now with emphasis on verifying coronagraphy & FQPMs. 
    
    This is written using Python's unittest framework. This is my first attempt using
    unittest, and it seems somewhat convoluted in terms of setup, particularly since there
    is no easy way to iterate over test cases with variable arguments like wavelength.
    So there's some work-arounds using global variables here.


    Main routines to look at:  test_run() and test_multiwave() 

    test_run is a convenience function to let you run all the numbered test suites in order,
    or to pick just one at a time. 

    MDP 2011-02-10

"""

############################################################################
#
#  Airy functions for comparison's sake
#
############################################################################
def airy_1d( aperture=6.5, wavelength=2e-6, length = 512, pixelscale=0.010, 
        obscuration=0.0, center=None, plot_=False):
    """ 1-dimensional Airy function PSF calculator 
    
    Parameters
    ----------
    aperture, wavelength : float
        aperture diam and wavelength in meters
    size : tuple
        array size
    pixelscale : 
        arcseconds

    Returns 
    --------
    r : array
        radius array in arcsec
    airy : array
        Array with the Airy function values, normalized to 1 at peak
    """

    center = (length-1)/2.
    r = np.arange(length)*pixelscale

    RADtoARCSEC = 360.*60*60/np.pi # ~ 206265
    v = np.pi * (r/ RADtoARCSEC) * aperture/wavelength
    e = obscuration
    
    airy =  1./(1-e**2)**2* ((2*scipy.special.jn(1,v) - e*2*scipy.special.jn(1,e*v))/v )**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248

    if plot_:
        plt.semilogy(r, airy)
        plt.xlabel("radius [arcsec]")
        plt.ylabel("PSF intensity")
    return r, airy



def airy_2d( aperture=6.5, wavelength=2e-6, shape=(512,512), pixelscale=0.010, 
        obscuration=0.0, center=None):
    """ 2-dimensional Airy function PSF calculator 
    
    Parameters
    ----------
    aperture, wavelength : float
        aperture diam and wavelength in meters
    size : tuple
        array size
    pixelscale : 
        arcseconds
    """

    if center is None:
        center = (np.asarray(shape)-1.)/2
    y, x = np.indices(shape)
    y -= center[0]
    x -= center[1]
    y *= pixelscale
    x *= pixelscale
    r = np.sqrt(x**2 + y**2)

    RADtoARCSEC = 360.*60*60/np.pi # ~ 206265
    v = np.pi * (r/ RADtoARCSEC) * aperture/wavelength
    e = obscuration

    
    airy =  1./(1-e**2)**2* ((2*scipy.special.jn(1,v) - e*2*beselj(1,e*v))/v )**2
    # see e.g. Schroeder, Astronomical Optics, 2nd ed. page 248





#
#def basic_test():
#    jwst = JWST_OTE("path/to/some/OPDs")
#    miri = jwst.MIRI
#    gstar = pysynphot('G2 star')
#
#    psf2 = miri.psf('imaging', center=(512,512), filter='F1500W', oversample=4, spectrum=gstar)
#
#    corPSF = miri.psf('lyot', filter='F2550W', decenter=0.01, oversample=4)


def check_wavefront(filename_or_hdulist, slice=0, ext=0, test='nearzero', comment=""):
    """ A helper routine to verify certain properties of a wavefront FITS file, 
    as requested by some test routine. """
    if isinstance(filename_or_hdulist, str):
        hdulist = fits.open(filename_or_hdulist)
        filename = filename_or_hdulist
    elif isinstance(filename_or_hdulist, fits.HDUList):
        hdulist = filename_or_hdulist
        filename = 'input HDUlist'
    else:
        raise 
    imstack = hdulist[ext].data
    im = imstack[slice,:,:]

    try:

        if test=='nearzero':
            assert(  np.all(np.abs(im) < np.finfo(im.dtype).eps*10))
            _log.info("Slice %d of %s %s is all essentially zero" % (slice, filename, comment))
            return True
        elif test == 'is_real':
            #assumes output type = 'all'
            cplx_im = imstack[1,:,:] * np.exp(1j*imstack[2,:,:])
            assert(  np.all( cplx_im.imag < np.finfo(im.dtype).eps*10))
            _log.info("File %s %s is essentially all real " % (filename, comment))
            return True

    except:
        _log.error("Test %s failed for %s " % (test, filename))
        return False

####################################################33
# Funky aperture for FFT tests.


class ParityTestAperture(poppy.AnalyticOpticalElement):
    """ Defines a circular pupil aperture with boxes cut out.
    This is mostly a test aperture

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0

    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.

    """

    def __init__(self, name=None,  radius=1.0, pad_factor = 1.5, **kwargs):
        if name is None: name = "Circle, radius=%.2f m" % radius
        poppy.AnalyticOpticalElement.__init__(self,name=name, **kwargs)
        self.radius = radius
        self.pupil_diam = pad_factor * 2* self.radius # for creating input wavefronts - let's pad a bit


    def getPhasor(self,wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, poppy.Wavefront):
            raise ValueError("CircularAperture getPhasor must be called with a Wavefront to define the spacing")
        assert (wave.planetype == poppy.poppy_core._PUPIL)

        y, x = wave.coordinates()
        r = np.sqrt(x**2+y**2) #* wave.pixelscale

        w_outside = np.where( r > self.radius)
        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0

        w_box1 = np.where( (r> self.radius*0.5) & (np.abs(x) < self.radius*0.1 ) & ( y < 0 ))
        w_box2 = np.where( (r> self.radius*0.75) & (np.abs(y) < self.radius*0.2) & ( x < 0 ))
        self.transmission[w_box1] = 0
        self.transmission[w_box2] = 0

        return self.transmission



####################################################33
#
#  Test Cases
#




_TEST_WAVELENGTH = 2e-6
_TEST_OVERSAMP = 2

class TestPoppy(unittest.TestCase):
    """ Base test case that allows setting wavelength and oversampling for the tests, 
    and cleans up temp files afterwards """
    # Note: you CANNOT override the __init__ method for a TestCase - this behaves in some
    # semi-horrible way due to complexities of the guts of how unittest works. 
    #
    def setUp(self):
        self.wavelength = _TEST_WAVELENGTH
        self.oversample = _TEST_OVERSAMP
        self.pixelscale = 0.010
        self.test_wavelengths = [1e-6, 2e-6, 3e-6]
    def tearDown(self):
        """ Clean up after tests """
        if 0: 
            os.delete('wavefront_plane*fits') # delete intermediate wavefront files
            os.delete('test*fits')

    def iter_wavelengths(self, function, *args, **kwargs):
        """ Iterate over the set of wavelengths to test, repeatedly calling a given function 
        This is implemented as a generator function, to allow the calling routine to examine intermediate
        data products before the next step of the iteration """

        for wl in self.test_wavelengths:
            _log.info("---- Testing wavelength = %e " % wl)
            self.wavelength=wl
            yield function(*args, **kwargs)

class Test_FOV_size(TestPoppy):
    def test_fov_size_pixels(self):
        """ Test the PSF field of view size is as requested, in pixels for a square aperture"""

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil(function='Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_pixels=100, oversample=1)
        
        psf = osys.calcPSF(wavelength=self.wavelength)

        self.assertEqual(psf[0].data.shape[0], 100)
        self.assertEqual(psf[0].data.shape[1], 100)

    def test_fov_size_arcsec(self):
        """ Test the PSF field of view size is as requested, in pixels for a square aperture"""

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil(function='Circle', radius=6.5/2)
        fov_arcsec = 2.0
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=fov_arcsec, oversample=1)
        
        psf = osys.calcPSF(wavelength=self.wavelength)

        self.assertEqual(psf[0].data.shape[0], round(fov_arcsec/self.pixelscale))
        self.assertEqual(psf[0].data.shape[1], round(fov_arcsec/self.pixelscale))


class Test_Pupils(TestPoppy):
    """ Basic functionality tests"""
    def test_analytic_pupils(self):
        """ Test circular, square, hexagonal pupils and their PSFs """

        def image_cut_1d(image, angle=0):
            """ Make a quick 1D cut through an image starting at the center """
            #y, x = np.indices(image)
            #y-= (image.shape[0]-1)/2
            #x-= (image.shape[1]-1)/2

            t = np.arange(image.shape[0])
            cx = np.cos(angle*np.pi/180)*t +  (image.shape[0]-1)/2
            cy = np.sin(angle*np.pi/180)*t +  (image.shape[1]-1)/2
            cx = np.asarray(np.round(cx), dtype=int)
            cy = np.asarray(np.round(cy), dtype=int)

            wg = np.where( (cx >=0) & (cy >=0) & (cx < image.shape[1]) & (cy < image.shape[0]))
            return image[cy[wg],cx[wg]]


        pupils = ['Circle', 'Hexagon', 'Square']
        angles = [[0], [0, 30], [0, 45]]
        effective_diams = [[2], [2,np.sqrt(3)], [2,2*np.sqrt(2)]]

        plt.clf()
        cuts = []
        for i in range(3):
            plt.subplot(2,3, i+1)
            osys = poppy.OpticalSystem("test", oversample=self.oversample)
            osys.addPupil(pupils[i])
            osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)

            psf = osys.calcPSF(wavelength=self.wavelength)

            poppy.utils.display_PSF(psf)
            plt.title(pupils[i])

            plt.subplot(2,3, i+4)
            for ang, diam in zip(angles[i], effective_diams[i]):
                cut = image_cut_1d(psf[0].data, ang)
                r = np.arange(cut.size) * 0.010
                cut /= cut.max() # normalize to peak=1
                plt.semilogy(r, cut, label='$\\theta = %d^o$' % ang )
                if i == 0:
                    radius, airyfn = airy_1d(diam, self.wavelength, pixelscale=self.pixelscale)
                    plt.plot(radius, airyfn, "k--", label='analytic')
            plt.gca().set_xbound(0,3)
            plt.gca().set_ybound(1e-13,1.5)

            # TODO - overplot perfect analytical PSFs.
            plt.legend(loc='upper right', frameon=False)

        self.assertTrue(True)
        #FIXME - compare cuts to airy function etc.


    def test_fits_pupil(self):
        """ Test pupils created from FITS files, including 3 different ways of setting the pixel scale:
            automatically (from FITS header), from a user-specified header keyword, from a user-specified value"""
        pupilfile = os.getenv('WEBBPSF_PATH', default= os.path.dirname(os.path.dirname(os.path.abspath(poppy.__file__))) +os.sep+"data" ) + os.sep+"pupil_RevV.fits"

        plt.clf()
        for i, pixelscale in enumerate([None, 'PUPLSCAL', 0.0064560*2]):
            ax = plt.subplot(2,3, i+1)
            osys = poppy.OpticalSystem("test", oversample=self.oversample)
            osys.addPupil(transmission=pupilfile, pixelscale=pixelscale)
            osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
            psf = osys.calcPSF(wavelength=self.wavelength)
            osys.planes[0].display(ax=ax, what='intensity')

            plt.subplot(2,3, i+4)
            poppy.utils.display_PSF(psf)
            plt.title("File = "+os.path.basename(pupilfile))
            plt.xlabel("Pixel scale = "+str(pixelscale))

            ## FIXME this needs an assertion to check

    def test_rotated_pupil(self):
        """ Test rotations, applied to pupils created from FITS files."""
        pupilfile = os.getenv('WEBBPSF_PATH', default= os.path.dirname(os.path.dirname(os.path.abspath(poppy.__file__))) +os.sep+"data" ) + os.sep+"pupil_RevV.fits"

        plt.clf()
        for i, rotinfo in enumerate([(0,'degrees'), (30,'degrees'),(np.pi/6,'radians')]):
            rotdist, rotunit = rotinfo
            ax = plt.subplot(2,3, i+1)
            osys = poppy.OpticalSystem("test", oversample=self.oversample)
            osys.addPupil(transmission=pupilfile)
            osys.addRotation(angle=rotdist, units=rotunit)
            osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
            psf = osys.calcPSF(wavelength=self.wavelength)
            osys.planes[0].display(ax=ax, what='intensity')

            plt.subplot(2,3, i+4)
            poppy.utils.display_PSF(psf)
            #plt.title("File = "+os.path.basename(pupilfile))
            #plt.xlabel("Pixel scale = "+str(pixelscale))
            plt.title("Rotated by %f %s" % (rotdist, rotunit))

            ## FIXME this needs an assertion to check


class Test_Normalizations(TestPoppy):
    def do_generic_normalization_test(self, osys):
        """ Test the PSF normalization: Ensure proper results are returned
             using all of the normalization options: 
                first plane = 1.0
                last plane = 1.0
                first plane = 2.0 (not physically meaningful, just a debugging check option
            """

        for norm in ['first', 'last', 'first=2']:
            plt.clf()
            psf = osys.calcPSF(wavelength=self.wavelength, normalize=norm, display_intermediates=True)
            poppy.utils.display_PSF(psf)
            tot = psf[0].data.sum()
            _log.info("Using normalization method=%s, the PSF total is\t%f" % (norm, tot))
            if norm =='last':
                self.assertAlmostEqual(abs(tot), 1 ) # the PSF's total on the computed array should be 1, or very close to it.
            elif norm == 'first=2':
                self.assertAlmostEqual(abs(tot), 2, delta=0.01 )  # this should be very roughly 1.
            else: 
                self.assertAlmostEqual(abs(tot), 1, delta=0.01 )  # this should be very roughly 1.

    def do_test_1_normalization(self):
        """ Test the PSF normalization for MFTs """

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil(function='Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=20.0) # use a large FOV so we grab essentially all the light and conserve flux
        self.do_generic_normalization_test(osys)

    def do_test_1_normalization_fft(self):
        """ Test the PSF normalization for FFTs"""

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil(function='Circle', radius=6.5/2)
        osys.addImage() # null plane to force FFT
        osys.addPupil() # null plane to force FFT
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=10.0) # use a large FOV so we grab essentially all the light and conserve flux
        self.do_generic_normalization_test(osys)

    def do_test_1_normalization_invMFT(self):
        """ Test the PSF normalization for Inverse MFTs """

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil(function='Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=10.0) # use a large FOV so we grab essentially all the light and conserve flux
        osys.addPupil() # this will force an inverse MFT
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=10.0) # use a large FOV so we grab essentially all the light and conserve flux
        self.do_generic_normalization_test(osys)


    def test_1_normalization_multiwave(self):
        results = [res for res in self.iter_wavelengths(self.do_test_1_normalization)]

    def test_1_normalization_fft_multiwave(self):
        results = [res for res in self.iter_wavelengths(self.do_test_1_normalization_fft)]

    def test_1_normalization_invMFT(self):
        results = [res for res in self.iter_wavelengths(self.do_test_1_normalization_invMFT)]


    def test_inverse_MFT(self):
        # Verify basic functionality of the Inverse MFT code. 
        poppy._FLUXCHECK=True

        test_ap = ParityTestAperture(radius=6.5/2)

        osys = poppy.OpticalSystem("test", oversample=4)
        osys.addPupil(test_ap)
        osys.addDetector(pixelscale=0.010, fov_arcsec=10.0) # use a large FOV so we grab essentially all the light and conserve flux
        psf1 = osys.calcPSF(wavelength=2e-6, normalize='first', display_intermediates=True)

        #osys.addPupil(test_ap)
        osys.addPupil() # this will force an inverse MFT
        osys.addDetector(pixelscale=0.010, fov_arcsec=10.0) # use a large FOV so we grab essentially all the light and conserve flux
        plt.clf()
        psf = osys.calcPSF(wavelength=2e-6, normalize='first', display_intermediates=True)

        # the intermediate PSF (after one MFT) should be essentially identical to the
        # final PSF (after an MFT, inverse MFT, and another MFT):
        self.assertTrue(   np.abs(psf1[0].data - psf[0].data).max()  < 1e-7 )


#class Test8(TestPoppy):
#    "Verify that extra padding around the aperture makes no difference "
#    def test_padding_numpyfft(self):
#        poppy._USE_FFTW3=False
#        self.do_test_8()
#    def test_padding_fftw(self):
#        poppy._USE_FFTW3=True
#        self.do_test_8()
#
#    def do_test_8(self):
#        """ Test point source shifting, given variable pupil array padding. """
#
#        angle = 36.
#        for pad_factor in [1.0, 1.1, 1.5, 2.0]: 
#            osys = poppy.OpticalSystem("test", oversample=self.oversample)
#            osys.addPupil('Circle', radius=6.5/2, pad_factor = pad_factor)
#            osys.addImage()
#            osys.addPupil('Circle', radius=6.5/2)
#            osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
#            
#            plt.clf()
#            poppy._FLUXCHECK=True
#            poppy._USE_FFTW3 = False
#
#
#            osys.source_offset_theta = angle
#            osys.source_offset_r =  0.5
#            psf = osys.calcPSF(wavelength=self.wavelength, display_intermediates=True)
#            #psf.writeto('test3_psf.fits', clobber=True)
#            # TODO check position
#            pos = fwcentroid(psf[0].data, halfwidth=10, threshold=1e-2)
#            # pos is the pixel coords in y,x in pixel units.
#            cenx = (psf[0].data.shape[0]-1)/2.0
#            offset = np.sqrt( (pos[0]-cenx)**2 + (pos[1]-cenx)**2)* self.pixelscale/self.oversample
#            _log.info("Desired offset is %f, measured is %f " % (osys.source_offset_r, offset))
#            self.assertAlmostEqual(osys.source_offset_r, offset, 3)
# 
#        # after the Lyot plane, the wavefront should be all real. 
#        #check_wavefront('wavefront_plane_004.fits', test='is_real')
#
 
#
#class Test10(TestPoppy):
#    "Test multiwavelength multiprocessor propagation "
#
#    def test_10_multiproc_numpyfft(self):
#        poppy._USE_FFTW3 = False
#        # multiprocessor and FFTW not sure if they play nicely all the time?
#        self.do_test_10_multiproc()
#
#
#    #def test_10_multiproc_fftw3(self):
#        #poppy._USE_FFTW3 = True
#        ## multiprocessor and FFTW not sure if they play nicely all the time?
#        #self.do_test_10_multiproc()
# 
#    def do_test_10_multiproc(self):
#
#        # for the sake of a reasonably realistic usage case, we use the BLC setup from Test 9
#        kind = 'circular'
#        radius = 6.5/2
#        lyot_radius = 6.5/2.5
#        osys = poppy.OpticalSystem("test", oversample=self.oversample)
#        osys.addPupil('Circle', radius=radius)
#        osys.addImage()
#        osys.addImage('BandLimitedCoron', kind=kind, sigma=5.0)
#        osys.addPupil()
#        osys.addPupil('Circle', radius=lyot_radius)
#        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=5.0)
#        osys.source_offset_r =  1.5 # make the PSF easy to see...
# 
#
#        nlam= 6
#        source = {'weights': [0.1]*nlam, 'wavelengths': np.linspace(2.0e-6, 3.0e-6, nlam)}
#
#        _log.info("Calculating multiprocess PSF")
#        times = []
#        times.append(time.time())
#        psf2 = osys.calcPSFmultiproc(source)
#        times.append(time.time())
#        tmulti =  times[-1]-times[-2]
#        _log.info(" Time for multiprocessor: %f s " % (tmulti))
#
#        _log.info("Calculating single process PSF")
#        times.append(time.time())
#        psf1 = osys.calcPSF(source['wavelengths'], source['weights'])
#        times.append(time.time())
#        tsing =  times[-1]-times[-2]
#        _log.info(" Time for single processor: %f s " % (tsing))
#
#
#        _log.info(" Speedup factor: %f " % (tsing/tmulti))
#        plt.clf()
#        ax = plt.subplot(1,3,1)
#        poppy.utils.display_PSF(psf1)
#        ax = plt.subplot(1,3,2)
#        poppy.utils.display_PSF(psf2)
#        ax = plt.subplot(1,3,3)
#
#        poppy.imshow_with_mouseover(psf1[0].data - psf2[0].data, ax=ax)
#
#        self.assertTrue(  (psf1[0].data == psf2[0].data).all()  )
#        _log.info("Exact same result achieved both ways.")
#
#
#class Test11(TestPoppy):
#    """ Test various AnalyticOpticalElements 
#    """
#
#    def test_thinlens(self):
#
#        radius = 6.5/2
#        lyot_radius = 6.5/2.5
#
#        nsteps = 5
#        plt.clf()
#        fig, axarr = plt.subplots(1,nsteps, squeeze=True, num=1)
#        psfs = []
#        for nwaves in range(nsteps):
#
#            osys = poppy.OpticalSystem("test", oversample=self.oversample)
#            osys.addPupil('Circle', radius=radius)
#            osys.addPupil(optic=poppy.ThinLens(nwaves=nwaves, reference_wavelength=self.wavelength)) 
#            osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=5.0)
#
#            psf = osys.calcPSF(wavelength=self.wavelength)
#            psfs.append(psf)
#
#            norm = matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-4)
#            poppy.imshow_with_mouseover(psf[0].data, ax=axarr[nwaves], norm=norm)
#
#        # FIXME need evaluation here. 
#        stop()
#
#
#
#    def test_scalar(self):
#        wave = poppy.Wavefront(npix=100, wavelength=self.wavelength)
#        nulloptic = poppy.ScalarTransmission()
#
#        self.assertEqual(nulloptic.getPhasor(wave), 1.0)
#
#        NDoptic = poppy.ScalarTransmission(transmission=1e-3)
#        self.assertEqual(NDoptic.getPhasor(wave), 1.0e-3)
#
#

#

#
#def test_defocus():
#    """ Test a ThinLens defocus element 
#    """
#    radius = 6.5/2
#    lyot_radius = 6.5/2.5
#
#    waves = [0, 0.5, 1, 2, 4, 8]
#    nsteps = len(waves)
#    plt.clf()
#    fig, axarr = plt.subplots(1,nsteps, squeeze=True, num=1)
#    psfs = []
#    for i, nwaves in enumerate(waves):
#
#        osys = poppy.OpticalSystem("test", oversample=2)
#        osys.addPupil('Circle', radius=radius)
#        lens =poppy.ThinLens(nwaves=nwaves, reference_wavelength=1e-6)
#        osys.addPupil(optic=lens)
#        osys.addDetector(pixelscale=0.010, fov_arcsec=10.0)
#
#        psf = osys.calcPSF(wavelength=1e-6)
#        psfs.append(psf)
#
#        norm = matplotlib.colors.LogNorm(vmin=1e-8, vmax=1e-4)
#        poppy.imshow_with_mouseover(psf[0].data, ax=axarr[i], norm=norm )
#        axarr[i].set_title('%.1f waves defocus' % nwaves)
#
#        #wf = osys.inputWavefront()
#        #wf2 = wf * osys.planes[0]
#        #wf3 = wf2* osys.planes[1]
#
#
#    #stop()
#
#
#
##################################################################################
#
#
#
#
#def test_multiwave(*args):
#    """ Run the full test suite, for multiple different wavelengths 
#    """
#    global _TEST_WAVELENGTH
#    for w in [1e-6, 2e-6, 5e-6]:
#        print("*"*70)
#        print("  Running tests with wavelength = %e m" % w)
#        test_run(wavelength=w, *args)
#
#
#

#
#
#def test_fftw3():
#   #test_MFT()
#    osys = poppy.OpticalSystem("Perfect JW", oversample=4)
#    osys.addPupil(transmission=path+"/pupil_RevV.fits", name='JW Pupil')
#    osys.addImage(function='fieldstop', name='20 arcsec stop', size=20)
#    osys.addImage(function='FQPM',wavelength=10.65e-6)
#    osys.addPupil(transmission="/Users/mperrin/software/newJWPSF/data/MIRI/coronagraph/MIRI_FQPMLyotStop.fits", name='MIRI FQPM Lyot')
#    osys.addDetector(0.032, name="Detector", fov_pixels=128)
#
#
#
#    nlam = 20
#    nlam = 3
#    source = {'wavelengths': np.linspace(10,15,nlam)*1e-6, 'weights': nlam* [1]}
#    #source = {'wavelengths': np.arange(10,11,0.25)*1e-6, 'weights': 4* [1]}
#    mono = {'wavelengths': [source['wavelengths'].mean()], 'weights': [1]}
#
#    import time
#
#    #print "-- mono, regular --"
#    #t1 = time.time()
#    #osys.calcPSF(mono).writeto('test_mono.fits', clobber=True)
#    #print "-- poly, single process --"
#    #t2 = time.time()
#    #osys.calcPSF(source).writeto('test_single.fits', clobber=True)
#    #print "-- poly, multi process --"
#    #t3 = time.time()
#    #osys.calcPSFmultiproc(source).writeto('test_multi.fits', clobber=True)
#    #t4 = time.time()
#    #print "for %d wavelengths: " % nlam
#    #for t, v in zip(['mono:', 'poly single:', 'poly multi:'], [t2-t1, t3-t2, t4-t3]):
#        #print "  Executed %s in %f seconds." % (t,v)
#
#    #print "poly single relative computation time:\t%f" % ( (t3-t2)/(t2-t1)/nlam )
#    #print "multi/single relative computation time:\t%f" % ( (t4-t3)/(t3-t2) )
#
#
#    print "-- poly, singleprocess, numpy fft--"
#    t2 = time.time()
#    _USE_FFTW3 = False
#    osys.calcPSF(source=source).writeto('test_numpyfft.fits', clobber=True)
#    print "-- poly, single process, fftw3 --"
#    t3 = time.time()
#    _USE_FFTW3 = True
#    osys.calcPSF(source=source).writeto('test_fftw3.fits', clobber=True)
#    t4 = time.time()
#
#
#    print "for %d wavelengths: " % nlam
#    for t, v in zip(['Numpy FFT', 'FFTW3'], [t3-t2, t4-t3]):
#        print "  Executed %s in %f seconds." % (t,v)
#
#
#
#def test_speed():
#
#    npix = 111
#    offset = (0.0, 0)
#    osys = OpticalSystem("Circle JW", oversample=1)
#    #osys.addPupil(function='Square', name = "6.5m square", size=6.5)
#    osys.addPupil(function='Circle', name = "6.5m circle", radius=6.5/2)
#    #osys.addImage()
#    #osys.addDetector(0.032, name="Detector", fov_pixels=128)
#    osys.addDetector(0.032/2, name="Detector", fov_pixels=npix)
#
#    osys.planes[-1].det_offset = offset
#
#    #_USE_FFTW3 = True
#    #_USE_FFTW3 = False
#    #_TIMETESTS= True
#
#    #res = osys.propagate_mono(2e-6,display_intermediates=False)
#
#    src = {'wavelengths': [2e-6], 'weights': [1.0]}
#    res = osys.calcPSF(src, display=True)
#
#
#    #plt.clf()
#    #plt.imshow(res[0].data)
#    res.writeto('test_ci_np%d_off%s.fits' %(npix, offset), clobber=True)
#
#
#def test_nonsquare_detector(oversample=1, pixelscale=0.010, wavelength=1e-6):
#        """ Test that the MFT supports non-square detectors """
#
#        osys = poppy.OpticalSystem("test", oversample=oversample)
#        osys.addPupil(function='Circle', radius=6.5/2)
#        osys.addDetector(pixelscale=pixelscale, fov_arcsec=(6,3)) 
#        #self.do_generic_normalization_test(osys)
#        psf = osys.calcPSF(wavelength=wavelength, display_intermediates=True)
#        poppy.utils.display_PSF(psf)


if __name__== "__main__":
    logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')
