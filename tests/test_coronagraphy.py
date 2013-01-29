import matplotlib.pyplot as plt
import numpy as np
import time

import poppy
from simple_tests import TestPoppy, check_wavefront, _log, fits, fwcentroid

"""
    Various tests of coronagraphic calculations
"""


def test_basic_FWPM_monochromatic():
    """ A very  basic test of FQPM calculation  : does it not crash?
    """

    plt.clf()

    osys = poppy.OpticalSystem("Perfect JW", oversample=4)
    osys.addPupil(transmission=path+"/pupil_RevV.fits", name='JW Pupil')
    osys.addImage(function='fieldstop', name='20 arcsec stop', size=20)
    osys.addImage(function='FQPM',wavelength=10.65e-6)
    osys.addPupil(transmission="/Users/mperrin/software/newJWPSF/data/MIRI/coronagraph/MIRI_FQPMLyotStop.fits", name='MIRI FQPM Lyot')
    osys.addDetector(0.032, name="Detector", fov_pixels=128)

    out = osys.propagate_mono(wavelength=10.65e-6, display_intermediates=True, save_intermediates=True)
    out.writeto('test_fft.fits',clobber=True)



class Test_FQPM_fieldmask(TestPoppy):
    """ Test FQPM plus field mask together. The check is that there should be very low flux in the final image plane """
    #def test

    def test_6(self): #oversample=2, verbose=True, wavelength=2e-6):
        """ Perfect circular case  with FQPM with fieldMask
        Test  ideal FQPM, with field mask. Verify proper behavior in Lyot plane"""

        poppy._IMAGECROP = 5 # plot images w/out zooming in on just the center.

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=6.5/2)
        osys.addPupil('FQPM_FFT_aligner')
        osys.addImage('FQPM', wavelength=self.wavelength)  # perfect FQPM for this wavelength
        osys.addImage('fieldstop', size=6.0)  
        osys.addPupil('FQPM_FFT_aligner', direction='backward')
        osys.addPupil('Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=10.0)

        plt.clf()
        poppy._FLUXCHECK=True
        poppy._USE_FFTW3 = False
        #poppy._USE_FFTW3 = True
        #logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')
        psf = osys.calcPSF(wavelength=self.wavelength, display_intermediates=True)
        self.assertLess(psf[0].data.sum(), 0.002) 
        _log.info("post-FQPM flux is appropriately low.")


class Test_FQPM_coron_offaxis(TestPoppy):
    """ Test the FQPM with field mask, off axis 
    
    When you are off axis the FQPM should not appreciably mess with the PSF.
    The check is seeing that the PSF centroid for an off-axis PSF is the same
    as it would be if there were no FQPM occulter


    """
    def test_7(self): 
 
        """ Perfect circular case  with FQPM with fieldMask off-axis 
        """

        poppy._IMAGECROP = 5 # plot images w/out zooming in on just the center.
        #oversample = 2
        #pixelscale = 0.1

        fov = 6

        osys1 = poppy.OpticalSystem(" no FQPM, offset", oversample=self.oversample)
        osys1.addPupil('Circle', radius=6.5/2)
        osys1.addPupil('FQPM_FFT_aligner')
        osys1.addImage('fieldstop', size=20.0)  
        osys1.addPupil('FQPM_FFT_aligner', direction='backward')
        osys1.addPupil('Circle', radius=6.5/2)
        osys1.addDetector(pixelscale=self.pixelscale, fov_arcsec=fov)

        osys2 = poppy.OpticalSystem("FQPM offset", oversample=self.oversample)
        osys2.addPupil('Circle', radius=6.5/2)
        osys2.addPupil('FQPM_FFT_aligner')
        osys2.addImage('FQPM', wavelength=self.wavelength)  # perfect FQPM for this wavelength
        osys2.addImage('fieldstop', size=20.0)  
        osys2.addPupil('FQPM_FFT_aligner', direction='backward')
        osys2.addPupil('Circle', radius=6.5/2)
        osys2.addDetector(pixelscale=self.pixelscale, fov_arcsec=fov)

        myoffset = 3.0
        osys1.source_offset_r = myoffset
        osys1.source_offset_theta = 45.
        osys2.source_offset_r = myoffset
        osys2.source_offset_theta = 45.
 
        poppy._FLUXCHECK=True
        plt.figure(1)
        plt.clf()
        psf1 = osys1.calcPSF(wavelength=self.wavelength, display_intermediates=True, save_intermediates=False)
        plt.figure(2)
        plt.clf()
        psf2 = osys2.calcPSF(wavelength=self.wavelength, display_intermediates=True, save_intermediates=False)

        plt.figure(3)
        plt.subplot(211)
        poppy.utils.display_PSF(psf1, title=osys1.name)
        plt.subplot(212)
        poppy.utils.display_PSF(psf2, title=osys2.name)

        pos1 = fwcentroid(psf1[0].data, halfwidth=10, threshold=1e-2)
        pos2 = fwcentroid(psf2[0].data, halfwidth=10, threshold=1e-2)

        rel_offset = np.sqrt(((np.array(pos1) - np.array(pos2))**2).sum())
        self.assertTrue(rel_offset < 1e-3 ) 
        _log.info("Source position does not appear to be affected by FQPMs for far off-axis sources")


class Test_BLC_coron(TestPoppy):
    "Test BLC corons. Verify reasonable on- and off- axis behavior. "
    def test_BLC_coron_circ(self):
        self.do_test_9(kind='circular')
    def test_BLC_coron_linear(self):
        self.do_test_9(kind='linear')
    def test_BLC_coron_circ_offset(self):
        self.do_test_9(kind='circular', offset=True)
    def test_BLC_coron_linear_offset(self):
        self.do_test_9(kind='linear', offset=True)
  
    def do_test_9(self, kind='circular', offset=False):
        _log.info("Testing BLC kind = "+kind)
        
        radius = 6.5/2
        lyot_radius = 6.5/2.5
        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=radius)
        osys.addImage()
        osys.addImage('BandLimitedCoron', kind=kind, sigma=5.0)
        osys.addPupil()
        osys.addPupil('Circle', radius=lyot_radius)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=5.0)
        poppy._FLUXCHECK=True


        if offset: 
            osys.source_offset_r =  2.0
        else: 
            osys.source_offset_r =  0.0
        poppy._FLUXCHECK= True
        plt.clf()
        psf, int_wfs = osys.calcPSF(wavelength=self.wavelength, display_intermediates=True, return_intermediates=True)

        if offset:
            # the flux should be relatively high
            # with the vast majority of the loss due just to the undersized Lyot stop
            self.assertGreater(psf[0].data.sum(), (lyot_radius/radius)**2 *0.95 ) 
            _log.info("For offset source, post-BLC flux is appropriately high")
        else:
            # after the Lyot plane, the wavefront should be all real. 
            lyot_wf = int_wfs[-2]
            lyot_wf_fits = lyot_wf.asFITS(what='all') # need to save this for the multiwave comparison in test_3_multiwave()
            self.assertTrue(check_wavefront(lyot_wf_fits, test='is_real', comment='(Lyot Plane)'))

            # and the flux should be low.
            self.assertLess(psf[0].data.sum(), 1e-4) 
            _log.info("post-BLC flux is appropriately low.")


class Test12(TestPoppy):
    """ Test SemiAnalyticCoronagraph code
    """

    def test_SAMC(self):
        self.do_test_SAMC()

    def do_test_SAMC(self, display_intermediates=False):
        """ Test semianalytic coronagraphic method

        """
        radius = 6.5/2
        lyot_radius = 6.5/2.5
        pixelscale = 0.010

        osys = poppy.OpticalSystem("test", oversample=8)
        osys.addPupil('Circle', radius=radius, name='Entrance Pupil')
        osys.addImage('CircularOcculter', radius = 0.1)
        osys.addPupil('Circle', radius=lyot_radius, name = "Lyot Pupil")
        osys.addDetector(pixelscale=pixelscale, fov_arcsec=5.0)


        plt.figure(1)
        sam_osys = poppy.SemiAnalyticCoronagraph(osys, oversample=8, occulter_box=0.15) 

        t0s = time.time()
        psf_sam = sam_osys.calcPSF(display_intermediates=display_intermediates)
        t1s = time.time()

        plt.figure(2)
        t0f = time.time()
        psf_fft = osys.calcPSF(display_intermediates=display_intermediates)
        t1f = time.time()

        plt.figure(3)
        plt.clf()
        plt.subplot(121)
        poppy.utils.display_PSF(psf_fft, title="FFT")
        plt.subplot(122)
        poppy.utils.display_PSF(psf_sam, title="SAM")

        print "Elapsed time, FFT:  %.3s" % (t1f-t0f)
        print "Elapsed time, SAM:  %.3s" % (t1s-t0s)


        maxdiff = np.abs(psf_fft[0].data - psf_sam[0].data).max()
        print "Max difference between results: ", maxdiff

        self.assertTrue( maxdiff < 1e-8)


###################################################################################
def test_blc(oversample=2, verbose=True, wavelength=2.1e-6, angle=0, kind='nircamcircular'):

    if wavelength == 2.1e-6: sigma=5.253
    elif wavelength == 3.35e-6: sigma=3.2927866
    else: sigma=2.5652

    blc = poppy.BandLimitedCoron('myBLC', kind, sigma=sigma, wavelength=wavelength)
    #wf = poppy.Wavefront( wavelength = wavelen, pixelscale = 0.010, npix=512)
    #blcphase = blc.getPhasor(wf)


    plt.clf()
    #plt.subplot(121)
    blc.display()

    #stop()


def test_blc_corons(oversample=2, verbose=True, wavelength=2e-6, angle=0, kind='circular'):
    """ Test point source shifting, given variable pupil array padding. """

    oversample=2
    pixelscale = 0.010
    wavelength = 4.6e-6
    poppy._IMAGECROP = 5

    osys = poppy.OpticalSystem("test", oversample=oversample)
    osys.addPupil('Circle', radius=6.5/2)
    osys.addImage()
    osys.addImage('BandLimitedCoron', kind=kind,  sigma=5.0)
    osys.addPupil()
    osys.addPupil('Circle', radius=6.5/2.5)
    osys.addDetector(pixelscale=pixelscale, fov_arcsec=3.0)
    
    plt.clf()
    poppy._FLUXCHECK=True
    poppy._USE_FFTW3 = False


    osys.source_offset_theta = angle
    osys.source_offset_r =  0.0
    psf = osys.calcPSF(wavelength=wavelength, display_intermediates=True)
            #psf.writeto('test3_psf.fits', clobber=True)
            # TODO check position

        # after the Lyot plane, the wavefront should be all real. 
        #check_wavefront('wavefront_plane_004.fits', test='is_real')


def test_blc2(oversample=2, verbose=True, wavelength=2e-6, angle=0, kind='circular', sigma=1.0, loc = 0.3998):
    import scipy
    x = np.linspace(-5, 5, 401)
    sigmar = sigma*x
    if kind == 'circular':
        trans = (1-  (2*scipy.special.jn(1,sigmar)/sigmar)**2)**2
    else: 
        trans = (1-  (np.sin(sigmar)/sigmar)**2)**2
    plt.clf()
    plt.plot(x, trans)
    plt.axhline(0.5, ls='--', color='k')


    plt.axvline(loc, ls='--', color='k')
    #plt.gca().set_xbound(loc*0.98, loc*1.02)
    wg = np.where(sigmar > 0.01)
    intfn = scipy.interpolate.interp1d(x[wg], trans[wg])
    print "Value at %.4f :\t%.4f" % (loc, intfn(loc))

    # figure out the FWHM
    #   cut out the portion of the curve from the origin to the first positive maximum
    wp = np.where(x > 0)
    xp = x[wp]
    transp = trans[wp]
    wm = np.argmax(transp)
    wg = np.where(( x>0 )& ( x<xp[wm]))
    interp = scipy.interpolate.interp1d(trans[wg], x[wg])
    print "For sigma = %.4f, HWHM occurs at %.4f" % (sigma, interp(0.5))


#def width_blc(desired_width, approx=None, plot=False):
#    """ The calculation of sigma parameters for the wedge BLC function is not straightforward.
#
#    This function numerically solves the relevant equation to determine the sigma required to 
#    acheive a given HWHM.
#
#    It uses recursion to iterate to a higher precision level.
#    """
#
#    loc = desired_width
#
#    if approx is None:
#        sigma = np.linspace(0, 20, 5000)
#    else: 
#        sigma = np.linspace(approx*0.9, approx*1.1, 100000.)
#    lhs = loc* np.sqrt(1 - np.sqrt(0.5))
#    rhs = np.sin(sigma * loc) / sigma
#    diff = np.abs(lhs - rhs)
#    wmin = np.where(diff == np.nanmin(diff))
#    sig_ans = sigma[wmin][0]
#
#    if approx: 
#        return sig_ans
#    else:
#        # use recursion
#        sig_ans = width_blc(loc, sig_ans)
#
#    if plot:
#        check =  (1-  (np.sin(sig_ans * loc)/sig_ans/loc)**2)**2
#        #plt.plot(sigma, lhs)
#        plt.clf()
#        plt.plot(sigma, rhs)
#        plt.axhline(lhs)
#
#        print "sigma = %f implies HWHM = %f" % (sig_ans, loc)
#        print " check: 0.5 == %f" % (check)
#    return sig_ans
#
#
#def calc_blc_wedge(deg=4, wavelength=2.1e-6):
#    """ This function determines the desired sigma coefficients required to 
#    achieve a wedge from 2 to 6 lam/D.
#
#    It returns the coefficients of a polynomial fit that maps from
#    nlambda/D to sigma. 
#
#    """
#    import scipy
#    r = np.linspace(2, 6, 161)
#    difflim = wavelen / 6.5 * 180.*60*60/np.pi 
#    sigs = [width_blc(difflim * ri) for ri in r]
#
#    pcs = scipy.polyfit(r, sigs, deg)
#    p = scipy.poly1d(pcs)
#    plt.plot(r, sigs, 'b')
#    plt.plot(r, p(r), "r--")
#    diffs = (sigs - p(r))
#    print "Poly fit:" +repr(pcs)
#    print "  fit rms: "+str(diffs.std())
#
#
#

