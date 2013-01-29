import matplotlib.pyplot as plt
import numpy as np

import poppy
from simple_tests import TestPoppy, check_wavefront, _log, fits, fwcentroid

"""

These are moderately simple tests

2: Is the wave in the Lyot plane essentially all real? (for coronagraph with no occulter)
3: First, verify the FQPM tilt behavior works as desired. 
    Then, test an ideal FQPM  
4: Verify ability to shift point sources.
    The Lyot plane should still be all real.

"""



class Test2(TestPoppy):
    """ Is the wave in the Lyot plane essentially all real? i.e. negligible imaginary part """
    def test_2_lyotreal_numpyfft(self):
        poppy._USE_FFTW3 = False
        self.do_test_2_lyotreal()
    def test_2_lyotreal_fftw(self):
        poppy._USE_FFTW3 = True
        self.do_test_2_lyotreal()
    def do_test_2_lyotreal(self):
        """ Test  no FQPM, no field mask. Verify proper behavior in Lyot plane"""
        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=6.5/2)
        osys.addImage()  # perfect image plane
        osys.addPupil('Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
        plt.clf()
        poppy._FLUXCHECK=True
        poppy._USE_FFTW3 = True
        psf = osys.calcPSF(wavelength=self.wavelength, save_intermediates=True, display_intermediates=True)
        psf.writeto('test2_psf.fits', clobber=True)

        # after the Lyot plane, the wavefront should be all real. 
        self.assertTrue(check_wavefront('wavefront_plane_002.fits', test='is_real', comment='(Lyot Plane)'))
    def test_multiwave(self):
        self.iter_wavelengths(self.test_2_lyotreal_fftw)

class Test3(TestPoppy):
    """ First, verify the FQPM tilt behavior works as desired. 
        Then, test an ideal FQPM  """

    def test_3_fqpm_tilt_numpyfft(self):
        poppy._USE_FFTW3 = False
        self.do_test_3_fqpm_tilt()
    def test_3_fqpm_tilt_fftw(self):
        poppy._USE_FFTW3 = True
        self.do_test_3_fqpm_tilt()
    def test_3_ideal_fqpm_numpyfft(self):
        poppy._USE_FFTW3 = False
        self.do_test_3_ideal_fqpm()
    def test_3_ideal_fqpm_fftw(self):
        poppy._USE_FFTW3 = True
        self.do_test_3_ideal_fqpm()
         
    def do_test_3_fqpm_tilt(self):
        """ Test FQPM tilting (no FQPM yet), no field mask. Verify proper behavior in Lyot plane"""

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=6.5/2)
        osys.addPupil('FQPM_FFT_aligner')
        osys.addImage()  # perfect image plane
        osys.addPupil('FQPM_FFT_aligner', direction='backward')
        osys.addPupil('Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
            #TODO testing of odd and even focal plane sizes?
        
        plt.clf()
        poppy._FLUXCHECK=True
        psf = osys.calcPSF(wavelength=self.wavelength, save_intermediates=True, display_intermediates=True)
        psf.writeto('test3a_psf.fits', clobber=True)

        # after the Lyot plane, the wavefront should be all real. 
        check_wavefront('wavefront_plane_004.fits', test='is_real', comment='(Lyot Plane)')

        cen = poppy.measure_centroid('wavefront_plane_002.fits', boxsize=50)
        head = fits.getheader('wavefront_plane_002.fits')
        desired_pos = (head['NAXIS1']-1)/2.0
        self.assertAlmostEqual( cen[0], desired_pos, delta=0.025) #within 1/50th of a pixel of desired pos?
        self.assertAlmostEqual( cen[1], desired_pos, delta=0.025) #within 1/50th of a pixel of desired pos?
                # This is likely dominated by uncertainties in the simple center measuring algorithm...

        _log.info("FQPM FFT half-pixel tilting is working properly in intermediate image plane")

        cen2 = poppy.measure_centroid('wavefront_plane_005.fits', boxsize=50)
        head2 = fits.getheader('wavefront_plane_005.fits')
        desired_pos2 = (head2['NAXIS1']-1)/2.0
        self.assertAlmostEqual( cen2[0], desired_pos2, delta=0.05) #within 1/20th of a pixel of desired pos?
                                    
        _log.info("FQPM FFT half-pixel tilting is working properly in final image plane")


    def do_test_3_ideal_fqpm(self):
        """ Test  ideal FQPM, no field mask. Verify proper behavior in Lyot plane"""


        #self.wavelength = 8e-6 # for ease of seeing details on screen make it bigger
        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=6.5/2)
        osys.addPupil('FQPM_FFT_aligner')
        osys.addImage('FQPM', wavelength=self.wavelength)  # perfect FQPM for this wavelength
        osys.addPupil('FQPM_FFT_aligner', direction='backward')
        osys.addPupil('Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)
        
        plt.clf()
        poppy._FLUXCHECK=True
        poppy._USE_FFTW3 = False
        #poppy._USE_FFTW3 = True
        #logging.basicConfig(level=logging.DEBUG,format='%(name)-10s: %(levelname)-8s %(message)s')
        psf, int_wfs = osys.calcPSF(wavelength=self.wavelength, save_intermediates=False, display_intermediates=True, return_intermediates=True)
        #psf.writeto('test3_psf.fits', clobber=True)
        lyot_wf = int_wfs[-2]
        lyot_wf.writeto("wavefront_plane_004.fits", what='all', clobber=True) # need to save this for the multiwave comparison in test_3_multiwave()

        # after the Lyot plane, the wavefront should be all real. 
        self.assertTrue(check_wavefront(lyot_wf.asFITS(what='all'), test='is_real', comment='(Lyot Plane)'))
        self.assertLess(psf[0].data.sum(), 0.002) 
        _log.info("post-FQPM flux is appropriately low.")

    def test_3_multiwave(self):
        """ Verify that the fluxes at the lyot planes are small and independent of wavelength"""
        lyot_fluxes = []
        for result in self.iter_wavelengths(self.do_test_3_ideal_fqpm):
            # The Lyot plane flux should be indep of wavelength
            im = fits.getdata('wavefront_plane_004.fits')
            lyot_fluxes.append(im[0,:,:].sum())
            self.assertLess(lyot_fluxes[-1], 0.005) 
        _log.info("Lyot plane fluxes : "+str(lyot_fluxes))

        for i in range(len(lyot_fluxes)-1):
            self.assertAlmostEqual(lyot_fluxes[i], lyot_fluxes[i+1])
        _log.info("Lyot plane is independent of wavelength. ")

class Test4(TestPoppy):
    """ Verify ability to shift point sources.
    The Lyot plane should still be all real. """
    def test_offsets_numpyfft(self):
        poppy._USE_FFTW3 = False
        self.do_4_source_offsets()
    def test_offsets_fftw(self):
        poppy._USE_FFTW3 = True
        self.do_4_source_offsets()

    def do_4_source_offsets(self, angle=0):
        #oversample=2, verbose=True, wavelength=2e-6, angle=0):
        """ Perfect circular case  no FQPM no field mask, off-axis PSF location

        Test point source shifting. no FQPM, no field mask. Verify point source goes to right spot in image plane"""

        osys = poppy.OpticalSystem("test", oversample=self.oversample)
        osys.addPupil('Circle', radius=6.5/2)
        osys.addImage()
        osys.addPupil('Circle', radius=6.5/2)
        osys.addDetector(pixelscale=self.pixelscale, fov_arcsec=3.0)

        plt.clf()
        poppy._FLUXCHECK=True

        osys.source_offset_theta = angle
        for i in range(15):
            osys.source_offset_r = i * 0.1
            psf = osys.calcPSF(wavelength=self.wavelength, display_intermediates=True)

            pos = fwcentroid(psf[0].data, halfwidth=10, threshold=1e-2)
            # pos is the pixel coords in y,x in pixel units.
            cenx = (psf[0].data.shape[0]-1)/2.0
            offset = np.sqrt( (pos[0]-cenx)**2 + (pos[1]-cenx)**2)* self.pixelscale/self.oversample
            _log.info("Desired offset is %f, measured is %f " % (osys.source_offset_r, offset))
            self.assertAlmostEqual(osys.source_offset_r, offset, 3)


    def test_multiwave_offsets(self):
        self.iter_wavelengths(self.do_4_source_offsets, angle=0.0)
        self.iter_wavelengths(self.do_4_source_offsets, angle=45.0)
