from .. import poppy_core as poppy
from .. import optics
import numpy as np
import astropy.io.fits as fits


def test_nonsquare_detector(oversample=1, pixelscale=0.010, wavelength=1e-6):
        """ Test that the MFT supports non-square detectors 
        
        Do this by first computing a square detector then two different
        rectangular detector grids, all from the same pupil and detector pixel sampling.
        Check that the center pixels of the various results are identical
        
        """
        # verified working properly on Jan 28 2013

        results = []
        #pl.figure(1)

        fovs_to_test = ( (3,3), (3,6), (6,3),(4,2), (3,11))
        for fov_arcsec in  fovs_to_test:
            #pl.clf()

            osys = poppy.OpticalSystem("test", oversample=oversample)
            circ = optics.CircularAperture(radius=6.5/2)
            osys.addPupil(circ)
            osys.addDetector(pixelscale=pixelscale, fov_arcsec=fov_arcsec) 
            psf = osys.calcPSF(wavelength=wavelength)
            #poppy.utils.display_PSF(psf)

            results.append(psf[0].data)

            #pl.draw()


        #pl.figure(2)
        psf0 = results[0]
        bx=10
        ceny = psf0.shape[0]/2
        cenx = psf0.shape[1]/2

        cut0 = psf0[ceny-bx:ceny+bx, cenx-bx:cenx+bx]

        #pl.subplot(1, len(fovs_to_test), 1)
        #pl.imshow(np.log10(cut0))
        #pl.title("peak from\nsquare array = "+str(fovs_to_test[0]))

        # the maximum ought to be the same no matter what 
        for i in range(1, len(fovs_to_test)):
            thispsf = results[i]
            assert(psf0.max() == thispsf.max())

            #pl.subplot(1, len(fovs_to_test),  i+1)

            ceny = thispsf.shape[0]/2
            cenx = thispsf.shape[1]/2

            thiscut = thispsf[ceny-bx:ceny+bx, cenx-bx:cenx+bx]
            #pl.imshow(np.log10(thiscut))

            #pl.title("peak from array =\n"+str(fovs_to_test[i]))

            #print (thiscut-cut0).sum()

            assert( (thiscut-cut0).sum()  == 0)

