from .. import poppy_core as poppy
from .. import optics
import numpy as np
import astropy.io.fits as fits


def test_nonsquare_detector_axes_lengths():
        """ Test that an MFT onto non-square detectors yields
        the requested axes lengths 

        """

        for fov_pixels in ( [100,100], [100,200], [200,100], [137, 511] ):
            osys = poppy.OpticalSystem()
            circ = optics.CircularAperture(radius=6.5/2)
            osys.add_pupil(circ)
            osys.add_detector(pixelscale=0.1, oversample=1, fov_pixels=fov_pixels )

            psf = osys.calc_psf(wavelength=1e-6)

            assert(psf[0].data.shape[0] == fov_pixels[0])
            assert(psf[0].data.shape[1] == fov_pixels[1])


def test_nonsquare_detector_values(oversample=1, pixelscale=0.010, wavelength=1e-6,
        verbose=False):
        """ Test that the MFT onto non-square detectors yields the same pixel
        values as onto square detectors

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
            osys.add_pupil(circ)
            osys.add_detector(pixelscale=pixelscale, fov_arcsec=fov_arcsec) 
            psf = osys.calc_psf(wavelength=wavelength)
            #poppy.utils.display_psf(psf)

            results.append(psf[0].data)

            #pl.draw()


        #pl.figure(2)
        psf0 = results[0]
        bx=10
        ceny = psf0.shape[0]//2
        cenx = psf0.shape[1]//2

        cut0 = psf0[ceny-bx:ceny+bx, cenx-bx:cenx+bx]

        #pl.subplot(1, len(fovs_to_test), 1)
        #pl.imshow(np.log10(cut0))
        #pl.title("peak from\nsquare array = "+str(fovs_to_test[0]))

        # the maximum ought to be the same no matter what 
        for i in range(1, len(fovs_to_test)):
            thispsf = results[i]
            if verbose: print("i = {}, shape={}, Maxes= {}, {}, diff={}".format(i,
                fovs_to_test[i], psf0.max(), thispsf.max(), psf0.max() - thispsf.max()))


            assert(np.allclose(psf0.max(), thispsf.max()))

            #pl.subplot(1, len(fovs_to_test),  i+1)

            ceny = thispsf.shape[0]//2
            cenx = thispsf.shape[1]//2

            thiscut = thispsf[ceny-bx:ceny+bx, cenx-bx:cenx+bx]
            #pl.imshow(np.log10(thiscut))

            #pl.title("peak from array =\n"+str(fovs_to_test[i]))

            #print (thiscut-cut0).sum()


            assert(np.allclose(thiscut, cut0))
