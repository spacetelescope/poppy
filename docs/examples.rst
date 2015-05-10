.. _examples:

Examples
============

Let's dive right in to some example code. 


For all of the following examples, you will have more informative text output when running the code
if you first enable Python's logging mechanism to display log messages to screen::

        import logging
        logging.basicConfig(level=logging.DEBUG)




A simple circular pupil
--------------------------

This is very simple, as it should be::

        osys = poppy.OpticalSystem()
        osys.addPupil( poppy.CircularAperture(radius=3))    # pupil radius in meters
        osys.addDetector(pixelscale=0.010, fov_arcsec=5.0)  # image plane coordinates in arcseconds

        psf = osys.calcPSF(2e-6)                            # wavelength in microns
        poppy.display_PSF(psf, title='The Airy Function')

.. image:: ./example_airy.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result

A complex segmented pupil
--------------------------

By combining multiple analytic optics together it is possible to create quite complex pupils::

        ap = poppy.MultiHexagonAperture(rings=3, flattoflat=2)           # 3 rings of 2 m segments yields 14.1 m circumscribed diameter
        sec = poppy.SecondaryObscuration(secondary_radius=1.5, n_supports=4, support_width=0.1)   # secondary with spiders
        atlast = poppy.CompoundAnalyticOptic( opticslist=[ap, sec], name='Mock ATLAST')           # combine into one optic

        atlast.display(npix=1024, colorbar_orientation='vertical')

.. image:: ./example_atlast_pupil.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result

And here's the PSF::

        osys = poppy.OpticalSystem()
        osys.addPupil(atlast)
        osys.addDetector(pixelscale=0.010, fov_arcsec=2.0)
        psf = osys.calcPSF(1e-6)

        poppy.display_PSF(psf, title="Mock ATLAST PSF")

.. image:: ./example_atlast_psf.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result




Multiple defocused PSFs
---------------------------

Defocus can be added using a lens::

        wavelen=1e-6
        nsteps = 4
        psfs = []
        for nwaves in range(nsteps):

            osys = poppy.OpticalSystem("test", oversample=2)
            osys.addPupil( poppy.CircularAperture(radius=3))    # pupil radius in meters
            osys.addPupil( poppy.ThinLens(nwaves=nwaves, reference_wavelength=wavelen, radius=3))
            osys.addDetector(pixelscale=0.01, fov_arcsec=4.0)

            psf = osys.calcPSF(wavelength=wavelen)
            psfs.append(psf)

            pl.subplot(1,nsteps, nwaves+1)
            poppy.display_PSF(psf, title='Defocused by {0} waves'.format(nwaves),
                colorbar_orientation='horizontal')

        
.. image:: ./example_defocus.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result




Band Limited Coronagraph with Off-Axis Source
-----------------------------------------------

As an example of a more complicated calculation, here's a NIRCam-style band limited coronagraph with the source not precisely centered::

    oversample=2
    pixelscale = 0.010  #arcsec/pixel
    wavelength = 4.6e-6

    osys = poppy.OpticalSystem("test", oversample=oversample)
    osys.addPupil(poppy.CircularAperture(radius=6.5/2))
    osys.addImage()
    osys.addImage(poppy.BandLimitedCoron(kind='circular',  sigma=5.0)) 
    osys.addPupil()
    osys.addPupil(poppy.CircularAperture(radius=6.5/2))
    osys.addDetector(pixelscale=pixelscale, fov_arcsec=3.0)

    osys.source_offset_theta = 45.
    osys.source_offset_r =  0.1  # arcsec
    psf = osys.calcPSF(wavelength=wavelength, display_intermediates=True)

.. image:: ./example_BLC_offset.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result



FQPM coronagraph
------------------

Four quadrant phase mask coronagraphs are a bit more complicated because one needs to ensure proper alignment of the
FFT result with the center of the phase mask. This is done using a virtual optic called an 'FQPM FFT aligner' as follows::

    optsys = poppy.OpticalSystem()
    optsys.addPupil( poppy.CircularAperture( radius=3))
    optsys.addPupil( poppy.FQPM_FFT_aligner())   # ensure the PSF is centered on the FQPM cross hairs
    optsys.addImage()  # empty image plane for "before the mask"
    optsys.addImage( poppy.IdealFQPM(wavelength=2e-6))
    optsys.addPupil( poppy.FQPM_FFT_aligner(direction='backward'))  # undo the alignment tilt after going back to the pupil plane
    optsys.addPupil( poppy.CircularAperture( radius=3)) # Lyot mask - change radius if desired
    optsys.addDetector(pixelscale=0.01, fov_arcsec=10.0)


    psf = optsys.calcPSF(wavelength=2e-6, display_intermediates=True)

.. image:: ./example_FQPM.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result


FQPM on an Obscured Aperture (demonstrates compound optics)
--------------------------------------------------------------

As a variation, we can add a secondary obscuration. This can be done by
creating a compound optic consisting of the circular outer aperture plus an
opaque circular obscuration. The latter we can make using the InverseTransmission class. ::


    primary = poppy.CircularAperture( radius=3)
    secondary = poppy.InverseTransmission( poppy.CircularAperture(radius=0.5) )
    aperture = poppy.CompoundAnalyticOptic( opticslist = [primary, secondary] )

    optsys = poppy.OpticalSystem()
    optsys.addPupil( aperture)
    optsys.addPupil( poppy.FQPM_FFT_aligner())   # ensure the PSF is centered on the FQPM cross hairs
    optsys.addImage( poppy.IdealFQPM(wavelength=2e-6))
    optsys.addPupil( poppy.FQPM_FFT_aligner(direction='backward'))  # undo the alignment tilt after going back to the pupil plane
    optsys.addPupil( poppy.CircularAperture( radius=3)) # Lyot mask - change radius if desired
    optsys.addDetector(pixelscale=0.01, fov_arcsec=10.0)

    optsys.display()

    psf = optsys.calcPSF(wavelength=2e-6, display_intermediates=True)


.. image:: ./example_FQPM_obscured.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result





Semi-analytic Coronagraph Calculations
----------------------------------------

In some cases, coronagraphy calculations can be sped up significantly using the semi-analytic algorithm of Soummer et al. 
This is implemented by first creating an OpticalSystem as usual, and then casting it to a SemiAnalyticCoronagraph class 
(which has a special customized propagation method implementing the alternate algorithm):


The following code performs the same calculation both ways and compares their speeds::

        radius = 6.5/2
        lyot_radius = 6.5/2.5
        pixelscale = 0.060

        osys = poppy.OpticalSystem("test", oversample=8)
        osys.addPupil( poppy.CircularAperture(radius=radius), name='Entrance Pupil')
        osys.addImage( poppy.CircularOcculter(radius = 0.1) )
        osys.addPupil( poppy.CircularAperture(radius=lyot_radius), name='Lyot Pupil')
        osys.addDetector(pixelscale=pixelscale, fov_arcsec=5.0)


        plt.figure(1)
        sam_osys = poppy.SemiAnalyticCoronagraph(osys, oversample=8, occulter_box=0.15)

        import time
        t0s = time.time()
        psf_sam = sam_osys.calcPSF(display_intermediates=True)
        t1s = time.time()

        plt.figure(2)
        t0f = time.time()
        psf_fft = osys.calcPSF(display_intermediates=True)
        t1f = time.time()

        plt.figure(3)
        plt.clf()
        plt.subplot(121)
        poppy.utils.display_PSF(psf_fft, title="FFT")
        plt.subplot(122)
        poppy.utils.display_PSF(psf_sam, title="SAM")

        print "Elapsed time, FFT:  %.3s" % (t1f-t0f)
        print "Elapsed time, SAM:  %.3s" % (t1s-t0s)


On my circa-2010 Mac Pro, the results are dramatic::

        Elapsed time, FFT:  62.
        Elapsed time, SAM:  4.1


