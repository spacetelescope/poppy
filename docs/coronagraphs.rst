Poppy has extensive functionality to faciliate the modeling of coronagraph point spread functions. In addition to the general summary of those capabilities here, see the examples in the notebooks subdirectory:
`POPPY Examples <https://github.com/mperrin/poppy/blob/master/notebooks/POPPY%20Examples.ipynb>`_
and
`MatrixFTCoronagraph_demo <https://github.com/mperrin/poppy/blob/master/notebooks/MatrixFTCoronagraph_demo.ipynb>`_.

=====================================================
Efficient Lyot coronagraph propagation
=====================================================

By default, an optical system defined in Poppy uses the Fast Fourier Transform (FFT) to propagate the scalar field between pupil and image planes. While the FFT is a powerful tool for general Fraunhofer diffraction calculations, it is rarely the most computationally efficient approach for a coronagraph model. Consider the two coronagraph schematics below, from `Zimmerman et al (2016) <http://dx.doi.org/10.1117/1.JATIS.2.1.011012>`_:

.. image:: ./Lyot_coronagraphs_diagram.png
   :height: 373px
   :width: 916px
   :scale: 10 %
   :alt: Schematics of two Lyot coronagraph design variants
   :align: right

The upper design in the figure representes the classical Lyot coronagraph and its widely implemented, optimized descendent, the `apodized pupil Lyot coronagraph (APLC) <http://dx.doi.org/10.1051/0004-6361:20021573>`_. In this case an intermediate focal plane (labeled B) is occulted by a round, opaque mask. By applying the principle of electromagnetic field superposition, combined with knowledge of how FFT complexity scales with array size, `Soummer et al. (2007) <http://dx.doi.org/10.1364/OE.15.015935>`_ showed that the number of operations needed to compute the PSF is greatly reduced by replacing the FFT with direct Fourier transfoms, implemented in a vectorized fashion and spatially restricted to the *occulted* region of the intermediate focal plane. This is the now widely-used **semi-analytical** computational method for numerically modeling Lyot coronagraphs.

The lower design in the above figure shows a slightly different Lyot coronagraph design case. Here the focal plane mask (FPM) is a diaphragm that restricts the outer edge of the transmitted field. `Zimmerman et al (2016) <http://dx.doi.org/10.1117/1.JATIS.2.1.011012>`_ showed how this design variant can solve the same starlight cancellation problem, in particular for the baseline design of WFIRST. With this FPM geometry, the superposition simplification of `Soummer et al. (2007) <http://dx.doi.org/10.1364/OE.15.015935>`_ is not valid. However, again the execution time is greatly reduced by using direct, vectorized Fourier transforms, now spatially restricted to the *transmitted* region of the intermediate focal plane.

In Poppy, two subclasses of OpticalSystem exploit the computational methods described above: SemiAnalyticCoronagraph and MatrixFTCoronagraph. Let's see how to make use of these subclasses to speed up Lyot corongraph PSF calculations.

Lyot coronagraph using the semi-analytical subclass
---------------------------------------------------

In this example we consider a Lyot coronagraph with a conventional, opaque occulting spot. This configuration corresponds to the upper half of the schematic described above.

The semi-analytic method is specified by first creating an OpticalSystem as usual, and then casting it to a SemiAnalyticCoronagraph class (which has a special customized propagation method implementing the alternate algorithm):

The following code performs the same calculation both with semi-analytical and FFT propagation, and compares their speeds::

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


.. image:: ./example_SAM_comparison.png
   :scale: 50%
   :align: center
   :alt: Sample calculation result


On a circa-2010 Mac Pro, the results are dramatic::

        Elapsed time, FFT:  62.
        Elapsed time, SAM:  4.1


Lyot coronagraph using the MatrixFTCoronagraph subclass
---------------------------------------------------------

This coronagraph uses an annular diaphragm in the intermediate focal plane, corresponding to the lower half of the diagram above. Again we will compare the execution time with the FFT case.::

        D = 2.
        wavelen = 1e-6
        ovsamp = 8



Band Limited Coronagraph
-------------------------

Depending on the specific implementation, a Lyot coronagraph with a band-limited occulter can also benefit from the semi-analytical method in Poppy. For additional band-limited coronagraph examples, see the JWST NIRCam coronagraph modes included in `WebbPSF <http://github.com/mperrin/webbpsf>`_.

As an example of a more complicated coronagraph PSF calculation than the ones above, here's a NIRCam-style band limited coronagraph with the source not precisely centered::

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
   :scale: 60%
   :align: center
   :alt: Sample calculation result
 
FQPM coronagraph
------------------

Due to the wide (ideally infinite) spatial extension of its focal plane phase-shifting optic, the four-quadrant phase mask (FQPM) coronagraphs relies on FFT propagation. Another unique complication of the FQPM coronagraph class is its array alignment requirement between the FFT result in the intermediate focal plane with the center of the phase mask. This is done using a virtual optic called an 'FQPM FFT aligner' as follows::

    optsys = poppy.OpticalSystem()
    optsys.addPupil( poppy.CircularAperture( radius=3, pad_factor=1.5)) #pad display area by 50%
    optsys.addPupil( poppy.FQPM_FFT_aligner())   # ensure the PSF is centered on the FQPM cross hairs
    optsys.addImage()  # empty image plane for "before the mask"
    optsys.addImage( poppy.IdealFQPM(wavelength=2e-6))
    optsys.addPupil( poppy.FQPM_FFT_aligner(direction='backward'))  # undo the alignment tilt after going back to the pupil plane
    optsys.addPupil( poppy.CircularAperture( radius=3)) # Lyot mask - change radius if desired
    optsys.addDetector(pixelscale=0.01, fov_arcsec=10.0)


    psf = optsys.calcPSF(wavelength=2e-6, display_intermediates=True)

.. image:: ./example_FQPM.png
   :scale: 60%
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
   :scale: 60%
   :align: center
   :alt: Sample calculation result

