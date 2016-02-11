Poppy has extensive functionality to faciliate the modeling of coronagraph point spread functions. In addition to the general summary of those capabilities here, see the examples in the notebooks subdirectory:
`POPPY Examples <https://github.com/mperrin/poppy/blob/master/notebooks/POPPY%20Examples.ipynb>`_
and
`MatrixFTCoronagraph_demo <https://github.com/mperrin/poppy/blob/master/notebooks/MatrixFTCoronagraph_demo.ipynb>`_.

=====================================================
Efficient computation of Lyot coronagraph propogation
=====================================================

By default, an optical system defined in Poppy uses the Fast Fourier Transform (FFT) to propagate the scalar field between pupil and image planes. While the FFT is a powerful tool for general Fraunhofer diffraction calculations, it is rarely the most efficient approach for a coronagraph system. Consider the two coronagraph schematics below, from `Zimmerman et al (2016) <http://dx.doi.org/10.1117/1.JATIS.2.1.011012>`_:

.. image:: ./Lyot_coronagraphs_diagram.png
   :height: 373px
   :width: 916px
   :scale: 10 %
   :alt: Schematics of two Lyot coronagraph design variants
   :align: right

The upper design in the figure representes the classical Lyot coronagraph and its widely implemented, optimized descendent, the apodized pupil Lyot coronagraph (APLC). In this case an intermediate focal plane (labeled B) is occulted by a round, opaque mask. By applying the laws of field superposition, combined with knowledge of FFT scaling complexity, `Soummer et al. (2007) <http://dx.doi.org/10.1364/OE.15.015935>`_ showed that the number of operations needed to compute the PSF is greatly reduced by replacing the FFT with direct Fourier transfoms, implemented in a vectorized fashion and spatially restricted to the *occulted* region of the intermediate focal plane. This is the now widely-used **semi-analytical** computational method for numerically modeling Lyot coronagraphs.

The lower design in the above figure shows a slightly different Lyot coronagraph design case. Here the focal plane mask (FPM) is a diaphragm that restricts the outer edge of the transmitted field. `Zimmerman et al (2016) <http://dx.doi.org/10.1117/1.JATIS.2.1.011012>`_ showed how this design variant also solves the Lyot coronagraph optimization problem, in particular for the baseline design of WFIRST. With this FPM geometry, the superposition trick used by `Soummer et al. (2007) <http://dx.doi.org/10.1364/OE.15.015935>`_ is not valid. However, again the execution time is greatly reduced by using direct, vectorized Fourier transforms, this time spatially restricted to the *transmitted* region of the intermediate focal plane.

Two subclasses of OpticalSystem have been defined to exploit these computational methods: SemiAnalyticCoronagraph and MatrixFTCoronagraph. Let's see how to make use of these subclasses to speed up Lyot corongraph PSF calculations.

Semi-analytical (occulting spot FPM) Lyot coronagraph
=====================================================


MatrixFT (diaphragm FPM) Lyot coronagraph
=========================================









   



