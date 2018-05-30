#  
#  Test functions for MatrixFTCoronagraph optical system subclass
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import poppy

import logging
_log = logging.getLogger('poppy_tests')

def test_MatrixFT_FFT_Lyot_propagation_equivalence(display=False):
    """ Using a simple Lyot coronagraph prescription,
    perform a simple numerical check for consistency
    between calc_psf result of standard (FFT) propagation and
    MatrixFTCoronagraph subclass of OpticalSystem."""

    D = 2.
    wavelen = 1e-6
    ovsamp = 4

    fftcoron_annFPM_osys = poppy.OpticalSystem( oversample=ovsamp )
    fftcoron_annFPM_osys.add_pupil( poppy.CircularAperture(radius=D/2) )
    spot = poppy.CircularOcculter( radius=0.4  )
    diaphragm = poppy.InverseTransmission( poppy.CircularOcculter( radius=1.2 ) )
    annFPM = poppy.CompoundAnalyticOptic( opticslist = [diaphragm, spot] )
    fftcoron_annFPM_osys.add_image( annFPM )
    fftcoron_annFPM_osys.add_pupil( poppy.CircularAperture(radius=0.9*D/2) )
    fftcoron_annFPM_osys.add_detector( pixelscale=0.05, fov_arcsec=3. )
    
    # Re-cast as MFT coronagraph with annular diaphragm FPM
    matrixFTcoron_annFPM_osys = poppy.MatrixFTCoronagraph( fftcoron_annFPM_osys, occulter_box=diaphragm.uninverted_optic.radius_inner )
    
    annFPM_fft_psf = fftcoron_annFPM_osys.calc_psf(wavelen)
    annFPM_mft_psf = matrixFTcoron_annFPM_osys.calc_psf(wavelen)

    diff_img = annFPM_mft_psf[0].data - annFPM_fft_psf[0].data
    abs_diff_img = np.abs(diff_img)
   
    if display: 
        plt.figure(figsize=(16,3))
        plt.subplot(131)
        poppy.display_psf(annFPM_fft_psf, vmin=1e-10, vmax=1e-6, title='Annular FPM Lyot coronagraph, FFT')
        plt.subplot(132)
        poppy.display_psf(annFPM_mft_psf, vmin=1e-10, vmax=1e-6, title='Annular FPM Lyot coronagraph, Matrix FT')
        plt.subplot(133)
        plt.imshow( (annFPM_mft_psf[0].data - annFPM_fft_psf[0].data), cmap='gist_heat')
        plt.colorbar()
        plt.title('Difference (MatrixFT - FFT)')
        plt.show()

    print("Max of absolute difference: %.10g" % np.max(abs_diff_img))

    assert( np.all(abs_diff_img < 2e-7) )
