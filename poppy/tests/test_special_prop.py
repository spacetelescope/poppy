# Test functions for specialized propagators 
import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import time
import pytest
try:
    import scipy
except ImportError:
    scipy = None

from .. import poppy_core
from .. import optics
from .. import special_prop

wavelen = 1e-6
radius = 6.5/2


def test_SAMC(fft_oversample=4, samc_oversample=8, npix=512,
        extra_test_plane=True, display=False):
    """ Test semianalytic coronagraphic method

    fft_oversample, samc_oversample : int
        Oversampling factor for resolution & precision
    extra_test_plane : bool
        Should we add an extra plane in the beginning? This allows easy testing of
        muliple cases in the updated flexible-about-number-of-planes SAMC implementation.

    """
    lyot_radius = 6.5/2.5
    pixelscale = 0.010

    osys = poppy_core.OpticalSystem("test", oversample=fft_oversample, npix=npix)
    osys.add_pupil( optics.CircularAperture(radius=radius, name='Entrance Pupil'))

    if extra_test_plane:
        osys.add_pupil( optics.CircularAperture(radius=radius, name='Extra test Pupil'))
    osys.add_image( optics.CircularOcculter( radius = 0.1) )
    osys.add_pupil( optics.CircularAperture(radius=lyot_radius, name = "Lyot Pupil"))
    osys.planes[-1].wavefront_display_hint='intensity'
    osys.add_detector(pixelscale=pixelscale, fov_arcsec=5.0)


    sam_osys = special_prop.SemiAnalyticCoronagraph(osys, oversample=samc_oversample, occulter_box=0.15,
            fpm_index = 2)

    if display:
        plt.figure()
    t_start_sam = time.time()
    psf_sam = sam_osys.calc_psf(display_intermediates=display)
    # also compute a version with the intermediate planes returned
    psf_sam_copy, intermediates= sam_osys.calc_psf(display_intermediates=display,
                    return_intermediates=True)
    t_stop_sam = time.time()

    print("SAMC calculation: {} s".format(t_stop_sam - t_start_sam))
    if display:
        plt.suptitle("Calculation using SAMC method")
        plt.figure()

    t_start_fft = time.time()
    psf_fft = osys.calc_psf(display_intermediates=display)
    t_stop_fft = time.time()
    print("Basic FFT calculation: {} s".format(t_stop_fft - t_start_fft))
    if display:
        plt.suptitle("Calculation using Basic FFT method")

    # The pixel by pixel difference should be small:
    maxdiff = np.abs(psf_fft[0].data - psf_sam[0].data).max()
    #print "Max difference between results: ", maxdiff

    assert( maxdiff < 1e-7)

    # and the overall flux difference should be small also:
    if fft_oversample<=4:
        thresh = 1e-4
    elif fft_oversample==6:
        thresh=5e-5
    elif fft_oversample>=8:
        thresh = 4e-6
    else:
        raise NotImplementedError("Don't know what threshold to use for oversample="+str(oversample))

    expected_total = 0.005615
    assert np.abs(psf_sam[0].data.sum()-expected_total) < thresh, "Summed total of PSF intensity did not match expectations"
    assert np.abs(psf_sam[0].data.sum()-expected_total)/expected_total < 0.003, "Summed total of PSF intensity was more than 0.3% away from expectation"

    # Check there are the expected number of intermediate planes, which for this
    # kind of propagation has some extras:
    assert len(intermediates) == len(osys)+2, "Unexpected number of returned optical planes"

    assert np.allclose(psf_sam[0].data, psf_sam_copy[0].data), "Didn't get same results with & without return_intermediates"
