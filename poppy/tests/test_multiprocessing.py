# Test functions for poppy multiprocessing

from .. import poppy_core
from .. import optics
from .. import conf
from .. import utils

import numpy as np
import astropy
import astropy.io.fits as fits
import sys
from distutils.version import LooseVersion
from astropy.tests.helper import remote_data

try:
    import pytest
    _HAVE_PYTEST = True
except:
    _HAVE_PYTEST = False


if _HAVE_PYTEST:

    #@pytest.mark.xfail
    # Just skip this test entirely for right now because sometimes it hangs the
    # entire Python process...

    @pytest.mark.skipif( (sys.version_info < (3,4,0) ),
            reason="Python 3.4 required for reliable forkserver start method")
    @pytest.mark.skipif(LooseVersion(astropy.__version__) <  LooseVersion('1.0.3'),
            reason="astropy >=1.0.3 required for tests of multiprocessing")
    def test_basic_multiprocessing():
        """For a simple optical system, test that single process and
        multiprocess calculation give the same results"""
        osys = poppy_core.OpticalSystem("test")
        pupil = optics.CircularAperture(radius=1)
        osys.add_pupil(pupil)
        osys.add_detector(pixelscale=0.1, fov_arcsec=5.0)

        source={'wavelengths': [1.0e-6, 1.1e-6, 1.2e-6, 1.3e-6], 'weights':[0.25, 0.25, 0.25, 0.25]}
        conf.use_fftw=False

        conf.use_multiprocessing=False
        psf_single = osys.calc_psf(source=source)

        conf.use_multiprocessing=True
        psf_multi = osys.calc_psf(source=source)

        assert np.allclose(psf_single[0].data, psf_multi[0].data), \
            "PSF from multiprocessing does not match PSF from single process"

        return psf_single, psf_multi


    @pytest.mark.skipif( (sys.version_info < (3,4,0) ),
            reason="Python 3.4 required for reliable forkserver start method")
    @pytest.mark.skipif(LooseVersion(astropy.__version__) <  LooseVersion('1.0.3'),
            reason="astropy >=1.0.3 required for tests of multiprocessing")
    def test_multiprocessing_intermediate_planes():
        """ Test that using multiprocessing you can retrieve the intermediate planes,
        and they are consistent with the intermediate planes from a
        single process calculation"""
        osys = poppy_core.OpticalSystem("test")
        osys.add_pupil(optics.CircularAperture(radius=1))
        osys.add_pupil(optics.CircularAperture(radius=0.5))
        osys.add_detector(pixelscale=0.1, fov_arcsec=2.0)

        source={'wavelengths': [1.0e-6, 1.1e-6, 1.2e-6, 1.3e-6], 'weights':[0.25, 0.25, 0.25, 0.25]}
        conf.use_fftw=False

        conf.use_multiprocessing=False
        psf_single, planes_single = osys.calc_psf(source=source, return_intermediates=True)

        conf.use_multiprocessing=True
        psf_multi, planes_multi = osys.calc_psf(source=source, return_intermediates=True)

        assert np.allclose(psf_single[0].data, psf_multi[0].data), \
            "PSF from multiprocessing does not match PSF from single process"


        assert len(planes_multi) == len(planes_single), \
            "Intermediate calculation planes from multiprocessing has wrong number of planes."

        for i in range(len(planes_single)):
            assert (np.allclose(planes_single[i].intensity, planes_multi[i].intensity)), \
                "Intermediate plane {} from multiprocessing does not match same plane from single process.".format(i)

        return psf_single, psf_multi


def test_estimate_nprocesses():
    """ Apply some basic functionality tests to the
    estimate nprocesses function.
    """
    osys = poppy_core.OpticalSystem("test")
    osys.add_pupil(optics.CircularAperture(radius=1))
    osys.add_pupil(optics.CircularAperture(radius=0.5))
    osys.add_detector(pixelscale=0.1, fov_arcsec=2.0)

    answer = utils.estimate_optimal_nprocesses(osys)

    #see if it's an int with a reasonable value
    assert type(answer) is int

    assert answer > 0, "Estimated optimal nprocesses must be positive integer"
    assert answer < 100, "Estimated optimal nprocesses is unreasonably large"

