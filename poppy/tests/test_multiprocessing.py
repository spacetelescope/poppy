#Test functions for poppy multiprocessing

from .. import poppy_core 
from .. import optics
from .. import conf

import numpy as np
import astropy.io.fits as fits

try:
    import pytest
    _HAVE_PYTEST = True
except:
    _HAVE_PYTEST = False


if _HAVE_PYTEST:

    #@pytest.mark.xfail
    # Just skip this test entirely for right now because sometimes it hangs the
    # entire Python process...
    @pytest.mark.skipif(True, reason="Intermittent Python interpreter hangs with multiprocessing")  
    def test_basic_multiprocessing():
        osys = poppy_core.OpticalSystem("test")
        pupil = optics.CircularAperture(radius=1)
        osys.addPupil(pupil) #function='Circle', radius=1)
        osys.addDetector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

        source={'wavelengths': [1.0e-6, 1.1e-6, 1.2e-6, 1.3e-6], 'weights':[0.25, 0.25, 0.25, 0.25]}
        conf.use_fftw=False

        conf.use_multiprocessing=False
        psf_single = osys.calcPSF(source=source)

        conf.use_multiprocessing=True
        psf_multi = osys.calcPSF(source=source)

        assert np.allclose(psf_single[0].data, psf_multi[0].data), \
            "PSF from multiprocessing does not match PSF from single process"

        return psf_single, psf_multi

