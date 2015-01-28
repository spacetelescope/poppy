# This file contains code for testing various error handlers and user interface edge cases,
# as opposed to testing the main body of functionality of the code.

from .. import poppy_core
from .. import optics

try:
    import pytest
    _HAVE_PYTEST = True
except:
    _HAVE_PYTEST = False




if _HAVE_PYTEST:
    def test_calcPSF_catch_invalid_wavelength():
        """ Test that it rejects incompatible wavelengths"""

        osys = poppy_core.OpticalSystem("test")
        pupil = optics.CircularAperture(radius=1)
        osys.addPupil(pupil) #function='Circle', radius=1)
        osys.addDetector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flu

        with pytest.raises(ValueError) as excinfo:
            psf = osys.calcPSF('cat')
        assert excinfo.value.message.startswith('You have specified an invalid wavelength to calcPSF:')


        source={'wavelengths': [1.0e-6, 1.1e-6, 1.2e-6, 1.3e-6], 'weights':[0.25, 0.25, 0.25, 0.25]}
        with pytest.raises(ValueError) as excinfo:
            psf = osys.calcPSF(source)
        assert excinfo.value.message.startswith('You have specified an invalid wavelength to calcPSF:')

