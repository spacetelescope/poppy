# This file contains code for testing various error handlers and user interface edge cases,
# as opposed to testing the main body of functionality of the code.

from .. import poppy_core
from .. import optics
from .. import matrixDFT

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

    def test_matrixDFT_catch_invalid_parameters():
        import numpy as np

        # invalid nlamD
        plane = np.zeros( (100,100))
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, 'not allowed', 100)   # wrong type
        assert excinfo.value.message.startswith("'nlamD' must be supplied as a scalar (for square arrays) or as ")
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, (1, 2, 3), 100)       # wrong dimensionality
        assert excinfo.value.message.startswith("'nlamD' must be supplied as a scalar (for square arrays) or as ")

        # invalid npix
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, 10, "invalid")       # wrong type
        assert excinfo.value.message.startswith("'npix' must be supplied as a scalar (for square arrays) or as ")
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, 10, (4,5,6))         # wrong dimensionality
        assert excinfo.value.message.startswith("'npix' must be supplied as a scalar (for square arrays) or as ")
        with pytest.raises(TypeError) as excinfo:
            matrixDFT.matrix_dft(plane, 10, 3.1415)          # must be an integer
        assert excinfo.value.message.startswith("'npix' must be supplied as integer value(s)")


        #invalid offset
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, 10, 50, offset=(1,2,3), centering='adjustable')
        assert excinfo.value.message.startswith("'offset' must be supplied as a 2-tuple with")


        # invalid centering
        with pytest.raises(ValueError) as excinfo:
            matrixDFT.matrix_dft(plane, 10, 50, centering='Diagonal')
        assert excinfo.value.message.startswith("Invalid centering style")


    def test_inverseTransmission_invalid_parameters():
        import numpy as np
        with pytest.raises(ValueError) as excinfo:
            optics.InverseTransmission()
        assert excinfo.value.message.startswith("Need to supply an valid optic to invert!")

        with pytest.raises(ValueError) as excinfo:
            optics.InverseTransmission(optic=np.ones((100,100)))
        assert excinfo.value.message.startswith("Need to supply an valid optic to invert!")





    def test_CircularAperture_invalid_parameters():
        with pytest.raises(TypeError) as excinfo:
            optics.CircularAperture(radius='a')
        assert excinfo.value.message.startswith("Argument 'radius' must be the radius of the pupil in meters")

