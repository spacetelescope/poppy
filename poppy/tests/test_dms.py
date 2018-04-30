# Tests for deformable mirror classes
from __future__ import (absolute_import, division, print_function, unicode_literals)

import matplotlib.pyplot as pl
import numpy as np
import astropy.io.fits as fits
import astropy.units as u

from .. import poppy_core
from .. import optics
from .. import dms


def test_basic_continuous_dm():
    """ A simple test for the deformable mirror code - can we move actuators, and
    does adding nonzero WFE result in decreased Strehl?"""

    dm = dms.ContinuousDeformableMirror()

    osys = poppy_core.OpticalSystem(npix=256)
    osys.add_pupil(optics.CircularAperture())
    osys.add_pupil(dm)
    osys.add_detector(0.010, fov_pixels=128)

    psf_perf = osys.calc_psf()

    for actx, acty in ( (3,7), (7,3)):
        dm.set_actuator(actx, acty, 1e-6) # 1000 nm = 1 micron
        assert np.allclose(dm.surface[acty, actx],  1e-6), "Actuator ({}, {}) did not move as expected using bare floats".format(actx,acty)

    for actx, acty in ( (5,2), (6,9)):
        dm.set_actuator(actx, acty, 1*u.nm) # 1 nm
        assert np.allclose(dm.surface[acty, actx],  1e-9), "Actuator ({}, {}) did not move as expected using astropy quantities".format(actx,acty)


    psf_aberrated = osys.calc_psf()

    peak_perf = psf_perf[0].data.max()
    peak_aberrated = psf_aberrated[0].data.max()
    assert peak_aberrated < peak_perf, "Adding nonzero WFE did not decrease the Strehl as expected."

    return psf_aberrated, psf_perf, osys

