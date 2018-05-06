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


def test_basic_hex_dm():
    """ A simple test for the hex segmented deformable mirror code -
    can we move actuators, and does adding nonzero WFE result in decreased Strehl?"""

    dm = dms.HexSegmentedDeformableMirror(rings=1)

    osys = poppy_core.OpticalSystem(npix=256)
    osys.add_pupil(optics.CircularAperture())
    osys.add_pupil(dm)
    osys.add_detector(0.010, fov_pixels=128)

    psf_perf = osys.calc_psf()

    for act in ( 3,6):
        dm.set_actuator(act, 1e-6, 0, 1e-4) # 1000 nm = 1 micron
        assert np.allclose(dm.surface[act,0],  1e-6), "Segment {} did not move as expected using bare floats".format(actx,acty)
        assert np.allclose(dm.surface[act,2],  1e-4), "Segment {} did not move as expected using bare floats".format(actx,acty)

    for act in ( 5,2):
        dm.set_actuator(act, 1*u.nm, 0, 0) # 1 nm
        assert np.allclose(dm.surface[act,0],  1e-9), "Segment {} did not move as expected in piston using astropy quantities".format(actx,acty)

    for act in ( 1,4):
        dm.set_actuator(act,  0, 1*u.milliradian, 0) # 1 nm
        assert np.allclose(dm.surface[act,1],  1e-3), "Segment {} did not move as expected in tilt using astropy quantities".format(actx,acty)



    psf_aberrated = osys.calc_psf()

    peak_perf = psf_perf[0].data.max()
    peak_aberrated = psf_aberrated[0].data.max()
    assert peak_aberrated < peak_perf, "Adding nonzero WFE did not decrease the Strehl as expected."

    return psf_aberrated, psf_perf, osys

