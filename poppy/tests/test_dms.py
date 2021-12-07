# Tests for deformable mirror classes

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


def test_cont_dm_flips(display=False):
    """ Test that we can flip a deformable mirror in X or Y
    """

    testdm1 = dms.ContinuousDeformableMirror(dm_shape=(5,5))
    testdm2 = dms.ContinuousDeformableMirror(dm_shape=(5,5), flip_x=True)
    testdm3 = dms.ContinuousDeformableMirror(dm_shape=(5,5), flip_y=True)

    for t in [testdm1, testdm2, testdm3]:
        t.set_actuator(0,2, 1e-6)
        t.set_actuator(1,1, -0.5e-6)

    w = poppy_core.Wavefront(npix=5)

    if display:
        import matplotlib.pyplot as plt
        testdm1.display(what='both', npix=5)
        testdm2.set_actuator(0,2, 1e-6)
        plt.figure()
        testdm2.display(what='both', npix=5)
        plt.figure()
        testdm3.display(what='both', npix=5)

    opd_1 = testdm1.get_opd(w)
    opd_2 = testdm2.get_opd(w)
    opd_3 = testdm3.get_opd(w)

    assert np.allclose(opd_1, opd_2[:, ::-1]), 'Problem with flip_x'
    assert np.allclose(opd_1, opd_3[::-1]), 'Problem with flip_y'


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


def test_hex_dm_rotation(npix=128, ptt = (0, 1e-6, 0), plot=False):
    """
    Verify the hexagonally segmented DM can be rotated without introducing unexpected errors in the
    OPD models. Verify rotating the DM also rotates the OPD accordingly, and does not change its values.

	npix : number of pixels for test OPD evaluation
	ptt : piston, tip, tilt for segment move for use in test OPD evaluation
    plot : make optional plots
    """
    dm = dms.HexSegmentedDeformableMirror(rings=2, rotation=0)
    dm.set_actuator(1, *ptt )

    ref_opd = dm.sample(what='opd', npix=npix)

    for rot in [90, -90, 180]:
        dm = dms.HexSegmentedDeformableMirror(rings=2, rotation=rot)
        dm.set_actuator(1, *ptt )
        opd = dm.sample(what='opd', npix=npix)

        assert np.allclose(opd.max(), ref_opd.max()), f"OPD max pixel changed unexpectedly for rotation = {rot}"
        nrotations = rot//90
        assert np.allclose(np.rot90(opd, k=nrotations), ref_opd), f"OPD rotation not as expected for rotation = {rot}, k={nrotations}"

        if plot:
            plt.figure()
            dm.display(what='opd', npix=npix, opd_vmax=1e-6, title=f'DM with {rot} rotation', colorbar_orientation='vertical')


def test_factor_of_two_surface_vs_wfe():
    """For both types of DM, test the factor of two option
    """

    # Create two DMs, one commanded in WFE and the other in surface
    dm = dms.ContinuousDeformableMirror()
    dm2 = dms.ContinuousDeformableMirror(include_factor_of_two=True)

    for actx, acty in ( (3,7), (7,3)):
        dm2.set_actuator(actx, acty, 1e-6) # 1000 nm = 1 micron
        dm.set_actuator(actx, acty, 1e-6) # 1000 nm = 1 micron

    w = poppy_core.Wavefront(npix=128, diam=dm._aperture.radius*2)
    assert np.allclose(dm.get_opd(w)*2, dm2.get_opd(w)), "The continuous DM response should be 2x greater if include_factor_of_two is set"

    # Now repeat for the hex dm
    hexdm = dms.HexSegmentedDeformableMirror(rings=1)
    hexdm2 = dms.HexSegmentedDeformableMirror(rings=1, include_factor_of_two=True)
    for act in ( 3,6):
        hexdm.set_actuator(act, 1e-6, 0, 1e-4) # 1000 nm = 1 micron
        hexdm2.set_actuator(act, 1e-6, 0, 1e-4) # 1000 nm = 1 micron
    w = poppy_core.Wavefront(npix=128, diam=hexdm.pupil_diam)
    assert np.allclose(hexdm.get_opd(w)*2, hexdm2.get_opd(w)), "The hexagonal DM response should be 2x greater if include_factor_of_two is set"


def test_basic_circular_dm():
    """ A simple test for the circularly segmented deformable mirror code -
    can we move actuators, and does adding nonzero WFE result in decreased Strehl?"""

    dm = dms.CircularSegmentedDeformableMirror(rings=1)

    osys = poppy_core.OpticalSystem(npix=256)
    osys.add_pupil(dm)
    osys.add_detector(0.010, fov_pixels=128)

    psf_perf = osys.calc_psf()

    for act in ( 3,6):
        dm.set_actuator(act, 1e-6, 0, 1e-7) # 1000 nm = 1 micron
        assert np.allclose(dm.surface[act,0],  1e-6), "Segment {} did not move as expected using bare floats".format(actx,acty)
        assert np.allclose(dm.surface[act,2],  1e-7), "Segment {} did not move as expected using bare floats".format(actx,acty)

    for act in ( 5,2):
        dm.set_actuator(act, 1*u.nm, 0, 0) # 1 nm
        assert np.allclose(dm.surface[act,0],  1e-9), "Segment {} did not move as expected in piston using astropy quantities".format(actx,acty)

    for act in ( 1,4):
        dm.set_actuator(act,  0, 1*u.microradian, 0) # 1 nm
        assert np.allclose(dm.surface[act,1],  1e-6), "Segment {} did not move as expected in tilt using astropy quantities".format(actx,acty)

    psf_aberrated = osys.calc_psf()

    peak_perf = psf_perf[0].data.max()
    peak_aberrated = psf_aberrated[0].data.max()
    assert peak_aberrated < peak_perf, "Adding nonzero WFE did not decrease the Strehl as expected."

    return psf_aberrated, psf_perf, osys
