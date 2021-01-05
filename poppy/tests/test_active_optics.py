
import poppy
import astropy.units as u
import numpy as np


def test_tip_tilt_optic():
    """Test basic operation of tip tilt optic class"""

    ap = poppy.HexagonAperture()

    tt = poppy.active_optics.TipTiltStage(ap, include_factor_of_two=False)

    wave = poppy.Wavefront(npix=128, diam=2*u.m)
    trans = ap.get_transmission(wave)
    assert np.allclose(tt.get_transmission(wave), trans), "Transmission does not match expectations"

    assert np.allclose(tt.get_opd(wave), 0), "OPD without tilt does not match expectation"

    for ztilt in [1e-6, 2e-6]:
        tt.set_tip_tilt(ztilt, 0)

        def rms(ar, mask ):
            return np.sqrt((ar[mask]**2).mean())

        rmstilt = rms(tt.get_opd(wave), trans != 0)

        assert np.allclose(rmstilt, ztilt, rtol=0.1), f"OPD with tilt {ztilt} does not match expectations. Expected {ztilt}, got {rmstilt} RMS"
        assert np.allclose(tt.get_transmission(wave), trans), "Transmission does not match expectations"



