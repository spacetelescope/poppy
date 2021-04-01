
import poppy
import astropy.units as u
import numpy as np


def test_TipTiltStage(display=False, verbose=False):
    """ Test tip tilt stage moves the PSF by the requested amount
    """
    ap = poppy.HexagonAperture(flattoflat=0.75*u.m)
    det = poppy.Detector(pixelscale=0.1*u.arcsec/u.pix, fov_pixels=128)

    tt = poppy.active_optics.TipTiltStage(ap, include_factor_of_two=False)

    wave = poppy.Wavefront(npix=128, diam=1*u.m)

    trans = ap.get_transmission(wave)
    assert np.allclose(tt.get_transmission(wave), trans), "Transmission does not match expectations"
    assert np.allclose(tt.get_opd(wave), 0), "OPD without tilt does not match expectation"

    for tx, ty in ( (0*u.arcsec, 1*u.arcsec),
                    (1*u.arcsec, 0*u.arcsec),
                    (-0.23*u.arcsec, 0.65*u.arcsec)):
        for include_factor_of_two in [True, False]:

            if verbose:
                print(f"Testing {tx}, {ty}, with include_factor_of_two={include_factor_of_two}")

            tt.include_factor_of_two = include_factor_of_two
            tt.set_tip_tilt(tx, ty)

            wave = poppy.Wavefront(npix=64, diam=1*u.m)
            wave *= ap
            wave *= tt

            if display:
                plt.figure()
                wave.display(what='both')
                plt.suptitle(f"Wavefront with {tx}, {ty}")

            wave.propagate_to(det)

            if display:
                plt.figure()
                wave.display()
                plt.title(f"PSF with {tx}, {ty}")


            cen = poppy.measure_centroid(wave.as_fits(), boxsize=5, relativeto='center', units='arcsec')

            factor = 2 if include_factor_of_two else 1
            assert np.isclose(cen[1]*u.arcsec, tx*factor, atol=1e-4), "X pos not as expected"
            assert np.isclose(cen[0]*u.arcsec, ty*factor, atol=1e-4), f"Y pos not as expected: {cen[0]*u.arcsec}, {ty*factor}"


