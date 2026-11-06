'''
Rachel Morgan, Summer 2020
'''

import astropy.units as u
import numpy as np

import poppy
from poppy import sub_sampled_optics


def test_ShackHartmannWFS(n_lenslets=2):
    """
    Test Shack Hartmann Wavefront Sensor class functionality.

    Verifies that spot centroid measurement changes when optical system reflects off a deformed mirror

    Parameters
    ----------
    n_lenslets = 2
        n_lenslets parameter for ShackHartmannWavefrontSensor; the number of lenslets
        per side of a square grid.

    """
    wavelength = 635*u.nm
    # define nominal shack hartmann wavefront sensor:
    shwfs = sub_sampled_optics.ShackHartmannWavefrontSensor(n_lenslets=n_lenslets)
    dm_size = shwfs.lenslet_pitch*24

    # setup flat wavefront and calculate nominal spot locations on SHWFS
    wf_flat = poppy.Wavefront(diam=dm_size, wavelength=wavelength, npix=int((shwfs.lenslet_pitch/shwfs.pixel_pitch).value*shwfs.n_lenslets*2))
    wf_flat *= poppy.CircularAperture(radius = dm_size/2)
    # sample flat wavefront:
    shwfs.sample_wf(wf_flat)
    shwfs.get_psfs()
    flat_centroid_list = shwfs.get_centroids()

    ## define DM
    act_x = 2
    act_y = 2
    stroke = .3e-6
    dm_actuator_pitch = dm_size/4
    dm = poppy.dms.ContinuousDeformableMirror(dm_shape=(4,4),
        actuator_spacing=dm_actuator_pitch, radius=dm_size/2,
        include_factor_of_two = True)
    dm.set_actuator(act_x, act_y, stroke)

    # define Wavefront object for simulation, reflect off DM
    wf = poppy.Wavefront(diam=dm_size, wavelength=wavelength, npix=int((shwfs.lenslet_pitch/shwfs.pixel_pitch).value*shwfs.n_lenslets*2))
    wf *= poppy.CircularAperture(radius = dm_size/2)
    wf *= dm

    #sample actual wf and propagate to detector:
    shwfs.sample_wf(wf)
    shwfs.get_psfs()
    # reconstruct wavefront and ensure that it is nonzero after reflecting off of deformed DM
    reconstruction = shwfs.reconstruct_wavefront(flat_centroid_list).value

    assert np.count_nonzero(reconstruction)>0, "Wavefront reconstruction was not non-zero as expected for input DM actuation"

    return np.count_nonzero(reconstruction)>0

if __name__ == "__main__":
    # execute only if run as a script
    test_ShackHartmannWFS()




def test_subapertures_array_api_strict(monkeypatch):
    """Sampling a wavefront into subapertures and re-tiling runs under a strict array API namespace."""
    import array_api_strict

    from .. import accel_math, geometry, matrixDFT, optics, poppy_core, utils

    def run():
        shwfs = sub_sampled_optics.ShackHartmannWavefrontSensor(n_lenslets=2)
        npix = int((shwfs.lenslet_pitch / shwfs.pixel_pitch).value * shwfs.n_lenslets * 2)
        wf = poppy.Wavefront(diam=shwfs.lenslet_pitch * 24, wavelength=635 * u.nm, npix=npix)
        wf *= poppy.CircularAperture(radius=shwfs.lenslet_pitch * 12)
        shwfs.sample_wf(wf)
        tiled = np.asarray(shwfs.get_wavefront_array().wavefront)
        shwfs.get_psfs()
        return [tiled, np.asarray(shwfs.get_wavefront_array().wavefront), shwfs.get_centroids()]

    monkeypatch.setattr(accel_math, "_USE_NUMEXPR", False)
    expected = run()
    for mod in (accel_math, poppy_core, utils, optics, geometry, matrixDFT, sub_sampled_optics):
        monkeypatch.setattr(mod, "xp", array_api_strict)
    for e, g in zip(expected, run()):
        np.testing.assert_allclose(g, e, rtol=1e-9, atol=1e-15)
