from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

from .. import poppy_core
from .. import optics
from .. import zernike
from .. import wfe

def test_ZernikeAberration():
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeAberration
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=1)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=WAVELENGTH)  # 10x10 meter square
    tl_wave *= pupil
    tl_wave *= lens

    zern_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=WAVELENGTH)  # 10x10 meter square
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_lens

    stddev = np.std(zern_wave.phase - tl_wave.phase)

    assert stddev < 1e-16, ("ZernikeAberration disagrees with ThinLens! stddev {}".format(stddev))

def test_ParameterizedAberration():
    # verify that we can reproduce the same behavior as ZernikeAberration
    # using ParameterizedAberration
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=1)

    zern_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=1e-6)  # 10x10 meter square
    zernike_wfe = wfe.ZernikeWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_wfe

    parameterized_distortion = wfe.ParameterizedWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        basis_factory=zernike.zernike_basis,
        radius=RADIUS
    )

    pd_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=1e-6) # 10x10 meter square
    pd_wave *= pupil
    pd_wave *= parameterized_distortion

    stddev = np.std(pd_wave.phase - zern_wave.phase)

    assert stddev < 1e-16, ("ParameterizedAberration disagrees with "
                            "ZernikeAberration! stddev {}".format(stddev))
