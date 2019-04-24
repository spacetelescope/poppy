
import numpy as np
import astropy.units as u

from .. import poppy_core
from .. import optics
from .. import zernike
from .. import wfe

NWAVES = 0.5
WAVELENGTH = 1e-6
RADIUS = 1.0
NPIX = 101
DIAM = 3.0

def test_ZernikeAberration():
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeAberration
    pupil = optics.CircularAperture(radius=RADIUS)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    tl_wave *= pupil
    tl_wave *= lens

    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    # need a negative sign in the following b/c of different sign conventions for
    # zernikes vs "positive" and "negative" lenses.
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, -NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_lens

    stddev = np.std(zern_wave.phase - tl_wave.phase)

    assert stddev < 1e-16, ("ZernikeAberration disagrees with ThinLens! stddev {}".format(stddev))

def test_wavefront_or_meters_decorator():
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    opd_waves_a = zernike_lens.get_opd(WAVELENGTH)
    opd_waves_b = zernike_lens.get_opd(poppy_core.Wavefront(wavelength=WAVELENGTH))

    stddev = np.std(opd_waves_a - opd_waves_b)
    assert stddev < 1e-16, "OPD map disagreement based on form of argument to get_opd!"

def test_zernike_get_opd():
    zernike_optic = wfe.ZernikeWFE(coefficients=[NWAVES * WAVELENGTH,], radius=RADIUS)
    opd_map = zernike_optic.get_opd(WAVELENGTH, units='meters')
    assert np.max(opd_map) == NWAVES * WAVELENGTH

    opd_map_waves = zernike_optic.get_opd(WAVELENGTH, units='waves')
    assert np.max(opd_map_waves) == NWAVES

def test_ParameterizedAberration():
    # verify that we can reproduce the same behavior as ZernikeAberration
    # using ParameterizedAberration
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=RADIUS)

    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=1e-6)
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

    pd_wave = poppy_core.Wavefront(npix=NPIX, diam=3.0, wavelength=1e-6)
    pd_wave *= pupil
    pd_wave *= parameterized_distortion

    np.testing.assert_allclose(pd_wave.phase, zern_wave.phase,
                               err_msg="ParameterizedAberration disagrees with ZernikeAberration")


def test_StatisticalPSDWFE():

    # Verify that we produce phase screen with input RMS WFE
    NPIX = 256    # 101 is too small and results in issues for this test
    def rms(opd):
        """Calculate the RMS WFE of a wf with a zero mean."""
        rms = np.sqrt(np.mean(np.square(opd)))
        return rms

    wvferr = 134*u.nm
    psd_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=1e-6)
    psd_wfe = wfe.StatisticalPSDWFE(index=3.0, wfe=wvferr, radius=RADIUS, seed=None)

    psd_opd = psd_wfe.get_opd(psd_wave)
    assert np.isclose(rms(psd_opd), wvferr.to(u.m).value), "WFE doesn't match input WFE."

    # Verify that we reproduce the correct spectral index of the PSD
    def radial_profile(image, center=None):
        """ Compute a radial profile of the image.
        Stripped down version of puppy.utils.radial_profile().
        Keeps only the functionality that is needed for this test.
        Parameters
        ----------
        image : array
            image to get a radial profile of
        center : tuple of floats
            Coordinates (x,y) of PSF center, in pixel units. Default is image center.
        Returns
        --------
        results : tuple
            Tuple containing (radius, profile).
        """

        y, x = np.indices(image.shape, dtype=float)
        if center is None:
            # get exact center of image
            # center = (image.shape[1]/2, image.shape[0]/2)
            center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

        x -= center[0]
        y -= center[1]

        r = np.sqrt(x ** 2 + y ** 2)

        # Use full image
        ind = np.argsort(r.flat)
        sr = r.flat[ind]  # sorted r
        sim = image.flat[ind]  # sorted image

        ri = sr.astype(int)  # sorted r as int
        deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
        rind = np.where(deltar)[0]
        nr = rind[1:] - rind[:-1]  # number in radius bin
        csim = np.nan_to_num(sim).cumsum(dtype=float)  # cumulative sum to figure out sums for each bin
        # np.nancumsum is implemented in >1.12
        tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
        radialprofile = tbin / nr

        # pre-pend the initial element that the above code misses.
        radialprofile2 = np.empty(len(radialprofile) + 1)
        if rind[0] != 0:
            radialprofile2[0] = csim[rind[0]] / (
                    rind[0] + 1)  # if there are multiple elements in the center bin, average them
        else:
            radialprofile2[0] = csim[0]  # otherwise if there's just one then just take it.
        radialprofile2[1:] = radialprofile
        rr = np.arange(
            len(radialprofile2)) +  0.5  # these should be centered in the bins, so add a half.

        return rr, radialprofile2

    inv_psd = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(psd_opd)))
    rad, prof = radial_profile(np.abs(inv_psd) ** 2, center=(int(NPIX/2), int(NPIX/2)))

    # Make comparison slope
    com = rad[5:] ** (-3)
    com *= prof[5] / com.max()

    # Verify that the slopes of prof and com are the same
    del_prof = prof[105] - prof[5]
    del_com = com[100] - com[0]

    assert np.isclose(del_prof, del_com), "Spectral indices do not match."
