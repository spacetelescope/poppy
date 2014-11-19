#Tests for individual Optic classes
from __future__ import (absolute_import, division, print_function, unicode_literals)

import matplotlib.pyplot as pl
import numpy as np
import astropy.io.fits as fits
import pytest

from .. import poppy_core 
from .. import optics
from .. import zernike
from .test_core import check_wavefront



wavelength=1e-6



#def test_OpticalElement():
#    pass


#def test_FITSOpticalElement():
#    pass

#def test_Rotation():
#    pass

def test_InverseTransmission():
    """ Verify this inverts the optic throughput appropriately"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    # vary uniform scalar transmission
    for transmission in np.arange(10, dtype=float)/10:

        optic = optics.ScalarTransmission(transmission=transmission)
        inverted = optics.InverseTransmission(optic)
        assert( np.all(  np.abs(optic.getPhasor(wave) - (1-inverted.getPhasor(wave))) < 1e-10 ))

    # vary 2d shape
    for radius in np.arange(10, dtype=float)/10:

        optic = optics.CircularAperture(radius=radius)
        inverted = optics.InverseTransmission(optic)
        assert( np.all(  np.abs(optic.getPhasor(wave) - (1-inverted.getPhasor(wave))) < 1e-10 ))


#------ Generic Analytic elements -----

def test_scalar_transmission():
    """ Verify this adjusts the wavefront intensity appropriately """
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    for transmission in [1.0, 1.0e-3, 0.0]:

        optic = optics.ScalarTransmission(transmission=transmission)
        assert( np.all(optic.getPhasor(wave) == transmission))


#------ Analytic Image Plane elements -----

def test_RectangularFieldStop():
    optic= optics.RectangularFieldStop(width=1, height=10)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1000 # 1/10 of the 1e4 element array


def test_SquareFieldStop():
    optic= optics.SquareFieldStop(size=2)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 400 # 1/10 of the 1e4 element array



def test_BarOcculter():
    optic= optics.BarOcculter(width=1, angle=0)
    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square

    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 9000 # 9/10 of the 1e4 element array

#def test_rotations_RectangularFieldStop():
#
#    # First let's do a rotation of the wavefront itself by 90^0 after an optic
#
#    # now try a 90^0 rotation for the field stop at that optic. Assuming perfect system w/ no aberrations when comparing rsults. ? 
#    fs = poppy_core.RectangularFieldStop(width=1, height=10, ang;le=90)
#    wave = poppy_core.Wavefront(npix=100, pixelscale=0.1, wavelength=1e-6) # 10x10 arcsec square
#
#    wave*= fs
#    assert wave.shape[0] == 100
#    assert fs.intensity.sum() == 1000 # 1/10 of the 1e4 element array
#
#


#------ Analytic Pupil Plane elements -----

def test_ParityTestAperture():
    """ Verify that this aperture is not symmetric in either direction"""
    wave = poppy_core.Wavefront(npix=100, wavelength=wavelength)

    array = optics.ParityTestAperture().getPhasor(wave)

    assert np.any(array[::-1,:] != array)
    assert np.any(array[:,::-1] != array)


def test_RectangleAperture():
    """ Test rectangular aperture
    based on areas of 2 different rectangles, 
    and also that the rotation works to swap the axes
    """
    optic= optics.RectangleAperture(width=5, height=3)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1500 # 50*30 pixels of the 1e4 element array

    optic= optics.RectangleAperture(width=2, height=7)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave*= optic
    assert wave.shape[0] == 100
    assert wave.intensity.sum() == 1400 # 50*30 pixels of the 1e4 element array


    optic1= optics.RectangleAperture(width=2, height=7, rotation=90)
    optic2= optics.RectangleAperture(width=7, height=2)
    wave1 = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave2= poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 arcsec square
    wave1*= optic1
    wave2*= optic2

    assert wave1.shape[0] == 100
    assert np.all(np.abs(wave1.intensity - wave2.intensity) < 1e-6)




def test_HexagonAperture(display=False):
    """ Tests creating hexagonal aperture """

    # should make hexagon PSF and compare to analytic expression
    optic= optics.HexagonAperture(side=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: optic.display()

@pytest.mark.xfail
def test_MultiHexagonAperture(display=False):
    # should make multihexagon PSF and compare to analytic expression
    optic = optics.MultiHexagonAperture(side=1, rings=3)

    def _old_hexCenter(hex_index):
        ring = optic._hexInRing(hex_index)

        # now count around from the starting point:
        index_in_ring = hex_index - optic._nHexesInsideRing(ring) + 1  # 1-based
        #print("hex %d is %dth in its ring" % (hex_index, index_in_ring))

        angle_per_hex = 2 * np.pi / optic._nHexesInRing(ring)  # angle in radians
        # Check generalized _hexCenter using older non-general code for first 3 rings
        xpos = None
        if ring <= 1:
            radius = (optic.flattoflat + optic.gap) * ring
            angle = angle_per_hex * (index_in_ring - 1)
        elif ring == 2:
            if np.mod(index_in_ring, 2) == 1:
                radius = (optic.flattoflat + optic.gap) * ring  # JWST 'B' segments
            else:
                radius = optic.side * 3 + optic.gap * np.sqrt(3.) / 2 * 2  # JWST 'C' segments
            angle = angle_per_hex * (index_in_ring - 1)
        elif ring == 3:
            if np.mod(index_in_ring, ring) == 1:
                radius = (optic.flattoflat + optic.gap) * ring  # JWST 'B' segments
                angle = angle_per_hex * (index_in_ring - 1)
            else:  # C-like segments (in pairs)
                ypos = 2.5 * (optic.flattoflat + optic.gap)
                xpos = 1.5 * optic.side + optic.gap * np.sqrt(3) / 4
                radius = np.sqrt(xpos ** 2 + ypos ** 2)
                Cangle = np.arctan2(xpos, ypos)

                if np.mod(index_in_ring, 3) == 2:
                    last_B_angle = ((index_in_ring - 1) // 3) * 3 * angle_per_hex
                    angle = last_B_angle + Cangle * np.mod(index_in_ring - 1, 3)
                else:
                    next_B_angle = (((index_in_ring - 1) // 3) * 3 + 3) * angle_per_hex
                    angle = next_B_angle - Cangle
                xpos = None
        # now clock clockwise around the ring (for rings <=3 only)
        if xpos is None:
            ypos = radius * np.cos(angle)
            xpos = radius * np.sin(angle)
        return ypos, xpos

    for i in optic.segmentlist:
            oldy, oldx = _old_hexCenter(i)
            newy, newx = optic._hexCenter(i)
            diffy, diffx = oldy - newy, oldx - newx
            assert oldy - newy == diffy, "wtf?"
            _err_msg = "Disagreement between general case and existing special-case code for " \
                   "MultiHexagonAperture._hexCenter (old - new = ({}, {}), seg={}, ring={})"
            assert np.isclose(oldy, newy) and np.isclose(oldx, newx), _err_msg.format(
                oldy - newy,
                oldx - newx,
                i,
                optic._hexInRing(i)
            )
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave *= optic
    if display:
        optic.display()


def test_NgonAperture(display=False):
    """ Test n-gon aperture

    Note we could better test this if we impemented symmetry checks using the rotation argument?
    """
    # should make n-gon PSF for n=4, 6 and compare to square and hex apertures
    optic= optics.NgonAperture(nsides=4, radius=1, rotation=45)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: 
        pl.subplot(131)
        optic.display()

    optic= optics.NgonAperture(nsides=5, radius=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: 
        pl.subplot(132)
        optic.display()



    optic= optics.NgonAperture(nsides=6, radius=1)
    wave = poppy_core.Wavefront(npix=100, diam=10.0, wavelength=1e-6) # 10x10 meter square
    wave*= optic
    if display: 
        pl.subplot(133)
        optic.display()



def test_ObscuredCircularAperture_Airy(display=False):
    """ Compare analytic 2d Airy function with the results of a POPPY
    numerical calculation of the PSF for a circular aperture.
    
    Note that we expect very close but not precisely perfect agreement due to
    the quantization of the POPPY PSF relative to a perfect geometric circle.
    """
    
    from ..misc import airy_2d

    pri_diam = 1
    sec_diam = 0.4
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=pri_diam, obscuration=sec_diam/pri_diam)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    osys.addPupil( 
            optics.CompoundAnalyticOptic( [optics.CircularAperture(radius=pri_diam/2) ,
                                           optics.SecondaryObscuration(secondary_radius=sec_diam/2, n_supports=0) ]) )
    osys.addDetector(pixelscale=0.010,fov_pixels=512, oversample=1)
    numeric = osys.calcPSF(wavelength=1.0e-6, display=False)

    # Comparison
    difference = numeric[0].data-analytic
    #assert np.all(np.abs(difference) < 3e-5)
    

    if display:
        from .. import utils
        #comparison of the two
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=1e-6, vmax=1e-2)

        pl.figure(figsize=(15,5))
        pl.subplot(141)
        ax1=pl.imshow(analytic, norm=norm)
        pl.title("Analytic")
        pl.subplot(142)
        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_PSF(numeric, vmin=1e-6, vmax=1e-2, colorbar=False)
        pl.title("Numeric")
        pl.subplot(143)
        ax2=pl.imshow(numeric[0].data-analytic, norm=norm)
        pl.title("Difference N-A")
        pl.subplot(144)
        ax2=pl.imshow(np.abs(numeric[0].data-analytic) < 1e-4)
        pl.title("Difference <1e-4")


def test_CompoundAnalyticOptic(display=False):
    wavelen = 2e-6
    nwaves = 2
    r = 3

    osys_compound = poppy_core.OpticalSystem()
    osys_compound.addPupil(
        optics.CompoundAnalyticOptic([
            optics.CircularAperture(radius=r),
            optics.ThinLens(nwaves=nwaves, reference_wavelength=wavelen,
                            radius=r)
        ])
    )
    osys_compound.addDetector(pixelscale=0.010, fov_pixels=512, oversample=1)
    psf_compound = osys_compound.calcPSF(wavelength=wavelen, display=False)

    osys_separate = poppy_core.OpticalSystem()
    osys_separate.addPupil(optics.CircularAperture(radius=r))    # pupil radius in meters
    osys_separate.addPupil(optics.ThinLens(nwaves=nwaves, reference_wavelength=wavelen,
                                           radius=r))
    osys_separate.addDetector(pixelscale=0.01, fov_pixels=512, oversample=1)
    psf_separate = osys_separate.calcPSF(wavelength=wavelen, display=False)

    if display:
        from matplotlib import pyplot as plt
        from poppy import utils
        plt.figure()
        plt.subplot(1, 2, 1)
        utils.display_PSF(psf_separate, title='From Separate Optics')
        plt.subplot(1, 2, 2)
        utils.display_PSF(psf_compound, title='From Compound Optics')

    difference = psf_compound[0].data - psf_separate[0].data

    assert np.all(np.abs(difference) < 1e-3)



def test_AsymmetricObscuredAperture(display=False):
    """  Test that we can run the code with asymmetric spiders
    """
    
    from ..misc import airy_2d

    pri_diam = 1
    sec_diam = 0.4
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=pri_diam, obscuration=sec_diam/pri_diam)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    osys.addPupil( 
            optics.CompoundAnalyticOptic( [optics.CircularAperture(radius=pri_diam/2) ,
                                           optics.AsymmetricSecondaryObscuration(secondary_radius=sec_diam/2, support_angle=[0,150,210], support_width=0.1) ]) )
    osys.addDetector(pixelscale=0.030,fov_pixels=512, oversample=1)
    if display: osys.display()
    numeric = osys.calcPSF(wavelength=1.0e-6, display=False)

    # Comparison
    difference = numeric[0].data-analytic
    #assert np.all(np.abs(difference) < 3e-5)
    

    if display:
        from .. import utils
        #from matplotlib.colors import LogNorm
        #norm = LogNorm(vmin=1e-6, vmax=1e-2)

        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_PSF(numeric, vmin=1e-8, vmax=1e-2, colorbar=False)
        #pl.title("Numeric")


def test_ThinLens(display=False):
    pupil_radius = 1

    pupil = optics.CircularAperture(radius=pupil_radius)
    # let's add < 1 wave here so we don't have to worry about wrapping
    lens = optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6, radius=pupil_radius)
    wave = poppy_core.Wavefront(npix=1024, diam=3.0, wavelength=1e-6)
    wave *= pupil
    wave *= lens

    # The Zernike is normalized so rho == 1.0 at pupil_radius, and evaluated at pixels where
    # rho <= 1.0. If there is no pixel centered exactly at pupil_radius (not unlikely), the
    # resulting phase array will only get within a fraction of a percent of pi/2. That is why
    # the threshold is comparatively high versus the one used below to compare two optical systems
    # that should have "exactly" the same output.
    assert np.abs(wave.phase.max() - np.pi/2) < 2e-4
    assert np.abs(wave.phase.min() + np.pi/2) < 2e-4

    # test to ensure null optical elements don't change ThinLens behavior
    # https://github.com/mperrin/poppy/issues/14
    osys = poppy_core.OpticalSystem()
    osys.addPupil(optics.CircularAperture(radius=1))
    for i in range(10):
        osys.addImage()
        osys.addPupil()

    osys.addPupil(optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6,
                                  radius=pupil_radius))
    osys.addDetector(pixelscale=0.01, fov_arcsec=3.0)
    psf = osys.calcPSF(wavelength=1e-6)

    osys2 = poppy_core.OpticalSystem()
    osys2.addPupil(optics.CircularAperture(radius=1))
    osys2.addPupil(optics.ThinLens(nwaves=0.5, reference_wavelength=1e-6,
                                   radius=pupil_radius))
    osys2.addDetector(pixelscale=0.01, fov_arcsec=3.0)
    psf2 = osys2.calcPSF()

    THRESHOLD = 1e-19
    assert np.std(psf[0].data - psf2[0].data) < THRESHOLD, (
        "ThinLens shouldn't be affected by null optical elements! Introducing extra image planes "
        "raised std(psf_with_extras - psf_without_extras) above {}".format(THRESHOLD)
    )

def test_ZernikeOptic():
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeOptic
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=1)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=WAVELENGTH)  # 10x10 meter square
    tl_wave *= pupil
    tl_wave *= lens

    zern_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=WAVELENGTH)  # 10x10 meter square
    zernike_lens = optics.ZernikeOptic(
        coefficients=[
            (2, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))),
        ],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_lens

    stddev = np.std(zern_wave.phase - tl_wave.phase)

    assert stddev < 1e-16, ("ZernikeOptic disagrees with ThinLens! stddev {}".format(stddev))

def test_ParameterizedDistortion():
    # verify that we can reproduce the same behavior as ZernikeOptic
    # using ParameterizedDistortion
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=1)

    zern_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=1e-6)  # 10x10 meter square
    zernike_lens = optics.ZernikeOptic(
        coefficients=[
            (2, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))),
            (1, -1, 2e-7),
            (2, 2, 3e-8)
        ],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_lens

    parameterized_distortion = optics.ParameterizedDistortion(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        basis_factory=zernike.zernike_basis,
        radius=RADIUS
    )

    pd_wave = poppy_core.Wavefront(npix=101, diam=3.0, wavelength=1e-6) # 10x10 meter square
    pd_wave *= pupil
    pd_wave *= parameterized_distortion

    stddev = np.std(pd_wave.phase - zern_wave.phase)

    assert stddev < 1e-16, ("ParameterizedDistortion disagrees with "
                            "ZernikeOptic! stddev {}".format(stddev))

