# Test functions for core poppy functionality
import os

import numpy as np
from poppy.accel_math import xp   # May be numpy, or CuPy on GPU
from astropy.io import fits
import astropy.units as u
import pytest
import matplotlib.pyplot as plt

try:
    import scipy
except ImportError:
    scipy = None

import poppy
from .. import poppy_core
from .. import optics

####### Test Common Infrastructure #######

def check_wavefront(filename_or_hdulist, slice=0, ext=0, test='nearzero', comment=""):
    """ A helper routine to verify certain properties of a wavefront FITS file,
    as requested by some test routine. """
    if isinstance(filename_or_hdulist, str):
        hdulist = fits.open(filename_or_hdulist)
        filename = filename_or_hdulist
    elif isinstance(filename_or_hdulist, fits.HDUList):
        hdulist = filename_or_hdulist
        filename = 'input HDUlist'
    imstack = xp.asarray(hdulist[ext].data)  # extra asarray helps with GPU compatibility here
    im = imstack[slice,:,:]

    if test=='nearzero':
        return xp.all(xp.abs(im) < xp.finfo(im.dtype).eps * 10)
    elif test == 'is_real':
        #assumes output type = 'all'
        cplx_im = imstack[1,:,:] * xp.exp(1j * imstack[2, :, :])
        return xp.all(cplx_im.imag < xp.finfo(im.dtype).eps * 10)

wavelength=2e-6


######### Core tests functions #########

def test_basic_functionality():
    """ For one specific geometry, test that we get the expected value based on a prior reference
    calculation."""
    osys = poppy_core.OpticalSystem("test", oversample=1)
    pupil = optics.CircularAperture(radius=1)
    osys.add_pupil(pupil) #function='Circle', radius=1)
    osys.add_detector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    psf = osys.calc_psf(wavelength=1.0e-6)
    # we need to be a little careful here due to floating point math comparison equality issues... Can't just do a strict equality
    assert abs(psf[0].data.max() - 0.201) < 0.001

    # test the (fairly trivial) description function.
    # This prints a string to screen and there's nothing returned.
    osys.describe()

def test_input_wavefront_size():

    # if absolutely nothing is set then the default is 1024.
    # the oversample parameter multiplies that *only* if padding
    # is applied during an FFT propagation; by default there's no effect
    # in the unpadded array.
    for oversamp in (1,2,4):
        osys = poppy_core.OpticalSystem("test", oversample=oversamp, pupil_diameter = 1*u.meter)
        #pupil = optics.CircularAperture(radius=1)
        wf = osys.input_wavefront()
        expected_shape = (1024,1024) if (wf.ispadded == False) else (1024*oversamp, 1024*oversamp)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)


    # test setting the size based on the npix parameter, with no optical system planes
    # (so it gets the diameter from the optical system object)
    for size in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1, npix=size, pupil_diameter = 1*u.meter)
        #pupil = optics.CircularAperture(radius=1)
        wf = osys.input_wavefront()
        expected_shape = (size,size)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)

    # test setting the size based on the npix parameter, with a non-null optical system
    # (so it infers the system diameter from the first optic's diameter)
    for size in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1, npix=size)
        osys.add_pupil(optics.CircularAperture(radius=1))
        wf = osys.input_wavefront()
        expected_shape = (size,size)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)


    # test setting the size based on an input optical element
    for npix in [512, 1024, 2001]:
        osys = poppy_core.OpticalSystem("test", oversample=1)
        pupil = optics.CircularAperture(radius=1)
        pupil_fits = pupil.to_fits(npix=npix)
        osys.add_pupil(transmission=pupil_fits)

        wf = osys.input_wavefront()
        expected_shape = (npix,npix)
        assert pupil_fits[0].data.shape == expected_shape, 'FITS array from optic element is not the expected size: is {} expects {}'.format(pupil_fits[0].data.shape,  expected_shape)
        assert wf.shape == expected_shape, 'Wavefront is not the expected size: is {} expects {}'.format(wf.shape,  expected_shape)



def test_CircularAperture_Airy(display=False):
    """ Compare analytic 2d Airy function with the results of a POPPY
    numerical calculation of the PSF for a circular aperture.

    Note that we expect very close but not precisely perfect agreement due to
    the quantization of the POPPY PSF relative to a perfect geometric circle.
    """

    from ..misc import airy_2d
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=1)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    pupil = optics.CircularAperture(radius=0.5)
    osys.add_pupil(pupil)
    osys.add_detector(pixelscale=0.010,fov_pixels=512, oversample=1)
    numeric = osys.calc_psf(wavelength=1.0e-6, display=False)

    # Comparison. Extra xp.array cast needed for the GPU case
    difference = xp.array(numeric[0].data) - analytic
    assert xp.all(xp.abs(difference) < 3e-5)

    if display:
        from .. import utils
        #comparison of the two
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=1e-6, vmax=1e-2)

        plt.figure(figsize=(15,5))
        plt.subplot(141)
        ax1=poppy.utils.imshow(analytic, norm=norm)
        plt.title("Analytic")
        plt.subplot(142)
        #ax2=pl.imshow(numeric[0].data, norm=norm)
        utils.display_psf(numeric, vmin=1e-6, vmax=1e-2, colorbar=False)
        plt.title("Numeric")
        plt.subplot(143)
        ax2=poppy.utils.imshow(difference, norm=norm)
        plt.title("Difference N-A")
        plt.subplot(144)
        ax2=poppy.utils.imshow(xp.abs(difference) < 3e-5)
        plt.title("Difference <1e-5")


def test_multiwavelength_opticalsystem():
    """
    Tests the ability to just provide wavelengths, weights directly
    """
    wavelengths = [2.0e-6, 2.1e-6, 2.2e-6]
    weights = [0.3, 0.5, 0.2]

    osys = poppy_core.OpticalSystem("test")
    pupil = optics.CircularAperture(radius=1)
    osys.add_pupil(pupil) #function='Circle', radius=1)
    osys.add_detector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux


    psf = osys.calc_psf(wavelength=wavelengths, weight=weights)
    assert psf[0].header['NWAVES'] == len(wavelengths), \
        "Number of wavelengths in PSF header does not match number requested"

    # Check weighted sum
    output = np.zeros_like(psf[0].data)
    for wavelength, weight in zip(wavelengths, weights):
        output += weight * osys.calc_psf(wavelength=wavelength)[0].data

    assert np.allclose(psf[0].data, output), \
        "Multi-wavelength PSF does not match weighted sum of individual wavelength PSFs"

    # test that it's also possible to display a progress bar for multi wave calculations
    psf = osys.calc_psf(wavelength=wavelengths, weight=weights, progressbar=True)

    return psf


def test_normalization():
    """ Test that we can compute a PSF and get the desired flux,
    depending on the normalization """
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.add_pupil(pupil) #function='Circle', radius=6.5/2)
    osys.add_detector(pixelscale=0.01, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    from .. import conf
    conf.enable_flux_tests  = True

    # we need to be a little careful here due to floating point math comparison equality issues... Can't just do a strict equality

    # this should be very very close to one
    psf_last = osys.calc_psf(wavelength=1.0e-6, normalize='last')
    assert abs(psf_last[0].data.sum() - 1) < 0.01

    # this should be a little further but still pretty close
    psf_first = osys.calc_psf(wavelength=1.0e-6, normalize='first')
    assert abs(psf_first[0].data.sum() - 1) < 0.01
    assert abs(psf_first[0].data.sum() - 1) > 0.0001

    # for the simple optical system above, the 'first' and 'exit_pupil' options should be equivalent:
    psf_exit_pupil = osys.calc_psf(wavelength=1.0e-6, normalize='exit_pupil')
    assert (psf_exit_pupil[0].data.sum() - 1) < 1e-9
    assert abs(psf_exit_pupil[0].data - psf_first[0].data).max() < 1e-10


    # and if we make an pupil stop with half the radius we should get 1/4 the light if normalized to 'first'
    # but normalized to 1 if normalized to last_pupil
    osys2 = poppy_core.OpticalSystem("test", oversample=2)
    osys2.add_pupil(  optics.CircularAperture(radius=6.5/2) )
    osys2.add_pupil(  optics.CircularAperture(radius=6.5/2/2) )
    osys2.add_detector(pixelscale=0.01, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    psf_small_pupil_first = osys2.calc_psf(wavelength=1.0e-6, normalize='first')
    psf_small_pupil_exit  = osys2.calc_psf(wavelength=1.0e-6, normalize='exit_pupil')
    psf_small_pupil_last  = osys2.calc_psf(wavelength=1.0e-6, normalize='last')
    # normalized for the output to 1 we should of course get 1
    assert abs(psf_small_pupil_last[0].data.sum() - 1) < 1e-9
    # normalized to the exit pupil we should get near but not exactly 1 (due to finite FOV)
    assert abs(psf_small_pupil_exit[0].data.sum() - 1) < 0.01
    assert abs(psf_small_pupil_exit[0].data.sum() - 1) > 0.0001
    # normalized to the entrance pupil we should get very close to 4x over the exit pupil one
    # (not totally sure why the agreement isn't closer - presumably due to finite sampling quantization
    #  of the discretized arrays)
    assert abs(psf_small_pupil_first[0].data.sum() *4 - psf_small_pupil_exit[0].data.sum()) < 1e-3


def test_fov_size_pixels():
    """ Test the PSF field of view size is as requested, in pixels for a square aperture"""

    # square FOV
    for size in (100, 137, 256):
        osys = poppy_core.OpticalSystem("test", oversample=2)
        pupil = optics.CircularAperture(radius=6.5/2)
        osys.add_pupil(pupil)
        osys.add_detector(pixelscale=0.1, fov_pixels=size, oversample=1)

        psf = osys.calc_psf(wavelength=1e-6)

        assert psf[0].data.shape[0] == size
        assert psf[0].data.shape[1] == size


    # rectangular FOV
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.add_pupil(pupil)
    osys.add_detector(pixelscale=0.1, fov_pixels=(100,200) , oversample=1)

    psf = osys.calc_psf(wavelength=1e-6)

    assert psf[0].data.shape[0] == 100
    assert psf[0].data.shape[1] == 200


def test_inverse_MFT():
    """
    Verify basic functionality of the Inverse MFT code.
    """

    fov_arcsec  = 5.0

    test_ap = optics.ParityTestAperture(radius=6.5/2, pad_factor=1.5)

    osys = poppy_core.OpticalSystem("test", oversample=4)
    osys.add_pupil(test_ap)
    osys.add_detector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    psf1 = osys.calc_psf(wavelength=wavelength, normalize='first', display_intermediates=False)

    #osys.add_pupil(test_ap)
    osys.add_pupil() # this will force an inverse MFT
    osys.add_detector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    #plt.clf()
    psf = osys.calc_psf(wavelength=wavelength, normalize='first', display_intermediates=False)

    # the intermediate PSF (after one MFT) should be essentially identical to the
    # final PSF (after an MFT, inverse MFT, and another MFT):
    assert(abs(psf1[0].data - psf[0].data).max() < 1e-7)


@pytest.mark.skipif(
    (scipy is None),
    reason='No SciPy installed'
)
def test_optic_resizing():
    '''
    Tests the rescaling functionality of OpticalElement.get_phasor(),
    by first creating an optic with a small pixel scale and then
    creating an optic with a large pixel scale, and checking the returned
    phasor of each has the dimensions of the input wavefront.
    '''

    # diameter 1 meter, pixel scale 2 mm
    inputwf = poppy_core.Wavefront(diam=1.0, npix=500)

    # Test rescaling from finer scales: diameter 1 meter, pixel scale 1 mm
    test_optic_small=fits.HDUList([fits.PrimaryHDU(np.zeros([1000, 1000]))])
    test_optic_small[0].header["PUPLSCAL"]=.001
    test_optic_small_element=poppy_core.FITSOpticalElement(transmission=test_optic_small)
    assert(test_optic_small_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling from coarser scales: diameter 1 meter, pixel scale 10 mm
    test_optic_large=fits.HDUList([fits.PrimaryHDU(np.zeros([100, 100]))])
    test_optic_large[0].header["PUPLSCAL"]=.01
    test_optic_large_element=poppy_core.FITSOpticalElement(transmission=test_optic_large)
    assert(test_optic_large_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to pad with extra zeros:
    # diameter 0.8 mm, pixel scale 1 mm
    test_optic_pad=fits.HDUList([fits.PrimaryHDU(np.zeros([800, 800]))])
    test_optic_pad[0].header["PUPLSCAL"]=.001
    test_optic_pad_element=poppy_core.FITSOpticalElement(transmission=test_optic_pad)
    assert(test_optic_pad_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to trim to a smaller size:
    # diameter 1.2 mm, pixel scale 1 mm
    test_optic_crop=fits.HDUList([fits.PrimaryHDU(np.zeros([1200, 1200]))])
    test_optic_crop[0].header["PUPLSCAL"]=.001
    test_optic_crop_element=poppy_core.FITSOpticalElement(transmission=test_optic_crop)
    assert(test_optic_crop_element.get_phasor(inputwf).shape ==inputwf.shape )


def test_unit_conversions():
    """ Test the astropy.Quantity unit conversions
    This is a modified version of test_CircularAperture
    """
    from ..misc import airy_2d
    import astropy.units as u
    # Analytic PSF for 1 meter diameter aperture
    analytic = airy_2d(diameter=1)
    analytic /= analytic.sum() # for comparison with poppy outputs normalized to total=1


    # Numeric PSF for 1 meter diameter aperture
    osys = poppy_core.OpticalSystem()
    pupil = optics.CircularAperture(radius=0.5)
    osys.add_pupil(pupil)
    osys.add_detector(pixelscale=0.010,fov_pixels=512, oversample=1)

    # test versions with 3 different ways of saying the wavelength:
    for wavelen in [1e-6, 1e-6*u.m, 1*u.micron]:
        numeric_psf = osys.calc_psf(wavelength=wavelen, display=False)

        # Comparison
        difference = xp.asarray(numeric_psf[0].data) - analytic
        assert xp.all(xp.abs(difference) < 3e-5)

def test_return_complex():
    osys =poppy_core.OpticalSystem()
    osys.add_pupil(optics.CircularAperture(radius=3))
    osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)
    psf = osys.calc_psf(2e-6,return_final=True)
    assert len(psf[1])==1 #make sure only one element was returned
    #test that the wavefront returned is the final wavefront:
    assert xp.allclose(psf[1][0].intensity, psf[0][0].data)


def test_displays(close=True):
    # Right now doesn't check the outputs are as expected in any way
    # TODO consider doing that? But it's hard given variations in matplotlib version etc

    # As a result this just tests that the code runs to completion, without any assessment
    # of the correctness of the output displays.
    import poppy

    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture())
    osys.add_detector(fov_pixels=128, pixelscale=0.01*u.arcsec/u.pixel)

    # Test optical system display
    # This implicitly exercises the optical element display paths, too
    osys.display()

    # Test PSF calculation with intermediate wavefronts
    plt.figure()
    psf = osys.calc_psf(display_intermediates=True)

    # Test PSF display
    plt.figure()
    poppy.display_psf(psf)

    # Test PSF display with other units too
    poppy.display_psf(psf, angular_coordinate_unit=u.urad)

    # Test PSF calculation with intermediate wavefronts and other units
    plt.figure()
    psf = osys.calc_psf(display_intermediates=True)
    osys2 = poppy.OpticalSystem()
    osys2.add_pupil(poppy.CircularAperture())
    osys2.add_detector(fov_pixels=128, pixelscale=0.05*u.urad/u.pixel)
    psf2, waves = osys.calc_psf(display_intermediates=True, return_intermediates=True)

    # Test wavefront display, implicitly including other units
    waves[-1].display()

    if close:
        plt.close('all')


def test_rotation_in_OpticalSystem(display=False, npix=1024):
    """ Test that rotation planes within an OpticalSystem work as
    expected to rotate the wavefront. We can get equivalent results
    by rotating an Optic a given amount counterclockwise, or rotating the wavefront
    in the same direction.
    """

    angles_and_tolerances = ((90, 1e-8), (45, 3e-7))

    for angle, atol in angles_and_tolerances:
        osys = poppy.OpticalSystem(npix=npix)
        osys.add_pupil(poppy.optics.ParityTestAperture(rotation=angle))
        osys.add_detector(fov_pixels=128, pixelscale=0.01)
        psf1 = osys.calc_psf()

        osys2 = poppy.OpticalSystem(npix=npix)
        osys2.add_pupil(poppy.optics.ParityTestAperture())
        osys2.add_rotation(angle=angle)  # note, same sign here.
        osys2.add_detector(fov_pixels=128, pixelscale=0.01)
        psf2 = osys2.calc_psf()

        if display:
            fig, axes = plt.subplots(figsize=(16, 5), ncols=2)
            poppy.display_psf(psf1, ax=axes[0])
            axes[0].set_title("Optic rotated {} deg".format(angle))
            poppy.display_psf(psf2, ax=axes[1])
            axes[1].set_title("Wavefront rotated {} deg".format(angle))

        assert xp.allclose(psf1[0].data, psf2[0].data, atol=atol), ("PSFs did not agree "
                                                                    f"within the requested tolerance, for angle={angle}."
                                                                    f"Max |difference| = {xp.max(xp.abs(psf1[0].data - psf2[0].data))}")

### Tests for OpticalElements defined in poppy_core###

def test_ArrayOpticalElement():
    import poppy
    y,x = xp.indices((10, 10)) # arbitrary something to stick in an optical element

    ar = poppy.ArrayOpticalElement(opd=x, transmission=y, pixelscale=1*u.meter/u.pixel)

    assert xp.allclose(ar.opd, x), "Couldn't set OPD"
    assert xp.allclose(ar.amplitude, y), "Couldn't set amplitude transmission"
    assert ar.pixelscale == 1*u.meter/u.pixel

def test_FITSOpticalElement(tmpdir):
    circ_fits = poppy.CircularAperture().to_fits(grid_size=3, npix=10)
    fn = str(os.path.join(tmpdir , "circle.fits"))
    circ_fits.writeto(fn, overwrite=True)

    # Test passing aperture via file on disk
    foe = poppy.FITSOpticalElement(transmission=fn)
    assert foe.amplitude_file == fn
    assert xp.allclose(foe.amplitude, circ_fits[0].data)

    # Test passing OPD via FITS object, along with unit conversion
    circ_fits[0].header['BUNIT'] = 'micron' # need unit for OPD
    foe = poppy.FITSOpticalElement(opd=circ_fits)
    assert foe.opd_file == 'supplied as fits.HDUList object'
    assert xp.allclose(foe.opd, circ_fits[0].data * 1e-6)

    # make a cube
    rect_mask = poppy.RectangleAperture().sample(grid_size=3, npix=10)
    rect_mask = poppy.accel_math.ensure_not_on_gpu(rect_mask)
    circ_mask = circ_fits[0].data
    circ_fits[0].data = np.stack([circ_mask, rect_mask])
    circ_fits[0].header['BUNIT'] = 'nm' # need unit for OPD
    fn2 = str(os.path.join(tmpdir, "cube.fits"))
    circ_fits.writeto(fn2, overwrite=True)

    # Test passing OPD as cube, with slice default, units of nanometers
    foe = poppy.FITSOpticalElement(opd=fn2)
    assert foe.opd_file == fn2
    assert foe.opd_slice == 0
    assert xp.allclose(foe.opd, circ_mask * 1e-9)

    # Same cube but now we ask for the next slice
    foe = poppy.FITSOpticalElement(opd=(fn2, 1))
    assert foe.opd_file == fn2
    assert foe.opd_slice == 1
    assert xp.allclose(foe.opd, rect_mask * 1e-9)

def test_OPD_in_waves_for_FITSOpticalElement():
    pupil_radius = 1 * u.m
    pupil = poppy.CircularAperture(radius=pupil_radius)
    reference_wavelength = 1 * u.um
    npix = 16
    single_wave_1um_lens = poppy.ThinLens(
        name='Defocus',
        nwaves=1,
        reference_wavelength=reference_wavelength,
        radius=pupil_radius
    )
    osys = poppy.OpticalSystem(oversample=1, npix=npix)
    osys.add_pupil(pupil)
    osys.add_pupil(single_wave_1um_lens)
    osys.add_detector(0.01 * u.arcsec / u.pixel, fov_pixels=3)
    # We have applied 1 wave of defocus at 1 um, so verify the center
    # has lower flux than at 2 um (it should be the 'hole' of the donut)
    psf_1um = osys.calc_psf(reference_wavelength)
    center_pixel_value = psf_1um[0].data[1,1]
    psf_2um = osys.calc_psf(2 * reference_wavelength)
    assert psf_2um[0].data[1,1] > psf_1um[0].data[1,1]
    # Now, use the single_wave_1um_lens optic to make a
    # wavelength-independent 1 wave defocus
    lens_as_fits = single_wave_1um_lens.to_fits(what='opd', npix=3 * npix // 2)
    lens_as_fits[0].header['BUNIT'] = 'radian'
    lens_as_fits[0].data *= 2 * xp.pi / reference_wavelength.to(u.m).value
    lens_as_fits_trans = single_wave_1um_lens.to_fits(what='amplitude', npix=3 * npix // 2)
    thin_lens_wl_indep = poppy.FITSOpticalElement(opd=lens_as_fits, transmission=lens_as_fits_trans, opdunits='radian')
    # We expect identical peak flux for all wavelengths, so check at 0.5x and 2x
    for prefactor in (0.5, 1.0, 2.0):
        osys = poppy.OpticalSystem(oversample=1, npix=npix)
        osys.add_pupil(pupil)
        osys.add_pupil(thin_lens_wl_indep)
        osys.add_detector(prefactor * 0.01 * u.arcsec / u.pixel, fov_pixels=3)
        psf = osys.calc_psf(wavelength=prefactor * u.um)
        assert xp.isclose(center_pixel_value, psf[0].data[1,1])

def test_fits_rot90_vs_ndimagerotate_consistency(plot=False):
    """Test that rotating a FITS HDUList via either of the two
    methods yields consistent results. This compares an exact
    90 degree rotation and an interpolating not-quite-90-deg rotation.
    Both methods should rotate counterclockwise and consistently.
    """
    letterf_hdu = poppy.optics.LetterFAperture().to_fits(npix=128)
    opt1 = poppy.FITSOpticalElement(transmission=letterf_hdu,
                                   rotation=90)
    opt2 = poppy.FITSOpticalElement(transmission=letterf_hdu,
                                   rotation=89.99999)
    assert xp.allclose(opt1.amplitude, opt2.amplitude, atol=1e-5)

    if plot:
        fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
        axes[0].imshow(opt1.amplitude)
        axes[0].set_title("Rot90")
        axes[1].imshow(opt2.amplitude)
        axes[1].set_title("ndimage rotate(89.9999)")

def test_analytic_vs_FITS_rotation_consistency(plot=False):
    """Test that rotating an AnalyticOpticalElement vs
    rotating a discretized version as a FITSOpticalElement
    are consistent in rotation direction (counterclockwise)
    and amount"""
    opt1 = poppy.optics.LetterFAperture(rotation=90)

    letterf_hdu = poppy.optics.LetterFAperture().to_fits(npix=128)
    opt2 = poppy.FITSOpticalElement(transmission=letterf_hdu,
                                    rotation=90)

    if plot:
        opt1.display()
        plt.figure()
        opt2.display()

    array1 = opt1.sample(npix=128)
    array2 = opt2.amplitude
    assert xp.allclose(array1, array2)

### OpticalSystem tests and related

def test_source_offsets_in_OpticalSystem(npix=128, fov_size=1, verbose=False):
    """Test source offsets within the field move in the expected
    directions and by the expected amounts

    The source offset positions are specified in the *output* detector coordinate frame,
    (i.e. for where the PSF should appear in the output image), but we create the
    wavefront in the entrance pupil coordinate frame. These may be different if
    there are coordinate transforms for axes flips or rotations. Therefore test several cases
    and ensure the output PSF appears in the expected location in each case.


    Parameters:
    ----------
    npix : int
        number of pixels
    fov_size :
        fov size in arcsec (pretty much arbitrary)
    """
    if npix < 110:
        raise ValueError("npix < 110 results in too few pixels for fwcentroid to work properly.")

    pixscale = fov_size / npix
    center_coords = xp.asarray((npix - 1, npix - 1)) / 2

    ref_psf1 = None  # below we will save and compare PSFs with transforms to one without.

    for transform in ['no', 'inversion', 'rotation', 'both']:

        osys = poppy.OpticalSystem(oversample=1, npix=npix)
        osys.add_pupil(poppy.CircularAperture(radius=1.0))
        if transform == 'inversion' or transform == 'both':
            if verbose:
                print("ADD INVERSION")
            osys.add_inversion(axis='y')
        if transform == 'rotation' or transform == 'both':
            if verbose:
                print("ADD ROTATION")
            osys.add_rotation(angle=12.5)
        osys.add_detector(pixelscale=pixscale, fov_pixels=npix)

        # a PSF with no offset should be centered
        psf0 = osys.calc_psf()
        cen = poppy.measure_centroid(psf0)
        assert xp.allclose(cen, center_coords), "PSF with no source offset should be centered"
        if verbose:
            print(f"PSF with no offset OK for system with {transform} transform.\n")

        # Compute a PSF with the source offset towards PA=0 (+Y), still within the FOV
        osys.source_offset_r = 0.3 * fov_size

        # Shift to PA=0 should move in +Y
        osys.source_offset_theta = 0
        psf1 = osys.calc_psf()
        cen = poppy.measure_centroid(psf1)
        assert xp.allclose((cen[0] - center_coords[0]) * pixscale, osys.source_offset_r,
                           rtol=0.1), "Measured centroid in Y did not match expected offset"
        assert xp.allclose(cen[1], center_coords[1], rtol=0.1), "Measured centroid in X should not shift for this test case"
        if verbose:
            print(f"PSF with +Y offset OK for system with {transform} transform.\n")

        if ref_psf1 is None:
            ref_psf1 = psf1
        else:
            assert xp.allclose(ref_psf1[0].data, psf1[0].data,
                               atol=1e-4), "PSF is inconsistent with the system without any transforms"

        # Shift to PA=90 should move in -X
        osys.source_offset_theta = 90
        psf2 = osys.calc_psf()
        cen = poppy.measure_centroid(psf2)
        assert xp.allclose((cen[1] - center_coords[1]) * pixscale, -osys.source_offset_r,
                           rtol=0.1), "Measured centroid in X did not match expected offset"
        assert xp.allclose(cen[0], center_coords[0], rtol=0.1), "Measured centroid in Y should not shift for this test case"

        if verbose:
            print(f"PSF with -X offset OK for system with {transform} transform.\n")

### Detector class unit test ###

try:
    import pytest
    _HAVE_PYTEST = True
except:
    _HAVE_PYTEST = False

def _exception_message_starts_with(excinfo, message_body):
    return excinfo.value.args[0].startswith(message_body)

def test_Detector_pixelscale_units():
    """ Detectors can take various kinds of units for pixel scales.
    Check that these work as expected."""

    import astropy.units as u

    # We can specify units in arcsec, and fov in pixels:
    test_det = poppy_core.Detector(pixelscale=0.01 * u.arcsec / u.pixel, fov_pixels=100)
    assert test_det.pixelscale == 0.01 * u.arcsec / u.pixel
    assert test_det.shape == (100, 100)

    # Or scale in arcsec, and fov in arcsec:
    test_det = poppy_core.Detector(pixelscale=0.01 * u.arcsec / u.pixel, fov_arcsec=10)
    assert test_det.pixelscale == 0.01 * u.arcsec / u.pixel
    assert test_det.shape == (1000, 1000)

    # Or scale in microradians, and fov in milliradians (converted to arcsec, since the fov_arcsec parameter needs arcsec):
    test_det = poppy_core.Detector(pixelscale=1 * u.urad / u.pixel, fov_arcsec=(1*u.mrad).to_value(u.arcsec))
    assert test_det.pixelscale == 1 * u.urad / u.pixel
    assert test_det.shape == (1000, 1000)

    # It also works to leave the scale unspecified in unit, which is interpreted as arcsec
    test_det = poppy_core.Detector(pixelscale=0.01, fov_arcsec=10)
    assert test_det.pixelscale == 0.01 * u.arcsec / u.pixel
    assert test_det.shape == (1000, 1000)

    # We can make the pixelscale in microns/pixel, and fov in pixels
    test_det = poppy_core.Detector(pixelscale=0.02 * u.meter / u.pixel, fov_pixels=200)
    assert test_det.pixelscale == 0.02 * u.meter / u.pixel
    assert test_det.shape == (200, 200)

    if _HAVE_PYTEST:
        with pytest.raises(ValueError) as excinfo:
            # But this will fail: pixelscale in microns/pixel and fov in arcsec
            test_det = poppy_core.Detector(pixelscale=20 * u.micron / u.pixel, fov_arcsec=10)
        assert _exception_message_starts_with(excinfo, "If you specify the detector pixelscale in microns/pixel "
                                                       "or other linear unit"), "Error message not as expected"

        with pytest.raises(ValueError) as excinfo:
            # This will also fail: pixelscale in microns/pixel and no fov spec
            test_det = poppy_core.Detector(pixelscale=20 * u.micron / u.pixel)
        assert _exception_message_starts_with(excinfo, "If you specify the detector pixelscale in microns/pixel "
                                                       "or other linear unit"), "Error message not as expected"

        with pytest.raises(ValueError) as excinfo:
            # This will also fail: pixelscale has garbage units
            test_det = poppy_core.Detector(pixelscale=1 * u.kiloparsec / u.week)
        assert _exception_message_starts_with(excinfo, "Argument 'pixelscale' to function"), \
            "Error message not as expected"

        with pytest.raises(u.UnitsError) as excinfo:
            # This will also fail: fov_pixels has garbage units
            test_det = poppy_core.Detector(pixelscale=20, fov_pixels=1 * u.kiloparsec / u.week)
        assert _exception_message_starts_with(excinfo, "Argument 'fov_pixels' to function"), \
            "Error message not as expected"


# Tests for CompoundOpticalSystem


def test_CompoundOpticalSystem():
    """ Verify basic functionality of concatenating optical systems
    """
    opt1 = poppy.SquareAperture()
    opt2 = poppy.CircularAperture(radius=0.55)

    # a single optical system
    osys = poppy.OpticalSystem()
    osys.add_pupil(opt1)
    osys.add_pupil(opt2)
    osys.add_detector(pixelscale=0.1, fov_pixels=128)

    # two systems, joined into a CompoundOpticalSystem
    osys1 = poppy.OpticalSystem()
    osys1.add_pupil(opt1)

    osys2 = poppy.OpticalSystem()
    osys2.add_pupil(opt2)
    osys2.add_detector(pixelscale=0.1, fov_pixels=128)

    cosys = poppy.CompoundOpticalSystem([osys1, osys2])

    # PSF calculations
    psf_simple = osys.calc_psf()
    psf_compound = cosys.calc_psf()

    xp.testing.assert_allclose(psf_simple[0].data, psf_compound[0].data,
                               err_msg="PSFs do not match between equivalent simple and compound optical systems")


    # check the planes
    assert len(cosys.planes) == len(osys1.planes)+len(osys2.planes)


# Tests for the inwave argument

def test_inwave_fraunhofer(plot=False):
    '''Verify basic functionality of the inwave kwarg for a basic OpticalSystem()'''
    npix=128
    oversample=2
    diam=2.4*u.m
    lambda_m = 0.5e-6*u.m
    # calculate the Fraunhofer diffraction pattern
    hst = poppy.OpticalSystem(pupil_diameter=diam, npix=npix, oversample=oversample)
    hst.add_pupil(poppy.CircularAperture(radius=diam.value/2))
    hst.add_pupil(poppy.SecondaryObscuration(secondary_radius=0.396,
                  support_width=0.0264,
                  support_angle_offset=45.0))
    hst.add_image(poppy.ScalarTransmission(planetype=poppy_core.PlaneType.image, name='focus'))

    if plot:
        plt.figure(figsize=(9,3))
    psf1,wfs1 = hst.calc_psf(wavelength=lambda_m, display_intermediates=plot, return_intermediates=True)
    
    # now test the system by inputting a wavefront first
    wfin = poppy.Wavefront(wavelength=lambda_m, npix=npix,
                           diam=diam, oversample=oversample)
    if plot:
        plt.figure(figsize=(9,3))
    psf2,wfs2 = hst.calc_psf(wavelength=lambda_m, display_intermediates=plot, return_intermediates=True,
                             inwave=wfin)
    
    wf = wfs1[-1].wavefront
    wf_no_in = wfs2[-1].wavefront
    
    assert xp.allclose(wf, wf_no_in), 'Results differ unexpectedly when using inwave argument in OpticalSystem().'

