# Test functions for core poppy functionality
import os

import numpy as np
from astropy.io import fits
import astropy.units as u
import pytest
try:
    import scipy
except ImportError:
    scipy = None

import poppy
from .. import poppy_core
from .. import optics

####### Test Common Infrastructre #######

def check_wavefront(filename_or_hdulist, slice=0, ext=0, test='nearzero', comment=""):
    """ A helper routine to verify certain properties of a wavefront FITS file,
    as requested by some test routine. """
    if isinstance(filename_or_hdulist, str):
        hdulist = fits.open(filename_or_hdulist)
        filename = filename_or_hdulist
    elif isinstance(filename_or_hdulist, fits.HDUList):
        hdulist = filename_or_hdulist
        filename = 'input HDUlist'
    imstack = hdulist[ext].data
    im = imstack[slice,:,:]


    if test=='nearzero':
        return np.all(np.abs(im) < np.finfo(im.dtype).eps*10)
    elif test == 'is_real':
        #assumes output type = 'all'
        cplx_im = imstack[1,:,:] * np.exp(1j*imstack[2,:,:])
        return np.all( cplx_im.imag < np.finfo(im.dtype).eps*10)

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
    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality
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

    # Comparison
    difference = numeric[0].data-analytic
    assert np.all(np.abs(difference) < 3e-5)

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
        ax2=pl.imshow(np.abs(numeric[0].data-analytic) < 3e-5)
        pl.title("Difference <1e-5")


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

    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality

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
    assert np.abs( psf_exit_pupil[0].data - psf_first[0].data).max()  < 1e-10


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



###    EXPECTED TO FAIL RIGHT NOW - Offsets don't work yet.
###    See https://github.com/mperrin/poppy/issues/40
import pytest
@pytest.mark.xfail
def test_fov_offset(scale=1.0):
    """ Test offsetting the field of view of a Detector
    This is distinct from offsetting the source! """
    from ..utils import measure_centroid

    size=100
    pixscale = 0.1

    # A PSF created on-axis with no offset
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.add_pupil(pupil)
    osys.add_detector(pixelscale=pixscale, fov_pixels=size, oversample=1)
    psf1 = osys.calc_psf()
    # The measured centroid should put it in the center of the array
    cent1 = measure_centroid(psf1, relativeto='center')
    poppy_core._log.info("On-axis PSF (no offset) centroid is:" + str(cent1))
    assert(abs(cent1[0]-0) < 1e-5)
    assert(abs(cent1[1]-0) < 1e-5)

    # Now create an equivalent PSF but offset the axes by 1 pixel in the first axis
    osys2 = poppy_core.OpticalSystem("test", oversample=2)
    osys2.add_pupil(pupil)
    osys2.add_detector(pixelscale=pixscale, fov_pixels=size, oversample=1, offset=(pixscale*scale,0))
    psf2 = osys2.calc_psf()
    # Its centroid shouldbe offset by a pixel
    poppy_core._log.info("Offset PSF (by ({0},0) pixels ) centroid is: {1}".format(str(scale), str(cent1)))
    cent2 = measure_centroid(psf2, relativeto='center')
    assert(abs(cent2[0]-scale) < 1e-5)
    assert(abs(cent2[1]-0) < 1e-5)


    # and do the same thing in the second axis (after the above works)



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
    assert(   np.abs(psf1[0].data - psf[0].data).max()  < 1e-7 )


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
    test_optic_small=fits.HDUList([fits.PrimaryHDU(np.zeros([1000,1000]))])
    test_optic_small[0].header["PUPLSCAL"]=.001
    test_optic_small_element=poppy_core.FITSOpticalElement(transmission=test_optic_small)
    assert(test_optic_small_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling from coarser scales: diameter 1 meter, pixel scale 10 mm
    test_optic_large=fits.HDUList([fits.PrimaryHDU(np.zeros([100,100]))])
    test_optic_large[0].header["PUPLSCAL"]=.01
    test_optic_large_element=poppy_core.FITSOpticalElement(transmission=test_optic_large)
    assert(test_optic_large_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to pad with extra zeros:
    # diameter 0.8 mm, pixel scale 1 mm
    test_optic_pad=fits.HDUList([fits.PrimaryHDU(np.zeros([800,800]))])
    test_optic_pad[0].header["PUPLSCAL"]=.001
    test_optic_pad_element=poppy_core.FITSOpticalElement(transmission=test_optic_pad)
    assert(test_optic_pad_element.get_phasor(inputwf).shape ==inputwf.shape )

    # Test rescaling where we have to trim to a smaller size:
    # diameter 1.2 mm, pixel scale 1 mm
    test_optic_crop=fits.HDUList([fits.PrimaryHDU(np.zeros([1200,1200]))])
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
        difference = numeric_psf[0].data-analytic
        assert np.all(np.abs(difference) < 3e-5)

def test_return_complex():
    osys =poppy_core.OpticalSystem()
    osys.add_pupil(optics.CircularAperture(radius=3))
    osys.add_detector(pixelscale=0.010, fov_arcsec=5.0)
    psf = osys.calc_psf(2e-6,return_final=True)
    assert len(psf[1])==1 #make sure only one element was returned
    #test that the wavefront returned is the final wavefront:
    assert np.allclose(psf[1][0].intensity,psf[0][0].data)


def test_displays():
    # Right now doesn't check the outputs are as expected in any way
    # TODO consider doing that? But it's hard given variations in matplotlib version etc
    import poppy
    import matplotlib.pyplot as plt

    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture())
    osys.add_detector(fov_pixels=128, pixelscale=0.01)

    osys.display()

    plt.figure()
    psf = osys.calc_psf(display_intermediates=True)

    plt.figure()
    #psf = osys.calc_psf(display_intermediates=True)
    poppy.display_psf(psf)


def test_rotation_in_OpticalSystem(display=False, npix=1024):
    """ Test that rotation planes within an OpticalSystem work as
    expected to rotate the wavefront. We can get equivalent results
    by rotating an Optic a given amount, or rotating the wavefront
    in the opposite direction.
    """

    angles_and_tolerances = ((90, 1e-8), (45, 3e-7))

    for angle, atol in angles_and_tolerances:
        osys = poppy.OpticalSystem(npix=npix)
        osys.add_pupil(poppy.optics.ParityTestAperture(rotation=angle))
        osys.add_detector(fov_pixels=128, pixelscale=0.01)

        if display: plt.figure()
        psf1 = osys.calc_psf(display=display)
        if display: plt.title("Optic rotated {} deg".format(angle))


        osys = poppy.OpticalSystem(npix=npix)
        osys.add_pupil(poppy.optics.ParityTestAperture())
        osys.add_rotation(angle=-angle)  # note, opposite sign here.
        osys.add_detector(fov_pixels=128, pixelscale=0.01)
        if display: plt.figure()
        psf2 = osys.calc_psf(display=display)
        if display: plt.title("Wavefront rotated {} deg".format(angle))


        assert np.allclose(psf1[0].data, psf2[0].data, atol=atol), ("PSFs did not agree "
            "within the requested tolerance")

### Tests for OpticalElements defined in poppy_core###

def test_ArrayOpticalElement():
    import poppy
    y,x = np.indices((10,10)) # arbitrary something to stick in an optical element

    ar = poppy.ArrayOpticalElement(opd=x, transmission=y, pixelscale=1*u.meter/u.pixel)

    assert np.allclose(ar.opd, x), "Couldn't set OPD"
    assert np.allclose(ar.amplitude, y), "Couldn't set amplitude transmission"
    assert ar.pixelscale == 1*u.meter/u.pixel

def test_FITSOpticalElement(tempdir='./'):
    circ_fits = poppy.CircularAperture().to_fits(grid_size=3, npix=10)
    fn = tempdir+"circle.fits"
    circ_fits.writeto(fn, overwrite=True)

    # Test passing aperture via file on disk
    foe = poppy.FITSOpticalElement(transmission=fn)
    assert foe.amplitude_file == fn
    assert np.allclose(foe.amplitude, circ_fits[0].data)

    # Test passing OPD via FITS object, along with unit conversion
    circ_fits[0].header['BUNIT'] = 'micron' # need unit for OPD
    foe = poppy.FITSOpticalElement(opd=circ_fits)
    assert foe.opd_file == 'supplied as fits.HDUList object'
    assert np.allclose(foe.opd, circ_fits[0].data*1e-6)

    # make a cube
    rect_mask = poppy.RectangleAperture().sample(grid_size=3, npix=10)
    circ_mask = circ_fits[0].data
    circ_fits[0].data = np.stack([circ_mask, rect_mask])
    circ_fits[0].header['BUNIT'] = 'nm' # need unit for OPD
    fn2 = tempdir+"cube.fits"
    circ_fits.writeto(fn2, overwrite=True)

    # Test passing OPD as cube, with slice default, units of nanometers
    foe = poppy.FITSOpticalElement(opd=fn2)
    assert foe.opd_file == fn2
    assert foe.opd_slice == 0
    assert np.allclose(foe.opd, circ_mask*1e-9)

    # Same cube but now we ask for the next slice
    foe = poppy.FITSOpticalElement(opd=(fn2, 1))
    assert foe.opd_file == fn2
    assert foe.opd_slice == 1
    assert np.allclose(foe.opd, rect_mask*1e-9)

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
    lens_as_fits[0].data *= 2 * np.pi / reference_wavelength.to(u.m).value
    thin_lens_wl_indep = poppy.FITSOpticalElement(opd=lens_as_fits, opdunits='radian')
    # We expect identical peak flux for all wavelengths, so check at 0.5x and 2x
    for prefactor in (0.5, 1.0, 2.0):
        osys = poppy.OpticalSystem(oversample=1, npix=npix)
        osys.add_pupil(pupil)
        osys.add_pupil(thin_lens_wl_indep)
        osys.add_detector(prefactor * 0.01 * u.arcsec / u.pixel, fov_pixels=3)
        psf = osys.calc_psf(wavelength=prefactor * u.um)
        assert np.isclose(center_pixel_value, psf[0].data[1,1])

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

    np.testing.assert_allclose(psf_simple[0].data, psf_compound[0].data,
                               err_msg="PSFs do not match between equivalent simple and compound optical systems")


    # check the planes
    assert len(cosys.planes) == len(osys1.planes)+len(osys2.planes)
