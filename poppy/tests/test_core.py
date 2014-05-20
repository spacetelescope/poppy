#Test functions for core poppy functionality

from .. import poppy_core 
from .. import optics

import numpy as np
import astropy.io.fits as fits


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

class ParityTestAperture(optics.AnalyticOpticalElement):
    """ Defines a circular pupil aperture with boxes cut out.
    This is mostly a test aperture

    Parameters
    ----------
    name : string
        Descriptive name
    radius : float
        Radius of the pupil, in meters. Default is 1.0

    pad_factor : float, optional
        Amount to oversize the wavefront array relative to this pupil.
        This is in practice not very useful, but it provides a straightforward way
        of verifying during code testing that the amount of padding (or size of the circle)
        does not make any numerical difference in the final result.

    """

    def __init__(self, name=None,  radius=1.0, pad_factor = 1.5, **kwargs):
        if name is None: name = "Circle, radius=%.2f m" % radius
        poppy_core.AnalyticOpticalElement.__init__(self,name=name, **kwargs)
        self.radius = radius
        self.pupil_diam = pad_factor * 2* self.radius # for creating input wavefronts - let's pad a bit


    def getPhasor(self,wave):
        """ Compute the transmission inside/outside of the occulter.
        """
        if not isinstance(wave, poppy_core.Wavefront):
            raise ValueError("CircularAperture getPhasor must be called with a Wavefront to define the spacing")
        #assert (wave.planetype == poppy._PUPIL)

        y, x = wave.coordinates()
        r = np.sqrt(x**2+y**2) #* wave.pixelscale

        w_outside = np.where( r > self.radius)
        self.transmission = np.ones(wave.shape)
        self.transmission[w_outside] = 0

        w_box1 = np.where( (r> self.radius*0.5) & (np.abs(x) < self.radius*0.1 ) & ( y < 0 ))
        w_box2 = np.where( (r> self.radius*0.75) & (np.abs(y) < self.radius*0.2) & ( x < 0 ))
        self.transmission[w_box1] = 0
        self.transmission[w_box2] = 0

        return self.transmission



######### Core tests functions #########




def test_basic_functionality():
    """ For one specific geometry, test that we get the expected value based on a prior reference
    calculation."""
    osys = poppy_core.OpticalSystem("test", oversample=1)
    pupil = optics.CircularAperture(radius=1)
    osys.addPupil(pupil) #function='Circle', radius=1)
    osys.addDetector(pixelscale=0.1, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux

    psf = osys.calcPSF(wavelength=1.0e-6)
    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality
    assert abs(psf[0].data.max() - 0.201) < 0.001


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
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=0.010,fov_pixels=512, oversample=1)
    numeric = osys.calcPSF(wavelength=1.0e-6, display=False)

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

#fits.writeto("test.fits", numeric[0].data-analytic)
#print a2.max()



def test_normalization():
    """ Test that we can compute a PSF and get the desired flux, 
    depending on the normalization """
    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.addPupil(pupil) #function='Circle', radius=6.5/2)
    osys.addDetector(pixelscale=0.01, fov_arcsec=5.0) # use a large FOV so we grab essentially all the light and conserve flux


    # we need to be a little careful here due to floating point math comparision equality issues... Can't just do a strict equality

    # this should be very very close to one
    psf = osys.calcPSF(wavelength=1.0e-6, normalize='last')
    assert (psf[0].data.sum() - 1) < 1e-9

    # this should be a little further but still pretty close
    psf = osys.calcPSF(wavelength=1.0e-6, normalize='first')
    assert abs(psf[0].data.sum() - 1) < 0.1



def test_fov_size_pixels():
    """ Test the PSF field of view size is as requested, in pixels for a square aperture"""

    osys = poppy_core.OpticalSystem("test", oversample=2)
    pupil = optics.CircularAperture(radius=6.5/2)
    osys.addPupil(pupil)
    osys.addDetector(pixelscale=0.1, fov_pixels=100, oversample=1)

    psf = osys.calcPSF(wavelength=1e-6)

    assert psf[0].data.shape[0] == 100
    assert psf[0].data.shape[1] == 100




def test_inverse_MFT():
    """
    Verify basic functionality of the Inverse MFT code. 
    """

    fov_arcsec  = 5.0

    test_ap = optics.ParityTestAperture(radius=6.5/2)

    osys = poppy_core.OpticalSystem("test", oversample=4)
    osys.addPupil(test_ap)
    osys.addDetector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    psf1 = osys.calcPSF(wavelength=wavelength, normalize='first', display_intermediates=False)

    #osys.addPupil(test_ap)
    osys.addPupil() # this will force an inverse MFT
    osys.addDetector(pixelscale=0.010, fov_arcsec=fov_arcsec) # use a large FOV so we grab essentially all the light and conserve flux
    #plt.clf()
    psf = osys.calcPSF(wavelength=wavelength, normalize='first', display_intermediates=False)

    # the intermediate PSF (after one MFT) should be essentially identical to the
    # final PSF (after an MFT, inverse MFT, and another MFT):
    assert(   np.abs(psf1[0].data - psf[0].data).max()  < 1e-7 )


