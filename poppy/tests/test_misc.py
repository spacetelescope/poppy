
import numpy as np
import matplotlib.pyplot as pl
import scipy

from ..misc import airy_1d, airy_2d, _RADtoARCSEC, _ARCSECtoRAD

airy_zeros = np.asarray([3.8317, 7.0156, 10.1735, 13.3237, 16.4706])/np.pi  # first several zeros of the Bessel function J1. See e.g. http://en.wikipedia.org/wiki/Airy_disk#Mathematical_details

def test_airy_1d(display=False):
    """ Compare analytic airy function results to the expected locations
    for the first three dark rings and the FWHM of the PSF."""
    lam = 1.0e-6
    D = 1.0
    r, airyprofile = airy_1d(wavelength=lam, diameter=D, length=20480, pixelscale=0.0001)

    # convert to units of lambda/D
    r_norm = r*_ARCSECtoRAD / (lam/D)
    if display:
        pl.semilogy(r_norm,airyprofile)
        pl.axvline(1.028/2, color='k', ls=':')
        pl.axhline(0.5, color='k', ls=':')
        pl.ylabel('Intensity relative to peak')
        pl.xlabel('Separation in $\lambda/D$')
        for rad in airy_zeros:
            pl.axvline(rad, color='red', ls='--')

    airyfn = scipy.interpolate.interp1d(r_norm, airyprofile)
    # test FWHM occurs at 1.028 lam/D, i.e. HWHM is at 0.514
    assert (airyfn(0.5144938) - 0.5) < 1e-5

    # test first minima occur near 1.22 lam/D, 2.23, 3.24 lam/D
    # TODO investigate/improve numerical precision here?
    for rad in airy_zeros:
        #print(rad, airyfn(rad), airyfn(rad+0.005))
        assert airyfn(rad) < airyfn(rad+0.0003)
        assert airyfn(rad) < airyfn(rad-0.0003)


def test_airy_2d(display=False):
    """ Test 2D airy function vs 1D function; both
    should yield the exact same results for a 1D cut across the 2d function.
    And we've already tested the 1D above...
    """

    fn2d = airy_2d(diameter=1.0, wavelength=1e-6, shape=(511, 511), pixelscale=0.010)
    r, fn1d = airy_1d(diameter=1.0, wavelength=1e-6, length=256, pixelscale=0.010)

    cut = fn2d[255, 255:].flatten()
    print cut.shape

    if display:
        
        pl.subplot(211)

        pl.semilogy(r, fn1d, label='1D')

        pl.semilogy(r, cut, label='2D', color='black', ls='--')

        pl.legend(loc='upper right')
        pl.axvline(0.251643, color='red', ls='--')
        pl.ylabel('Intensity relative to peak')
        pl.xlabel('Separation in $\lambda/D$')
 
        ax=pl.subplot(212)
        pl.plot(r, cut-fn1d)
        ax.set_ylim(-1e-8, 1e-8)
        pl.ylabel('Difference')
        pl.xlabel('Separation in $\lambda/D$')

    #print fn1d[0], cut[0]
    #print np.abs(fn1d-cut) #< 1e-9
    assert np.all( np.abs(fn1d-cut) < 1e-9)

    #return fn2d

