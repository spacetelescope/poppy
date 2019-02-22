
import numpy as np
import matplotlib.pyplot as plt
import scipy

from ..misc import airy_1d, airy_2d, sinc2_2d, _RADtoARCSEC, _ARCSECtoRAD

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
        plt.semilogy(r_norm,airyprofile)
        plt.axvline(1.028/2, color='k', ls=':')
        plt.axhline(0.5, color='k', ls=':')
        plt.ylabel('Intensity relative to peak')
        plt.xlabel('Separation in $\\lambda/D$')
        for rad in airy_zeros:
            plt.axvline(rad, color='red', ls='--')

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
    print(cut.shape)

    if display:
        
        plt.subplot(211)

        plt.semilogy(r, fn1d, label='1D')

        plt.semilogy(r, cut, label='2D', color='black', ls='--')

        plt.legend(loc='upper right')
        plt.axvline(0.251643, color='red', ls='--')
        plt.ylabel('Intensity relative to peak')
        plt.xlabel('Separation in $\\lambda/D$')
 
        ax=plt.subplot(212)
        plt.plot(r, cut-fn1d)
        ax.set_ylim(-1e-8, 1e-8)
        plt.ylabel('Difference')
        plt.xlabel('Separation in $\\lambda/D$')

    #print fn1d[0], cut[0]
    #print np.abs(fn1d-cut) #< 1e-9
    assert np.all( np.abs(fn1d-cut) < 1e-9)

    #return fn2d

def test_sinc2_2d(display=False):
    """ Test 2D Sinc function vs 1D function.

    """
    fn2d = sinc2_2d(width=1.0, height=0.5, wavelength=1e-6, shape=(511, 511), pixelscale=0.010)
    x = np.arange(256)
    #r, fn1d = airy_1d(diameter=1.0, wavelength=1e-6, length=256, pixelscale=0.010)

    cut_h = fn2d[255, 255:].flatten()
    cut_v = fn2d[255:, 255].flatten()


    # test shape and centering
    assert fn2d.shape == (511, 511)
    assert fn2d[255,255] == 1.0

    # and the horizontal axis should be 2x as spaced out as the vertcal, given the rectangular aperture above.
    assert cut_v[20] == cut_h[10]
    assert cut_v[200] == cut_h[100]

    if display:
        import matplotlib
        plt.clf()
        plt.subplot(211)

        plt.imshow(fn2d, norm=matplotlib.colors.LogNorm() )

        #plt.semilogy(r, fn1d, label='1D')

        ax=plt.subplot(212)
        plt.semilogy(x, cut_h, label='2D cut horizontal', color='red', ls='-')
        plt.semilogy(x, cut_v, label='2D cut vertical', color='black', ls='-')

        plt.legend(loc='upper right')
        plt.ylabel('Intensity relative to peak')
        plt.xlabel('Separation in $\\lambda/D$')
 
        #plt.plot(r, cut-fn1d)
        #ax.set_ylim(-1e-8, 1e-8)
        #plt.ylabel('Difference')
        #plt.xlabel('Separation in $\lambda/D$')

    #print fn1d[0], cut[0]
    #print np.abs(fn1d-cut) #< 1e-9
    #assert np.all( np.abs(fn1d-cut) < 1e-9)

    #return fn2d


