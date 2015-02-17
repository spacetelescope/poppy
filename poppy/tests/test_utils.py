import numpy as np
import astropy.io.fits as fits

from .. import utils

def test_padToSize():
    square = np.ones((300,300))

    for desiredshape in [ (500, 500), (400,632), (2048, 312)]:
        newshape = utils.padToSize(square, desiredshape).shape 
        for i in [0,1]: assert newshape[i] == desiredshape[i]



# Utility function used in test_measure_FWHM
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    From https://gist.github.com/andrewgiessel/4635563
    As variously modified by marshall
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = (size-1)*.5
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def test_measure_FWHM(display=False):
    """ Test the utils.measure_FWHM function

    Current implementation can be off by a
    couple percent for small FWHMs that are only
    marginally well sampled by the array pixels, so
    the allowed tolerance for measured_fwhm = input_fwhm
    is that it's allowed to be off by a couple percent.

    """

    desired = (3, 4.5, 5, 8, 12)
    tolerance= (0.07, 0.03, 0.02, 0.02, 0.02)

    for des, tol in zip(desired, tolerance):


        desired_fwhm = des #4.0 # pixels
        pxscl = 0.010

        center=(24.5,26.25)
        ar = makeGaussian(50, fwhm=desired_fwhm, center=center)

        testfits = fits.HDUList(fits.PrimaryHDU(ar))
        testfits[0].header['PIXELSCL'] = pxscl

        meas_fwhm = utils.measure_fwhm(testfits, center=center)
        print "Measured FWHM: {0:.4f} arcsec, {1:.4f} pixels ".format(meas_fwhm, meas_fwhm/pxscl)

        reldiff =  np.abs((meas_fwhm/pxscl) - desired_fwhm ) / desired_fwhm 
        print "Desired: {0:.4f}. Relative difference: {1:.4f}    Tolerance: {2:.4f}".format(desired_fwhm, reldiff, tol)
        assert( reldiff < tol )


