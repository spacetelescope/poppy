#
#  Test functions for subpixel geometry code
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from .. import poppy_core
from .. import optics
from .. import geometry

# Test routines for antialiased fractional circle

def test_area_minimal_circle():
    """ Test that A = pi * r**2, where r==1 """
    res = geometry.filled_circle_aa((2,2), 0.5, 0.5, 1)
    assert np.abs(res.sum() -np.pi) < 1e-7


def test_clipping():

    res = geometry.filled_circle_aa((20,20), 10.5, 10.5, 1, clip=True,cliprange=(0,1))

    assert res.min() >= 0.0
    assert res.max() <= 1.0


# Come up with some representative plausible test cases for whcih we know the answers


# Test effect of shifting the center of the image by integer pixels

# Test effect of shifting the center of the image by fractional pixels
    # cross correlation of shifted & unshifted to demonstrate 1/2 pixel shifts?

# Test using subpixel scaling of incput X and Y arrays

# Test the specific case at fault here.







def test_pixwt():
    """
    Test subpixel circular aperture (pixwt function)
    via comparison to values calculated from IDL pixwt.pro
    """

    def check_pixwt_result( args, expected=1, tolerance=1e-5):
        assert np.abs(geometry.pixwt(*args)-expected).max() < tolerance

    # radius 1 circle
    check_pixwt_result((0,0,1,0,0), expected=1.0)
    check_pixwt_result((0,0,1,1,1), expected=0.07878669)

    # check with array args
    xa = np.asarray([0,0,1,1,2,-1])
    ya = np.asarray([0,1,1,0,0,-1])
    exp = np.asarray([ 1.0000000, 0.45661139, 0.078786805, 0.45661139, 8.9406967e-08, 0.078786790 ])
    check_pixwt_result((0,0,1,xa,ya), expected=exp)


    # radius 10 circle
    check_pixwt_result((0,0,10,0,9), expected=1.0)
    check_pixwt_result((0,0,10,0,10), expected=0.49584007)
    check_pixwt_result((0,0,10,0,11), expected=0)

    check_pixwt_result((0,0,10,1,10), expected=0.44564247)
    check_pixwt_result((0,0,10,2,10), expected=0.29352379)

    # offset origin
    check_pixwt_result((0,1,10,0,11), expected=0.49584007)
    check_pixwt_result((20,1,10,20,11), expected=0.49584007)


def test_filled_aa_circle(plot=True):
    """ Test the function that computes a filled, anti-aliased circle


    """

    # test some specific values in comparison to the pixwt function

    # brute force run pixwt on some array
    y,x = np.indices((100,100))
    res_1 = geometry.pixwt(50,50,40,x,y)


    res_2 = geometry.filled_circle_aa( (100,100), 50, 50, 40, 1)
    assert np.max(np.abs(res_1,res_2) < 1e-5)
