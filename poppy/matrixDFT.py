#! /usr/bin/env  python 
"""
    MatrixDFT: Matrix-based discrete Fourier transforms for computing PSFs. 

    See Soummer et al. 2007 JOSA

    The main user interface in this module is a class MatrixFourierTransform. 
    Internally this will call one of several subfunctions depending on the 
    specified centering type. These have to do with where the (0,0) element of the Fourier
    transform is located, i.e. where the PSF center ends up.

        - 'FFTSTYLE' centered on one pixel
        - 'SYMMETRIC' centerd on crosshairs between middle pixel
        - 'ADJUSTIBLE', always centered in output array depending on whether it is even/odd

    'ADJUSTIBLE' is the default.


    This module was originally called "Slow Fourier Transform", and this
    terminology still appears in some places in the code.  Note that this is
    'slow' only in the sense that if you perform the exact same calculation as
    an FFT, the FFT algorithm is much faster. However this algorithm gives you
    much more flexibility in choosing array sizes and sampling, and often lets
    you replace "fast calculations on very large arrays" with "relatively slow
    calculations on much smaller ones". 



    Example
    -------
    mf = matrixDFT.MatrixFourierTransform()
    result = mf.perform(pupilArray, focalplane_size, focalplane_npix)




    History
    -------
    Code originally by A. Sivaramakrishnan
    2010-11-05 Revised normalizations for flux conservation consistent
        with Soummer et al. 2007. Updated documentation.  -- M. Perrin
    2011-2012: Various enhancements, detailed history not kept, sorry.
    2012-05-18: module renamed SFT.py -> matrixDFT.py
    2012-09-26: minor big fixes

"""
#from __future__ import (absolute_import, division, print_function, unicode_literals)
from __future__ import print_function

__all__ = ['MatrixFourierTransform']

import numpy as np
import astropy.io.fits as fits

import logging
_log = logging.getLogger('poppy')



# master routine combining all centering types
def DFT_combined(pupil, nlamD, npix, offset=(0.0,0.0), inverse=False, centering='FFTSTYLE', **kwargs):
    """

    This function attempts to merge and unify the behaviors of all the other DFT routines
    into one master, flexible routines. It thus should subsume:
        DFT_fftstyle
        DFT_fftstyle_rect
        DFT_adjustible
        DFT_symmetric

    As of Jan 2013, this works OK for even-sized arrays, but for odd-sized arrays it
    appears to get the sense of fftstyle and symmetry flipped. Some debugging still needed?
    See the code in tests/test_matrixDFT.py. -MP

    Parameters
    ----------
    pupil : 2d ndarray
        pupil array (n by m). This can also be an image array if you're computing an
        inverse transformation.
    nlamD : float
        size of focal plane array, in units of lam/D
        (corresponds to 'm' in Soummer et al. 2007 4.2)
    npix : int
        number of pixels per side side of destination plane array
        (corresponds to 'N_B' in Soummer et al. 2007 4.2)
        This will be the # of pixels in the image plane for a forward
        transformation, in the pupil plane for an inverse. 

    inverse : bool
        Is this a forward or inverse transformation?
    centering : string
        What type of centering convention should be used for this FFT? 
        'FFTYSTYLE', 'SYMMETRIC', 'ADJUSTIBLE'



    """

    npupY, npupX = pupil.shape[0:2]

    if np.isscalar(npix): 
        npixY, npixX = npix, npix
    else:
        npixY, npixX = npix[0:2]   

    if not np.isscalar(nlamD):  
        nlamDY, nlamDX = nlamD[0:2]
    else:
        nlamDY, nlamDX = nlamD, nlamD

    if inverse:
        dX = nlamDX / float(npupX)
        dY = nlamDY / float(npupY)
        dU = 1.0/float(npixY)
        dV = 1.0/float(npixX)
    else:
        dU = nlamDX / float(npixX)
        dV = nlamDX / float(npixX)
        dX = 1.0/float(npupX)
        dY = 1.0/float(npupY)

#
#    du = nlamD / float(npix)
#    dv = nlamD / float(npix)
#
#    dx = 1.0/float(npup)
#    dy = 1.0/float(npup)
#
    if centering.upper() == 'FFTSTYLE':
        Xs = (np.arange(npupX) - (npupX/2)) * dX
        Ys = (np.arange(npupY) - (npupY/2)) * dY

        Us = (np.arange(npixY) - npixX/2) * dU
        Vs = (np.arange(npixX) - npixY/2) * dV
    elif centering.upper() == 'ADJUSTIBLE':
        Xs = (np.arange(npupX) - float(npupX)/2.0 - offset[1] + 0.5) * dX
        Ys = (np.arange(npupY) - float(npupY)/2.0 - offset[0] + 0.5) * dY

        Us = (np.arange(npixY) - float(npixX)/2.0 - offset[1] + 0.5) * dU
        Vs = (np.arange(npixX) - float(npixY)/2.0 - offset[0] + 0.5) * dV
    elif centering.upper() == 'SYMMETRIC':
        Xs = (np.arange(npupX) - float(npupX)/2.0 + 0.5) * dX
        Ys = (np.arange(npupY) - float(npupY)/2.0 + 0.5) * dY

        Us = (np.arange(npixY) - float(npixX)/2.0 + 0.5) * dU
        Vs = (np.arange(npixX) - float(npixY)/2.0 + 0.5) * dV





    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)

    #print("")
    #print dx, dy, du, dv
    #print("")

    if inverse:
        expYV = expYV.T.copy()
        t1 = np.dot(expYV, pupil)
        t2 = np.dot(t1, expXU)
    else:
        expXU = expXU.T.copy()
        t1 = np.dot(expXU, pupil)
        t2 = np.dot(t1, expYV)

    #return  nlamD/(npup*npix) *   t2 * dx * dy
    norm_coeff = np.sqrt(  ( nlamDY* nlamDX) / (npupY*npupX*npixY*npixX))
    #return  float(nlamD)/(npup*npix) *   t2 
    return  norm_coeff *   t2 



# FFTSTYLE centering
def DFT_fftstyle(pupil, nlamD, npix, inverse=False, **kwargs):
    """

    Compute an "FFTSTYLE" matrix fourier transform.
    This means that the zero-order term is put all in a 
    single pixel.

    Parameters
    ----------
    pupil
        pupil array (n by n)
    nlamD
        size of focal plane array, in units of lam/D
        (corresponds to 'm' in Soummer et al. 2007 4.2)
    npix
        number of pixels per side side of focal plane array
        (corresponds to 'N_B' in Soummer et al. 2007 4.2)


    """

    npup = pupil.shape[0]

    du = nlamD / float(npix)
    dv = nlamD / float(npix)

    dx = 1.0/float(npup)
    dy = 1.0/float(npup)

    Xs = (np.arange(npup) - (npup/2)) * dx
    Ys = (np.arange(npup) - (npup/2)) * dy

    Us = (np.arange(npix) - npix/2) * du
    Vs = (np.arange(npix) - npix/2) * dv

    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)

    #print("")
    #print( dx, dy, du, dv)
    #print("")

    if inverse:
        expYV = expYV.T.copy()
        t1 = np.dot(expYV, pupil)
        t2 = np.dot(t1, expXU)
    else:
        expXU = expXU.T.copy()
        t1 = np.dot(expXU, pupil)
        t2 = np.dot(t1, expYV)

    #return  nlamD/(npup*npix) *   t2 * dx * dy
    return  float(nlamD)/(npup*npix) *   t2 


# FFTSTYLE centering, rectangular pupils supported
def DFT_fftstyle_rect(pupil, nlamD, npix, inverse=False):
    """

    Compute an "FFTSTYLE" matrix fourier transform.
    This means that the zero-order term is put all in a 
    single pixel.

    This version supports rectangular, non-square arrays,
    in which case nlamD and npix should be 2-element
    tuples or lists, using the usual Pythonic order (Y,X)

    Parameters
    ----------
    pupil
        pupil array (n by n)
    nlamD
        size of focal plane array, in units of lam/D
        (corresponds to 'm' in Soummer et al. 2007 4.2)
    npix
        number of pixels per side side of focal plane array
        (corresponds to 'N_B' in Soummer et al. 2007 4.2)


    """

    npupY, npupX = pupil.shape[0:2]

    if np.isscalar(npix): #hasattr(npix, '__getitem__'):
        npixY, npixX = npix, npix
    else:
        npixY, npixX = npix[0:2]   

    if np.isscalar(nlamD): # hasattr(nlamD, '__getitem__'):
        nlamDY, nlamDX = nlamD, nlamD
    else:
        nlamDY, nlamDX = nlamD[0:2]


    if inverse:
        dX = nlamDX / float(npupX)
        dY = nlamDY / float(npupY)
        dU = 1.0/float(npixY)
        dV = 1.0/float(npixX)
    else:
        dU = nlamDX / float(npixX)
        dV = nlamDX / float(npixX)
        dX = 1.0/float(npupX)
        dY = 1.0/float(npupY)

    Xs = (np.arange(npupX) - (npupX/2)) * dX
    Ys = (np.arange(npupY) - (npupY/2)) * dY

    Us = (np.arange(npixX) - npixX/2) * dU
    Vs = (np.arange(npixY) - npixY/2) * dV

    YV = np.outer(Ys, Vs)
    XU = np.outer(Xs, Us)

    expYV = np.exp(-2.0 * np.pi * 1j * YV)  
    expXU = np.exp(-2.0 * np.pi * 1j * XU)

    expYV = expYV.T.copy()
    t1 = np.dot(expYV, pupil)
    t2 = np.dot(t1, expXU)

    if inverse:
        #print expYV.shape, pupil.shape, expXU.shape
        t2 = t2[::-1, ::-1]
    #else:
        #expYV = expYV.T.copy()                          # matrix npixY * npupY
        #t1 = np.dot(expYV, pupil)                       # matrix npixY * npupX?     dot prod combined last axis of expYV and first axis of pupil.
        #t2 = np.dot(t1, expXU)                          # matrix npixY * npixX
        #print expYV.shape, pupil.shape, expXU.shape

    #print ""

    #return  nlamD/(npup*npix) *   t2 * dx * dy
    # normalization here is almost certainly wrong:
    norm_coeff = np.sqrt(  float( nlamDY* nlamDX) / (npupY*npupX*npixY*npixX))
    #mean_npup = np.sqrt(npupY**2+npupX**2)
    #mean_npix = np.sqrt(npixY**2+npixX**2)
    return  norm_coeff *   t2 


# SYMMETRIC centering : PSF centered between 4 pixels
def DFT_symmetric(pupil, nlamD, npix, **kwargs):
    """
    Compute a "SYMMETRIC" matrix fourier transform. 
    This means that the zero-order term is spread evenly
    between the center 4 pixels.

    Parameters
    ----------
    pupil
        pupil array (n by n)
    nlamD
        size of focal plane array, in units of lam/D
        (corresponds to 'm' in Soummer et al. 2007 4.2)
    npix
        number of pixels per side side of focal plane array
        (corresponds to 'N_B' in Soummer et al. 2007 4.2)


    """



    npup = pupil.shape[0]

    du = nlamD / float(npix)
    dv = nlamD / float(npix)

    dx = 1.0/float(npup)
    dy = 1.0/float(npup)

    Xs = (np.arange(npup) - float(npup)/2.0 + 0.5) * dx
    Ys = (np.arange(npup) - float(npup)/2.0 + 0.5) * dy

    Us = (np.arange(npix) - float(npix)/2.0 + 0.5) * du
    Vs = (np.arange(npix) - float(npix)/2.0 + 0.5) * dv

    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)
    expXU = expXU.T.copy()

    t1 = np.dot(expXU, pupil)
    t2 = np.dot(t1, expYV)
    #print ""
    #print dx, dy, du, dv
    #print ""


    #return t2 * dx * dy
    return  float(nlamD)/(npup*npix) *   t2 

## --- OBSOLETED BY DFT_adjustible_rect just below ----
## 
## # ADJUSTIBLE centering: PSF centered on array regardless of parity
## def DFT_adjustible(pupil, nlamD, npix, offset=(0.0,0.0), inverse=False, **kwargs):
##     """
##     Compute an adjustible-center matrix fourier transform. 
## 
##     For an output array with ODD size n,
##     the PSF center will be at the center of pixel (n-1)/2
##     
##     For an output array with EVEN size n, 
##     the PSF center will be in the corner between pixel (n/2-1,n/2-1) and (n/2,n/2)
## 
##     Those coordinates all assume Python/IDL style pixel coordinates running from
##     (0,0) up to (n-1, n-1). 
## 
##     Parameters
##     ----------
##     pupil : array
##         pupil array (n by n)
##     nlamD : float or tuple
##         size of focal plane array, in units of lam/D
##         (corresponds to 'm' in Soummer et al. 2007 4.2)
##     npix : float or tuple
##         number of pixels per side side of focal plane array
##         (corresponds to 'N_B' in Soummer et al. 2007 4.2)
##     offset: tuple
##         an offset in pixels relative to the above
## 
##     """
## 
## 
##     npup = pupil.shape[0]
## 
##     du = nlamD / float(npix)
##     dv = nlamD / float(npix)
## 
##     dx = 1.0/float(npup)
##     dy = 1.0/float(npup)
## 
##     Xs = (np.arange(npup) - float(npup)/2.0 - offset[1] + 0.5) * dx
##     Ys = (np.arange(npup) - float(npup)/2.0 - offset[0] + 0.5) * dy
## 
##     Us = (np.arange(npix) - float(npix)/2.0 - offset[1] + 0.5) * du
##     Vs = (np.arange(npix) - float(npix)/2.0 - offset[0] + 0.5) * dv
## 
##     XU = np.outer(Xs, Us)
##     YV = np.outer(Ys, Vs)
## 
## 
##     expXU = np.exp(-2.0 * np.pi * 1j * XU)
##     expYV = np.exp(-2.0 * np.pi * 1j * YV)
##     expXU = expXU.T.copy()
## 
##     t1 = np.dot(expXU, pupil)
##     t2 = np.dot(t1, expYV)
## 
##     #return t2 * dx * dy
##     return  float(nlamD)/(npup*npix) *   t2 
## 

# ADJUSTIBLE centering:PSF centered on array regardless of parity, with rectangular pupils allowed
def DFT_adjustible_rect(pupil, nlamD, npix, offset=(0.0,0.0), inverse=False, **kwargs):
    """
    Compute an adjustible-center matrix fourier transform. 

    For an output array with ODD size n,
    the PSF center will be at the center of pixel (n-1)/2
    
    For an output array with EVEN size n, 
    the PSF center will be in the corner between pixel (n/2-1,n/2-1) and (n/2,n/2)

    Those coordinates all assume Python/IDL style pixel coordinates running from
    (0,0) up to (n-1, n-1). 


    This version supports rectangular, non-square arrays,
    in which case nlamD and npix should be 2-element
    tuples or lists, using the usual Pythonic order (Y,X)



    Parameters
    ----------
    pupil : array
        pupil array (n by n)
    nlamD : float or tuple
        size of focal plane array, in units of lam/D
        (corresponds to 'm' in Soummer et al. 2007 4.2)
    npix : float or tuple
        number of pixels per side side of focal plane array
        (corresponds to 'N_B' in Soummer et al. 2007 4.2)
    offset: tuple
        an offset in pixels relative to the above

    """


    npupY, npupX = pupil.shape[0:2]

    if np.isscalar(npix): 
        npixY, npixX = npix, npix
    else:
        npixY, npixX = npix[0:2]   

    if np.isscalar(nlamD):  
        nlamDY, nlamDX = nlamD, nlamD
    else:
        nlamDY, nlamDX = nlamD[0:2]


    if inverse:
        dX = nlamDX / float(npupX)
        dY = nlamDY / float(npupY)
        dU = 1.0/float(npixY)
        dV = 1.0/float(npixX)
    else:
        dU = nlamDX / float(npixX)
        dV = nlamDX / float(npixX)
        dX = 1.0/float(npupX)
        dY = 1.0/float(npupY)

    Xs = (np.arange(npupX) - float(npupX)/2.0 - offset[1] + 0.5) * dX
    Ys = (np.arange(npupY) - float(npupY)/2.0 - offset[0] + 0.5) * dY

    Us = (np.arange(npixX) - float(npixX)/2.0 - offset[1] + 0.5) * dU
    Vs = (np.arange(npixY) - float(npixY)/2.0 - offset[0] + 0.5) * dV

    YV = np.outer(Ys, Vs)
    XU = np.outer(Xs, Us)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)

    expYV = expYV.T.copy()
    t1 = np.dot(expYV, pupil)
    t2 = np.dot(t1, expXU)

    if inverse:
        #print expYV.shape, pupil.shape, expXU.shape
        t2 = t2[::-1, ::-1]

    norm_coeff = np.sqrt(  float( nlamDY* nlamDX) / (npupY*npupX*npixY*npixX))
    return  norm_coeff *   t2 

# back compatibility aliases for older/more confusing terminology: 
## SFT1 = DFT_fftstyle 
## SFT1rect = DFT_fftstyle_rect 
## SFT2 = DFT_symmetric
## SFT3 = DFT_adjustible
## SFT3rect = DFT_adjustible_rect



class MatrixFourierTransform:
    """Implements a discrete matrix Fourier transform for optical 
    propagation, following the algorithms discussed in 
    Soummer et al. 2007 JOSA 15 24

    Parameters
    ----------
    centering : string
        Either 'SYMMETRIC', 'FFTSTYLE', or 'ADJUSTIBLE'. 
        Sets whether the DFT result is centered at pixel n/2+1 (FFTSTYLE) 
        or on the crosshairs between the central pixels (SYMMETRIC),
        or exactly centered in the array no matter what (ADJUSTIBLE). 
        Default is FFTSTYLE. 


    Example
    -------
    mft = MatrixFourierTransform()
    mft.perform(pupilArray, focalplane_size, focalplane_npix)


    History
    -------
    Code by Sivaramakrishnan based on Soummer et al.
    2010-01 Documentation updated by Perrin
    2013-01 'choice' keyword renamed to 'centering' for clarity. 'choice' is retained
        as an option for back compatibility, however it is deprecated.

    """

    def __init__(self, centering="FFTSTYLE", verbose=False):

        self.verbose=verbose

        self._dft_fns = {'FFTSTYLE':DFT_fftstyle, "SYMMETRIC":DFT_symmetric, "ADJUSTIBLE":DFT_adjustible_rect, 'FFTRECT': DFT_fftstyle_rect}
        #self.centering_methods= ("FFTSTYLE", "SYMMETRIC", "ADJUSTIBLE", 'FFTRECT')
        centering = centering.upper()
        if centering not in self._dft_fns.keys():
            raise ValueError("Error: centering method must be one of [%s]" % ', '.join(self._dft_fns.keys()))
        self.centering = centering


        if self.verbose:
            _log.info("This instance of MatrixFourierTransform is a(n) {0}  set-up calling {1} ".format(centering, self._dft_fns[centering]))
        _log.debug("MatrixDFT initialized using centering type = {0}".format(centering))

    def perform(self, pupil, nlamD, npix, **kwargs):
        """ Forward Fourier Transform 

        Parameters
        --------------
        pupil : 2D ndarray
            Real or complex valued 2D array representing the input image to transform
        nlamD : float
            Size of desired output region in lambda/D units, assuming that the pupil fills the
            input array. I.e. this is in units of the spatial frequency that is just Nyquist sampled
            by the input array
        npix : int
            Number of pixels to use for representing across that region lambda/D units in size.

        Returns a 2D complex valued Fourier transform array.

        """

        dft_fn_to_call = self._dft_fns[self.centering]
        _log.debug("MatrixDFT mode {0} calling {1}".format( self.centering, str(dft_fn_to_call)))

        if not np.isscalar(nlamD) or not np.isscalar(npix):
            if self.centering == 'FFTSTYLE' or self.centering=='SYMMETRIC':
                raise RuntimeError('The selected MatrixDFT centering mode, {0}, does not support rectangular arrays.'.format(self.centering))
        return dft_fn_to_call(pupil, nlamD, npix, **kwargs)


    def inverse(self, image, nlamD, npix):
        """ Inverse Fourier Transform 

        Parameters
        --------------
        image : 2D ndarray
            Real or complex valued 2D array representing the input image to transform, which
            presumably is the result of some previous forward transform.
        nlamD : float
            Size of desired output region in lambda/D units, assuming that the pupil fills the
            input array. I.e. this is in units of the spatial frequency that is just Nyquist sampled
            by the input array
        npix : int
            Number of pixels to use for representing across that region lambda/D units in size.

        Returns a 2D complex valued Fourier transform array.


        """
        return self.perform(image, nlamD, npix, inverse=True)

#  This function is used nowhere in poppy or webbpsf, and it really
#  does not do much of anything useful - so let's delete it. Aug 2014.
#    def performFITS(hdulist, focalplane_size, focalplane_npix):
#        """ Perform an MFT, and return the result as a fits.HDUlist """
#        newHDUlist = hdulist.copy()
#        newim = self.perform(hdulist[0].data, focalplane_size, focalplane_npix)
#
#        newHDUlist[0].data = newim
#        #TODO fits header keyword updates
#
#        return newHDUlist

## SlowFourierTransform = MatrixFourierTransform  # back compatible name

#---------------------------------------------------------------------
#  Test functions 
#  are now in tests/test_matrixDFT.py

