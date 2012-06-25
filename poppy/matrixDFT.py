#! /usr/bin/env  python 
"""
    Matrix DFT  

    Matrix-based Fourier transforms for computing PSFs. 
    See Soummer et al. 2007 JOSA

    This module was originally called "Slow Fourier Transform", and the function names still are. 
    Note that this is 'slow' only in the sense that if you perform the exact same
    calculation as an FFT, the FFT algorithm is much faster. However this algorithm
    gives you much more flexibility in choosing array sizes and sampling, and often lets
    you replace "fast calculations on very large arrays" with 
    "relatively slow calculations on much smaller ones". 


    This module contains the following four functions for different types of discrete
    FTs. There is also a class SlowFourierTransform() which is a useful wrapper.
        - SFT1: 'FFTSTYLE' centered on one pixel
        - SFT2: 'SYMMETRIC' centerd on crosshairs between middle pixel
        - SFT3: 'ADJUSTIBLE', always centered in output array depending on whether it is even/odd
        - SFT4: 'ROTATABLE'; not properly implemented do not use

    There are also SFT1rect and SFT3rect versions which support non-square arrays for both
    forward and inverse transformations.


    Example
    -------
    sf = matrixDFT.SlowFourierTransform()
    result = sf.perform(pupilArray, focalplane_size, focalplane_npix)




    History
    -------
    Code originally by A. Sivaramakrishnan
    2010-11-05 Revised normalizations for flux conservation consistent
        with Soummer et al. 2007. Updated documentation.  -- M. Perrin
    2011-2012: Various enhancements, detailed history not kept, sorry.
    2012-05-18: module renamed SFT.py -> matrixDFT.py

"""
from __future__ import division # always floating point


import numpy as np
import pyfits

try:
    __IPYTHON__
    from IPython.Debugger import Tracer; stop = Tracer()
except:
    def stop(): 
        pass


#import SimpleFits as SF


# FFTSTYLE
def SFT1(pupil, nlamD, npix, **kwargs):
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

    print ""
    print dx, dy, du, dv
    print ""

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

# FFTSTYLE
def SFT1rect(pupil, nlamD, npix, inverse=False):
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

    if hasattr(npix, '__getitem__'):
        npixY, npixX = npix[0:2]   
    else:
        npixY, npixX = npix, npix

    if hasattr(nlamD, '__getitem__'):
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

    Xs = (np.arange(npupX) - (npupX/2)) * dX
    Ys = (np.arange(npupY) - (npupY/2)) * dY

    Us = (np.arange(npixX) - npixX/2) * dU
    Vs = (np.arange(npixY) - npixY/2) * dV

    YV = np.outer(Ys, Vs)
    XU = np.outer(Xs, Us)

    expYV = np.exp(-2.0 * np.pi * 1j * YV)  
    expXU = np.exp(-2.0 * np.pi * 1j * XU)

    #print ""
    #print dX, dY, dU, dV
    #print (npupY, npupX, nlamDY, nlamDX, npixY, npixY)
    #print YV.shape, XU.shape
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
    norm_coeff = np.sqrt(  ( nlamDY* nlamDX) / (npupY*npupX*npixY*npixX))
    #mean_npup = np.sqrt(npupY**2+npupX**2)
    #mean_npix = np.sqrt(npixY**2+npixX**2)
    return  norm_coeff *   t2 


# SYMMETRIC
def SFT2(pupil, nlamD, npix, **kwargs):
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


# ADJUSTIBLE
def SFT3(pupil, nlamD, npix, offset=(0.0,0.0), inverse=False, **kwargs):
    """
    Compute an adjustible-center matrix fourier transform. 

    For an output array with ODD size n,
    the PSF center will be at the center of pixel (n-1)/2
    
    For an output array with EVEN size n, 
    the PSF center will be in the corner between pixel (n/2-1,n/2-1) and (n/2,n/2)

    Those coordinates all assume Python/IDL style pixel coordinates running from
    (0,0) up to (n-1, n-1). 

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


    npup = pupil.shape[0]

    du = nlamD / float(npix)
    dv = nlamD / float(npix)

    dx = 1.0/float(npup)
    dy = 1.0/float(npup)

    Xs = (np.arange(npup) - float(npup)/2.0 - offset[1] + 0.5) * dx
    Ys = (np.arange(npup) - float(npup)/2.0 - offset[0] + 0.5) * dy

    Us = (np.arange(npix) - float(npix)/2.0 - offset[1] + 0.5) * du
    Vs = (np.arange(npix) - float(npix)/2.0 - offset[0] + 0.5) * dv

    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)
    expXU = expXU.T.copy()

    t1 = np.dot(expXU, pupil)
    t2 = np.dot(t1, expYV)

    #return t2 * dx * dy
    return  float(nlamD)/(npup*npix) *   t2 


# ADJUSTIBLE
def SFT3rect(pupil, nlamD, npix, offset=(0.0,0.0), inverse=False, **kwargs):
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

    if np.isscalar(npix): #hasattr(npix, '__len__'):
        npixY, npixX = npix, npix
    else:
        npixY, npixX = npix[0:2]   

    if not np.isscalar(nlamD):  #hasattr(nlamD, '__getitem__'):
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

    Xs = (np.arange(npupX) - float(npupX)/2.0 - offset[1] + 0.5) * dX
    Ys = (np.arange(npupY) - float(npupY)/2.0 - offset[0] + 0.5) * dY

    Us = (np.arange(npixX) - float(npixX)/2.0 - offset[1] + 0.5) * dU
    Vs = (np.arange(npixX) - float(npixY)/2.0 - offset[0] + 0.5) * dV

    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)

    expYV = expYV.T.copy()
    t1 = np.dot(expYV, pupil)
    t2 = np.dot(t1, expXU)

    if inverse:
        #print expYV.shape, pupil.shape, expXU.shape
        t2 = t2[::-1, ::-1]

    norm_coeff = np.sqrt(  ( nlamDY* nlamDX) / (npupY*npupX*npixY*npixX))
    return  norm_coeff *   t2 



# ROTATABLE 
def SFT4(pupil, nlamD, npix, offset=(0.0,0.0), angle=0.0, **kwargs):
    """
    Compute an adjustible-center, rotatable matrix fourier transform. 

    For an output array with ODD size n,
    the PSF center will be at the center of pixel (n-1)/2
    
    For an output array with EVEN size n, 
    the PSF center will be in the corner between pixel (n/2-1,n/2-1) and (n/2,n/2)

    Those coordinates all assume IDL or Python style pixel coordinates running from
    (0,0) up to (n-1, n-1). 

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
    offset
        an offset in pixels relative to the above

    """

    rotation = 45.0
    cosr = np.cos( np.radians(rotation))
    sinr = np.sin( np.radians(rotation))


    npup = pupil.shape[0]

    du = nlamD / float(npix)
    dv = nlamD / float(npix)

    dx = 1.0/float(npup)
    dy = 1.0/float(npup)

    # Xs and Ys are, unsurprisingly, the X and Y coordinates for the pupil array.
    # I think? Though then why does it make it work properly to shift them here like this?
    # Actually it should make NO difference. A shift in the pupil plane just induces a phase tilt
    # in the image plane, which we don't care about for PSFs since we just measure total intensity.
      #Yep, confirmed numerically with some tests. Applying shifts here is unnecessary!
        #Xs = (np.arange(npup) - float(npup)/2.0 - offset[1] + 0.5) * dx
        #Ys = (np.arange(npup) - float(npup)/2.0 - offset[0] + 0.5) * dy

    Xs = (np.arange(npup) - float(npup)/2.0 ) * dx
    Ys = (np.arange(npup) - float(npup)/2.0 ) * dy



    # OK, a 2D FFT can be computed as a the result of two separate 1D transforms...


    # Aaaargh this is not going to work.
    #
    Us = (np.arange(npix) - float(npix)/2.0 - offset[1] + 0.5) * du
    Vs = (np.arange(npix) - float(npix)/2.0 - offset[0] + 0.5) * dv
    Us.shape = (1,npix)
    Vs.shape = (npix,1)

    UsR =  cosr*Us + sinr*Vs
    VsR = -sinr*Us + cosr*Vs


    XU = np.outer(Xs, Us)
    YV = np.outer(Ys, Vs)


    expXU = np.exp(-2.0 * np.pi * 1j * XU)
    expYV = np.exp(-2.0 * np.pi * 1j * YV)
    expXU = expXU.T.copy()

    t1 = np.dot(expXU, pupil)
    t2 = np.dot(t1, expYV)

    #return t2 * dx * dy
    return  float(nlamD)/(npup*npix) *   t2 



class SlowFourierTransform:
    """Implements a discrete matrix Fourier transform for optical 
    propagation, following the algorithms discussed in 
    Soummer et al. 2007 JOSA 15 24

    Parameters
    ----------
    choice : string
        Either 'SYMMETRIC', 'FFTSTYLE', or 'ADJUSTIBLE'. 
        Sets whether the DFT result is centered at pixel n/2+1 (FFTSTYLE) 
        or on the crosshairs between the central pixels (SYMMETRIC),
        or exactly centered in the array no matter what (ADJUSTIBLE). Default is FFTSTYLE. 


    Example
    -------
    sft = SlowFourierTransform()
    sft.perform(pupilArray, focalplane_size, focalplane_npix)


    History
    -------
    Code by Sivaramakrishnan based on Soummer et al.
    2010-01 Documentation updated by Perrin

    """

    def __init__(self, choice="FFTSTYLE", verbose=False):

        self.verbose=verbose

        self.choices = ("FFTSTYLE", "SYMMETRIC", "ADJUSTIBLE", 'FFTRECT')
        self.correctoffset = {self.choices[0]: 0.5, self.choices[1]: 0.0, self.choices[2]:-1, self.choices[3]: 0.5 }
        if choice not in self.choices:
            raise ValueError("Error: choice must be one of [%s]" % ', '.join(self.choices))
        self.choice = choice

        fns = {'FFTSTYLE':SFT1, "SYMMETRIC":SFT2, "ADJUSTIBLE":SFT3rect, 'FFTRECT': SFT1rect}

        if self.verbose:
            #print choice
            #print "Announcement  - This instance of SlowFourierTransform uses SFT2"
            print "This instance of SFT is a(n) %s  set-up calling %s " % (choice, fns[choice])
        self.perform = fns[choice]

    def offset(self):
        return self.correctoffset[self.choice]


    def inverse(self, image, nlamD, npix):
        return self.perform(image, nlamD, npix, inverse=True)


    def performFITS(hdulist, focalplane_size, focalplane_npix):
        """ Perform an MFT, and return the result as a pyfits.HDUlist """
        newHDUlist = hdulist.copy()
        newim = self.perform(hdulist[0].data, focalplane_size, focalplane_npix)

        newHDUlist[0].data = newim
        #TODO fits header keyword updates

        return newHDUlist


#---------------------------------------------------------------------
#  Test functions 

def euclid2(s, c=None):

	if c is None:
		c = (0.5*float(s[0]),  0.5*float(s[1]))

	y, x = np.indices(s)
	r2 = (x - c[0])**2 + (y - c[1])**2

	return r2

def makedisk(s=None, c=None, r=None, inside=1.0, outside=0.0, grey=None, t=None):
	
	# fft style or sft asymmetric style - center = nx/2, ny/2
	# see ellipseDriver.py for details on symm...

	disk = np.where(euclid2(s, c=c) <= r*r, inside, outside)
	return disk



def test_SFT(choice='FFTSTYLE', outdir='.', outname='SFT1'):
    import os

    print "Testing SFT, style = "+choice

    def complexinfo(a, str=None):

        if str:
            print 
            print "\t", str
        re = a.real.copy()
        im = a.imag.copy()
        print "\t%.2e  %.2g  =  re.sum im.sum" % (re.sum(), im.sum())
        print "\t%.2e  %.2g  =  abs(re).sum abs(im).sum" % (abs(re).sum(), abs(im).sum())


    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    sft1 = SlowFourierTransform(choice=choice)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    pyfits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    a = sft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum() 
    post = (abs(a)**2).sum() 
    ratio = post / pre
    calcr = 1./(u**2 *npix**2)     # multiply post by this to make them equal
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str="sft1 asf")
    #print 
    asf = a.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"asf.fits", data=asf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)


def test_SFT_rect(choice='FFTRECT', outdir='.', outname='SFT1R_', npix=None, sampling=10., nlamd=None):
    """
    Test matrix DFT, including non-square arrays, in both the
    forward and inverse directions.

    This is an exact equivalent (in Python) of test_matrix_DFT in matrix_dft.pro (in IDL)
    They should give identical results.

    """
    import os
    import matplotlib
    import matplotlib.pyplot as P



    print "Testing SFT, style = "+choice

    def complexinfo(a, str=None):

        if str:
            print 
            print "\t", str
        re = a.real.copy()
        im = a.imag.copy()
        print "\t%.2e  %.2g  =  re.sum im.sum" % (re.sum(), im.sum())
        print "\t%.2e  %.2g  =  abs(re).sum abs(im).sum" % (abs(re).sum(), abs(im).sum())


    npupil = 156
    pctr = int(npupil/2)
    s = (npupil,npupil)


    # make things rectangular:
    if nlamd is None and npix is None:
        nlamd = (10,20)
        npix = [val*sampling for val in nlamd] #(100, 200) 
    elif npix is None:
        npix = [val*sampling for val in nlamd] #(100, 200) 
    elif nlamd is None:
        nlamd = [val/sampling for val in npix]
    u = nlamd
    print u
    #(u, float(u)/npix[0]*npix[1])
    #npix = (npix, 2*npix)


    # FFT style
    sft1 = SlowFourierTransform(choice=choice)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil[0:60, 0:60] = 0
    pupil[0:10] = 0

    pupil /= np.sqrt(pupil.sum())

    P.clf()
    P.subplots_adjust(left=0.02, right=0.98)
    P.subplot(141)

    pmx = pupil.max()
    P.imshow(pupil, vmin=0, vmax=pmx*1.5)


    pyfits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    a = sft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum() 
    post = (abs(a)**2).sum() 
    ratio = post / pre
    calcr = 1./(1.0*u[0]*u[1] *npix[0]*npix[1])     # multiply post by this to make them equal
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str="sft1 asf")
    #print 
    asf = a.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"asf.fits", data=asf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)

    ax=P.subplot(142)
    P.imshow(asf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
    ax.set_title='ASF'

    ax=P.subplot(143)
    P.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
    ax.set_title='PSF'

    P.subplot(144)

    pupil2 = sft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    P.imshow( pupil2r, vmin=0,vmax=pmx*1.5*0.01) # FIXME flux normalization is not right?? I think this has to do with squaring the pupil here, that's all.
    P.gca().set_title='back to pupil'
    P.draw()
    print "Post-inverse FFT total: "+str( abs(pupil2r).sum() )
    print "Post-inverse pupil max: "+str(pupil2r.max())

    stop()


def test_SFT_center( npix=100, outdir='.', outname='SFT1'):
    choice='ADJUSTIBLE'
    import os

    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    sft1 = SlowFourierTransform(choice=choice)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    pyfits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    a = sft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum() 
    post = (abs(a)**2).sum() 
    ratio = post / pre
    calcr = 1./(u**2 *npix**2)     # multiply post by this to make them equal
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str="sft1 asf")
    #print 
    asf = a.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"asf.fits", data=asf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    pyfits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)



def test_inverse():
    import matplotlib
    import matplotlib.pyplot as P

    choice='ADJUSTIBLE'
    choice='SYMMETRIC'
    import os

    npupil = 300 #156
    pctr = int(npupil/2)
    npix = 100 #1024
    u = 20 #100    # of lam/D

    npix, u = 2000, 200
    s = (npupil,npupil)




    # FFT style
    sft1 = SlowFourierTransform(choice=choice)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)
    pupil /= np.sqrt(pupil.sum())

    pupil[100:200, 30:50] = 0
    pupil[0:50, 140:160] = 0

    P.subplot(141)
    P.imshow(pupil)

    print "Pupil 1 total:", pupil.sum() 

    a = sft1.perform(pupil, u, npix)

    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    print "PSF total", psf.sum()
 
    P.subplot(142)
    P.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))

    P.subplot(143)

    pupil2 = sft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    P.imshow( pupil2r)

    print "Pupil 2 total:", pupil2r.sum() 



    a2 = sft1.perform(pupil2r, u, npix)
    psf2 = (a2*a2.conjugate()).real.copy()
    print "PSF total", psf2.sum()
    P.subplot(144)
    P.imshow(psf2, norm=matplotlib.colors.LogNorm(1e-8, 1.0))




if __name__ == "__main__":

    #test_SFT('FFTSTYLE', outname='SFT1')
    #test_SFT('SYMMETRIC', outname='SFT2')
    pass
