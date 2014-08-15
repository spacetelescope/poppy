#  
#  Test functions for matrix DFT code
#
#

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io.fits as fits

import os
from .. import poppy_core 
from .. import optics
from .. import matrixDFT


import logging
_log = logging.getLogger('poppy_tests')


def complexinfo(a, str=None):
    """ Print some info about the dum of real and imaginary parts of an array
    """

    if str:
        print 
        print "\t", str
    re = a.real.copy()
    im = a.imag.copy()
    _log.debug("\t%.2e  %.2g  =  re.sum im.sum" % (re.sum(), im.sum()))
    _log.debug("\t%.2e  %.2g  =  abs(re).sum abs(im).sum" % (abs(re).sum(), abs(im).sum()))



def euclid2(s, c=None):
    """ Compute Euclidean distance between points across an 2d ndarray

    Paramters
    ----------
    s : tuple
        shape of array
    c : tuple
        coordinates of point to measure distance from

    """

    if c is None:
        c = (0.5*float(s[0]),  0.5*float(s[1]))

    y, x = np.indices(s)
    r2 = (x - c[0])**2 + (y - c[1])**2

    return r2

def makedisk(s=None, c=None, r=None, inside=1.0, outside=0.0, grey=None, t=None):
    """ Create a 2D ndarray containing a uniform circular aperture with
    given center and size
    """

    
    # fft style or sft asymmetric style - center = nx/2, ny/2
    # see ellipseDriver.py for details on symm...

    disk = np.where(euclid2(s, c=c) <= r*r, inside, outside)
    return disk




def test_MFT_flux_conservation(centering='FFTSTYLE', outdir=None, outname='test_MFT_flux', precision=0.01):
    """
    Test that MFTing a circular aperture is properly normalized to be flux conserving.

    This test is limited primarily by how finely the arrays are sampled. The function implements tests to
    better than 1% or 0.1%, selectable using the 'precision' argument. 

    Parameters
    -----------

    outdir : path
        Directory path to output diagnostic FITS files. If not specified, files will not be written.
    precision : float, either 0.01 or 0.001
        How precisely to expect flux conservation; it will not be strictly 1.0 given any finite array size. 
        This function usesd predetermined MFT array sizes based on the desired precision level of the test.
    """

    # Set up constants for either a more precise test or a less precise but much 
    # faster test:
    print("Testing MFT flux conservation for centering = "+centering)
    if precision ==0.001:
        npupil = 800
        npix = 4096
        u = 400    # of lam/D. Must be <= the Nyquist frequency of the pupil sampling or there
                   #           will be aliased copies of the PSF.
    elif precision==0.01: 
        npupil = 400
        npix = 2048
        u = 200    # of lam/D. Must be <= the Nyquist frequency of the pupil sampling or there
                   #           will be aliased copies of the PSF.
    else:
        raise NotImplementedError('Invalid value for precision.')



    # Create pupil
    ctr = (float(npupil)/2.0, float(npupil)/2.0 )
    pupil = makedisk(s=(npupil, npupil), c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)
    pupil /= np.sqrt(pupil.sum())
    if outdir is not None:
        fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    # MFT setup style and execute
    mft = matrixDFT.MatrixFourierTransform(centering=centering, verbose=True)
    a = mft.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum()    # normalized area of input pupil, should be 1 by construction
    post = (abs(a)**2).sum()       # 
    ratio = post / pre
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)



    if outdir is not None:
        complexinfo(a, str="mft1 asf")
        asf = a.real.copy()
        fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
        cpsf = a * a.conjugate()
        psf = cpsf.real.copy()
        #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
        fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)

    assert np.abs(1.0 - ratio) < precision

def test_MFT_fluxconsv_all_types(centering=None, **kwargs):

    test_MFT_flux_conservation(centering='FFTSTYLE', **kwargs)
    test_MFT_flux_conservation(centering='SYMMETRIC', **kwargs)
    test_MFT_flux_conservation(centering='ADJUSTIBLE', **kwargs)
    test_MFT_flux_conservation(centering='FFTRECT', **kwargs)




def test_DFT_rect(centering='FFTRECT', outdir='.', outname='DFT1R_', npix=None, sampling=10., nlamd=None):
    """
    Test matrix DFT, including non-square arrays, in both the
    forward and inverse directions.

    This is an exact equivalent (in Python) of test_matrix_DFT in matrix_dft.pro (in IDL)
    They should give identical results.

    """

    print "Testing DFT, style = "+centering


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
    mft1 = matrixDFT.MatrixFourierTransform(centering=centering)

    #ctr = (float(npupil)/2.0 + mft1.offset(), float(npupil)/2.0 + mft1.offset())
    ctr = (float(npupil)/2.0 , float(npupil)/2.0)
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil[0:60, 0:60] = 0
    pupil[0:10] = 0

    pupil /= np.sqrt(pupil.sum())

    plt.clf()
    plt.subplots_adjust(left=0.02, right=0.98)
    plt.subplot(141)

    pmx = pupil.max()
    plt.imshow(pupil, vmin=0, vmax=pmx*1.5)


    fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    a = mft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum() 
    post = (abs(a)**2).sum() 
    ratio = post / pre
    calcr = 1./(1.0*u[0]*u[1] *npix[0]*npix[1])     # multiply post by this to make them equal
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str=",ft1 asf")
    #print 
    asf = a.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"asf.fits", data=asf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)

    ax=plt.subplot(142)
    plt.imshow(asf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
    ax.set_title='ASF'

    ax=plt.subplot(143)
    plt.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
    ax.set_title='PSF'

    plt.subplot(144)

    pupil2 = mft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    plt.imshow( pupil2r, vmin=0,vmax=pmx*1.5*0.01) # FIXME flux normalization is not right?? I think this has to do with squaring the pupil here, that's all.
    plt.gca().set_title='back to pupil'
    plt.draw()
    print "Post-inverse FFT total: "+str( abs(pupil2r).sum() )
    print "Post-inverse pupil max: "+str(pupil2r.max())

    plt.suptitle('Matrix DFT with rectangular arrays using centering={0}'.format(centering))

    plt.savefig('test_DFT_rectangular_results_{0}.pdf'.format(centering))

def test_DFT_rect_adj():
    """ Repeat DFT rectangle check, but for am adjustible FFT centering 
    """
    test_DFT_rect(centering='ADJUSTIBLE', outname='DFT1Radj_')

def test_DFT_center( npix=100, outdir='.', outname='DFT1'):
    centering='ADJUSTIBLE'

    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    mft1 = matrixDFT.MatrixFourierTransform(centering=centering)

    ctr = (float(npupil)/2.0, float(npupil)/2.0 )
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    a = mft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum() 
    post = (abs(a)**2).sum() 
    ratio = post / pre
    calcr = 1./(u**2 *npix**2)     # multiply post by this to make them equal
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str="mft1 asf")
    #print 
    asf = a.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"asf.fits", data=asf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)



def test_inverse( centering='SYMMETRIC'):
    """ Test repeated transformations between pupil and image

    TODO FIXME - this needs some assertions added
    """


    npupil = 300 #156
    pctr = int(npupil/2)
    npix = 100 #1024
    u = 20 #100    # of lam/D

    npix, u = 2000, 200
    s = (npupil,npupil)




    mft1 = matrixDFT.MatrixFourierTransform(centering=centering)

    ctr = (float(npupil)/2.0, float(npupil)/2.0 )
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)
    pupil /= np.sqrt(pupil.sum())

    pupil[100:200, 30:50] = 0
    pupil[0:50, 140:160] = 0

    plt.subplot(141)
    plt.imshow(pupil)

    print "Pupil 1 total:", pupil.sum() 

    a = mft1.perform(pupil, u, npix)

    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    print "PSF total", psf.sum()
 
    plt.subplot(142)
    plt.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))

    plt.subplot(143)

    pupil2 = mft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    plt.imshow( pupil2r)

    print "Pupil 2 total:", pupil2r.sum() 



    a2 = mft1.perform(pupil2r, u, npix)
    psf2 = (a2*a2.conjugate()).real.copy()
    print "PSF total", psf2.sum()
    plt.subplot(144)
    plt.imshow(psf2, norm=matplotlib.colors.LogNorm(1e-8, 1.0))


def run_all_MFS_tests_DFT(outdir='.', outname='DFT1'):
    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    #mft1 = MatrixFourierTransform(centering=centering)


    # make a pupil
    ctr = (float(npupil)/2.0 , float(npupil)/2.0)  # in middle of array (though this should make no difference)
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.00, t=np.float32, grey=0)
    pupil /= np.sqrt(pupil.sum())

    fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

    npix=512
    a1 = DFT_combined(pupil, u, npix, centering='FFTSTYLE')
    a2 = DFT_combined(pupil, u, npix, centering='SYMMETRIC')
    a3 = DFT_combined(pupil, u, npix, centering='ADJUSTIBLE')
    a4 = DFT_fftstyle(pupil, u, npix)
    a5 = DFT_symmetric(pupil, u, npix)

    fits.writeto(outdir+os.sep+outname+"_a1_fft.fits",(a1*a1.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_a2_sym.fits",(a2*a2.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_a3_adj.fits",(a3*a3.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_a4_fftr.fits",(a4*a4.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_a5_symr.fits",(a5*a5.conjugate()).real, clobber=True) 

    npix=513
    b1 = DFT_combined(pupil, u, npix, centering='FFTSTYLE')
    b2 = DFT_combined(pupil, u, npix, centering='SYMMETRIC')
    b3 = DFT_combined(pupil, u, npix, centering='ADJUSTIBLE')
    b4 = DFT_fftstyle(pupil, u, npix)
    b5 = DFT_symmetric(pupil, u, npix)


    fits.writeto(outdir+os.sep+outname+"_b1_fft.fits",(b1*b1.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_b2_sym.fits",(b2*b2.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_b3_adj.fits",(b3*b3.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_b4_fftr.fits",(b4*b4.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_b5_symr.fits",(b5*b5.conjugate()).real, clobber=True) 


    u2 = (u, u/4)
    npix2=(512, 128)
    c1 = DFT_combined(pupil, u2, npix2, centering='FFTSTYLE')
    c2 = DFT_combined(pupil, u2, npix2, centering='SYMMETRIC')
    c3 = DFT_combined(pupil, u2, npix2, centering='ADJUSTIBLE')
    c4 = DFT_fftstyle_rect(pupil, u2, npix2)
    c5 = DFT_adjustible_rect(pupil, u2, npix2)

    fits.writeto(outdir+os.sep+outname+"_c1_fft.fits",(c1*c1.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_c2_sym.fits",(c2*c2.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_c3_adj.fits",(c3*c3.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_c4_fftr.fits",(c4*c4.conjugate()).real, clobber=True) 
    fits.writeto(outdir+os.sep+outname+"_c5_adjr.fits",(c5*c5.conjugate()).real, clobber=True) 


    for c, label in zip([c1, c2, c3, c4,c5], ['comb-fft', 'comb-sym', 'comb-adj', 'fft_rect', 'adj_rect']) :
        print label, c.shape

def test_check_invalid_centering():
    """ intentionally invalid CENTERING option to test the error message part of the code. 
    """
    try:
        import pytest
    except:
        poppy._log.warning('Skipping test test_check_invalid_centering because pytest is not installed.')
        return # We can't do this test if we don't have the pytest.raises function.

    # MFT setup style and execute

    with pytest.raises(ValueError) as excinfo:
        mft = matrixDFT.MatrixFourierTransform(centering='some garbage value', verbose=True)
    assert excinfo.value.message == 'Error: centering method must be one of [SYMMETRIC, ADJUSTIBLE, FFTRECT, FFTSTYLE]'



