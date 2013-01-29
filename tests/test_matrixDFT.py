#  
#  Test functions for matrix DFT code
#
#

import matplotlib
import matplotlib.pyplot as plt

import os

from poppy.matrixDFT import *

import logging
_log = logging.getLogger('poppy-test')


def complexinfo(a, str=None):

    if str:
        print 
        print "\t", str
    re = a.real.copy()
    im = a.imag.copy()
    print "\t%.2e  %.2g  =  re.sum im.sum" % (re.sum(), im.sum())
    print "\t%.2e  %.2g  =  abs(re).sum abs(im).sum" % (abs(re).sum(), abs(im).sum())



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





def test_DFT(centering='FFTSTYLE', outdir='.', outname='DFT1'):

    print "Testing DFT, style = "+centering


    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    sft1 = MatrixFourierTransform(centering=centering)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

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
    fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)


def test_DFT_all_types():

    test_DFT(centering='FFTSTYLE')
    test_DFT(centering='SYMMETRIC')
    test_DFT(centering='ADJUSTIBLE')
    test_DFT(centering='FFTRECT')




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
    sft1 = MatrixFourierTransform(centering=centering)

    #ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
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

    pupil2 = sft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    plt.imshow( pupil2r, vmin=0,vmax=pmx*1.5*0.01) # FIXME flux normalization is not right?? I think this has to do with squaring the pupil here, that's all.
    plt.gca().set_title='back to pupil'
    plt.draw()
    print "Post-inverse FFT total: "+str( abs(pupil2r).sum() )
    print "Post-inverse pupil max: "+str(pupil2r.max())

    plt.suptitle('Matrix DFT with rectangular arrays using centering={0}'.format(centering))

    plt.savefig('test_DFT_rectangular_results_{0}.pdf'.format(centering))

def test_DFT_rect_adj():
    test_DFT_rect(centering='ADJUSTIBLE', outname='DFT1Radj_')

def test_DFT_center( npix=100, outdir='.', outname='DFT1'):
    centering='ADJUSTIBLE'

    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    sft1 = MatrixFourierTransform(centering=centering)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", clobber=True)

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
    fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", clobber=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), clobber='y')
    fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", clobber=True)



def test_inverse():

    centering='ADJUSTIBLE'
    centering='SYMMETRIC'

    npupil = 300 #156
    pctr = int(npupil/2)
    npix = 100 #1024
    u = 20 #100    # of lam/D

    npix, u = 2000, 200
    s = (npupil,npupil)




    # FFT style
    sft1 = MatrixFourierTransform(centering=centering)

    ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)
    pupil /= np.sqrt(pupil.sum())

    pupil[100:200, 30:50] = 0
    pupil[0:50, 140:160] = 0

    plt.subplot(141)
    plt.imshow(pupil)

    print "Pupil 1 total:", pupil.sum() 

    a = sft1.perform(pupil, u, npix)

    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    print "PSF total", psf.sum()
 
    plt.subplot(142)
    plt.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))

    plt.subplot(143)

    pupil2 = sft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    plt.imshow( pupil2r)

    print "Pupil 2 total:", pupil2r.sum() 



    a2 = sft1.perform(pupil2r, u, npix)
    psf2 = (a2*a2.conjugate()).real.copy()
    print "PSF total", psf2.sum()
    plt.subplot(144)
    plt.imshow(psf2, norm=matplotlib.colors.LogNorm(1e-8, 1.0))


def test_DFT_combined(outdir='.', outname='DFT1'):


    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil,npupil)


    # FFT style
    #sft1 = MatrixFourierTransform(centering=centering)

    #ctr = (float(npupil)/2.0 + sft1.offset(), float(npupil)/2.0 + sft1.offset())

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


