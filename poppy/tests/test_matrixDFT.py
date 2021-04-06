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
from .test_errorhandling import _exception_message_starts_with


import logging
_log = logging.getLogger('poppy_tests')


def complexinfo(a, str=None):
    """ Print some info about the dum of real and imaginary parts of an array
    """

    if str:
        print()
        print("\t", str)
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
    print("Testing MFT flux conservation for centering = {}".format(centering))
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
        fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", overwrite=True)

    # MFT setup style and execute
    mft = matrixDFT.MatrixFourierTransform(centering=centering, verbose=True)
    a = mft.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum()    # normalized area of input pupil, should be 1 by construction
    post = (abs(a)**2).sum()       #
    ratio = post / pre
    print("Pre-FFT  total: "+str( pre))
    print("Post-FFT total: "+str( post ))
    print("Ratio:          "+str( ratio))



    if outdir is not None:
        complexinfo(a, str="mft1 asf")
        asf = a.real.copy()
        fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", overwrite=True)
        cpsf = a * a.conjugate()
        psf = cpsf.real.copy()
        #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), overwrite='y')
        fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", overwrite=True)

    assert np.abs(1.0 - ratio) < precision

def test_MFT_fluxconsv_all_types(centering=None, **kwargs):

    test_MFT_flux_conservation(centering='FFTSTYLE', **kwargs)
    test_MFT_flux_conservation(centering='SYMMETRIC', **kwargs)
    test_MFT_flux_conservation(centering='ADJUSTABLE', **kwargs)
    test_MFT_flux_conservation(centering='FFTRECT', **kwargs)




def test_DFT_rect(centering='FFTSTYLE', outdir=None, outname='DFT1R_', npix=None, sampling=10., nlamd=None, display=False):
    """
    Test matrix DFT, including non-square arrays, in both the
    forward and inverse directions.

    This is an exact equivalent (in Python) of Marshall Perrin's
    test_matrix_DFT in matrix_dft.pro (in IDL)
    They should give identical results. However, this function doesn't actually
    check that since that would require having IDL...
    Instead it just checks that the sizes of the output arrays
    are as requested.

    """

    _log.info("Testing DFT, style = "+centering)


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
    _log.info("Requested sampling in pixels: "+str(npix))
    _log.info("Requested sampling in lam/D units: "+str(u))
    #(u, float(u)/npix[0]*npix[1])
    #npix = (npix, 2*npix)


    # FFT style
    _log.info('init with centering='+ centering)
    mft1 = matrixDFT.MatrixFourierTransform(centering=centering)

    #ctr = (float(npupil)/2.0 + mft1.offset(), float(npupil)/2.0 + mft1.offset())
    ctr = (float(npupil)/2.0 , float(npupil)/2.0)
    #print ctr
    pupil = makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil[0:60, 0:60] = 0
    pupil[0:10] = 0

    pupil /= np.sqrt(pupil.sum())

    if display:
        plt.clf()
        plt.subplots_adjust(left=0.02, right=0.98)
        plt.subplot(141)

        pmx = pupil.max()
        plt.imshow(pupil, vmin=0, vmax=pmx*1.5)


    if outdir is not None:
        fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", overwrite=True)

    _log.info('performing MFT with pupil shape: '+ str(pupil.shape)+ ' nlamd: '+ str( nlamd)+ '  npix: '+ str(npix))
    a = mft1.perform(pupil, nlamd, npix)


    _log.info('Shape of MFT result: '+str(a.shape))

    assert( a.shape[0] == npix[0] )
    assert( a.shape[1] == npix[1] )




    pre = (abs(pupil)**2).sum()
    post = (abs(a)**2).sum()
    ratio = post / pre
    calcr = 1./(1.0*u[0]*u[1] *npix[0]*npix[1])     # multiply post by this to make them equal
    _log.info( "Pre-FFT  total: "+str( pre))
    _log.info( "Post-FFT total: "+str( post ))
    _log.info( "Ratio:          "+str( ratio))
    #_log.info( "Calc ratio  :   "+str( calcr))
    #_log.info( "uncorrected:    "+str( ratio/calcr))


    #_log.info(complexinfo(a, str=",ft1 asf"))
    asf = a.real.copy()
    if outdir is not None:
        fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", overwrite=True)
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    if outdir is not None:
        fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", overwrite=True)

    # Inverse transform:
    pupil2 = mft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real

    assert(pupil2.shape[0] == pupil.shape[0] )
    assert(pupil2.shape[1] == pupil.shape[1] )


    if display:
        ax=plt.subplot(142)
        plt.imshow(asf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
        ax.set_title('ASF')

        ax=plt.subplot(143)
        plt.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))
        ax.set_title('PSF')

        plt.subplot(144)

        plt.imshow(np.abs(pupil2))
        plt.gca().set_title('back to pupil')
        plt.draw()
        plt.suptitle('Matrix DFT with rectangular arrays using centering={0}'.format(centering))

        plt.savefig('test_DFT_rectangular_results_{0}.pdf'.format(centering))

    _log.info( "Post-inverse FFT total: "+str( abs(pupil2r).sum() ))
    _log.info( "Post-inverse pupil max: "+str(pupil2r.max()))


def test_DFT_rect_adj():
    """ Repeat DFT rectangle check, but for adjustable FFT centering
    """
    test_DFT_rect(centering='ADJUSTABLE', outname='DFT1Radj_')

def test_DFT_center( npix=100, outdir=None, outname='DFT1'):
    centering='ADJUSTABLE'

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

    if outdir is not None:
        fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", overwrite=True)

    a = mft1.perform(pupil, u, npix)

    pre = (abs(pupil)**2).sum()
    post = (abs(a)**2).sum()
    ratio = post / pre
    calcr = 1./(u**2 *npix**2)     # multiply post by this to make them equal
    print("Pre-FFT  total: "+str( pre))
    print("Post-FFT total: "+str( post ))
    print("Ratio:          "+str( ratio))
    #print "Calc ratio  :   "+str( calcr)
    #print "uncorrected:    "+str( ratio/calcr)


    complexinfo(a, str="mft1 asf")
    #print
    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    #SF.SimpleFitsWrite(fn=outdir+os.sep+outname+"psf.fits", data=psf.astype(np.float32), overwrite='y')
    if outdir is not None:
        fits.PrimaryHDU(asf.astype(np.float32)).writeto(outdir+os.sep+outname+"asf.fits", overwrite=True)
        fits.PrimaryHDU(psf.astype(np.float32)).writeto(outdir+os.sep+outname+"psf.fits", overwrite=True)

def test_DFT_rect_fov_sampling(fov_npix = (500,1000), pixelscale=0.03, display=False):
    """ Test that we can create a rectangular FOV which nonetheless
    is properly sampled in both the X and Y directions as desired.
    In this case specifically we test that we can get the a symmetric
    PSF (same pixel scale in both X and Y) even when the overall FOV
    is rectangular. This tests some of the low level normalizations and
    scaling factors within the matrixDFT code.
    """

    osys = poppy_core.OpticalSystem(oversample=1)
    osys.add_pupil(optics.CircularAperture())
    osys.add_detector(pixelscale=0.02, fov_pixels=fov_npix)
    osys.add_pupil(optics.CircularAperture())
    osys.add_detector(pixelscale=0.02, fov_pixels=fov_npix)


    psf, intermediates = osys.calc_psf(wavelength=1e-6, return_intermediates=True)

    delta = 100

    plane=1

    cut_h = intermediates[plane].intensity[fov_npix[0]//2, fov_npix[1]//2-delta:fov_npix[1]//2+delta]
    cut_v = intermediates[plane].intensity[fov_npix[0]//2-delta:fov_npix[0]//2+delta, fov_npix[1]//2]

    assert(np.all(np.abs(cut_h-cut_v) < 1e-12))




    if display:
        plt.subplot(311)
        poppy.display_psf(psf)

        plt.subplot(312)
        plt.semilogy(cut_h, label='horizontal')
        plt.semilogy(cut_v, label='vertical', color='red', ls='--')
        plt.legend()

        plt.subplot(313)
        plt.plot(cut_h-cut_v, label='difference')

        plt.tight_layout()



def test_inverse( centering='SYMMETRIC', display=False):
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

    if display:
        plt.subplot(141)
        plt.imshow(pupil)

    print("Pupil 1 total:", pupil.sum())

    a = mft1.perform(pupil, u, npix)

    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()
    print("PSF total", psf.sum())

    if display:
        plt.subplot(142)
        plt.imshow(psf, norm=matplotlib.colors.LogNorm(1e-8, 1.0))

        plt.subplot(143)

    pupil2 = mft1.inverse(a, u, npupil)
    pupil2r = (pupil2 * pupil2.conjugate()).real
    if display:
        plt.imshow( pupil2r)

    print("Pupil 2 total:", pupil2r.sum())



    a2 = mft1.perform(pupil2r, u, npix)
    psf2 = (a2*a2.conjugate()).real.copy()
    print("PSF total", psf2.sum())
    if display:
        plt.subplot(144)
        plt.imshow(psf2, norm=matplotlib.colors.LogNorm(1e-8, 1.0))


def run_all_MFS_tests_DFT(outdir=None, outname='DFT1'):
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

    if outdir is not None:
        fits.PrimaryHDU(pupil.astype(np.float32)).writeto(outdir+os.sep+outname+"pupil.fits", overwrite=True)

    npix=512
    a1 = DFT_combined(pupil, u, npix, centering='FFTSTYLE')
    a2 = DFT_combined(pupil, u, npix, centering='SYMMETRIC')
    a3 = DFT_combined(pupil, u, npix, centering='ADJUSTABLE')
    a4 = DFT_fftstyle(pupil, u, npix)
    a5 = DFT_symmetric(pupil, u, npix)

    if outdir is not None:
        fits.writeto(outdir+os.sep+outname+"_a1_fft.fits",(a1*a1.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_a2_sym.fits",(a2*a2.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_a3_adj.fits",(a3*a3.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_a4_fftr.fits",(a4*a4.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_a5_symr.fits",(a5*a5.conjugate()).real, overwrite=True)

    npix=513
    b1 = DFT_combined(pupil, u, npix, centering='FFTSTYLE')
    b2 = DFT_combined(pupil, u, npix, centering='SYMMETRIC')
    b3 = DFT_combined(pupil, u, npix, centering='ADJUSTABLE')
    b4 = DFT_fftstyle(pupil, u, npix)
    b5 = DFT_symmetric(pupil, u, npix)


    if outdir is not None:
        fits.writeto(outdir+os.sep+outname+"_b1_fft.fits",(b1*b1.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_b2_sym.fits",(b2*b2.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_b3_adj.fits",(b3*b3.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_b4_fftr.fits",(b4*b4.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_b5_symr.fits",(b5*b5.conjugate()).real, overwrite=True)


    u2 = (u, u/4)
    npix2=(512, 128)
    c1 = DFT_combined(pupil, u2, npix2, centering='FFTSTYLE')
    c2 = DFT_combined(pupil, u2, npix2, centering='SYMMETRIC')
    c3 = DFT_combined(pupil, u2, npix2, centering='ADJUSTABLE')
    c4 = DFT_fftstyle_rect(pupil, u2, npix2)
    c5 = DFT_adjustable_rect(pupil, u2, npix2)

    if outdir is not None:
        fits.writeto(outdir+os.sep+outname+"_c1_fft.fits",(c1*c1.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_c2_sym.fits",(c2*c2.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_c3_adj.fits",(c3*c3.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_c4_fftr.fits",(c4*c4.conjugate()).real, overwrite=True)
        fits.writeto(outdir+os.sep+outname+"_c5_adjr.fits",(c5*c5.conjugate()).real, overwrite=True)


    for c, label in zip([c1, c2, c3, c4,c5], ['comb-fft', 'comb-sym', 'comb-adj', 'fft_rect', 'adj_rect']) :
        print(label, c.shape)

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
    assert _exception_message_starts_with(excinfo, "'centering' must be one of [ADJUSTABLE, SYMMETRIC, FFTSTYLE]")

def test_parity_MFT_forward_inverse(display = False):
    """ Test that transforming from a pupil, to an image, and back to the pupil
    leaves you with the same pupil as you had in the first place.

    In other words it doesn't flip left/right or up/down etc.

    See https://github.com/mperrin/webbpsf/issues/35

    **  See also: test_fft.test_parity_FFT_forward_inverse() for a  **
    **  parallel function to this.                                  **


    """

    # set up optical system with 2 pupil planes and 2 image planes

    # use the same exact image plane sampling as in the FFT case
    # This is a bit slower but ensures quantitative agreement.

    pixscale = 0.03437746770784939
    npix=2048
    sys = poppy_core.OpticalSystem()
    sys.add_pupil(optics.ParityTestAperture())
    sys.add_detector(pixelscale=pixscale, fov_pixels=npix)
    sys.add_pupil()
    sys.add_detector(pixelscale=pixscale, fov_pixels=npix)

    psf, planes = sys.calc_psf(display=display, return_intermediates=True)

    # the wavefronts are padded by 0s. With the current API the most convenient
    # way to ensure we get unpadded versions is via the as_fits function.
    p0 = planes[0].as_fits(what='intensity', includepadding=False)
    p1 = planes[1].as_fits(what='intensity', includepadding=False)
    p2 = planes[2].as_fits(what='intensity', includepadding=False)

    # for checking the overall parity it's sufficient to check the intensity.
    # we can have arbitrarily large differences in phase for regions with
    # intensity =0, so don't check the complex field or phase here.


    absdiff = (np.abs(p0[0].data - p2[0].data))
    maxabsdiff = np.max(absdiff)
    # TODO this test could be more stringent if we used a better aperture
    # which is band-limited in the FFT so you don't get all the
    # Gibbs effect ringing after these two FFTs.
    assert (maxabsdiff < 1e-6)

    if display:
        nplanes = len(planes)
        for i, plane in enumerate(planes):
            ax = plt.subplot(2,nplanes,i+1)
            plane.display(ax = ax)
            plt.title("Plane {0}".format(i))


        plt.subplot(2,nplanes,nplanes+1)
        plt.imshow(absdiff)
        plt.title("Abs(Pupil0-Pupil2)")
        plt.colorbar()
        print(maxabsdiff)

def test_MFT_FFT_equivalence(display=False, displaycrop=None):
    """ Test that the MFT transform is numerically equivalent to the
    FFT, if calculated on the correct sampling. """

    centering='FFTSTYLE' # needed if you want near-exact agreement!

    imgin = optics.ParityTestAperture().sample(wavelength=1e-6, npix=256)

    npix = imgin.shape
    nlamD = np.asarray(imgin.shape)
    mft = matrixDFT.MatrixFourierTransform(centering=centering)
    mftout = mft.perform(imgin, nlamD, npix)

    # SIGN CONVENTION: with our adopted sign conventions, forward propagation requires an inverse fft
    # This differs from behavior in versions of poppy prior to 1.0.
    # Further, note that the numpy normalization convention includes 1/n for the inverse transform and 1 for
    # the forward transform, while we want to more symmetrically apply 1/sqrt(n) in both directions.
    fftout = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(imgin))) * np.sqrt(imgin.shape[0] * imgin.shape[1])

    norm_factor = abs(mftout).sum()

    absdiff = abs(mftout-fftout) / norm_factor

    assert(np.all(absdiff < 1e-10))

    if display:
        plt.figure(figsize=(18,3))

        plt.subplot(141)
        plt.imshow(np.abs(imgin))
        plt.colorbar()
        if displaycrop is not None:
            plt.xlim(*displaycrop)
            plt.ylim(*displaycrop)
        print('img input sum =', np.sum(imgin))

        plt.subplot(142)
        plt.imshow(np.abs(mftout))
        plt.colorbar()
        if displaycrop is not None:
            plt.xlim(*displaycrop)
            plt.ylim(*displaycrop)
        print('mft output sum =', np.sum(np.abs(mftout)))

        plt.subplot(143)
        plt.imshow(np.abs(fftout))
        plt.colorbar()
        if displaycrop is not None:
            plt.xlim(*displaycrop)
            plt.ylim(*displaycrop)
        print('fft output sum =', np.sum(np.abs(fftout)))

        plt.subplot(144)
        plt.imshow(np.abs(mftout - fftout))
        plt.colorbar()
        if displaycrop is not None:
            plt.xlim(*displaycrop)
            plt.ylim(*displaycrop)
        print('(mft - fft) output sum =', np.sum(np.abs(mftout - fftout)))

        return mftout, fftout


def test_MFT_FFT_equivalence_in_OpticalSystem(tmpdir, display=False, source_offset=1):
    """ Test that propagating Wavefronts through an OpticalSystem
    using an MFT and an FFT give equivalent results.

    This is a somewhat higher level test that involves all the
    Wavefront class's _propagateTo() machinery, which is not
    tested in the above function. Hence the two closely related tests.

    This test now includes a source offset, to test equivalnce of handling for
    nonzero WFE, in this case for tilts.
    """


    # Note that the Detector class and Wavefront propagation always uses
    # ADJUSTABLE-style MFTs (output centered in the array)
    # which is not compatible with FFT outputs for even-sized arrays.
    # Thus in order to get an exact equivalence, we have to set up our
    # OpticalSystem so that it, very unusually, uses an odd size for
    # its input wavefront. The easiest way to do this is to discretize
    # an AnalyticOpticalElement onto a specific grid.

    fn = str(tmpdir / "test.fits")
    fits511 = optics.ParityTestAperture().to_fits(fn, wavelength=1e-6, npix=511)
    pup511 = poppy_core.FITSOpticalElement(transmission=fits511)


    # set up simple optical system that will just FFT
    fftsys = poppy_core.OpticalSystem(oversample=1)
    fftsys.add_pupil(pup511)
    fftsys.add_image()
    fftsys.source_offset_r = source_offset
    fftsys.source_offset_theta = 90

    fftpsf, fftplanes = fftsys.calc_psf(display=False, return_intermediates=True)

    # set up equivalent using an MFT, tuned to get the exact same scale
    # for the image plane
    mftsys = poppy_core.OpticalSystem(oversample=1)
    mftsys.add_pupil(pup511)
    mftsys.add_detector(pixelscale=fftplanes[1].pixelscale , fov_pixels=fftplanes[1].shape, oversample=1) #, offset=(pixscale/2, pixscale/2))
    mftsys.source_offset_r = source_offset
    mftsys.source_offset_theta = 90

    mftpsf, mftplanes = mftsys.calc_psf(display=False, return_intermediates=True)


    if display:
        import poppy
        plt.figure(figsize=(15,4))
        plt.subplot(131)
        poppy.display_psf(fftpsf, title="FFT PSF")
        plt.subplot(132)
        poppy.display_psf(mftpsf, title='MFT PSF')
        plt.subplot(133)
        poppy.display_psf_difference(fftpsf, mftpsf, title='Diff FFT-MFT')



    assert( np.all(  np.abs(mftpsf[0].data-fftpsf[0].data) < 1e-10 ))
