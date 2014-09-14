#
# Poppy utility functions
#
# These provide various utilities to measure the PSF's properties in certain ways, display it on screen etc. 
#

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.ndimage
import matplotlib
import logging
_log = logging.getLogger('poppy')
import astropy.io.fits as fits


_Strehl_perfect_cache = {} # dict for caching perfect images used in Strehl calcs.


__all__ = [ 'display_PSF', 'display_PSF_difference', 'display_EE', 'display_profiles', 'radial_profile',
    'measure_EE', 'measure_radial', 'measure_fwhm', 'measure_sharpness', 'measure_centroid', 'measure_strehl', 'measure_anisotropy',
    'specFromSpectralType']


###########################################################################
#
#    Display functions 
#


def imshow_with_mouseover(image, ax=None,  *args, **kwargs):
    """ wrapper for matplotlib imshow that displays the value under the cursor position
    
    Wrapper for pyplot.imshow that sets up a custom mouseover display formatter
    so that mouse motions over the image are labeled in the status bar with
    pixel numerical value as well as X and Y coords.

    Why this behavior isn't the matplotlib default, I have no idea...
    """
    if ax is None: ax = plt.gca()
    myax = ax.imshow(image, *args, **kwargs)
    aximage = ax.images[0].properties()['array']
    # need to account for half pixel offset of array coordinates for mouseover relative to pixel center,
    # so that the whole pixel from e.g. ( 1.5, 1.5) to (2.5, 2.5) is labeled with the coordinates of pixel (2,2)


    # We use the extent and implementation to map back from the data coord to pixel coord
    # There is probably an easier way to do this...
    imext = ax.images[0].get_extent()  # returns [-X, X, -Y, Y]
    imsize = ax.images[0].get_size()   # returns [sY, sX]g
    # map data coords back to pixel coords:
    #pixx = (x - imext[0])/(imext[1]-imext[0])*imsize[1]
    #pixy = (y - imext[2])/(imext[3]-imext[2])*imsize[0]
    # and be sure to clip appropriatedly to avoid array bounds errors
    report_pixel = lambda x, y : "(%6.3f, %6.3f)     %-12.6g" % \
        (x,y,   aximage[np.floor( (y - imext[2])/(imext[3]-imext[2])*imsize[0]  ).clip(0,imsize[0]-1),\
                        np.floor( (x - imext[0])/(imext[1]-imext[0])*imsize[1]  ).clip(0,imsize[1]-1)])

        #(x,y, aximage[np.floor(y+0.5),np.floor(x+0.5)])   # this works for regular pixels w/out an explicit extent= call
    ax.format_coord = report_pixel

    return ax


def display_PSF(HDUlist_or_filename=None, ext=0,
    vmin=1e-8,vmax=1e-1, scale='log', cmap = matplotlib.cm.jet, 
        title=None, imagecrop=None, adjust_for_oversampling=False, normalize='None', crosshairs=False, markcentroid=False, colorbar=True, colorbar_orientation='vertical',
        pixelscale='PIXELSCL', ax=None, return_ax=False):
    """Display nicely a PSF from a given HDUlist or filename 

    This is extensively configurable. In addition to making an attractive display, for
    interactive usage this function provides a live display of the pixel value at a
    given (x,y) as you mouse around the image. 
    
    Parameters
    ----------
    HDUlist_or_filename : fits.HDUlist or string
        FITS file containing image to display.
    ext : int
        FITS extension. default = 0
    vmin, vmax : float
        min and max for image display scaling
    scale : str
        'linear' or 'log', default is log
    cmap : matplotlib.cm.Colormap instance
        Colormap to use. Default is matplotlib.cm.jet
    ax : matplotlib.Axes instance
        Axes to display into.
    title : string, optional
    imagecrop : float
        size of region to display (default is whole image)
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to normalize total flux=1. Default is no normalization.
    adjust_for_oversampling : bool
        rescale to conserve surface brightness for oversampled PSFs? 
        (making this True conserves surface brightness but not total flux)
        default is False, to conserve total flux.
    markcentroid : bool
        Draw a crosshairs at the image centroid location?
        Centroiding is computed with the JWST-standard moving box algorithm.
    colorbar : bool
        Draw a colorbar?
    colorbar_orientation : str
        either 'horizontal' or 'vertical'; default is vertical.
    pixelscale : str or float
        if str, interpreted as the FITS keyword name for the pixel scale in arcsec/pixels.
        if float, used as the pixelscale directly.



    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    if adjust_for_oversampling:

        try:
            scalefactor = HDUlist[ext].header['OVERSAMP']**2
        except:
            _log.error("Could not determine oversampling scale factor; therefore NOT rescaling fluxes.")
            scalefactor=1
        im = HDUlist[ext].data *scalefactor
    else: im = HDUlist[ext].data.copy() # don't change normalization of actual input array, work with a copy!

    if normalize.lower() == 'peak':
        _log.debug("Displaying image normalized to peak = 1")
        im /= im.max()
    elif normalize.lower() =='total':
        _log.debug("Displaying image normalized to PSF total = 1")
        im /= im.sum()

    if scale == 'linear':
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else: 
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    if isinstance(pixelscale, basestring):
        halffov_x = HDUlist[ext].header[pixelscale]*HDUlist[ext].data.shape[1]/2
        halffov_y = HDUlist[ext].header[pixelscale]*HDUlist[ext].data.shape[0]/2
    else:
        try: 
            pixelscale = float(pixelscale)
        except:
            _log.warning("Provided pixelscale is neither float nor str; cannot use it. Using default=1 instead.")
            pixelscale = 1.0
        halffov_x = pixelscale*HDUlist[ext].data.shape[1]/2
        halffov_y = pixelscale*HDUlist[ext].data.shape[0]/2
    unit="arcsec"
    extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]


    ax = imshow_with_mouseover( im   ,extent=extent,cmap=cmap, norm=norm, ax=ax)
    if imagecrop is not None:
        halffov_x = min( (imagecrop/2, halffov_x))
        halffov_y = min( (imagecrop/2, halffov_y))
    ax.set_xbound(-halffov_x, halffov_x)
    ax.set_ybound(-halffov_y, halffov_y)
    if crosshairs: 
        ax.axhline(0,ls=":", color='k')
        ax.axvline(0,ls=":", color='k')


    if title is None:
        try:
            fspec = "%s, %s" % (HDUlist[ext].header['INSTRUME'], HDUlist[ext].header['FILTER'])
        except: 
            fspec= str(HDUlist_or_filename)
        title="PSF sim for "+fspec
    ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(ax.images[0], ax=ax, orientation=colorbar_orientation)
        if scale.lower() =='log':
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), np.log10(vmax/vmin)+1)
            if colorbar_orientation=='horizontal' and vmax==1e-1 and vmin==1e-8: ticks = [1e-8, 1e-6, 1e-4,  1e-2, 1e-1] # looks better
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticks)
        if normalize.lower() == 'peak':
            cb.set_label('Intensity relative to peak pixel')
        else: 
            cb.set_label('Fractional intensity per pixel')

    if markcentroid:
        _log.info("measuring centroid to mark on plot...")
        ceny, cenx = measure_centroid(HDUlist, ext=ext, units='arcsec', relativeto='center', boxsize=20, threshhold=0.1)
        ax.plot(cenx, ceny, 'k+', markersize=15, markeredgewidth=1)
        _log.info("centroid: (%f, %f) " % (cenx, ceny))
        plt.draw()

    if return_ax:
        if colorbar: return (ax, cb)
        else: return ax


def display_PSF_difference(HDUlist_or_filename1=None, HDUlist_or_filename2=None, ext1=0, ext2=0, vmax=1e-4, title=None, imagecrop=None, adjust_for_oversampling=False, crosshairs=False, colorbar=True, colorbar_orientation='vertical', print_=False, ax=None, return_ax=False, vmin=None,
        normalize=False, normalize_to_second=False):
    """Display nicely the difference of two PSFs from given files 

    The two files may be FITS files on disk or FITS HDUList objects in memory. The two must have the same 
    shape and size.
    
    Parameters
    ----------
    HDUlist_or_filename1,2 : fits.HDUlist or string
        FITS files containing image to difference
    ext1, ext2 : int
        FITS extension. default = 0
    vmax : float
        for the  scaling
    title : string, optional
    imagecrop : float
        size of region to display (default is whole image)
    normalize : bool
        Display (difference image)/(mean image) instead of just the difference image.
    adjust_for_oversampling : bool
        rescale to conserve surface brightness for oversampled PSFs? 
        (making this True conserves surface brightness but not total flux)
        default is False, to conserve total flux.
    """
    if isinstance(HDUlist_or_filename1, basestring):
        HDUlist1 = fits.open(HDUlist_or_filename1)
    elif isinstance(HDUlist_or_filename1, fits.HDUList):
        HDUlist1 = HDUlist_or_filename1
    else: raise ValueError("input must be a filename or HDUlist")
    if isinstance(HDUlist_or_filename2, basestring):
        HDUlist2 = fits.open(HDUlist_or_filename2)
    elif isinstance(HDUlist_or_filename2, fits.HDUList):
        HDUlist2 = HDUlist_or_filename2
    else: raise ValueError("input must be a filename or HDUlist")


    if adjust_for_oversampling:
        scalefactor = HDUlist1[ext1].header['OVERSAMP']**2
        im1 = HDUlist1[ext1].data *scalefactor
        scalefactor = HDUlist2[ext2].header['OVERSAMP']**2
        im2 = HDUlist1[ext2].data *scalefactor
    else: 
        im1 = HDUlist1[ext1].data
        im2 = HDUlist2[ext2].data

    diff_im = im1-im2

    if normalize:
        avg_im = (im1+im2)/2
        diff_im /= avg_im
        cbtitle = 'Image difference / average  (per pixel)' #Relative intensity difference per pixel'
    if normalize_to_second:
        diff_im /= im2
        cbtitle = 'Image difference / original (per pixel)' #Relative intensity difference per pixel'
    else:
        cbtitle = 'Intensity difference per pixel'

    if vmin is None:
        vmin = -vmax



    if print_:
        rms_diff = np.sqrt((diff_im**2).mean())
        print("RMS of difference image: {0}".format(rms_diff))

    norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.cm.gray
    halffov_x = HDUlist1[ext1].header['PIXELSCL']*HDUlist1[ext1].data.shape[1]/2
    halffov_y = HDUlist1[ext1].header['PIXELSCL']*HDUlist1[ext1].data.shape[0]/2
    unit="arcsec"
    extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]


    ax = imshow_with_mouseover( diff_im   ,extent=extent,cmap=cmap, norm=norm, ax=ax)
    if imagecrop is not None:
        halffov_x = min( (imagecrop/2, halffov_x))
        halffov_y = min( (imagecrop/2, halffov_y))
    ax.set_xbound(-halffov_x, halffov_x)
    ax.set_ybound(-halffov_y, halffov_y)
    if crosshairs: 
        ax.axhline(0,ls=":", color='k')
        ax.axvline(0,ls=":", color='k')


    if title is None:
        try:
            fspec= str(HDUlist_or_filename1) +"-"+str(HDUlist_or_filename2)
            #fspec = "Difference Image " # "%s, %s" % (HDUlist[ext].header['INSTRUME'], HDUlist[ext].header['FILTER'])
        except: 
            fspec= str(HDUlist_or_filename1) +"-"+str(HDUlist_or_filename2)
        title="Difference of "+fspec
    ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(ax.images[0], ax=ax, orientation=colorbar_orientation)
        #ticks = np.logspace(np.log10(vmin), np.log10(vmax), np.log10(vmax/vmin)+1)
        #if vmin == 1e-8 and vmax==1e-1: 
            #ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        #ticks = [vmin, -0.5*vmax, 0, 0.5*vmax, vmax]
        #cb.set_ticks(ticks)
        #cb.set_ticklabels(ticks)
        cb.set_label(cbtitle)
    if return_ax:
        if colorbar: return (ax, cb)
        else: return ax


def display_EE(HDUlist_or_filename=None,ext=0, overplot=False, ax=None, mark_levels=True ):
    """ Display Encircled Energy curve for a PSF

    The azimuthally averaged encircled energy is plotted as a function of radius.

    Parameters
    ----------
    HDUlist_or_filename : fits.HDUlist or string
        FITS file containing image to display encircled energy for.
    ext : bool
        FITS extension to use. Default is 0
    overplot : bool
        whether to overplot or clear and produce an new plot. Default false
    ax : matplotlib Axes instance
        axis to plot into. If not provided, current axis will be used. 
    mark_levels : bool
        If set, mark and label on the plots the radii for 50%, 80%, 95% encircled energy.
        Default is True
 
    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename,ext=ext)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")


    radius, profile, EE = radial_profile(HDUlist, EE=True)

    if not overplot:
        if ax is None: 
            plt.clf()
            ax = plt.subplot(111)

    ax.plot(radius, EE) #, nonposy='clip')
    if not overplot:
        ax.set_xlabel("Radius [arcsec]")
        ax.set_ylabel("Encircled Energy")

    if mark_levels:
        for level in [0.5, 0.8, 0.95]:
            EElev = radius[np.where(EE > level)[0][0]]
            yoffset = 0 if level < 0.9 else -0.05 
            plt.text(EElev+0.1, level+yoffset, 'EE=%2d%% at r=%.3f"' % (level*100, EElev))


def display_profiles(HDUlist_or_filename=None,ext=0, overplot=False, title=None, **kwargs):
    """ Produce two plots of PSF radial profile and encircled energy

    See also the display_EE function.

    Parameters
    ----------
    HDUlist_or_filename1,2 : fits.HDUlist or string
        FITS files containing image to difference
    ext : bool
        FITS extension to use. Default is 0
    overplot : bool
        whether to overplot or clear and produce an new plot. Default false
    title : string, optional
        Title for plot
 
    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename,ext=ext)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")


    radius, profile, EE = radial_profile(HDUlist, EE=True, **kwargs)

    if title is None:
        try:
            title= "%s, %s" % (HDUlist[ext].header['INSTRUME'], HDUlist[ext].header['FILTER'])
        except: 
            title= str(HDUlist_or_filename)

    if not overplot:
        plt.clf()
        plt.title(title)
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("PSF radial profile")
    plt.subplot(2,1,1)
    plt.semilogy(radius, profile)

    fwhm = 2*radius[np.where(profile < profile[0]*0.5)[0][0]]
    plt.text(fwhm, profile[0]*0.5, 'FWHM = %.3f"' % fwhm)

    plt.subplot(2,1,2)
    #plt.semilogy(radius, EE, nonposy='clip')
    plt.plot(radius, EE, color='r') #, nonposy='clip')
    if not overplot:
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("Encircled Energy")

    for level in [0.5, 0.8, 0.95]:
        if (EE>level).any():
            EElev = radius[np.where(EE > level)[0][0]]
            yoffset = 0 if level < 0.9 else -0.05 
            plt.text(EElev+0.1, level+yoffset, 'EE=%2d%% at r=%.3f"' % (level*100, EElev))


def radial_profile(HDUlist_or_filename=None, ext=0, EE=False, center=None, stddev=False, binsize=None, maxradius=None):
    """ Compute a radial profile of the image. 

    This computes a discrete radial profile evaluated on the provided binsize. For a version
    interpolated onto a continuous curve, see measure_radial().

    Code taken pretty much directly from pydatatut.pdf

    Parameters
    ----------
    HDUlist_or_filename : string
        what it sounds like.
    ext : int
        Extension in FITS file
    EE : bool
        Also return encircled energy (EE) curve in addition to radial profile?
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center. 
    binsize : float
        size of step for profile. Default is pixel size.
    stddev : bool
        Compute standard deviation in each radial bin, not average?


    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data
    pixelscale = HDUlist[ext].header['PIXELSCL']


    if maxradius is not None:
        raise NotImplemented("add max radius")


    if binsize is None:
        binsize=pixelscale

    y,x = np.indices(image.shape)
    if center is None:
        # get exact center of image
        #center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple( (a-1)/2.0 for a in image.shape[::-1])

    r = np.sqrt( (x-center[0])**2 + (y-center[1])**2) *pixelscale / binsize # radius in bin size steps
    ind = np.argsort(r.flat)


    sr = r.flat[ind]
    sim = image.flat[ind]
    ri = sr.astype(int)
    deltar = ri[1:]-ri[:-1] # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=float) # cumulative sum to figure out sums for each bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile=tbin/nr

    #pre-pend the initial element that the above code misses.
    radialprofile2 = np.empty(len(radialprofile)+1)
    if rind[0] != 0:
        radialprofile2[0] =  csim[rind[0]] / (rind[0]+1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]                       # otherwise if there's just one then just take it. 
    radialprofile2[1:] = radialprofile
    rr = np.arange(len(radialprofile2))*binsize + binsize*0.5  # these should be centered in the bins, so add a half.

    if stddev:
        stddevs = np.zeros_like(radialprofile2)
        r_pix = r * binsize
        for i, radius in enumerate(rr):
            if i == 0: wg = np.where(r < radius+ binsize/2)
            else: 
                wg = np.where( (r_pix >= (radius-binsize/2)) &  (r_pix < (radius+binsize/2)))
                #wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
            stddevs[i] = image[wg].std()
        return (rr, stddevs)

    if not EE:
        return (rr, radialprofile2)
    else:
        #weighted_profile = radialprofile2*2*np.pi*(rr/rr[1])
        #EE = np.cumsum(weighted_profile)
        EE = csim[rind]
        return (rr, radialprofile2, EE) 


###########################################################################
#
#    PSF evaluation functions 
#

def measure_EE(HDUlist_or_filename=None, ext=0, center=None, binsize=None):
    """ measure encircled energy vs radius and return as an interpolator
    
    Returns a function object which when called returns the Encircled Energy inside a given radius, 
    for any arbitrary desired radius smaller than the image size.



    Parameters
    ----------
    HDUlist_or_filename : string
        Either a fits.HDUList object or a filename of a FITS file on disk
    ext : int
        Extension in that FITS file
    center : tuple of floats
        Coordinates (x,y) of PSF center. Default is image center. 
    binsize: 
        size of step for profile. Default is pixel size.

    Returns
    --------
    encircled_energy: function
        A function which will return the encircled energy interpolated to any desired radius.


    Examples
    --------
    >>> EE = measure_EE("someimage.fits")
    >>> print "The EE at 0.5 arcsec is ", EE(0.5)

    """

    rr, radialprofile2, EE = radial_profile(HDUlist_or_filename, ext, EE=True, center=center, binsize=binsize)

    # append the zero at the center
    rr_EE = rr + (rr[1]-rr[0])/1  # add half a binsize to this, because the EE is measured inside the
                                  # outer edge of each annulus. 
    rr0 = np.concatenate( ([0], rr_EE)) 
    EE0 = np.concatenate( ([0], EE))


    EE_fn = scipy.interpolate.interp1d(rr0, EE0,kind='cubic', bounds_error=False)

    return EE_fn


def measure_radial(HDUlist_or_filename=None, ext=0, center=None, binsize=None):
    """ measure azimuthally averaged radial profile of a PSF.
    
    Returns a function object which when called returns the mean value at a given radius.

    Parameters
    ----------
    HDUlist_or_filename : string
        what it sounds like.
    ext : int
        Extension in FITS file
    center : tuple of floats
        Coordinates (x,y) of PSF center. Default is image center. 
    binsize: 
        size of step for profile. Default is pixel size.

    Returns
    --------
    radial_profile: function
        A function which will return the mean PSF value at any desired radius.


    Examples
    --------
    >>> rp = measure_radial("someimage.fits")
    >>> radius = np.linspace(0, 5.0, 100)
    >>> plot(radius, rp(radius), label="PSF")

    """

    rr, radialprofile, EE = radial_profile(HDUlist_or_filename, ext, EE=True, center=center, binsize=binsize)

    radial_fn = scipy.interpolate.interp1d(rr, radialprofile,kind='cubic', bounds_error=False)

    return radial_fn


def measure_fwhm(HDUlist_or_filename=None, ext=0, center=None, level=0.5):
    """ Measure FWHM by interpolation of the radial profile 

    This measures the full width at half maximum for the supplied PSF, 
    or optionally the full width at some other fraction of max.

    Parameters
    ----------
    HDUlist_or_filename, ext : string, int
        Same as above
    center : tuple
        center to compute around.  Default is image center.
    level : float
        Fraction of max to compute for; default is 0.5 for Half Max. 
        You can also measure widths at other levels e.g. FW at 10% max
        by setting level=0.1


    """

    rr, radialprofile, EE = radial_profile(HDUlist_or_filename, ext, EE=True, center=center)
    rpmax = radialprofile.max()

    wlower = np.where(radialprofile < rpmax *level)
    wmin = np.min(wlower[0])
    # go just a bit beyond the half way mark
    winterp = np.arange(0, wmin+2, dtype=int)[::-1]

    if len(winterp) < 6: kind='linear'
    else: kind = 'cubic'

    interp_hw = scipy.interpolate.interp1d(radialprofile[winterp], rr[winterp], kind=kind)
    return 2*interp_hw(rpmax*level)
 

def measure_sharpness(HDUlist_or_filename=None, ext=0):
    """ Compute image sharpness, the sum of pixel squares.

    See Makidon et al. JWST-STScI-001157 for a discussion of this image metric
    and its relationship to noise equivalent pixels.

    Parameters
    ----------
    HDUlist_or_filename, ext : string, int
        Same as above
 
    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")


    # TODO or check that the oversampling factor is 1
    try:
        detpixels = HDUlist['DET_SAMP']
    except:
        raise ValueError("You can only measure sharpness for an image with an extension giving the rebinned actual detector pixel values.""")

    sharpness =  (detpixels.data**2).sum()
    return sharpness


def measure_centroid(HDUlist_or_filename=None, ext=0, slice=0, boxsize=20, verbose=False, units='pixels', relativeto='origin', **kwargs):
    """ Measure the center of an image via center-of-mass

    The centroid method used is the floating-box center of mass algorithm by
    Jeff Valenti et al., which has been adopted for JWST target acquisition 
    measurements on orbit.
    See JWST technical reports JWST-STScI-001117 and JWST-STScI-001134 for details.

    Parameters
    ----------
    HDUlist_or_filename : string
        Either a fits.HDUList object or a filename of a FITS file on disk
    ext : int
        Extension in that FITS file
    slice : int, optional
        If that extension is a 3D datacube, which slice (plane) of that datacube to use
    boxsize : int
        Half box size for centroid
    relativeto : string
        either 'origin' for relative to pixel (0,0) or 'center' for relative to image center. Default is 'origin'
    units : string
        either 'pixels' for position in pixels or 'arcsec' for arcseconds. 
        Relative to the relativeto parameter point in either case.
 

    Returns
    -------
    CoM : array_like
        [Y, X] coordinates of center of mass.

    """
    from .fwcentroid import fwcentroid

    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data
    
    if image.ndim >=3:  # handle datacubes gracefully
        image = image[slice,:,:]


    cent_of_mass = fwcentroid(image, halfwidth=boxsize, **kwargs)

    if verbose: print("Center of mass: (%.4f, %.4f)" % (cent_of_mass[1], cent_of_mass[0]))

    if relativeto == 'center':
        imcen = np.array([ (image.shape[0]-1)/2., (image.shape[1]-1)/2. ])
        cent_of_mass  = tuple( np.array(cent_of_mass) -  imcen)


    if units == 'arcsec':
        pixelscale = HDUlist[ext].header['PIXELSCL']
        cent_of_mass = tuple( np.array(cent_of_mass) *pixelscale)

    return cent_of_mass


def measure_strehl(HDUlist_or_filename=None, ext=0, slice=0, center=None, display=True, verbose=True, cache_perfect=False):
    """ Estimate the Strehl ratio for a PSF.
    
    This requires computing a simulated PSF with the same
    properties as the one under analysis.

    Note that this calculation will not be very accurate unless both PSFs are well sampled,
    preferably several times better than Nyquist. See 
    `Roberts et al. 2004 SPIE 5490 <http://adsabs.harvard.edu/abs/2004SPIE.5490..504R>`_
    for a discussion of the various possible pitfalls when calculating Strehl ratios. 

    Parameters
    ----------
    HDUlist_or_filename : string
        Either a fits.HDUList object or a filename of a FITS file on disk
    ext : int
        Extension in that FITS file
    slice : int, optional
        If that extension is a 3D datacube, which slice (plane) of that datacube to use
    center : tuple
        center to compute around.  Default is image center. If the center is on the
        crosshairs between four pixels, then the mean of those four pixels is used.
        Otherwise, if the center is in a single pixel, then that pixel is used. 
    print_, display : bool
        control whether to print the results or display plots on screen. 

    cache_perfect : bool
        use caching for perfect images? greatly speeds up multiple calcs w/ same config

    Returns
    ---------
    strehl : float
        Strehl ratio as a floating point number between 0.0 - 1.0
  
    """
    if isinstance(HDUlist_or_filename, basestring):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else: raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data
    header = HDUlist[ext].header

    if image.ndim >=3:  # handle datacubes gracefully
        image = image[slice,:,:]

 
    if center is None:
        # get exact center of image
        #center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple( (a-1)/2.0 for a in image.shape[::-1])



    # Compute a comparison image
    _log.info("Now computing image with zero OPD for comparison...")
    inst = Instrument(header['INSTRUME'])
    inst.filter = header['FILTER']
    inst.pupilopd = None # perfect image
    inst.pixelscale = header['PIXELSCL'] * header['OVERSAMP'] # same pixel scale pre-oversampling
    cache_key = (header['INSTRUME'], header['FILTER'], header['PIXELSCL'], header['OVERSAMP'],  header['FOV'],header['NWAVES'])
    try:
        comparison_psf = _Strehl_perfect_cache[cache_key]
    except:
        comparison_psf = inst.calcPSF(fov_arcsec = header['FOV'], oversample=header['OVERSAMP'], nlambda=header['NWAVES'])
        if cache_perfect: _Strehl_perfect_cache[cache_key ] = comparison_psf

    comparison_image = comparison_psf[0].data

    if (int(center[1]) == center[1]) and (int(center[0]) == center[0]):
        # individual pixel
        meas_peak =           image[center[1], center[0]]
        ref_peak = comparison_image[center[1], center[0]]
    else:
        # average across a group of 4
        bot = [np.floor(f) for f in center]
        top = [np.ceil(f)+1 for f in center]
        meas_peak =           image[bot[1]:top[1], bot[0]:top[0]].mean()
        ref_peak = comparison_image[bot[1]:top[1], bot[0]:top[0]].mean()
    strehl = (meas_peak/ref_peak)

    if display:
        plt.clf()
        plt.subplot(121)
        display_PSF(HDUlist, title="Observed PSF")
        plt.subplot(122)
        display_PSF(comparison_psf, title="Perfect PSF")
        plt.gcf().suptitle("Strehl ratio = %.3f" % strehl) 


    if verbose:
        print("Measured peak:  {0:.3g}".format(meas_peak))
        print("Reference peak: {0:.3g}".format(ref_peak))
        print("  Strehl ratio: {0:.3f}".format(strehl))

    return strehl


def measure_anisotropy(HDUlist_or_filename=None, ext=0, slice=0, boxsize=50):
    raise NotImplementedError("measure_anisotropy is not yet implemented.")

###########################################################################
#
#    Array manipulation utility functions 
#


def padToOversample(array, oversample):
    """ Add zeros around the edge of an array, for a given desired FFT integer oversampling ratio

    Parameters
    ----------
    array :  ndarray
        A 2D array representing some image
    oversample : int
        Padding factor for expanding the array

    Returns
    -------
    padded_array : ndarray
        A larger array containing mostly zeros but with the input array in the center.

    See Also
    ---------
    padToSize
    """
    npix = array.shape[0]
    padded = np.zeros(shape=(npix*oversample, npix*oversample), dtype=array.dtype)
    n0 = float(npix)*(oversample - 1)/2
    n1 = n0+npix
    n0 = int(round(n0)) # because astropy test_plugins enforces integer indices
    n1 = int(round(n1))
    padded[n0:n1, n0:n1] = array
    return padded

def padToSize(array, padded_shape):
    """ Add zeros around the edge of an array, to reach a specific defined size and shape.
    This is similar to padToOversample but is more flexible.

    Parameters
    ----------
    array :  ndarray
        A 2D array representing some image
    padded_shape :  tuple of 2 elements
        Desired size for the padded array.

    Returns
    -------
    padded_array : ndarray
        A larger array containing mostly zeros but with the input array in the center.


    See Also
    ---------
    padToOversample 
    """

    if len(padded_shape) < 2: 
        outsize0 = padded_shape
        outside1 = padded_shape
    else:
        outsize0 = padded_shape[0]
        outsize1 = padded_shape[1]
    #npix = array.shape[0]
    padded = np.zeros(shape=padded_shape, dtype=array.dtype)
    n0 = (outsize0 - array.shape[0])/2  # pixel offset for the inner array
    m0 = (outsize1 - array.shape[1])/2  # pixel offset in second dimension
    n1 = n0+array.shape[0]
    m1 = m0+array.shape[1]
    n0 = int(round(n0)) # because astropy test_plugins enforces integer indices
    n1 = int(round(n1))
    m0 = int(round(m0))
    m1 = int(round(m1))
    padded[n0:n1, m0:m1] = array
    return padded

def removePadding(array,oversample):
    " Remove zeros around the edge of an array, assuming some integer oversampling padding factor "
    npix = array.shape[0] / oversample
    n0 = float(npix)*(oversample - 1)/2
    n1 = n0+npix
    n0 = int(round(n0))
    n1 = int(round(n1))
    return array[n0:n1,n0:n1].copy()


def rebin_array(a = None, rc=(2,2), verbose=False):
    """ Rebin array by an integer factor while conserving flux

    Perform simple-minded flux-conserving binning... clip trailing
    size mismatch: eg a 10x3 array binned by 3 results in a 3x1 array

    Parameters
    ----------
    a : array_like
        input array
    rc : two-element tuple 
        (nrows, ncolumns) desired for rebinned array
    verbose : bool
        output additional status text?


    anand@stsci.edu

    """

    r, c = rc

    R = a.shape[0]
    C = a.shape[1]

    nr = int(R / r)
    nc = int(C / c)

    b = a[0:nr, 0:nc].copy()
    b = b * 0

    for ri in range(0, nr):
        Rlo = ri * r
        if verbose:
            print("row loop")
        for ci in range(0, nc):
            Clo = ci * c
            b[ri, ci] = np.add.reduce(a[Rlo:Rlo+r, Clo:Clo+c].copy().flat)
            if verbose:
                print("    [%d:%d, %d:%d]" % (Rlo,Rlo+r, Clo,Clo+c))
                print("%4.0f"  %   np.add.reduce(a[Rlo:Rlo+r, Clo:Clo+c].copy().flat))
    return b


def krebin(a, shape):
    """ Fast Rebinning with flux conservation

    New shape must be an integer divisor of the current shape.

    This algorithm is much faster than rebin_array

    Parameters
    ----------
    a : array_like
        input array
    shape : two-element tuple 
        (nrows, ncolumns) desired for rebinned array


    """
    # Klaus P's fastrebin from web
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

###########################################################################
#
#    Other utility functions 
#



def specFromSpectralType(sptype, return_list=False, catalog=None):
    """Get Pysynphot Spectrum object from a user-friendly spectral type string.

    Given a spectral type such as 'A0IV' or 'G2V', this uses a fixed lookup table
    to determine an appropriate spectral model from Castelli & Kurucz 2004 or 
    the Phoenix model grids. Depends on pysynphot and CDBS. This is just a
    convenient access function.

    Parameters
    -----------
    catalog: str
        'ck04' for Castelli & Kurucz 2004, 'phoenix' for Phoenix models.
        If not set explicitly, the code will check if the phoenix models are
        present inside the $PYSYN_CDBS directory. If so, those are the default;
        otherwise, it's CK04.

    """
    import pysynphot


    if catalog is None: 
        import os
        cdbs = os.getenv('PYSYN_CDBS')
        if os.path.exists( os.path.join(os.getenv('PYSYN_CDBS'), 'grid', 'phoenix')):
            catalog='phoenix'
        elif os.path.exists( os.path.join(os.getenv('PYSYN_CDBS'), 'grid', 'ck04models')):
            catalog='ck04'
        else:
            raise IOError("Could not find either phoenix or ck04models subdirectories of $PYSYN_CDBS/grid")

    if catalog.lower()  =='ck04':
        catname='ck04models'

        # Recommended lookup table into the CK04 models (from 
        # the documentation of that catalog?)
        lookuptable = {
            "O3V":   (50000, 0.0, 5.0),
            "O5V":   (45000, 0.0, 5.0),
            "O6V":   (40000, 0.0, 4.5),
            "O8V":   (35000, 0.0, 4.0),
            "O5I":   (40000, 0.0, 4.5),
            "O6I":   (40000, 0.0, 4.5),
            "O8I":   (34000, 0.0, 4.0),
            "B0V":   (30000, 0.0, 4.0),
            "B3V":   (19000, 0.0, 4.0),
            "B5V":   (15000, 0.0, 4.0),
            "B8V":   (12000, 0.0, 4.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "B0I":   (26000, 0.0, 3.0),
            "B5I":   (14000, 0.0, 2.5),
            "A0V":   (9500, 0.0, 4.0),
            "A5V":   (8250, 0.0, 4.5),
            "A0I":   (9750, 0.0, 2.0),
            "A5I":   (8500, 0.0, 2.0),
            "F0V":   (7250, 0.0, 4.5),
            "F5V":   (6500, 0.0, 4.5),
            "F0I":   (7750, 0.0, 2.0),
            "F5I":   (7000, 0.0, 1.5),
            "G0V":   (6000, 0.0, 4.5),
            "G5V":   (5750, 0.0, 4.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "G0I":   (5500, 0.0, 1.5),
            "G5I":   (4750, 0.0, 1.0),
            "K0V":   (5250, 0.0, 4.5),
            "K5V":   (4250, 0.0, 4.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "K0I":   (4500, 0.0, 1.0),
            "K5I":   (3750, 0.0, 0.5),
            "M0V":   (3750, 0.0, 4.5),
            "M2V":   (3500, 0.0, 4.5),
            "M5V":   (3500, 0.0, 5.0),
            "M0III": (3750, 0.0, 1.5),
            "M0I":   (3750, 0.0, 0.0),
            "M2I":   (3500, 0.0, 0.0)}
    elif catalog.lower() =='phoenix':
        catname='phoenix'
        # lookup table used in JWST ETCs
        lookuptable = {
            "O3V":   (45000, 0.0, 4.0),
            "O5V":   (41000, 0.0, 4.5),
            "O7V":   (37000, 0.0, 4.0),
            "O9V":   (33000, 0.0, 4.0),
            "B0V":   (30000, 0.0, 4.0),
            "B1V":   (25000, 0.0, 4.0),
            "B3V":   (19000, 0.0, 4.0),
            "B5V":   (15000, 0.0, 4.0),
            "B8V":   (12000, 0.0, 4.0),
            "A0V":   (9500, 0.0, 4.0),
            "A1V":   (9250, 0.0, 4.0),
            "A3V":   (8250, 0.0, 4.0),
            "A5V":   (8250, 0.0, 4.0),
            "F0V":   (7250, 0.0, 4.0),
            "F2V":   (7000, 0.0, 4.0),
            "F5V":   (6500, 0.0, 4.0),
            "F8V":   (6250, 0.0, 4.5),
            "G0V":   (6000, 0.0, 4.5),
            "G2V":   (5750, 0.0, 4.5),
            "G5V":   (5750, 0.0, 4.5),
            "G8V":   (5500, 0.0, 4.5),
            "K0V":   (5250, 0.0, 4.5),
            "K2V":   (4750, 0.0, 4.5),
            "K5V":   (4250, 0.0, 4.5),
            "K7V":   (4000, 0.0, 4.5),
            "M0V":   (3750, 0.0, 4.5),
            "M2V":   (3500, 0.0, 4.5),
            "M5V":   (3500, 0.0, 5.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "M0III": (3750, 0.0, 1.5),
            "O6I":   (39000, 0.0, 4.5),
            "O8I":   (34000, 0.0, 4.0),
            "B0I":   (26000, 0.0, 3.0),
            "B5I":   (14000, 0.0, 2.5),
            "A0I":   (9750, 0.0, 2.0),
            "A5I":   (8500, 0.0, 2.0),
            "F0I":   (7750, 0.0, 2.0),
            "F5I":   (7000, 0.0, 1.5),
            "G0I":   (5500, 0.0, 1.5),
            "G5I":   (4750, 0.0, 1.0),
            "K0I":   (4500, 0.0, 1.0),
            "K5I":   (3750, 0.0, 0.5),
            "M0I":   (3750, 0.0, 0.0),
            "M2I":   (3500, 0.0, 0.0)}

    if return_list:
        sptype_list = lookuptable.keys()
        def sort_sptype(typestr):
            letter = typestr[0]
            lettervals = {'O':0, 'B': 10, 'A': 20,'F': 30, 'G':40, 'K': 50, 'M':60}
            value = lettervals[letter]*1.0
            value += int(typestr[1])
            if "III" in typestr: value += .3
            elif "I" in typestr: value += .1
            elif "V" in typestr: value += .5
            return value
        sptype_list.sort(key=sort_sptype)
        sptype_list.insert(0,"Flat spectrum in F_nu")
        sptype_list.insert(0,"Flat spectrum in F_lambda")
        # add a variety of spectral type slopes, per request from Dean Hines
        for slope in [-3, -2, -1.5, -1, -0.75, -0.5, 0.5, 0.75, 1.0, 1.5, 2, 3]:
            sptype_list.insert(0,"Power law F_nu ~ nu^(%s)" % str(slope))
        #sptype_list.insert(0,"Power law F_nu ~ nu^(-0.75)")
        #sptype_list.insert(0,"Power law F_nu ~ nu^(-1.0)")
        #sptype_list.insert(0,"Power law F_nu ~ nu^(-1.5)")
        #sptype_list.insert(0,"Power law F_nu ~ nu^(-2.0)")
        return sptype_list


    if "Flat" in sptype:
        if sptype == "Flat spectrum in F_nu":    spec = pysynphot.FlatSpectrum( 1, fluxunits = 'fnu')
        elif sptype == "Flat spectrum in F_lambda":  spec= pysynphot.FlatSpectrum( 1, fluxunits = 'flam')
        spec.convert('flam')
        return spec*(1./spec.flux.mean())
    if 'Power law' in sptype:
        import re
        ans = re.search('\((.*)\)', sptype)
        if ans is None: raise ValueError("Invalid power law specification cannot be parsed to get exponent")
        exponent = float(ans.groups(0)[0])
        # note that Pysynphot's PowerLaw class implements a power law in terms of lambda, not nu.
        # but since nu = clight/lambda, it's just a matter of swapping the sign on the exponent. 

        spec = pysynphot.PowerLaw(1, (-1)*exponent, fluxunits='fnu')
        spec.convert('flam')
        spec *= (1./spec.flux.mean())
        spec.name = sptype
        return spec
    else: 
        keys = lookuptable[sptype]
        try:
            return pysynphot.Icat(catname,keys[0], keys[1], keys[2])
        except:
            errmsg = "Could not find a match in catalog {0} for key {1}. Check that is a valid name in the lookup table, and/or that pysynphot is installed properly.".format(catname, sptype)
            _log.critical(errmsg)
            raise LookupError(errmsg)


###################################################################33
#
#     Multiprocessing and FFT helper functions 


def estimate_optimal_nprocesses(osys, nwavelengths=None, padding_factor=None, memory_fraction=0.5):
    """ Attempt to estimate a reasonable number of processes to use for a multi-wavelength calculation.

    This is not entirely obvious because this can be either CPU- or memory-limited, and you don't want
    to just spawn nwavelengths processes necessarily. 
    
    Here we attempt to estimate how many such calculations can happen in
    parallel without swapping to disk, with a mixture of empiricism and conservatism.
    One really does not want to end up swapping to disk with huge arrays.

    NOTE: Requires psutil package. Otherwise defaults to just 4?

    Parameters
    -----------
    osys : OpticalSystem instance
        The optical system that we will be calculating for. 
    nwavelengths : int
        Number of wavelengths. Sets maximum # of processes.
    padding_factor : int
        How many copies of the wavefront array per calculation
    memory_fraction : float
        What fraction of total system physical RAM should webbPSF make use of?
        This is in attempt to make it play nicely with whatever else you're running...
    """

    from . import conf
    try:
        import psutil
    except:
        _log.debug("No psutil package available, cannot estimate optimal nprocesses.")
        return 4

    wfshape = osys.inputWavefront().shape
    # will we do an FFT or not?
    propinfo = osys._propagation_info()
    if 'FFT' in propinfo['steps']:
        wavefrontsize = wfshape[0]*wfshape[1]*osys.oversample**2 *  16 # 16 bytes = complex double size 
        _log.debug('FFT propagation with array={0}, oversample = {1} uses {2} bytes'.format(wfshape[0], osys.oversample, wavefrontsize))
        # The following is a very rough estimate
        # empirical tests show that an 8192x8192 propagation results in Python sessions with ~4 GB memory allocation. using FFTW
        # usingg mumpy FT, the memory usage per process can exceed 5 GB for an 8192x8192 propagation.
        padding_factor = 4  if conf.use_fftw else 5
    else:
        # oversampling not relevant for memory size in MFT mode
        wavefrontsize = wfshape[0]*wfshape[1] *  16 # 16 bytes = complex double size 
        _log.debug('MFT propagation with array={0} uses {2} bytes'.format(wfshape[0], osys.oversample, wavefrontsize))
        padding_factor = 1

    mem_per_prop = wavefrontsize * padding_factor
    mem_per_output = propinfo['output_size']*8

    # total memory needed is the sum of memory for the propagation plus memory to hold the results
    #avail_ram = psutil.phymem_usage().total * memory_fraction
    avail_ram = psutil.virtual_memory().available
    avail_ram -= 2* 1024.**3   # always leave at least 2 GB extra padding - let's be cautious to make sure we don't swap.
    recommendation = int(np.floor(float(avail_ram) / (mem_per_prop+mem_per_output)))
       
    if recommendation > psutil.NUM_CPUS: recommendation = psutil.NUM_CPUS
    if nwavelengths is not None:
        if recommendation > nwavelengths: recommendation = nwavelengths

    _log.info("estimated optimal # of processes is {0}".format(recommendation))
    return recommendation


def fftw_save_wisdom(filename=None):
    """ Save accumulated FFTW wisdom to a file 

    By default this file will be in the user's astropy configuration directory.
    (Another location could be chosen - this is simple and works easily cross-platform.)
    
    Parameters
    ------------
    filename : string, optional
        Filename to use (instead of the default, poppy_fftw_wisdom.txt)
    """
    import os
    import astropy.config
    try:
        import pyfftw
    except:
        return # FFTW is not present, therefore this is a null op

    if filename is None:
        filename=os.path.join( astropy.config.get_config_dir(), "poppy_fftw_wisdom.txt")


    double, single, longdouble = pyfftw.export_wisdom()
    f = open(filename, 'w')
    f.write("# FFTW wisdom information saved by POPPY\n")
    f.write("#   Do not edit this file by hand; that will likely break it.\n")
    f.write("# See http://hgomersall.github.io/pyFFTW/pyfftw/pyfftw.html?#wisdom-functions for more information\n")
    f.write('# the following three lines are the double, single, and longdouble wisdoms\n')
    f.write(double.replace('\n','\\n')+'\n')
    f.write(single.replace('\n','\\n')+'\n')
    f.write(longdouble.replace('\n','\\n')+'\n')
    _log.debug("FFTW wisdom saved to "+filename)


def fftw_load_wisdom(filename=None):
    """ Read accumulated FFTW wisdom previously saved in previously saved in a file 

    By default this file will be in the user's astropy configuration directory.
    (Another location could be chosen - this is simple and works easily cross-platform.)
    
    Parameters
    ------------
    filename : string, optional
        Filename to use (instead of the default, poppy_fftw_wisdom.txt)
    """
 
    import os
    from astropy import config
    try:
        import pyfftw
    except:
        return # FFTW is not present, therefore this is a null op

    if filename is None:
        filename=os.path.join( config.get_config_dir(), "poppy_fftw_wisdom.txt")
 
    if not os.path.exists(filename): return # gracefully ignore the case of lacking wisdom yet.

    _log.debug("Trying to reload wisdom from file "+filename)
    try:
        lines = open(filename,'r').readlines()
        # the first four lines are comments and should be ignored.
        wisdom = [lines[i].replace(r'\n', '\n') for i in [4,5,6]]
        wisdom = tuple(wisdom)
    except:
        _log.debug("ERROR - wisdom tuple could not be loaded from file :"+filename)
        return False

    success = pyfftw.import_wisdom(wisdom)
    _log.debug("Reloaded double precision wisdom: "+str(success[0]))
    _log.debug("Reloaded single precision wisdom: "+str(success[1]))
    _log.debug("Reloaded longdouble precision wisdom: "+str(success[2]))
    
    return True




