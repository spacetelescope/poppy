#
# Poppy utility functions
#
# These provide various utilities to measure the PSF's properties in certain ways, display it on screen etc.
#

import json
import logging
import os.path
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage
import warnings

from astropy import config
import astropy.units as u

import astropy.io.fits as fits

import poppy

try:
    import pyfftw
except ImportError:
    pyfftw = None

_log = logging.getLogger('poppy')

_loaded_fftw_wisdom = False


class FFTWWisdomWarning(RuntimeWarning):
    pass


__all__ = ['display_psf', 'display_psf_difference', 'display_ee', 'measure_ee', 'measure_radius_at_ee',
           'display_profiles', 'radial_profile',
           'measure_radial', 'measure_fwhm', 'measure_sharpness', 'measure_centroid',
           'measure_anisotropy', 'spectrum_from_spectral_type',
           'specFromSpectralType', 'removePadding']


###########################################################################
#
#    Display functions
#


def imshow_with_mouseover(image, ax=None, *args, **kwargs):
    """Wrapper for matplotlib imshow that displays the value under the
    cursor position

    Wrapper for pyplot.imshow that sets up a custom mouseover display
    formatter so that mouse motions over the image are labeled in the
    status bar with pixel numerical value as well as X and Y coords.
    """
    if ax is None:
        ax = plt.gca()
    ax.imshow(image, *args, **kwargs)
    aximage = ax.images[0].properties()['array']
    # need to account for half pixel offset of array coordinates for mouseover relative to pixel center,
    # so that the whole pixel from e.g. ( 1.5, 1.5) to (2.5, 2.5) is labeled with the coordinates of pixel (2,2)

    # We use the extent and implementation to map back from the data coord to pixel coord
    # There is probably an easier way to do this...
    imext = ax.images[0].get_extent()  # returns [-X, X, -Y, Y]
    imsize = ax.images[0].get_size()  # returns [sY, sX]g

    def report_pixel(x, y):
        # map data coords back to pixel coords
        # and be sure to clip appropriatedly to avoid array bounds errors
        img_y = np.floor((y - imext[2]) / (imext[3] - imext[2]) * imsize[0])
        img_y = int(img_y.clip(0, imsize[0] - 1))

        img_x = np.floor((x - imext[0]) / (imext[1] - imext[0]) * imsize[1])
        img_x = int(img_x.clip(0, imsize[1] - 1))

        return "(%6.3f, %6.3f)     %-12.6g" % (x, y, aximage[img_y, img_x])

    ax.format_coord = report_pixel
    return ax


def display_psf(hdulist_or_filename, ext=0, vmin=1e-7, vmax=1e-1,
                scale='log', cmap=None, title=None, imagecrop=None,
                adjust_for_oversampling=False, normalize='None',
                crosshairs=False, markcentroid=False, colorbar=True,
                colorbar_orientation='vertical', pixelscale='PIXELSCL',
                ax=None, return_ax=False, interpolation=None, cube_slice=None,
                angular_coordinate_unit=u.arcsec):
    """Display nicely a PSF from a given hdulist or filename

    This is extensively configurable. In addition to making an attractive display, for
    interactive usage this function provides a live display of the pixel value at a
    given (x,y) as you mouse around the image.

    Parameters
    ----------
    hdulist_or_filename : fits.hdulist or string
        FITS file containing image to display.
    ext : int
        FITS extension. default = 0
    vmin, vmax : float
        min and max for image display scaling
    scale : str
        'linear' or 'log', default is log
    cmap : matplotlib.cm.Colormap instance or None
        Colormap to use. If not given, taken from user's
        `poppy.conf.cmap_sequential` (Default: 'gist_heat').
    title : string, optional
        Set the plot title explicitly.
    imagecrop : float
        size of region to display (default is whole image)
    adjust_for_oversampling : bool
        rescale to conserve surface brightness for oversampled PSFs?
        (Making this True conserves surface brightness but not
        total flux.) Default is False, to conserve total flux.
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to
        normalize total flux=1. Default is no normalization.
    crosshairs : bool
        Draw a crosshairs at the image center (0, 0)? Default: False.
    markcentroid : bool
        Draw a crosshairs at the image centroid location?
        Centroiding is computed with the JWST-standard moving box
        algorithm. Default: False.
    colorbar : bool
        Draw a colorbar on the image?
    colorbar_orientation : 'vertical' (default) or 'horizontal'
        How should the colorbar be oriented? (Note: Updating a plot and
        changing the colorbar orientation is not supported. When replotting
        in the same axes, use the same colorbar orientation.)
    pixelscale : str or float
        if str, interpreted as the FITS keyword name for the pixel scale in arcsec/pixels.
        if float, used as the pixelscale directly.
    ax : matplotlib.Axes instance
        Axes to display into.
    return_ax : bool
        Return the axes to the caller for later use? (Default: False)
        When True, this function returns a matplotlib.Axes instance, or a
        tuple of (ax, cb) where the second is the colorbar Axes.
    interpolation : string
        Interpolation technique for PSF image. Default is None,
        meaning it is taken from matplotlib's `image.interpolation`
        rcParam.
    cube_slice : int or None
        if input PSF is a datacube from calc_datacube, which slice
        of the cube should be displayed?
    angular_coordinate_unit : astropy Unit
        Coordinate unit to use for axes display. Default is arcseconds.
    """
    if isinstance(hdulist_or_filename, str):
        hdulist = fits.open(hdulist_or_filename)
    elif isinstance(hdulist_or_filename, fits.HDUList):
        hdulist = hdulist_or_filename
    else:
        raise ValueError("input must be a filename or FITS HDUList object")

    # Get a handle on the input image
    if hdulist[ext].data.ndim == 2:
        im0 = hdulist[ext].data
        psf_array_shape = hdulist[ext].data.shape
    elif hdulist[ext].data.ndim == 3:
        if cube_slice is None:
            raise ValueError("To display a PSF datacube, you must set cube_slice=<#>.")
        else:
            im0 = hdulist[ext].data[cube_slice]
            psf_array_shape = hdulist[ext].data.shape[1:]
    else:
        raise RuntimeError("Unsupported image dimensionality.")

    # Normalization
    if adjust_for_oversampling:
        try:
            scalefactor = hdulist[ext].header['OVERSAMP'] ** 2
        except KeyError:
            _log.error("Could not determine oversampling scale factor; "
                       "therefore NOT rescaling fluxes.")
            scalefactor = 1
        im = im0 * scalefactor
    else:
        # don't change normalization of actual input array, work with a copy!
        im = im0.copy()

    if normalize.lower() == 'peak':
        _log.debug("Displaying image normalized to peak = 1")
        im /= im.max()
    elif normalize.lower() == 'total':
        _log.debug("Displaying image normalized to PSF total = 1")
        im /= im.sum()

    if scale == 'linear':
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    if isinstance(pixelscale, str):
        pixelscale = hdulist[ext].header[pixelscale]
    else:
        pixelscale = float(pixelscale)

    if angular_coordinate_unit != u.arcsec:
        coordinate_rescale = (1 * u.arcsec).to_value(angular_coordinate_unit)
        pixelscale *= coordinate_rescale
    else:
        coordinate_rescale = 1
    halffov_x = pixelscale * psf_array_shape[1] / 2.0
    halffov_y = pixelscale * psf_array_shape[0] / 2.0

    unit_label = str(angular_coordinate_unit)
    extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]

    if cmap is None:
        cmap = getattr(matplotlib.cm, poppy.conf.cmap_sequential)
    # update and get (or create) image axes
    ax = imshow_with_mouseover(
        im,
        extent=extent,
        cmap=cmap,
        norm=norm,
        ax=ax,
        interpolation=interpolation,
        origin='lower'
    )
    ax.set_xlabel(unit_label)

    if markcentroid:
        _log.info("measuring centroid to mark on plot...")
        ceny, cenx = measure_centroid(hdulist, ext=ext, units='arcsec', relativeto='center', boxsize=20, threshold=0.1)
        ceny *= coordinate_rescale  # if display coordinate unit isn't arcseconds, rescale the centroid accordingly
        cenx *= coordinate_rescale
        ax.plot(cenx, ceny, 'k+', markersize=15, markeredgewidth=1)
        _log.info("centroid: (%f, %f) " % (cenx, ceny))

    if imagecrop is not None:
        halffov_x = min((imagecrop / 2.0, halffov_x))
        halffov_y = min((imagecrop / 2.0, halffov_y))
    ax.set_xbound(-halffov_x, halffov_x)
    ax.set_ybound(-halffov_y, halffov_y)
    if crosshairs:
        ax.axhline(0, ls=':', color='k')
        ax.axvline(0, ls=':', color='k')
    if title is None:
        try:
            fspec = "%s, %s" % (hdulist[ext].header['INSTRUME'], hdulist[ext].header['FILTER'])
        except KeyError:
            fspec = str(hdulist_or_filename)
        title = "PSF sim for " + fspec
    ax.set_title(title)

    if colorbar:
        if ax.images[0].colorbar is not None:
            # Reuse existing colorbar axes (Issue #21)
            colorbar_axes = ax.images[0].colorbar.ax
            cb = plt.colorbar(
                ax.images[0],
                ax=ax,
                cax=colorbar_axes,
                orientation=colorbar_orientation
            )
        else:
            cb = plt.colorbar(
                ax.images[0],
                ax=ax,
                orientation=colorbar_orientation
            )
        if scale.lower() == 'log':
            ticks = np.logspace(np.log10(vmin), np.log10(vmax), int(np.round(np.log10(vmax / vmin) + 1)))
            if colorbar_orientation == 'horizontal' and vmax == 1e-1 and vmin == 1e-8:
                ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]  # looks better
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticks)
        if normalize.lower() == 'peak':
            cb.set_label('Intensity relative to peak pixel')
        else:
            cb.set_label('Fractional intensity per pixel')

    if return_ax:
        if colorbar:
            return ax, cb
        else:
            return ax


def display_psf_difference(hdulist_or_filename1=None, hdulist_or_filename2=None,
                           ext1=0, ext2=0, vmin=None, vmax=1e-4, title=None,
                           imagecrop=None, adjust_for_oversampling=False,
                           crosshairs=False, cmap=None, colorbar=True,
                           colorbar_orientation='vertical',
                           ax=None, return_ax=False,
                           normalize=False, normalize_to_second=False):
    """Display nicely the difference of two PSFs from given files

    The two files may be FITS files on disk or FITS HDUList objects in memory. The two must have the same
    shape and size.

    Parameters
    ----------
    hdulist_or_filename1, hdulist_or_filename2 : fits.HDUlist or string
        FITS files containing images to difference
    ext1, ext2 : int
        FITS extension. default = 0
    vmin, vmax : float
        Image intensity scaling min and max.
    title : string, optional
        Title for plot.
    imagecrop : float
        Size of region to display (default is whole image).
    adjust_for_oversampling : bool
        Rescale to conserve surface brightness for oversampled PSFs?
        (Making this True conserves surface brightness but not total flux.)
        Default is False, to conserve total flux.
    crosshairs : bool
        Plot crosshairs over array center?
    cmap : matplotlib.cm.Colormap instance or None
        Colormap to use. If not given, use standard gray colormap.
    colorbar : bool
        Draw a colorbar on the image?
    colorbar_orientation : 'vertical' (default) or 'horizontal'
        How should the colorbar be oriented? (Note: Updating a plot and
        changing the colorbar orientation is not supported. When replotting
        in the same axes, use the same colorbar orientation.)
    print\\_ : bool
        Print RMS difference value for the images? (Default: False)
    ax : matplotlib.Axes instance
        Axes to display into.
    return_ax : bool
        Return the axes to the caller for later use? (Default: False)
        When True, this function returns a matplotlib.Axes instance, or a
        tuple of (ax, cb) where the second is the colorbar Axes.
    normalize : bool
        Display (difference image)/(mean image) instead of just
        the difference image. Mutually exclusive to `normalize_to_second`.
        (Default: False)
    normalize_to_second : bool
        Display (difference image)/(second image) instead of just
        the difference image. Mutually exclusive to `normalize`.
        (Default: False)

    """
    if isinstance(hdulist_or_filename1, str):
        hdulist1 = fits.open(hdulist_or_filename1)
    elif isinstance(hdulist_or_filename1, fits.HDUList):
        hdulist1 = hdulist_or_filename1
    else:
        raise ValueError("input must be a filename or HDUlist")
    if isinstance(hdulist_or_filename2, str):
        hdulist2 = fits.open(hdulist_or_filename2)
    elif isinstance(hdulist_or_filename2, fits.HDUList):
        hdulist2 = hdulist_or_filename2
    else:
        raise ValueError("input must be a filename or HDUlist")

    if adjust_for_oversampling:
        scalefactor = hdulist1[ext1].header['OVERSAMP'] ** 2
        im1 = hdulist1[ext1].data * scalefactor
        scalefactor = hdulist2[ext2].header['OVERSAMP'] ** 2
        im2 = hdulist1[ext2].data * scalefactor
    else:
        im1 = hdulist1[ext1].data
        im2 = hdulist2[ext2].data

    diff_im = im1 - im2

    if normalize and not normalize_to_second:
        avg_im = (im1 + im2) / 2
        diff_im /= avg_im
        cbtitle = 'Image difference / average  (per pixel)'  # Relative intensity difference per pixel'
    elif normalize_to_second and not normalize:
        diff_im /= im2
        cbtitle = 'Image difference / original (per pixel)'  # Relative intensity difference per pixel'
    else:
        cbtitle = 'Intensity difference per pixel'

    if vmin is None:
        vmin = -vmax

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if cmap is None:
        cmap = matplotlib.cm.gray
    halffov_x = hdulist1[ext1].header['PIXELSCL'] * hdulist1[ext1].data.shape[1] / 2
    halffov_y = hdulist1[ext1].header['PIXELSCL'] * hdulist1[ext1].data.shape[0] / 2
    extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]

    ax = imshow_with_mouseover(diff_im, extent=extent, cmap=cmap, norm=norm, ax=ax,
                               origin='lower')
    if imagecrop is not None:
        halffov_x = min((imagecrop / 2, halffov_x))
        halffov_y = min((imagecrop / 2, halffov_y))
    ax.set_xbound(-halffov_x, halffov_x)
    ax.set_ybound(-halffov_y, halffov_y)
    if crosshairs:
        ax.axhline(0, ls=":", color='k')
        ax.axvline(0, ls=":", color='k')

    if title is None:
        title = "Difference of " + str(hdulist_or_filename1) + "-" + str(hdulist_or_filename2)
    ax.set_title(title)

    if colorbar:
        cb = plt.colorbar(ax.images[0], ax=ax, orientation=colorbar_orientation)
        # ticks = np.logspace(np.log10(vmin), np.log10(vmax), int(np.round(np.log10(vmax/vmin)+1)))
        # if vmin == 1e-8 and vmax==1e-1:
        # ticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        # ticks = [vmin, -0.5*vmax, 0, 0.5*vmax, vmax]
        # cb.set_ticks(ticks)
        # cb.set_ticklabels(ticks)
        cb.set_label(cbtitle)
    if return_ax:
        if colorbar:
            return ax, cb
        else:
            return ax


def display_ee(hdulist_or_filename=None, ext=0, overplot=False, ax=None, mark_levels=True, **kwargs):
    """ Display Encircled Energy curve for a PSF

    The azimuthally averaged encircled energy is plotted as a function of radius.

    Parameters
    ----------
    hdulist_or_filename : fits.HDUlist or string
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
    if isinstance(hdulist_or_filename, str):
        hdu_list = fits.open(hdulist_or_filename)
    elif isinstance(hdulist_or_filename, fits.HDUList):
        hdu_list = hdulist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    radius, profile, ee = radial_profile(hdu_list, ee=True, ext=ext, **kwargs)

    if not overplot:
        if ax is None:
            plt.clf()
            ax = plt.subplot(111)

    ax.plot(radius, ee)  # , nonposy='clip')
    if not overplot:
        ax.set_xlabel("Radius [arcsec]")
        ax.set_ylabel("Encircled Energy")

    if mark_levels:
        for level in [0.5, 0.8, 0.95]:
            ee_lev = radius[np.where(ee > level)[0][0]]
            yoffset = 0 if level < 0.9 else -0.05
            plt.text(ee_lev + 0.1, level + yoffset, 'EE=%2d%% at r=%.3f"' % (level * 100, ee_lev))


def display_profiles(hdulist_or_filename=None, ext=0, overplot=False, title=None, **kwargs):
    """ Produce two plots of PSF radial profile and encircled energy

    See also the display_ee function.

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
    if isinstance(hdulist_or_filename, str):
        hdu_list = fits.open(hdulist_or_filename, ext=ext)
    elif isinstance(hdulist_or_filename, fits.HDUList):
        hdu_list = hdulist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    radius, profile, ee = radial_profile(hdu_list, ee=True, ext=ext, **kwargs)

    if title is None:
        try:
            title = "%s, %s" % (hdu_list[ext].header['INSTRUME'], hdu_list[ext].header['FILTER'])
        except KeyError:
            title = str(hdulist_or_filename)

    if not overplot:
        plt.clf()
        plt.title(title)
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("PSF radial profile")
    plt.subplot(2, 1, 1)
    plt.semilogy(radius, profile)

    fwhm = 2 * radius[np.where(profile < profile[0] * 0.5)[0][0]]
    plt.text(fwhm, profile[0] * 0.5, 'FWHM = %.3f"' % fwhm)

    plt.subplot(2, 1, 2)
    # plt.semilogy(radius, ee, nonposy='clip')
    plt.plot(radius, ee, color='r')  # , nonposy='clip')
    if not overplot:
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("Encircled Energy")

    for level in [0.5, 0.8, 0.95]:
        if (ee > level).any():
            ee_lev = radius[np.where(ee > level)[0][0]]
            yoffset = 0 if level < 0.9 else -0.05
            plt.text(ee_lev + 0.1, level + yoffset, 'EE=%2d%% at r=%.3f"' % (level * 100, ee_lev))


def radial_profile(hdulist_or_filename=None, ext=0, ee=False, center=None, stddev=False, binsize=None, maxradius=None,
                   normalize='None', pa_range=None, slice=0):
    """ Compute a radial profile of the image.

    This computes a discrete radial profile evaluated on the provided binsize. For a version
    interpolated onto a continuous curve, see measure_radial().

    Code taken pretty much directly from pydatatut.pdf

    Parameters
    ----------
    hdulist_or_filename : string
        FITS HDUList object or path to a FITS file.
        NaN values in the FITS data array are treated as masked and ignored in computing bin statistics.
    ext : int
        Extension in FITS file
    ee : bool
        Also return encircled energy (EE) curve in addition to radial profile?
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center.
    binsize : float
        size of step for profile. Default is pixel size.
    stddev : bool
        Compute standard deviation in each radial bin, not average?
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to normalize total flux=1.
        Default is no normalization (i.e. retain whatever normalization was used in computing the PSF itself)
    maxradius : float, optional
        Maximum radius to compute radial profile to. If not set, will be computed for all radii within the image.
    pa_range : list of floats, optional
        Optional specification for [min, max] position angles to be included in the radial profile.
        I.e. calculate that profile only for some wedge, not the full image. Specify the PA in degrees
        counterclockwise from +Y axis=0. Note that you can specify ranges across zero using negative numbers,
        such as pa_range=[-10,10].  The allowed PA range runs from -180 to 180 degrees.
    slice: integer, optional
        Slice into a datacube, for use on cubes computed by calc_datacube. Default 0 if a
        cube is provided with no slice specified.

    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """
    if isinstance(hdulist_or_filename, str):
        hdu_list = fits.open(hdulist_or_filename)
    elif isinstance(hdulist_or_filename, fits.HDUList):
        hdu_list = hdulist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    if hdu_list[ext].header['NAXIS'] == 3:
        # data cube, so pick out just one slice
        image = hdu_list[ext].data[slice].copy()  # don't change normalization of actual input array, work with a copy!
    else:
        image = hdu_list[ext].data.copy()  # don't change normalization of actual input array, work with a copy!

    if normalize.lower() == 'peak':
        _log.debug("Calculating profile with PSF normalized to peak = 1")
        image /= image.max()
    elif normalize.lower() == 'total':
        _log.debug("Calculating profile with PSF normalized to total = 1")
        image /= image.sum()

    pixelscale = hdu_list[ext].header['PIXELSCL']

    if binsize is None:
        binsize = pixelscale

    y, x = np.indices(image.shape, dtype=float)
    if center is None:
        # get exact center of image
        # center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

    x -= center[0]
    y -= center[1]

    r = np.sqrt(x ** 2 + y ** 2) * pixelscale / binsize  # radius in bin size steps

    if pa_range is None:
        # Use full image
        ind = np.argsort(r.flat)
        sr = r.flat[ind]  # sorted r
        sim = image.flat[ind]  # sorted image

    else:
        # Apply the PA range restriction
        pa = np.rad2deg(np.arctan2(-x, y))  # Note the (-x,y) convention is needed for astronomical PA convention
        mask = (pa >= pa_range[0]) & (pa <= pa_range[1])
        ind = np.argsort(r[mask].flat)
        sr = r[mask].flat[ind]
        sim = image[mask].flat[ind]

    ri = sr.astype(int)  # sorted r as int
    deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]  # number in radius bin
    csim = np.nan_to_num(sim).cumsum(dtype=float)  # cumulative sum to figure out sums for each bin
    # np.nancumsum is implemented in >1.12
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
    radialprofile = tbin / nr

    # pre-pend the initial element that the above code misses.
    radialprofile2 = np.empty(len(radialprofile) + 1)
    if rind[0] != 0:
        radialprofile2[0] = csim[rind[0]] / (
                rind[0] + 1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]  # otherwise if there's just one then just take it.
    radialprofile2[1:] = radialprofile

    # Compute radius values corresponding to the measured points in the radial profile.
    # including handling the case where the innermost pixel may be more
    # than one pixel from the center. This can happen if pa_range is not None, since for
    # small ranges < 45 deg or so the innermost pixel that's valid in the mask may be
    # more than one pixel from the center. It can also happen if we are computing a
    # radial profile centered on an offset source outside of the FOV.
    rr = np.arange(ri.min(), ri.min() + len(
        radialprofile2)) * binsize + binsize * 0.5  # these should be centered in the bins, so add a half.

    if maxradius is not None:
        crop = rr < maxradius
        rr = rr[crop]
        radialprofile2 = radialprofile2[crop]

    if stddev:
        stddevs = np.zeros_like(radialprofile2)
        r_pix = r * binsize
        for i, radius in enumerate(rr):
            if i == 0:
                wg = np.where(r < radius + binsize / 2)
            else:
                wg = np.where((r_pix >= (radius - binsize / 2)) & (r_pix < (radius + binsize / 2)))
                # wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
            stddevs[i] = np.nanstd(image[wg])
        return rr, stddevs

    if not ee:
        return rr, radialprofile2
    else:
        ee = csim[rind]
        return rr, radialprofile2, ee


###########################################################################
#
#    PSF evaluation functions
#

def measure_ee(hdulist_or_filename=None, ext=0, center=None, binsize=None, normalize='None'):
    """ measure encircled energy vs radius and return as an interpolator

    Returns a function object which when called returns the Encircled Energy inside a given radius,
    for any arbitrary desired radius smaller than the image size.



    Parameters
    ----------
    hdulist_or_filename : string
        Either a fits.HDUList object or a filename of a FITS file on disk
    ext : int
        Extension in that FITS file
    center : tuple of floats
        Coordinates (x,y) of PSF center. Default is image center.
    binsize:
        size of step for profile. Default is pixel size.
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to normalize total flux=1.
        Default is no normalization (i.e. retain whatever normalization was used in computing the PSF itself)

    Returns
    --------
    encircled_energy: function
        A function which will return the encircled energy interpolated to any desired radius.


    Examples
    --------
    >>> ee = measure_ee("someimage.fits")  # doctest: +SKIP
    >>> print("The EE at 0.5 arcsec is ", ee(0.5))  # doctest: +SKIP

    """

    rr, radialprofile2, ee = radial_profile(hdulist_or_filename, ext, ee=True, center=center, binsize=binsize,
                                            normalize=normalize)

    # append the zero at the center
    rr_ee = rr + (rr[1] - rr[0]) / 2.0  # add half a binsize to this, because the ee is measured inside the
    # outer edge of each annulus.
    rr0 = np.concatenate(([0], rr_ee))
    ee0 = np.concatenate(([0], ee))

    ee_fn = scipy.interpolate.interp1d(rr0, ee0, kind='cubic', bounds_error=False)

    return ee_fn


def measure_radius_at_ee(hdulist_or_filename=None, ext=0, center=None, binsize=None, normalize='None'):
    """ measure encircled energy vs radius and return as an interpolator
    Returns a function object which when called returns the radius for a given Encircled Energy. This is the
    inverse function of measure_ee

    Parameters
    ----------
    hdulist_or_filename : string
        Either a fits.HDUList object or a filename of a FITS file on disk
    ext : int
        Extension in that FITS file
    center : tuple of floats
        Coordinates (x,y) of PSF center. Default is image center.
    binsize:
        size of step for profile. Default is pixel size.
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to normalize total flux=1.
        Default is no normalization (i.e. retain whatever normalization was used in computing the PSF itself)

    Returns
    --------
    radius: function
        A function which will return the radius of a desired encircled energy.

    Examples
    --------
    >>> ee = measure_radius_at_ee("someimage.fits")  # doctest: +SKIP
    >>> print("The EE is 50% at {} arcsec".format(ee(0.5)))  # doctest: +SKIP
    """

    rr, radialprofile2, ee = radial_profile(hdulist_or_filename, ext, ee=True, center=center, binsize=binsize,
                                            normalize=normalize)

    # append the zero at the center
    rr_ee = rr + (rr[1] - rr[0]) / 2.0  # add half a binsize to this, because the EE is measured inside the
    # outer edge of each annulus.
    rr0 = np.concatenate(([0], rr_ee))
    ee0 = np.concatenate(([0], ee))

    radius_at_ee_fn = scipy.interpolate.interp1d(ee0, rr0, kind='cubic', bounds_error=False)

    return radius_at_ee_fn


def measure_radial(hdulist_or_filename=None, ext=0, center=None, binsize=None):
    """ measure azimuthally averaged radial profile of a PSF.

    Returns a function object which when called returns the mean value at a given radius.

    Parameters
    ----------
    hdulist_or_filename : string
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
    >>> rp = measure_radial("someimage.fits")  # doctest: +SKIP
    >>> radius = np.linspace(0, 5.0, 100)
    >>> plt.plot(radius, rp(radius), label="PSF")  # doctest: +SKIP

    """

    rr, radialprofile, ee = radial_profile(hdulist_or_filename, ext, ee=True, center=center, binsize=binsize)

    radial_fn = scipy.interpolate.interp1d(rr, radialprofile, kind='cubic', bounds_error=False)

    return radial_fn


def measure_fwhm(hdulist_or_filename, ext=0, center=None, plot=False, threshold=0.1):
    """ Improved version of measuring FWHM, without any binning of image data.

    Method: Pick out the image pixels which are above some threshold relative to the
    peak intensity, then fit a Gaussian to those. Infer the FWHM based on the width of
    the Gaussian.


    Parameters
    ----------
    hdulist_or_filename : string
        what it sounds like.
    ext : int
        Extension in FITS file
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center.
    threshold : float
        Fraction relative to the peak pixel that is used to select the bright peak pixels
        used in fitting the Gaussian. Default is 0.1, i.e. pixels brighter that 0.1 of
        the maximum will be included. This is chosen semi-arbitrarily to include most of
        the peak but exclude the first Airy ring for typical cases.
    plot : bool
        Display a diagnostic plot.

    Returns
    -------
    fwhm : float
        FWHM in arcseconds

    """
    from astropy.modeling import models, fitting

    if isinstance(hdulist_or_filename, str):
        hdulist = fits.open(hdulist_or_filename)
    elif isinstance(hdulist_or_filename, fits.HDUList):
        hdulist = hdulist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    image = hdulist[ext].data.copy()  # don't change normalization of actual input array; work with a copy
    image = image / image.max()  # Normalize the copy to peak=1

    pixelscale = hdulist[ext].header['PIXELSCL']

    _log.debug("Pixelscale is {} arcsec/pix.".format(pixelscale, ))

    # Prepare array r with radius in arcseconds
    y, x = np.indices(image.shape, dtype=float)
    if center is None:
        # get exact center of image
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])
    _log.debug("Using PSF center = {}".format(center))
    x -= center[0]
    y -= center[1]
    r = np.sqrt(x ** 2 + y ** 2) * pixelscale  # radius in arcseconds

    # Select pixels above that threshold
    wpeak = np.where(image > threshold)  # note, image is normalized to peak=1 above
    _log.debug("Using {} pixels above {} of peak".format(len(wpeak[0]), threshold))

    rpeak = r[wpeak]
    impeak = image[wpeak]

    # Determine initial guess for Gaussian parameters
    if 'DIFFLMT' in hdulist[ext].header:
        std_guess = hdulist[ext].header['DIFFLMT'] / 2.354
    else:
        std_guess = measure_fwhm_radprof(hdulist, ext=ext, center=center, nowarn=True) / 2.354
    _log.debug("Initial guess Gaussian sigma= {} arcsec".format(std_guess))

    # Determine best fit Gaussian parameters
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=std_guess)
    g_init.mean.fixed = True

    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, rpeak, impeak)
    _log.debug("Fit results for Gaussian: {}, {}".format(g.amplitude, g.stddev))

    # Convert from the fit result sigma parameter to FWHM.
    # note, astropy fitting doesn't constrain the stddev to be positive for some reason.
    # so take abs value here.
    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.abs(g.stddev)

    if plot:
        plt.loglog(rpeak, impeak, linestyle='none', marker='o', alpha=0.5)
        rmin = rpeak[rpeak != 0].min()
        plotr = np.linspace(rmin, rpeak.max(), 30)

        plt.plot(plotr, g(plotr))
        plt.xlabel("Radius [arcsec]")
        plt.ylabel("Intensity relative to peak")

        plt.axhline(0.5, ls=":")
        plt.axvline(fwhm / 2, ls=':')
        plt.text(0.1, 0.2, 'FWHM={:.4f} arcsec'.format(fwhm), transform=plt.gca().transAxes, )

        plt.gca().set_ylim(threshold * .5, 2)

    return fwhm


def measure_fwhm_radprof(HDUlist_or_filename=None, ext=0, center=None, level=0.5, nowarn=False):
    """ Measure FWHM by interpolation of the radial profile.
    This version is old/deprecated; see the new measure_fwhm instead.

    However, this function is kept, for now, to provide a robust, simple backup
    method which can be used to determine the initial guess for the model-fitting
    approach in the newer measure_fwhm function.

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
    nowarn : bool
        Set this to suppress the warning display that this function is deprecated.
        But you probably shouldn't; only use this if you know what you're doing.


    Returns
    -------
    fwhm : float
        FWHM in arcseconds

    """

    if not nowarn:
        import warnings
        warnings.warn("measure_fwhm_radprof uses a deprecated, older algorithm. "
                      "measure_fwhm is preferred in most cases.",
                      DeprecationWarning)

    rr, radialprofile, ee = radial_profile(HDUlist_or_filename, ext, ee=True, center=center)
    rpmax = radialprofile.max()

    wlower = np.where(radialprofile < rpmax * level)
    if len(wlower[0]) == 0:
        raise ValueError(
            "The supplied array's pixel values never go below {0:.2f} of its maximum, {1:.3g}. " +
            "Cannot measure FWHM.".format(level, rpmax))
    wmin = np.min(wlower[0])
    # go just a bit beyond the half way mark
    winterp = np.arange(0, wmin + 2, dtype=int)[::-1]

    if len(winterp) < 6:
        kind = 'linear'
    else:
        kind = 'cubic'

    interp_hw = scipy.interpolate.interp1d(radialprofile[winterp], rr[winterp], kind=kind)
    return 2 * interp_hw(rpmax * level)


def measure_sharpness(HDUlist_or_filename=None, ext=0):
    """ Compute image sharpness, the sum of pixel squares.

    See Makidon et al. JWST-STScI-001157 for a discussion of this image metric
    and its relationship to noise equivalent pixels.

    Parameters
    ----------
    HDUlist_or_filename, ext : string, int
        Same as above

    """
    if isinstance(HDUlist_or_filename, str):
        hdulist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        hdulist = HDUlist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    # TODO or check that the oversampling factor is 1
    try:
        detpixels = hdulist['DET_SAMP']
    except KeyError:
        raise ValueError(
            "You can only measure sharpness for an image with an extension giving the rebinned " +
            "actual detector pixel values.""")

    sharpness = (detpixels.data ** 2).sum()
    return sharpness


def measure_centroid(HDUlist_or_filename=None, ext=0, slice=0, boxsize=20, verbose=False, units='pixels',
                     relativeto='origin', **kwargs):
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
    verbose : bool
        Be more verbose


    Returns
    -------
    CoM : array_like
        [Y, X] coordinates of center of mass.

    """
    from .fwcentroid import fwcentroid

    if isinstance(HDUlist_or_filename, str):
        HDUlist = fits.open(HDUlist_or_filename)
    elif isinstance(HDUlist_or_filename, fits.HDUList):
        HDUlist = HDUlist_or_filename
    else:
        raise ValueError("input must be a filename or HDUlist")

    image = HDUlist[ext].data

    if image.ndim >= 3:  # handle datacubes gracefully
        image = image[slice, :, :]

    cent_of_mass = fwcentroid(image, halfwidth=boxsize, **kwargs)

    if verbose:
        print("Center of mass: (%.4f, %.4f)" % (cent_of_mass[1], cent_of_mass[0]))

    if relativeto == 'center':
        imcen = np.array([(image.shape[0] - 1) / 2., (image.shape[1] - 1) / 2.])
        cent_of_mass = tuple(np.array(cent_of_mass) - imcen)

    if units == 'arcsec':
        pixelscale = HDUlist[ext].header['PIXELSCL']
        cent_of_mass = tuple(np.array(cent_of_mass) * pixelscale)

    return cent_of_mass


def measure_anisotropy(HDUlist_or_filename=None, ext=0, slice=0, boxsize=50):
    raise NotImplementedError("measure_anisotropy is not yet implemented.")


###########################################################################
#
#    Array manipulation utility functions
#


def pad_to_oversample(array, oversample):
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
    n = int(np.round(npix * oversample))
    padded = np.zeros(shape=(n, n), dtype=array.dtype)
    n0 = float(npix) * (oversample - 1) / 2
    n1 = n0 + npix
    n0 = int(round(n0))  # because astropy test_plugins enforces integer indices
    n1 = int(round(n1))
    padded[n0:n1, n0:n1] = array
    return padded


def pad_to_size(array, padded_shape):
    """ Add zeros around the edge of an array, to reach a specific defined size and shape.
    This is similar to pad_to_oversample but is more flexible.

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
    pad_to_oversample, pad_or_crop_to_shape
    """

    if len(padded_shape) < 2:
        outsize0 = padded_shape
        outsize1 = padded_shape
    else:
        outsize0 = padded_shape[0]
        outsize1 = padded_shape[1]
    # npix = array.shape[0]
    padded = np.zeros(shape=padded_shape, dtype=array.dtype)
    n0 = (outsize0 - array.shape[0]) // 2  # pixel offset for the inner array
    m0 = (outsize1 - array.shape[1]) // 2  # pixel offset in second dimension
    n1 = n0 + array.shape[0]
    m1 = m0 + array.shape[1]
    n0 = int(round(n0))  # because astropy test_plugins enforces integer indices
    n1 = int(round(n1))
    m0 = int(round(m0))
    m1 = int(round(m1))
    padded[n0:n1, m0:m1] = array
    return padded


def pad_or_crop_to_shape(array, target_shape):
    """ Adapt an array to match a desired shape, by
    adding zero pixels to pad, or cropping out pixels as needed.
    (Implicitly assumes the arrays have comparable pixel scale and units)

    Parameters
    ----------
    array : complex ndarray
        The phasor, produced by some call to get_phasor of an OpticalElement
    target_shape : 2-tuple
        The shape we should pad or crop that phasor to

    Returns
    -------
    new_phasor : complex ndarray
        A copy of the phasor modified to have the desired array size

    See Also
    ---------
    pad_to_oversample, pad_to_size

    """

    if array.shape == target_shape:
        return array

    lx, ly = array.shape
    lx_w, ly_w = target_shape
    border_x = np.abs(lx - lx_w) // 2
    border_y = np.abs(ly - ly_w) // 2

    if (lx < lx_w) or (ly < ly_w):
        _log.debug("Array shape " + str(array.shape) + " is smaller than desired shape " + str(
            [lx_w, ly_w]) + "; will attempt to zero-pad the array")

        resampled_array = np.zeros(shape=(lx_w, ly_w), dtype=array.dtype)
        resampled_array[border_x:border_x + lx, border_y:border_y + ly] = array
        _log.debug("  Padded with a {:d} x {:d} border to "
                   " match the desired shape".format(border_x, border_y))

    else:
        _log.debug("Array shape " + str(array.shape) + " is larger than desired shape " + str(
            [lx_w, ly_w]) + "; will crop out just the center part.")
        resampled_array = array[border_x:border_x + lx_w, border_y:border_y + ly_w]
        _log.debug("  Trimmed a border of {:d} x {:d} pixels "
                   "to match the desired shape".format(border_x, border_y))
    return resampled_array


def remove_padding(array, oversample):
    """ Remove zeros around the edge of an array, assuming some integer oversampling padding factor """
    npix = array.shape[0] / oversample
    n0 = float(npix) * (oversample - 1) / 2
    n1 = n0 + npix
    n0 = int(round(n0))
    n1 = int(round(n1))
    return array[n0:n1, n0:n1].copy()


# Back compatibility alias:
removePadding = remove_padding


def rebin_array(a=None, rc=(2, 2), verbose=False):
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
            b[ri, ci] = np.add.reduce(a[Rlo:Rlo + r, Clo:Clo + c].copy().flat)
            if verbose:
                print("    [%d:%d, %d:%d]" % (Rlo, Rlo + r, Clo, Clo + c))
                print("%4.0f" % np.add.reduce(a[Rlo:Rlo + r, Clo:Clo + c].copy().flat))
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
    sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
    return a.reshape(sh).sum(-1).sum(1)


###########################################################################
#
#    Unit Handling
#


class BackCompatibleQuantityInput(object):
    # Modified from code in astropy.units.decorators.py
    # See http://docs.astropy.org/en/stable/_modules/astropy/units/decorators.html

    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        """
        A decorator for validating the units of arguments to functions.
        This is a *variant* of the quantity_input decorator provided by astropy;
        the difference is the handling of bare input numbers without units.

        When given such an input, this function will silently & without complaint apply
        the specified unit as a default. The astropy version will raise a ValueError
        that it was expecting a Quantity.  The benefit is this approach allows back
        compatibility with functions originally written to accept floating point values
        implicitly in meters.

        Unit specifications can be provided as keyword arguments to the decorator,
        or by using Python 3's function annotation syntax. Arguments to the decorator
        take precedence over any function annotations present.

        A `~astropy.units.UnitsError` will be raised if the unit attribute of
        the argument is not equivalent to the unit specified to the decorator
        or in the annotation.

        Where an equivalency is specified in the decorator, the function will be
        executed with that equivalency in force.

        Notes
        -----

        The checking of arguments inside variable arguments to a function is not
        supported (i.e. \*arg or \**kwargs).

        Examples
        --------
        The desired type of the input parameter can be specified as an argument to
        the decorator, or via an annotation on the function argument itself:
        .. code-block:: python3

            import poppy.utils
            @poppy.utils.back_compatible_quantity_input(mylength=u.meter)
            def myfunction(mylength):
                return mylength**2


        .. code-block:: python3

            import poppy.utils
            @poppy.utils.back_compatible_quantity_input
            def myfunction(mylength: u.meter):
                return mylength**2


        """
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self

    def __init__(self, func=None, **kwargs):
        self.equivalencies = kwargs.pop('equivalencies', [])
        self.decorator_kwargs = kwargs

    def __call__(self, wrapped_function):
        from astropy.utils.decorators import wraps
        from astropy.units import UnitsError, add_enabled_equivalencies, Quantity
        import inspect

        # Extract the function signature for the function we are wrapping.
        wrapped_signature = inspect.signature(wrapped_function)

        # Define a new function to return in place of the wrapped one
        @wraps(wrapped_function)
        def unit_check_wrapper(*func_args, **func_kwargs):
            # Bind the arguments to our new function to the signature of the original.
            bound_args = wrapped_signature.bind(*func_args, **func_kwargs)

            # Iterate through the parameters of the original signature
            for param in wrapped_signature.parameters.values():
                # We do not support variable arguments (*args, **kwargs)
                if param.kind in (inspect.Parameter.VAR_KEYWORD,
                                  inspect.Parameter.VAR_POSITIONAL):
                    continue
                # Catch the (never triggered) case where bind relied on a default value.
                if param.name not in bound_args.arguments and param.default is not param.empty:
                    bound_args.arguments[param.name] = param.default

                # Get the value of this parameter (argument to new function)
                arg = bound_args.arguments[param.name]
                # don't try to apply a unit to an argument which is not present
                if arg is None:
                    continue

                # Get target unit, either from decorator kwargs or annotations
                if param.name in self.decorator_kwargs:
                    target_unit = self.decorator_kwargs[param.name]
                else:
                    target_unit = param.annotation

                # If the target unit is empty, then no unit was specified so we
                # move past it
                if target_unit is not inspect.Parameter.empty:
                    if not isinstance(arg, Quantity):
                        # if we're going to make something a quantity it had better
                        # be compatible with float ndarray
                        try:
                            tmp = np.asarray(arg, dtype=float)
                        except (ValueError, TypeError):
                            raise ValueError("Argument '{0}' to function '{1}'"
                                             " must be a number (not '{3}'), and convertable to"
                                             " units='{2}'.".format(param.name,
                                                                    wrapped_function.__name__,
                                                                    target_unit.to_string(), arg))

                    try:
                        equivalent = arg.unit.is_equivalent(target_unit,
                                                            equivalencies=self.equivalencies)

                        if not equivalent:
                            raise UnitsError("Argument '{0}' to function '{1}'"
                                             " must be in units convertable to"
                                             " '{2}'.".format(param.name,
                                                              wrapped_function.__name__,
                                                              target_unit.to_string()))

                    # Either there is no .unit or no .is_equivalent
                    except AttributeError:
                        if hasattr(arg, "unit"):
                            error_msg = "a 'unit' attribute without an 'is_equivalent' method"
                            raise TypeError("Argument '{0}' to function '{1}' has {2}. "
                                            "You may want to pass in an astropy Quantity instead."
                                            .format(param.name, wrapped_function.__name__, error_msg))
                        else:
                            # apply the default unit here, without complaint
                            # print("Updating: "+param.name)
                            bound_args.arguments[param.name] = arg * target_unit

            # Call the original function with any equivalencies in force.
            with add_enabled_equivalencies(self.equivalencies):
                # print("Args:   {}".format(bound_args.args))
                # print("KWArgs: {}".format(bound_args.kwargs))
                return wrapped_function(*bound_args.args, **bound_args.kwargs)

        return unit_check_wrapper


quantity_input = BackCompatibleQuantityInput.as_decorator


###########################################################################
#
#    Other utility functions
#


def spectrum_from_spectral_type(sptype, return_list=False, catalog=None):
    """Get synphot Spectrum object from a user-friendly spectral type string.

    Given a spectral type such as 'A0IV' or 'G2V', this uses a fixed lookup table
    to determine an appropriate spectral model from Castelli & Kurucz 2004 or
    the Phoenix model grids. Depends on synphot, stsynphot, and CDBS. This is just a
    convenient access function.

    Parameters
    -----------
    sptype : str
        Spectral type, like "G0V"
    catalog : str
        'ck04' for Castelli & Kurucz 2004, 'phoenix' for Phoenix models.
        If not set explicitly, the code will check if the phoenix models are
        present inside the $PYSYN_CDBS directory. If so, those are the default;
        otherwise, it's CK04.
    return_list : bool
        Return list of allowed spectral types. This is deprecated and unused now,
        but at one point was used in the deprecated GUI functionality.
    """
    try:
        from stsynphot import grid_to_spec
    except ImportError:
        raise ImportError("Need stsynphot for this functionality")
    from synphot import SourceSpectrum
    from synphot import units as syn_u
    from synphot.models import ConstFlux1D, PowerLawFlux1D

    if catalog is None:
        import os
        cdbs = os.getenv('PYSYN_CDBS')
        if cdbs is None:
            raise EnvironmentError("Environment variable $PYSYN_CDBS must be defined for synphot")
        if os.path.exists(os.path.join(os.getenv('PYSYN_CDBS'), 'grid', 'phoenix')):
            catalog = 'phoenix'
        elif os.path.exists(os.path.join(os.getenv('PYSYN_CDBS'), 'grid', 'ck04models')):
            catalog = 'ck04'
        else:
            raise IOError("Could not find either phoenix or ck04models subdirectories of $PYSYN_CDBS/grid")

    if catalog.lower() == 'ck04':
        catname = 'ck04models'

        # Recommended lookup table into the CK04 models (from
        # the documentation of that catalog?)
        lookuptable = {
            "O3V": (50000, 0.0, 5.0),
            "O5V": (45000, 0.0, 5.0),
            "O6V": (40000, 0.0, 4.5),
            "O8V": (35000, 0.0, 4.0),
            "O5I": (40000, 0.0, 4.5),
            "O6I": (40000, 0.0, 4.5),
            "O8I": (34000, 0.0, 4.0),
            "B0V": (30000, 0.0, 4.0),
            "B3V": (19000, 0.0, 4.0),
            "B5V": (15000, 0.0, 4.0),
            "B8V": (12000, 0.0, 4.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "B0I": (26000, 0.0, 3.0),
            "B5I": (14000, 0.0, 2.5),
            "A0V": (9500, 0.0, 4.0),
            "A5V": (8250, 0.0, 4.5),
            "A0I": (9750, 0.0, 2.0),
            "A5I": (8500, 0.0, 2.0),
            "F0V": (7250, 0.0, 4.5),
            "F5V": (6500, 0.0, 4.5),
            "F0I": (7750, 0.0, 2.0),
            "F5I": (7000, 0.0, 1.5),
            "G0V": (6000, 0.0, 4.5),
            "G5V": (5750, 0.0, 4.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "G0I": (5500, 0.0, 1.5),
            "G5I": (4750, 0.0, 1.0),
            "K0V": (5250, 0.0, 4.5),
            "K5V": (4250, 0.0, 4.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "K0I": (4500, 0.0, 1.0),
            "K5I": (3750, 0.0, 0.5),
            "M0V": (3750, 0.0, 4.5),
            "M2V": (3500, 0.0, 4.5),
            "M5V": (3500, 0.0, 5.0),
            "M0III": (3750, 0.0, 1.5),
            "M0I": (3750, 0.0, 0.0),
            "M2I": (3500, 0.0, 0.0)}
    elif catalog.lower() == 'phoenix':
        catname = 'phoenix'
        # lookup table used in JWST ETCs
        lookuptable = {
            "O3V": (45000, 0.0, 4.0),
            "O5V": (41000, 0.0, 4.5),
            "O7V": (37000, 0.0, 4.0),
            "O9V": (33000, 0.0, 4.0),
            "B0V": (30000, 0.0, 4.0),
            "B1V": (25000, 0.0, 4.0),
            "B3V": (19000, 0.0, 4.0),
            "B5V": (15000, 0.0, 4.0),
            "B8V": (12000, 0.0, 4.0),
            "A0V": (9500, 0.0, 4.0),
            "A1V": (9250, 0.0, 4.0),
            "A3V": (8250, 0.0, 4.0),
            "A5V": (8250, 0.0, 4.0),
            "F0V": (7250, 0.0, 4.0),
            "F2V": (7000, 0.0, 4.0),
            "F5V": (6500, 0.0, 4.0),
            "F8V": (6250, 0.0, 4.5),
            "G0V": (6000, 0.0, 4.5),
            "G2V": (5750, 0.0, 4.5),
            "G5V": (5750, 0.0, 4.5),
            "G8V": (5500, 0.0, 4.5),
            "K0V": (5250, 0.0, 4.5),
            "K2V": (4750, 0.0, 4.5),
            "K5V": (4250, 0.0, 4.5),
            "K7V": (4000, 0.0, 4.5),
            "M0V": (3750, 0.0, 4.5),
            "M2V": (3500, 0.0, 4.5),
            "M5V": (3500, 0.0, 5.0),
            "B0III": (29000, 0.0, 3.5),
            "B5III": (15000, 0.0, 3.5),
            "G0III": (5750, 0.0, 3.0),
            "G5III": (5250, 0.0, 2.5),
            "K0III": (4750, 0.0, 2.0),
            "K5III": (4000, 0.0, 1.5),
            "M0III": (3750, 0.0, 1.5),
            "O6I": (39000, 0.0, 4.5),
            "O8I": (34000, 0.0, 4.0),
            "B0I": (26000, 0.0, 3.0),
            "B5I": (14000, 0.0, 2.5),
            "A0I": (9750, 0.0, 2.0),
            "A5I": (8500, 0.0, 2.0),
            "F0I": (7750, 0.0, 2.0),
            "F5I": (7000, 0.0, 1.5),
            "G0I": (5500, 0.0, 1.5),
            "G5I": (4750, 0.0, 1.0),
            "K0I": (4500, 0.0, 1.0),
            "K5I": (3750, 0.0, 0.5),
            "M0I": (3750, 0.0, 0.0),
            "M2I": (3500, 0.0, 0.0)}

    if return_list:
        sptype_list = list(lookuptable.keys())

        def sort_sptype(typestr):
            letter = typestr[0]
            lettervals = {'O': 0, 'B': 10, 'A': 20, 'F': 30, 'G': 40, 'K': 50, 'M': 60}
            value = lettervals[letter] * 1.0
            value += int(typestr[1])
            if "III" in typestr:
                value += .3
            elif "I" in typestr:
                value += .1
            elif "V" in typestr:
                value += .5
            return value

        sptype_list.sort(key=sort_sptype)
        sptype_list.insert(0, "Flat spectrum in F_nu")
        sptype_list.insert(0, "Flat spectrum in F_lambda")
        # add a variety of spectral type slopes, per request from Dean Hines
        for slope in [-3, -2, -1.5, -1, -0.75, -0.5, 0.5, 0.75, 1.0, 1.5, 2, 3]:
            sptype_list.insert(0, "Power law F_nu ~ nu^(%s)" % str(slope))
        # sptype_list.insert(0,"Power law F_nu ~ nu^(-0.75)")
        # sptype_list.insert(0,"Power law F_nu ~ nu^(-1.0)")
        # sptype_list.insert(0,"Power law F_nu ~ nu^(-1.5)")
        # sptype_list.insert(0,"Power law F_nu ~ nu^(-2.0)")
        return sptype_list

    if "Flat" in sptype:
        if sptype == "Flat spectrum in F_nu":
            spec = SourceSpectrum(ConstFlux1D, amplitude=1 * syn_u.FNU)
        elif sptype == "Flat spectrum in F_lambda":
            spec = SourceSpectrum(ConstFlux1D, amplitude=1 * syn_u.FLAM)

        return spec
    if 'Power law' in sptype:
        import re
        ans = re.search(r'\((.*)\)', sptype)
        if ans is None:
            raise ValueError("Invalid power law specification cannot be parsed to get exponent")
        exponent = float(ans.groups(0)[0])
        # note that synphot's PowerLaw class implements a power law in terms of lambda, not nu.
        # but since nu = clight/lambda, it's just a matter of swapping the sign on the exponent.

        spec = SourceSpectrum(
            PowerLawFlux1D, amplitude=1 * syn_u.FNU, x_0=1 * u.AA,
            alpha=-exponent, meta={'name': sptype})

        return spec
    else:
        keys = lookuptable[sptype]
        try:
            return grid_to_spec(catname, keys[0], keys[1], keys[2])
        except IOError:
            errmsg = ("Could not find a match in catalog {0} for key {1}. Check that is a valid name in the " +
                      "lookup table, and/or that synphot is installed properly.".format(catname, sptype))
            _log.critical(errmsg)
            raise LookupError(errmsg)


# Back compatibility allias
specFromSpectralType = spectrum_from_spectral_type


# ##################################################################
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
    except ImportError:
        _log.debug("No psutil package available, cannot estimate optimal nprocesses.")
        return 4

    wfshape = osys.input_wavefront().shape
    # will we do an FFT or not?
    propinfo = osys._propagation_info()
    if 'FFT' in propinfo['steps']:
        wavefrontsize = wfshape[0] * wfshape[1] * osys.oversample ** 2 * 16  # 16 bytes = complex double size
        _log.debug('FFT propagation with array={0}, oversample = {1} uses {2} bytes'.format(wfshape[0], osys.oversample,
                                                                                            wavefrontsize))
        # The following is a very rough estimate
        # empirical tests show that an 8192x8192 propagation results in Python sessions with ~4 GB memory used w/ FFTW
        # usingg mumpy FT, the memory usage per process can exceed 5 GB for an 8192x8192 propagation.
        padding_factor = 4 if conf.use_fftw else 5
    else:
        # oversampling not relevant for memory size in MFT mode
        wavefrontsize = wfshape[0] * wfshape[1] * 16  # 16 bytes = complex double size
        _log.debug('MFT propagation with array={0} uses {2} bytes'.format(wfshape[0], osys.oversample, wavefrontsize))
        padding_factor = 1

    mem_per_prop = wavefrontsize * padding_factor
    mem_per_output = propinfo['output_size'] * 8

    # total memory needed is the sum of memory for the propagation plus memory to hold the results
    # avail_ram = psutil.phymem_usage().total * memory_fraction
    avail_ram = psutil.virtual_memory().available
    avail_ram -= 2 * 1024. ** 3  # always leave at least 2 GB extra  - let's be cautious to make sure we don't swap.
    recommendation = int(np.floor(float(avail_ram) / (mem_per_prop + mem_per_output)))

    if recommendation > psutil.cpu_count():
        recommendation = psutil.cpu_count()
    if nwavelengths is not None:
        if recommendation > nwavelengths:
            recommendation = nwavelengths

    _log.info("estimated optimal # of processes is {0}".format(recommendation))
    return recommendation


def fftw_save_wisdom(filename=None):
    """ Save accumulated FFTW wisdom to a file

    By default this file will be in the user's astropy configuration directory.
    (Another location could be chosen - this is simple and works easily cross-platform.)

    Parameters
    ------------
    filename : string, optional
        Filename to use (instead of the default, poppy_fftw_wisdom.json)
    """

    from .accel_math import _FFTW_INIT
    if filename is None:
        filename = os.path.join(config.get_config_dir(), "poppy_fftw_wisdom.json")

    # PyFFTW exports as bytestrings, but `json` uses only real strings in Python 3.x+
    double, single, longdouble = pyfftw.export_wisdom()
    wisdom = {
        'double': double.decode('ascii'),
        'single': single.decode('ascii'),
        'longdouble': longdouble.decode('ascii'),
        '_FFTW_INIT': pickle.dumps(_FFTW_INIT, protocol=0).decode('ascii')
        # ugly to put a pickled string inside JSON
        # but native JSON turns tuples into lists and we need to
        # preserve tuple-ness for use in fftw_load_wisdom
        # edit: try saving entire dict instead of just keys for py3 compat
    }

    with open(filename, 'w') as wisdom_file:
        json.dump(wisdom, wisdom_file)
    _log.debug("FFTW wisdom saved to " + filename)


def fftw_load_wisdom(filename=None):
    """Read accumulated FFTW wisdom previously saved in previously saved in a file

    By default this file will be in the user's astropy configuration directory.
    (Another location could be chosen - this is simple and works easily cross-platform.)

    Parameters
    ------------
    filename : string, optional
        Filename to use (instead of the default, poppy_fftw_wisdom.json)
    """
    from .accel_math import _FFTW_INIT
    global _loaded_fftw_wisdom
    if _loaded_fftw_wisdom:
        _log.debug("Already loaded wisdom prior to this calculation, not reloading.")
        return
    if filename is None:
        filename = os.path.join(config.get_config_dir(), "poppy_fftw_wisdom.json")

    if not os.path.exists(filename):
        return  # No wisdom yet, but that's not an error

    _log.debug("Trying to reload wisdom from file " + filename)
    with open(filename) as wisdom_file:
        try:
            wisdom = json.load(wisdom_file)
        except ValueError:  # catches json.JSONDecodeError on Python 3.x too
            warnings.warn("Unable to parse FFTW wisdom in {}. "
                          "The file may be corrupt.".format(filename), FFTWWisdomWarning)
            return

    # Python 3.x+ doesn't let us use ascii implicitly, but PyFFTW only accepts bytestrings
    # in this version...
    wisdom_tuple = (wisdom['double'].encode('ascii'),
                    wisdom['single'].encode('ascii'),
                    wisdom['longdouble'].encode('ascii'))

    success_double, success_single, success_longdouble = pyfftw.import_wisdom(wisdom_tuple)

    _log.debug("Reloaded double precision wisdom: {}".format(success_double))
    _log.debug("Reloaded single precision wisdom: {}".format(success_single))
    _log.debug("Reloaded longdouble precision wisdom: {}".format(success_longdouble))

    try:
        saved_fftw_init = pickle.loads(wisdom['_FFTW_INIT'].encode('ascii'))
        for key in saved_fftw_init.keys():
            _FFTW_INIT[key] = True
        _log.debug("Reloaded _FFTW_INIT list of optimized array sizes ")
    except (TypeError, KeyError, AttributeError):

        _log.warning(
            "Could not parse saved _FFTW_INIT info; this is OK but FFTW will need to repeat its " +
            "optimization measurements (automatically). ")

    _loaded_fftw_wisdom = True
