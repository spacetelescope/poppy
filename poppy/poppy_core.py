from __future__ import (absolute_import, division, print_function, unicode_literals)
import multiprocessing
import copy
import time
import enum
import warnings
import textwrap

import six

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
import matplotlib

import astropy.io.fits as fits
import astropy.units as u

from .matrixDFT import MatrixFourierTransform
from . import utils
from . import conf

import logging
_log = logging.getLogger('poppy')

__all__ = ['Wavefront',  'OpticalSystem', 'SemiAnalyticCoronagraph', 'MatrixFTCoronagraph',
           'OpticalElement', 'FITSOpticalElement', 'Rotation', 'Detector']

# Setup infrastructure for FFTW
_FFTW_INIT = {}  # dict of array sizes for which we have already performed the required FFTW planning step
_FFTW_FLAGS = ['measure']
try:
    # try to import FFTW to see if it is available
    import pyfftw
    _FFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    _FFTW_AVAILABLE = False


# internal constants for types of plane
class PlaneType(enum.Enum):
    unspecified = 0
    pupil = 1               # pupil plane
    image = 2               # image plane
    detector = 3
    rotation = 4            # coordinate system rotation
    intermediate = 5        # arbitrary plane between pupil and image
    inversion = 6           # coordinate system inversion (flip axes, e.g. like going through focus)

_PUPIL = PlaneType.pupil
_IMAGE = PlaneType.image
_DETECTOR = PlaneType.detector  # specialized type of image plane
_ROTATION = PlaneType.rotation  # not a real optic, just a coordinate transform
_INTERMED = PlaneType.intermediate  # for Fresnel propagation

_RADIANStoARCSEC = 180.*60*60 / np.pi



def _wrap_propagate_for_multiprocessing(args):
    """ This is an internal helper routine for parallelizing computations across multiple processors.

    Python's multiprocessing module allows easy execution of tasks across
    many CPUs or even distinct machines. It relies on Python's pickle mechanism to
    serialize and pass objects between processes. One side effect of this is
    that object instance methods cannot be pickled on their own, and thus cannot be easily
    invoked in other processes.

    Here, we work around that by pickling the entire object and argument list, packed
    as a tuple, transmitting that to the new process, and then unpickling that,
    unpacking the results, and *then* at last making our instance method call.
    """
    optical_system, wavelength, retain_intermediates, normalize, usefftwflag = args
    conf.use_fftw = usefftwflag  # passed in from parent process

    # we're in a different Python interpreter process so we
    # need to load the wisdom here too
    if conf.use_fftw and _FFTW_AVAILABLE:
        utils._loaded_fftw_wisdom = False
        utils.fftw_load_wisdom()

    return optical_system.propagate_mono(wavelength, retain_intermediates=retain_intermediates, normalize=normalize)


class Wavefront(object):
    """ A class representing a monochromatic wavefront that can be transformed between
    pupil and image planes (but not to intermediate planes, yet).

    In a pupil plane, a wavefront object `wf` has

        * `wf.diam`,         a diameter in meters
        * `wf.pixelscale`,   a scale in meters/pixel

    In an image plane, it has

        * `wf.fov`,          a field of view in arcseconds
        * `wf.pixelscale`,   a  scale in arcsec/pixel


    Use the `wf.propagate_to()` method to transform a wavefront between conjugate planes. This will update those
    properties as appropriate.

    By default, `Wavefronts` are created in a pupil plane. Set `pixelscale=#` to make an image plane instead.

    Parameters
    ----------
    wavelength : float
        Wavelength of light in meters
    npix : int
        Size parameter for wavefront array to create, per side.
    diam : float, optional
        For _PUPIL wavefronts, sets physical size corresponding to npix. Units are meters.
        At most one of diam or pixelscale should be set when creating a wavefront.
    pixelscale : float, optional
        For _IMAGE PLANE wavefronts, use this pixel scale.
    oversample : int, optional
        how much to oversample by in FFTs. Default is 2.
        Note that final propagations to Detectors use a different algorithmg
        and, optionally, a separate oversampling factor.
    dtype : numpy.dtype, optional
        default is double complex.

    """

    @utils.quantity_input(wavelength=u.meter, diam=u.meter, pixelscale=u.arcsec/u.pixel)
    def __init__(self, wavelength=2e-6*u.meter, npix=1024, dtype=np.complex128, diam=8.0*u.meter,
                 oversample=2, pixelscale=None):

        self._last_transform_type = None  # later used to track MFT vs FFT pixel coord centering in coordinates()
        self.oversample = oversample

        self.wavelength = wavelength
        """Wavelength in meters (or other unit if specified)"""

        self.diam = diam          # pupil plane size in meters
        """Diameter in meters. Applies to a pupil plane only."""
        self.fov = None                                     # image plane size in arcsec
        """Field of view in arcsec. Applies to an image plane only."""
        self.pixelscale = None
        "Pixel scale, in arcsec/pixel or meters/pixel depending on plane type"

        if pixelscale is None:
            self.pixelscale = self.diam / (npix*u.pixel)        # scale in meters/pix or arcsec/pix, as appropriate
            self.planetype = _PUPIL                             # are we at image or pupil?
        else:
            self.pixelscale = pixelscale                        # scale in meters/pix or arcsec/pix, as appropriate
            self.planetype = _IMAGE
        self._image_centered = 'array_center'                   # one of 'array_center', 'pixel', 'corner'
                                                                # This records where the coordinate origin is
                                                                # in image planes, and depends on how the imageg
                                                                # plane was produced (e.g. FFT implies pixel)
        "Are FT'ed image planes centered on a pixel or on a corner between pixels? "
        self.wavefront = np.ones((npix, npix), dtype=dtype)   # the actual complex wavefront array
        self.ispadded = False                               # is the wavefront padded for oversampling?
        self.history = []
        "List of strings giving a descriptive history of actions performed on the wavefront. Saved to FITS headers."
        self.history.append("Created wavefront: wavelength={0}, diam={1}".format(self.wavelength, self.diam))
        self.history.append(" using array size %s" % (self.wavefront.shape,))
        self.location = 'Entrance Pupil'
        "Descriptive string for where a wavefront is instantaneously located. Used mostly for titling displayed plots."

    def __str__(self):
        # TODO add switches for image/pupil planes
        return """Wavefront:
        wavelength = {}
        shape = {}
        sampling = {}""".format(self.wavelength.to(u.micron), self.wavefront.shape, self.pixelscale)

    def copy(self):
        """Return a copy of the wavefront as a different object."""
        return copy.deepcopy(self)

    def normalize(self):
        """Set this wavefront's total intensity to 1 """
        self.wavefront /= np.sqrt(self.total_intensity)

    def __imul__(self, optic):
        """Multiply a Wavefront by an OpticalElement or scalar"""
        if isinstance(optic, CoordinateTransform):
            return self     # a coord transform doesn't actually affect the wavefront via multiplication,
                            # but instead via forcing a call to rotate() or invert() in propagate_to...
        elif (isinstance(optic, float)) or isinstance(optic, int):
            self.wavefront *= optic  # it's just a scalar
            self.history.append("Multiplied WF by scalar value " + str(optic))
            return self

        if not isinstance(optic, OpticalElement):
            raise ValueError('Wavefronts can only be *= multiplied by OpticalElements or scalar values')

        if isinstance(optic, Detector):
            # detectors don't modify a wavefront.
            return self

        phasor = optic.get_phasor(self)

        if not np.isscalar(phasor) and phasor.size > 1:
            # actually isscalar() does not handle the case of a 1-element array properly
            assert self.wavefront.shape == phasor.shape

        self.wavefront *= phasor
        msg = "  Multiplied WF by phasor for "+str(optic)
        _log.debug(msg)
        self.history.append(msg)
        self.location = 'after '+optic.name
        return self

    def __mul__(self, optic):
        """ Multiply a wavefront by an OpticalElement or scalar """
        new = self.copy()
        new *= optic
        return new
    __rmul__ = __mul__  # either way works.

    def __iadd__(self, wave):
        """Add another wavefront to this one"""
        if not isinstance(wave, Wavefront):
            raise ValueError('Wavefronts can only be summed with other Wavefronts')

        if not self.wavefront.shape[0] == wave.wavefront.shape[0]:
            raise ValueError('Wavefronts can only be added if they have the same size and shape')

        self.wavefront += wave.wavefront
        self.history.append("Summed with another wavefront!")
        return self

    def __add__(self, wave):
        new = self.copy()
        new += wave
        return new

    def as_fits(self, what='intensity', includepadding=False, **kwargs):
        """ Return a wavefront as a pyFITS HDUList object

        Parameters
        -----------
        what : string
            what kind of data to write. Must be one of 'all', 'parts', 'intensity', or 'complex'.
            The default is to write a file containing intensity, amplitude, and phase in a data cube
            of shape (3, N, N). 'parts' omits intensity and produces a (2, N, N) array.
            'intensity' and 'phase' write out 2D arrays with the corresponding values.
        includepadding : bool
            include any "padding" region, if present, in the returned FITS file?
        """

        def get_unpadded(attribute_array):
            if self.planetype == _PUPIL and self.ispadded and not includepadding:
                return utils.removePadding(attribute_array.copy(), self.oversample)
            else:
                return attribute_array.copy()

        if what.lower() == 'all':
            intens = get_unpadded(self.intensity)
            outarr = np.zeros((3, intens.shape[0], intens.shape[1]))
            outarr[0, :, :] = intens
            outarr[1, :, :] = get_unpadded(self.amplitude)
            outarr[2, :, :] = get_unpadded(self.phase)
            outFITS = fits.HDUList(fits.PrimaryHDU(outarr))
            outFITS[0].header['PLANE1'] = 'Wavefront Intensity'
            outFITS[0].header['PLANE2'] = 'Wavefront Amplitude'
            outFITS[0].header['PLANE3'] = 'Wavefront Phase'
        elif what.lower() == 'parts':
            amp = get_unpadded(self.amplitude)
            outarr = np.zeros((2, amp.shape[0], amp.shape[1]))
            outarr[0, :, :] = amp
            outarr[1, :, :] = get_unpadded(self.phase)
            outFITS = fits.HDUList(fits.PrimaryHDU(outarr))
            outFITS[0].header['PLANE1'] = 'Wavefront Amplitude'
            outFITS[0].header['PLANE2'] = 'Wavefront Phase'
        elif what.lower() == 'intensity':
            outFITS = fits.HDUList(fits.PrimaryHDU(get_unpadded(self.intensity)))
            outFITS[0].header['PLANE1'] = 'Wavefront Intensity'
        elif what.lower() == 'phase':
            outFITS = fits.HDUList(fits.PrimaryHDU(get_unpadded(self.phase)))
            outFITS[0].header['PLANE1'] = 'Phase'
        else:
            raise ValueError("Unknown string for what to return: "+what)

        outFITS[0].header['WAVELEN'] = (self.wavelength.to(u.meter).value, 'Wavelength in meters')
        outFITS[0].header['DIFFLMT'] = ((self.wavelength/self.diam*u.radian).to(u.arcsec).value,
                                         'Diffraction limit lambda/D in arcsec')
        outFITS[0].header['OVERSAMP'] = (self.oversample, 'Oversampling factor for FFTs in computation')
        outFITS[0].header['DET_SAMP'] = (self.oversample, 'Oversampling factor for MFT to detector plane')
        if self.planetype ==_IMAGE:
            outFITS[0].header['PIXELSCL'] = (self.pixelscale.to(u.arcsec/u.pixel).value,
                                             'Scale in arcsec/pix (after oversampling)')
            fov_arcsec = self.fov.to(u.arcsec).value
            if np.isscalar(fov_arcsec):
                outFITS[0].header['FOV'] = (fov_arcsec, 'Field of view in arcsec (full array)')
            else:
                outFITS[0].header['FOV_X'] = (fov_arcsec[1], 'Field of view in arcsec (full array), X direction')
                outFITS[0].header['FOV_Y'] = (fov_arcsec[0], 'Field of view in arcsec (full array), Y direction')
        else:
            outFITS[0].header['PIXELSCL'] = (self.pixelscale.to(u.meter/u.pixel).value, 'Pixel scale in meters/pixel')
            outFITS[0].header['DIAM'] =  (self.diam.to(u.meter).value, 'Pupil diameter in meters (not incl padding)')

        for h in self.history:
            outFITS[0].header.add_history(h)

        return outFITS

    def writeto(self, filename, clobber=True, **kwargs):
        """Write a wavefront to a FITS file.

        Parameters
        -----------
        filename : string
            filename to use
        what : string
            what to write. Must be one of 'parts', 'intensity', 'complex'
        clobber : bool, optional
            overwhat existing? default is True

        Returns
        -------
        outfile: file on disk
            The output is written to disk.

        """
        self.as_fits(**kwargs).writeto(filename, clobber=clobber)
        _log.info("  Wavefront saved to %s" % filename)

    def display(self, what='intensity', nrows=1, row=1, showpadding=False,
                imagecrop=None, pupilcrop=None,
                colorbar=False, crosshairs=False, ax=None, title=None, vmin=None,
                vmax=None, scale=None, use_angular_coordinates=None):
        """Display wavefront on screen

        Parameters
        ----------
        what : string
           What to display. Must be one of {intensity, phase, best}.
           'Best' implies to display the phase if there is nonzero OPD,
           or else display the intensity for a perfect pupil.
        nrows : int
            Number of rows to display in current figure (used for
            showing steps in a calculation)
        row : int
            Which row to display this one in?
        vmin, vmax : floats
            min and maximum values to display. When left unspecified, these default
            to [0, intens.max()] for linear (scale='linear') intensity plots,
            [1e-6*intens.max(), intens.max()] for logarithmic (scale='log') intensity
            plots, and [-0.25, 0.25] waves for phase plots.
        scale : string
            'log' or 'linear', to define the desired display scale type for
            intensity. Default is log for image planes, linear otherwise.
        imagecrop : float, optional
            Crop the displayed image to a smaller region than the full array.
            For image planes in angular coordinates, this is given in units of
            arcseconds. The default is 5, so only the innermost 5x5 arcsecond
            region will be shown. This default may be changed in the
            POPPY config file. If the image size is < 5 arcsec then the
            entire image is displayed.
            For planes in linear physical coordinates such as pupils, this
            is given in units of meters, and the default is no cropping
            (i.e. the entire array will be displayed unless this keyword
            is set explicitly).
        showpadding : bool, optional
            For wavefronts that have been padded with zeros for oversampling,
            show the entire padded arrays, or just the good parts?
            Default is False, to show just the central region of interest.
        colorbar : bool
            Display colorbar
        crosshairs : bool
            Display a crosshairs indicator showing the axes centered on (0,0)
        ax : matplotlib Axes, optional
            axes to display into. If not set, will create new axes.
        use_angular_coordinates : bool, optional
            Should the axes be labeled in angular units of arcseconds?
            This is used by FresnelWavefront, where non-angular
            coordinates are possible everywhere. When using Fraunhofer
            propagation, this should be left as None so that the
            coordinates are inferred from the planetype attribute.
            (Default: None, infer coordinates from planetype)

        Returns
        -------
        figure : matplotlib figure
            The current figure is modified.
        """
        if scale is None:
            scale = 'log' if self.planetype == _IMAGE else 'linear'

        intens = self.intensity.copy()

        # make a version of the phase where we try to mask out
        # areas with particularly low intensity
        phase = self.phase.copy()
        mean_intens = np.mean(intens[intens !=0])
        phase[np.where(intens < mean_intens/100)] = np.nan
        amp = self.amplitude

        y, x = self.coordinates()
        if self.planetype == _PUPIL and self.ispadded and not showpadding:
            intens = utils.removePadding(intens, self.oversample)
            phase = utils.removePadding(phase, self.oversample)
            amp = utils.removePadding(amp, self.oversample)
            y = utils.removePadding(y, self.oversample)
            x = utils.removePadding(x, self.oversample)

        # extent specifications need to include the *full* data region, including the half pixel
        # on either side outside of the pixel center coordinates.  And remember to swap Y and X.
        # Recall that for matplotlib,
        #    extent = [xmin, xmax, ymin, ymax]
        # in this case those are coordinates in units of pixels. Recall that we define pixel
        # coordinates to be at the *center* of the pixel, so we compute here the coordinates at the
        # outside of those pixels.
        # This is needed to get the coordinates right when displaying very small arrays

        halfpix = self.pixelscale.value*0.5
        extent = [x.min()-halfpix, x.max()+halfpix, y.min()-halfpix, y.max()+halfpix]

        if use_angular_coordinates is None:
            use_angular_coordinates = self.planetype == _IMAGE

        unit = 'arcsec' if use_angular_coordinates else 'm'

        # implement semi-intellegent selection of what to display, if the user wants
        if what == 'best':
            if self.planetype == _IMAGE:
                what = 'intensity'  # always show intensity for image planes
            elif phase[np.where(np.isfinite(phase))].sum() == 0:
                what = 'intensity'  # for perfect pupils
            # FIXME re-implement this in some better way that doesn't depend on
            # optic positioning in the plot grid!
            # elif int(row) > 2:
                #what = 'intensity'  # show intensity for coronagraphic downstream propagation.
            else:
                what = 'phase'  # for aberrated pupils

        # compute plot parameters for the subplot grid
        nc = int(np.ceil(np.sqrt(nrows)))
        nr = int(np.ceil(float(nrows) / nc))
        if (nrows - nc * (nc - 1) == 1) and (nr > 1):  # avoid just one alone on a row by itself...
            nr -= 1
            nc += 1

        # prepare color maps and normalizations for intensity and phase
        if vmax is None:
            if what == 'phase':
               vmax = 0.25
            else:
               vmax = intens.max()
        if scale == 'linear':
            if vmin is None:
                if what == 'phase':
                    vmin = -0.25
                else:
                    vmin = 0
            norm_inten = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap_inten = getattr(matplotlib.cm, conf.cmap_pupil_intensity)
            cmap_inten.set_bad('0.0')
        else:
            if vmin is None:
                vmin = vmax*1e-6
            norm_inten = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            cmap_inten = getattr(matplotlib.cm, conf.cmap_sequential)
            cmap_inten.set_bad(cmap_inten(0))
        cmap_phase = getattr(matplotlib.cm, conf.cmap_diverging)
        cmap_phase.set_bad('0.3')
        norm_phase = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


        def wrap_lines_title(title):
            # Helper fn to add line breaks, tweaked to
            # put in particular places for aesthetics
            for prep in ['after','before']:
                if prep in title:
                    part1, part2 = title.split(prep)
                    return part1+prep+"\n"+"\n".join(textwrap.wrap(part2,30))
            return"\n".join(textwrap.wrap(title,30))



        # now display the chosen selection..
        if what == 'intensity':
            if ax is None:
                ax = plt.subplot(nr, nc, int(row))

            utils.imshow_with_mouseover(
                intens,
                ax=ax,
                extent=extent,
                norm=norm_inten,
                cmap=cmap_inten,
                origin='lower'
            )
            if title is None:
                title = wrap_lines_title("Intensity " + self.location)
            ax.set_title(title)
            ax.set_xlabel(unit)
            if colorbar:
                plt.colorbar(ax.images[0], orientation='vertical', shrink=0.8)
            plot_axes = [ax]
            to_return = ax
        elif what == 'phase':
            # Display phase in waves.
            if ax is None:
                ax = plt.subplot(nr, nc, int(row))
            utils.imshow_with_mouseover(
                phase / (np.pi * 2),
                ax=ax,
                extent=extent,
                norm=norm_phase,
                cmap=cmap_phase,
                origin='lower'
            )
            if title is None:
                title = wrap_lines_title("Phase " + self.location)
            plt.title(title)
            plt.xlabel(unit)
            if colorbar:
                plt.colorbar(ax.images[0], orientation='vertical', shrink=0.8)

            plot_axes = [ax]
            to_return = ax

        elif what == 'both':
            ax1 = plt.subplot(nrows, 2, (row * 2) - 1)
            plt.imshow(amp, extent=extent, cmap=cmap_inten, norm=norm_inten, origin='lower')
            plt.title("Wavefront amplitude")
            plt.ylabel(unit)
            plt.xlabel(unit)

            if colorbar:
                plt.colorbar(orientation='vertical', shrink=0.8)

            ax2 = plt.subplot(nrows, 2, row * 2)
            plt.imshow(phase, extent=extent, cmap=cmap_phase, norm=norm_phase, origin='lower')
            if colorbar:
                plt.colorbar(orientation='vertical', shrink=0.8)

            plt.xlabel(unit)
            plt.title("Wavefront phase [radians]")

            plot_axes = [ax1, ax2]
            to_return = (ax1, ax2)
        else:
            raise ValueError("Invalid value for what to display; must be 'amplitude', 'phase', or 'both'.")

        # now apply axes cropping and/or overplots, if requested.
        for ax in plot_axes:
            if crosshairs:
                ax.axhline(0, ls=":", color='white')
                ax.axvline(0, ls=":", color='white')

            if use_angular_coordinates:
                if imagecrop is None:
                    imagecrop = conf.default_image_display_fov

            if imagecrop is not None:
                cropsize_x = min((imagecrop/2, intens.shape[1]/2.*self.pixelscale.value))
                cropsize_y = min((imagecrop/2, intens.shape[0]/2.*self.pixelscale.value))
                ax.set_xbound(-cropsize_x, cropsize_x)
                ax.set_ybound(-cropsize_y, cropsize_y)

            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

        plt.draw()
        return to_return

    # add convenient properties for intensity, phase, amplitude, total_flux
    @property
    def amplitude(self):
        """Electric field amplitude of the wavefront """
        return np.abs(self.wavefront)

    @property
    def intensity(self):
        """Electric field intensity of the wavefront (i.e. field amplitude squared)"""
        return np.abs(self.wavefront)**2

    @property
    def phase(self):
        """Phase of the wavefront, in radians"""
        return np.angle(self.wavefront)

    @property
    def shape(self):
        """ Shape of the wavefront array"""
        return self.wavefront.shape

    @property
    def total_intensity(self):
        """Integrated intensity over the entire spatial/angular extent of the wavefront"""
        return self.intensity.sum()

    # methods for wavefront propagation:
    def propagate_to(self, optic):
        """Propagates a wavefront object to the next optic in the list.
        Modifies this wavefront object itself.

        Transformations between pupil and detector planes use MFT or inverse MFT.
        Transformations between pupil and other (non-detector) image planes use FFT or inverse FFT.
        Transformations from any frame through a rotation plane simply rotate the wavefront accordingly.

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.
        """
        if self.planetype == optic.planetype:
            _log.debug("  Wavefront and optic %s already at same plane type, no propagation needed." % optic.name)
            return
        else:
            msg = "  Propagating wavefront to %s. " % str(optic)
            _log.debug(msg)
            self.history.append(msg)
        _log.debug("conf.use_fftw is "+str(conf.use_fftw))

        if optic.planetype == _ROTATION:     # rotate
            self.rotate(optic.angle)
            self.location = 'after '+optic.name
        elif optic.planetype == PlaneType.inversion:     # invert coordinates
            self.invert(axis=optic.axis)
            self.location = 'after '+optic.name
        elif optic.planetype == _DETECTOR and self.planetype == _PUPIL:    # MFT pupil to detector
            self._propagate_mft(optic)
            self.location = 'before '+optic.name
        elif optic.planetype == _PUPIL and self.planetype == _IMAGE and self._last_transform_type == 'MFT':
            # inverse MFT detector to pupil
            # n.b. transforming _PUPIL -> _DETECTOR results in self.planetype == _IMAGE
            # while setting _last_transform_type to MFT
            self._propagate_mft_inverse(optic)
            self.location = 'before '+optic.name
        elif self.planetype == _IMAGE and optic.planetype == _DETECTOR:
            raise NotImplementedError('image plane directly to detector propagation (resampling!) not implemented yet')
        else:
            self._propagate_fft(optic)           # FFT pupil to image or image to pupil
            self.location = 'before '+optic.name

    def _propagate_fft(self, optic):
        """ Propagate from pupil to image or vice versa using a padded FFT

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.

        """
        # To use FFTW, it must both be enabled and the library itself has to be present
        _USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE)

        if self.oversample > 1 and not self.ispadded:  # add padding for oversampling, if necessary
            assert self.oversample == optic.oversample
            self.wavefront = utils.pad_to_oversample(self.wavefront, self.oversample)
            self.ispadded = True
            if optic.verbose:
                _log.debug("    Padded WF array for oversampling by %dx" % self.oversample)
            self.history.append("    Padded WF array for oversampling by %dx" % self.oversample)

        method = 'pyfftw' if _USE_FFTW  else 'numpy'  # for logging
        _log.info("using {1} FFT of {0} array".format(str(self.wavefront.shape), method))
        # Set up for computation - figure out direction & normalization
        if self.planetype == _PUPIL and optic.planetype == _IMAGE:
            FFT_direction = 'forward'
            normalization_factor = 1./ self.wavefront.shape[0] # correct for numpy fft

            do_fft = pyfftw.interfaces.numpy_fft.fft2 if _USE_FFTW else np.fft.fft2

            #(pre-)update state:
            self.planetype=_IMAGE
            self.pixelscale = (self.wavelength / self.diam * u.radian / self.oversample).to(u.arcsec) / u.pixel
            self.fov = self.wavefront.shape[0] * u.pixel * self.pixelscale
            self.history.append('   FFT {},  to _IMAGE  scale={}'.format(self.wavefront.shape, self.pixelscale))

        elif self.planetype == _IMAGE and optic.planetype ==_PUPIL:
            FFT_direction = 'backward'
            normalization_factor =  self.wavefront.shape[0]  # correct for numpy fft
            do_fft = pyfftw.interfaces.numpy_fft.ifft2 if _USE_FFTW else np.fft.ifft2

            #(pre-)update state:
            self.planetype=_PUPIL
            self.pixelscale = self.diam *self.oversample / (self.wavefront.shape[0]*u.pixel)
            self.history.append('   FFT {},  to PUPIL scale={}'.format(self.wavefront.shape, self.pixelscale))

        # do FFT
        if conf.enable_flux_tests: _log.debug("\tPre-FFT total intensity: "+str(self.total_intensity))
        if conf.enable_speed_tests: t0 = time.time()

        if FFT_direction =='backward': self.wavefront = np.fft.ifftshift(self.wavefront)

        _log.debug("using {2} FFT of {0} array, direction={1}".format(str(self.wavefront.shape), FFT_direction, method))
        if _USE_FFTW:
            if (self.wavefront.shape, FFT_direction) not in _FFTW_INIT:
                # The first time you run FFTW to transform a given size, it does a speed test to
                # determine optimal algorithm that is destructive to your chosen array.
                # So only do that test on a copy, not the real array:
                _log.info("Evaluating PyFFT optimal algorithm for %s, direction=%s" % (
                    str(self.wavefront.shape), FFT_direction))

                pyfftw.interfaces.cache.enable()
                pyfftw.interfaces.cache.set_keepalive_time(30)

                test_array = np.zeros(self.wavefront.shape)
                test_array = do_fft(test_array, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                    threads=multiprocessing.cpu_count())

                _FFTW_INIT[(self.wavefront.shape, FFT_direction)] = True

            self.wavefront = do_fft(self.wavefront, overwrite_input=True, planner_effort='FFTW_MEASURE',
                                    threads=multiprocessing.cpu_count())
        else:
            self.wavefront = do_fft(self.wavefront)

        if FFT_direction == 'forward':
            self.wavefront = np.fft.fftshift(self.wavefront)
            # FFT produces pixel-centered images by default, unless the _image_centered param
            # has already been set by an FQPM_FFT_aligner class
            if self._image_centered != 'corner':
                self._image_centered = 'pixel'
        self.wavefront = self.wavefront * normalization_factor
        self._last_transform_type = 'FFT'

        if conf.enable_speed_tests:
            t1 = time.time()
            _log.debug("\tTIME %f s\t for the FFT" % (t1-t0))

        if conf.enable_flux_tests:
            _log.debug("\tPost-FFT total intensity: "+str(self.total_intensity))

    def _propagate_mft(self, det):
        """ Compute from pupil to an image using the Soummer et al. 2007 MFT algorithm

        Parameters
        -----------
        det : OpticalElement, must be of type DETECTOR
            The target optical plane to propagate to."""

        assert self.planetype == _PUPIL
        assert det.planetype == _DETECTOR

        if self.ispadded:
            # pupil plane is padded - trim that out since it's not needed
            self.wavefront = utils.removePadding(self.wavefront, self.oversample)
            self.ispadded = False
        self._preMFT_pupil_shape =self.wavefront.shape   # save for possible inverseMFT
        self._preMFT_pupil_pixelscale = self.pixelscale  # save for possible inverseMFT

        # the arguments for the matrixDFT are
        # - wavefront (assumed to fill the input array)
        # - focal plane size in lambda/D units
        # - number of pixels on a side in focal plane array.

        # extract everything from Quantities to regular scalars here
        lamD = ((self.wavelength / self.diam)*u.radian).to(u.arcsec).value

        det_fov_lamD = det.fov_arcsec.to(u.arcsec).value / lamD
        det_calc_size_pixels = det.fov_pixels.to(u.pixel).value * det.oversample

        mft = MatrixFourierTransform(centering='ADJUSTABLE', verbose=False)
        if not np.isscalar(det_fov_lamD): #hasattr(det_fov_lamD,'__len__'):
            msg= '    Propagating w/ MFT: {:.4f}     fov=[{:.3f},{:.3f}] lam/D    npix={} x {}'.format(
                    det.pixelscale/det.oversample, det_fov_lamD[0], det_fov_lamD[1],
                    det_calc_size_pixels[0], det_calc_size_pixels[1])
        else:
            msg= '    Propagating w/ MFT: {:.4f}     fov={:.3f} lam/D    npix={:d}'.format(
                    det.pixelscale/det.oversample, det_fov_lamD, int(det_calc_size_pixels))
        _log.debug(msg)
        self.history.append(msg)
        det_offset = det.det_offset if hasattr(det, 'det_offset') else (0,0)


        _log.debug('      MFT method = '+mft.centering)

        # det_offset controls how to shift the PSF.
        # it gives the coordinates (X, Y) relative to the exact center of the array
        # for the location of the phase center of a converging perfect spherical wavefront.
        # This is where a perfect PSF would be centered. Of course any tilts, comas, etc, from the OPD
        # will probably shift it off elsewhere for an entirely different reason, too.
        self.wavefront = mft.perform(self.wavefront, det_fov_lamD, det_calc_size_pixels, offset=det_offset)
        _log.debug("     Result wavefront: at={0} shape={1} intensity={2:.3g}".format(
            self.location, str(self.shape), self.total_intensity))
        self._last_transform_type = 'MFT'

        self.planetype=_IMAGE
        self.fov = det.fov_arcsec
        self.pixelscale = det.fov_arcsec / det_calc_size_pixels / u.pixel

        if not np.isscalar(self.pixelscale.value):
            # check for rectangular arrays
            if self.pixelscale[0] == self.pixelscale[1]:
                self.pixelscale = self.pixelscale[0]
                # we're in a rectangular array with same scale in both directions, so treat pixelscale as a scalar
            else:
                raise NotImplementedError(
                    'Different pixel scales in X and Y directions (i.e. non-square pixels) not yet supported.')

    def _propagate_mft_inverse(self, pupil, pupil_npix=None):
        """ Compute from an image to a pupil using the Soummer et al. 2007 MFT algorithm
        This allows transformation back from an arbitrarily-sampled 'detector' plane to a pupil.

        This is only used if transforming back from a 'detector' type plane to a pupil, for instance
        inside the semi-analytic coronagraphy algorithm, but is not used in more typical propagations.

        """

        assert self.planetype == _IMAGE
        assert pupil.planetype == _PUPIL

        # the arguments for the matrixDFT are
        # - wavefront (assumed to fill the input array)
        # - focal plane size in lambda/D units
        # - number of pixels on a side in focal plane array.

        # extract everything from Quantities to regular scalars here
        lamD = (self.wavelength / self.diam * u.radian).to(u.arcsec).value

        det_fov_lamD = self.fov.to(u.arcsec).value / lamD

        # try to transform to whatever the intrinsic scale of the next pupil is.
        # but if this ends up being a scalar (meaning it is an AnalyticOptic) then
        # just go back to our own prior shape and pixel scale.
        if pupil_npix is None:
            if pupil.shape is not None and pupil.shape[0] != 1:
                pupil_npix = pupil.shape[0]
            else:
                pupil_npix = self._preMFT_pupil_shape[0]

        mft = MatrixFourierTransform(centering='ADJUSTABLE', verbose=False)

        # these can be either scalar or 2-element lists/tuples/ndarrays
        msg_pixscale = ('{0:.4f}'.format(self.pixelscale) if np.isscalar(self.pixelscale.value) else
                        '{0:.4f} x {1:.4f} arcsec/pix'.format(self.pixelscale.value[0], self.pixelscale.value[1]))
        msg_det_fov  = ('{0:.4f} lam/D'.format(det_fov_lamD) if np.isscalar(det_fov_lamD) else
                        '{0:.4f} x {1:.4f}  lam/D'.format(det_fov_lamD[0], det_fov_lamD[1]))

        msg= '    Propagating w/ InvMFT:  scale={0}    fov={1}    npix={2:d} x {2:d}'.format(
            msg_pixscale, msg_det_fov, pupil_npix)
        _log.debug(msg)
        self.history.append(msg)
        # det_offset = (0,0)  # det_offset not supported for InvMFT (yet...)

        self.wavefront = mft.inverse(self.wavefront, det_fov_lamD, pupil_npix)
        self._last_transform_type = 'InvMFT'

        self.planetype = _PUPIL
        self.pixelscale = self.diam / self.wavefront.shape[0] / u.pixel

    def tilt(self, Xangle=0.0, Yangle=0.0):
        """ Tilt a wavefront in X and Y.

        Recall from Fourier optics (although this is straightforwardly rederivable by drawing triangles)
        that for a wavefront tilted by some angle theta in radians, that a point r meters from the center of
        the pupil has:

            extra_pathlength = sin(theta) * r
            extra_waves = extra_pathlength/ wavelength = r * sin(theta) / wavelength

        So we calculate the U and V arrays (corresponding to r for the pupil, in meters from the center)
        and then multiply by the appropriate trig factors for the angle.

        The sign convention is chosen such that positive Yangle tilts move the star upwards in the
        array at the focal plane. (This is sort of an inverse of what physically happens in the propagation
        to or through focus, but we're ignoring that here and trying to just work in sky coords)

        Parameters
        ----------
        Xangle, Yangle : float
            tilt angles, specified in arcseconds

        """
        if self.planetype == _IMAGE:
            raise NotImplementedError("Are you sure you want to tilt a wavefront in an _IMAGE plane?")

        if np.abs(Xangle) > 0 or np.abs(Yangle) > 0:
            xangle_rad = Xangle * (np.pi / 180 / 60 / 60)
            yangle_rad = Yangle * (np.pi / 180 / 60 / 60)

            npix = self.wavefront.shape[0]
            V, U = np.indices(self.wavefront.shape, dtype=float)
            V -= (npix - 1) / 2.0
            V *= self.pixelscale
            U -= (npix - 1) / 2.0
            U *= self.pixelscale

            tiltphasor = np.exp(2.0j * np.pi * (U * xangle_rad + V * yangle_rad) / self.wavelength.to(u.meter).value)
            self.wavefront *= tiltphasor
            self.history.append("Tilted wavefront by "
                                "X={:2.2}, Y={:2.2} arcsec".format(Xangle, Yangle))

        else:
            _log.warn("Wavefront.tilt() called, but requested tilt was zero. No change.")

    def rotate(self, angle=0.0):
        """Rotate a wavefront by some amount, using spline interpolation

        Parameters
        ----------
        angle : float
            Angle to rotate, in degrees counterclockwise.

        """
        # self.wavefront = scipy.ndimage.interpolation.rotate(self.wavefront, angle, reshape=False)
        # Huh, the ndimage rotate function does not work for complex numbers. That's weird.
        # so let's treat the real and imaginary parts individually
        # FIXME TODO or would it be better to do this on the amplitude and phase?
        rot_real = scipy.ndimage.interpolation.rotate(self.wavefront.real, angle, reshape=False)
        rot_imag = scipy.ndimage.interpolation.rotate(self.wavefront.imag, angle, reshape=False)
        self.wavefront = rot_real + 1.j*rot_imag

        self.history.append('Rotated by {:.2f} degrees, CCW'.format(angle))

    def invert(self, axis='both'):
        """Invert coordinates, i.e. flip the direction of the X and Y axes

        This models the inversion of axes signs that happens for instance when a beam
        passes through a focus.

        Parameters
        ------------
        axis : string
            either 'both', 'x', or 'y', for which axes to invert

        """
        if axis.lower() == 'both':
            self.wavefront = self.wavefront[::-1, ::-1]
        elif axis.lower() == 'x':
            self.wavefront = self.wavefront[:, ::-1]
        elif axis.lower() == 'y':
            self.wavefront = self.wavefront[::-1]
        else:
            raise ValueError("Invalid/unknown value for the 'axis' parameter. Must be 'x', 'y', or 'both'.")
        self.history.append('Inverted axis direction for {} axes'.format(axis.upper()))

    # note: the following are implemented as static methods to
    # allow for reuse outside of this class in the Zernike polynomial
    # caching mechanisms. See zernike.py.
    @staticmethod
    def pupil_coordinates(shape, pixelscale):
        """Utility function to generate coordinates arrays for a pupil
        plane wavefront

        Parameters
        ----------

        shape : tuple of ints
            Shape of the wavefront array
        pixelscale : float or 2-tuple of floats
            the pixel scale in meters/pixel, optionally different in
            X and Y
        """
        y, x = np.indices(shape, dtype=float)
        pixelscale_mpix = pixelscale.to(u.meter/u.pixel).value if isinstance(pixelscale, u.Quantity) else pixelscale
        if not np.isscalar(pixelscale_mpix):
            pixel_scale_x, pixel_scale_y = pixelscale_mpix
        else:
            pixel_scale_x, pixel_scale_y = pixelscale_mpix, pixelscale_mpix

        y -= (shape[0] - 1) / 2.0
        x -= (shape[1] - 1) / 2.0

        return pixel_scale_y * y, pixel_scale_x * x

    @staticmethod
    def image_coordinates(shape, pixelscale, last_transform_type, image_centered):
        """Utility function to generate coordinates arrays for an image
        plane wavefront

        Parameters
        ----------

        shape : tuple of ints
            Shape of the wavefront array
        pixelscale : float or 2-tuple of floats
            the pixelscale in meters/pixel, optionally different in
            X and Y
        last_transform_type : string
            Was the last transformation on the Wavefront an FFT
            or an MFT?
        image_centered : string
            Was POPPY trying to keeping the center of the image on
            a pixel, crosshairs ('array_center'), or corner?
        """
        y, x = np.indices(shape, dtype=float)
        pixelscale_arcsecperpix = pixelscale.to(u.arcsec/u.pixel).value
        if not np.isscalar(pixelscale_arcsecperpix):
            pixel_scale_x, pixel_scale_y = pixelscale_arcsecperpix
        else:
            pixel_scale_x, pixel_scale_y = pixelscale_arcsecperpix, pixelscale_arcsecperpix

        # in most cases, the x and y values are centered around the exact center of the array.
        # This is not true in general for FFT-produced image planes where the center is in the
        # middle of one single pixel (the 0th-order term of the FFT), even though that means that
        # the PSF center is slightly offset from the array center.
        # On the other hand, if we used the FQPM FFT Aligner optic, then that forces the PSF center
        # to the exact center of an array.

        # The following are just relevant for the FFT-created images, not for the Detector MFT
        # image at the end.
        if last_transform_type == 'FFT':
            # FFT array sizes will always be even, right?
            if image_centered == 'pixel':
                # so this goes to an integer pixel
                y -= shape[0] / 2.0
                x -= shape[1] / 2.0
            elif image_centered == 'array_center' or image_centered == 'corner':
                # and this goes to a pixel center
                y -= (shape[0] - 1) / 2.0
                x -= (shape[1] - 1) / 2.0
        else:
            # MFT produced images are always exactly centered.
            y -= (shape[0] - 1) / 2.0
            x -= (shape[1] - 1) / 2.0

        return pixel_scale_y * y, pixel_scale_x * x

    def coordinates(self):
        """ Return Y, X coordinates for this wavefront, in the manner of numpy.indices()

        This function knows about the offset resulting from FFTs. Use it whenever computing anything
        measured in wavefront coordinates.

        Returns
        -------
        Y, X :  array_like
            Wavefront coordinates in either meters or arcseconds for pupil and image, respectively
        """

        if self.planetype == _PUPIL:
            return type(self).pupil_coordinates(self.shape, self.pixelscale)
        elif self.planetype == _IMAGE:
            return Wavefront.image_coordinates(self.shape, self.pixelscale,
                                               self._last_transform_type, self._image_centered)
        else:
            raise RuntimeError("Unknown plane type (should be pupil or image!)")

    asFITS = as_fits  # pep8 compliant alias; will be fully renamed in a future version.
    totalIntensity = total_intensity  # pep8 compliant synonym; will be fully renamed in future version
    propagateTo = propagate_to  # pep8 compliant synonym

    _propagateFFT = _propagate_fft  # back compatibility alias for older non-pep8 name. REMOVE IN 0.6 RELEASE
    _propagateMFT = _propagate_mft  # likewise   REMOVE IN 0.6 RELEASE
    _propagateMFTinverse = _propagate_mft_inverse # likewise   REMOVE IN 0.6 RELEASE


#------  Optical System classes -------
class OpticalSystem(object):
    """ A class representing a series of optical elements,
    either Pupil, Image, or Detector planes, through which light
    can be propagated.

    The difference between
    Image and Detector planes is that Detectors have fixed pixels
    in terms of arcsec/pixel regardless of wavelength (computed via
    MFT) while Image planes have variable pixels scaled in terms of
    lambda/D. Pupil planes are some fixed size in meters, of course.

    Parameters
    ----------
    name : string
        descriptive name of optical system
    oversample : int
        Either how many times *above* Nyquist we should be
        (for pupil or image planes), or how many times a fixed
        detector pixel will be sampled. E.g. `oversample=2` means
        image plane sampling lambda/4*D (twice Nyquist) and
        detector plane sampling 2x2 computed pixels per real detector
        pixel.  Default is 2.
    verbose : bool
        whether to be more verbose with log output while computing
    pupil_diameter : astropy.Quantity of dimension length
        Diameter of entrance pupil. Defaults to size of first optical element
        if unspecified, or else 1 meter.


    """
    def __init__(self, name="unnamed system", verbose=True, oversample=2,
            npix=1024, pupil_diameter=None):
        self.name = name
        self.verbose=verbose
        self.planes = []                    # List of OpticalElements
        self.oversample = oversample
        self.npix = npix
        self.pupil_diameter = pupil_diameter

        self.source_offset_r = 0 # = np.zeros((2))     # off-axis tilt of the source, in ARCSEC
        self.source_offset_theta = 0 # in degrees CCW

        self.intermediate_wfs = None        #
        if self.verbose:
            _log.info("Initialized OpticalSystem: "+self.name)

    # Methods for adding or manipulating optical planes:

    def _add_plane(self, optic, index=None, logstring=""):
        """ utility helper function for adding a generic plane """
        if index is None:
            self.planes.append(optic)
        else:
            self.planes.insert(index, optic)
        if self.verbose: _log.info("Added {}: {}".format(logstring,optic.name))
        return optic

    def add_pupil(self, optic=None, function=None, index=None, **kwargs):
        """ Add a pupil plane optic from file(s) giving transmission or OPD

          1) from file(s) giving transmission and/or OPD
                [set arguments `transmission=filename` and/or `opd=filename`]
          2) from an already-created :py:class:`OpticalElement` object
                [set `optic=that object`]

        Parameters
        ----------
        optic : poppy.OpticalElement, optional
            An already-created :py:class:`OpticalElement` object you would like to add
        function : string, optional
            Deprecated. The name of some analytic function you would like to use.
            Optional `kwargs` can be used to set the parameters of that function.
            Allowable function names are Circle, Square, Hexagon, Rectangle, and FQPM_FFT_Aligner
        opd, transmission : string, optional
            Filenames of FITS files describing the desired optic.
        index : int
            Index into the optical system's planes for where to add the new optic. Defaults to
            appending the optic to the end of the plane list.

        Returns
        -------
        poppy.OpticalElement subclass
            The pupil optic added (either `optic` passed in, or a new OpticalElement created)


        Note: Now you can use the optic argument for either an OpticalElement or a string function name,
        and it will do the right thing depending on type.  Both existing arguments are left for compatibility for now.


        Any provided parameters are passed to :py:class:`OpticalElement`.


        """
        if function is not None:
            import warnings
            warnings.warn("The function argument to addPupil is deprecated. Please provide an Optic object instead.",
                          DeprecationWarning)
        if optic is None and function is not None:
            # ease of use: 'function' input and providing 'optic' parameter as a string are synonymous.
            optic = function

        if isinstance(optic, OpticalElement):
            # OpticalElement object provided.
            # We can use it directly, but make sure the plane type is set.
            optic.planetype = _PUPIL
        elif isinstance(optic, six.string_types):
            # convenience code to instantiate objects from a string name.
            raise NotImplementedError('Setting optics based on strings is now deprecated.')
        elif optic is None and len(kwargs) > 0: # create image from files specified in kwargs
            # create image from files specified in kwargs
            optic = FITSOpticalElement(planetype=_PUPIL, oversample=self.oversample, **kwargs)
        elif optic is None and len(kwargs) == 0: # create empty optic.
            from .import optics
            optic = optics.ScalarTransmission() # placeholder optic, transmission=100%
            optic.planetype=_PUPIL
        else:
            raise TypeError("Not sure how to handle an Optic input of the provided type, {0}".format(
                str(optic.__class__)))

        return self._add_plane(optic, index=index, logstring="pupil plane")

    def add_image(self, optic=None, function=None, index=None, **kwargs):
        """ Add an image plane optic to the optical system

        That image plane optic can be specified either

          1) from file(s) giving transmission or OPD
                [set arguments `transmission=filename` and/or `opd=filename`]
          2) from an analytic function
                [set `function='circle, fieldstop, bandlimitedcoron, or FQPM'`
                and set additional kwargs to define shape etc.
          3) from an already-created OpticalElement object
                [set `optic=that object`]

        Parameters
        ----------
        optic : poppy.OpticalElement
            An already-created OpticalElement you would like to add
        function: string
            Name of some analytic function to add.
            Optional `kwargs` can be used to set the parameters of that function.
            Allowable function names are CircularOcculter, fieldstop, BandLimitedCoron, FQPM
        opd, transmission : string
            Filenames of FITS files describing the desired optic.
        index : int
            Index into the optical system's planes for where to add the new optic. Defaults to
            appending the optic to the end of the plane list.


        Returns
        -------
        poppy.OpticalElement subclass
            The pupil optic added (either `optic` passed in, or a new OpticalElement created)

        Notes
        ------

        Now you can use the optic argument for either an OpticalElement or a
        string function name, and it will do the right thing depending on type.
        Both existing arguments are left for back compatibility for now.



        """

        if isinstance(optic, six.string_types):
            function = optic
            optic = None

        if optic is None:
            from .import optics
            if function == 'CircularOcculter':
                fn = optics.CircularOcculter
            elif function == 'BarOcculter':
                fn = optics.BarOcculter
            elif function == 'fieldstop':
                fn = optics.FieldStop
            elif function == 'BandLimitedCoron':
                fn = optics.BandLimitedCoron
            elif function == 'FQPM':
                fn = optics.IdealFQPM
            elif function is not None:
                raise ValueError("Analytic mask type '%s' is unknown." % function)
            elif len(kwargs) > 0: # create image from files specified in kwargs
                fn = FITSOpticalElement
            else:
                fn = optics.ScalarTransmission # placeholder optic, transmission=100%

            optic = fn(oversample=self.oversample, **kwargs)
            optic.planetype=_IMAGE
        else:
            optic.planetype = _IMAGE
            optic.oversample = self.oversample # these need to match...

        return self._add_plane(optic, index=index, logstring="image plane")

    def add_rotation(self, angle=0.0, index=None, *args, **kwargs):
        """
        Add a clockwise or counterclockwise rotation around the optical axis

        Parameters
        -----------
        angle : float
            Rotation angle, counterclockwise. By default in degrees.
        index : int
            Index into the optical system's planes for where to add the new optic. Defaults to
            appending the optic to the end of the plane list.

        Returns
        -------
        poppy.Rotation
            The rotation added to the optical system
        """
        optic = Rotation(angle=angle, *args, **kwargs)
        return self._add_plane(optic, index=index, logstring="rotation plane")

    def add_inversion(self, index=None, *args, **kwargs):
        """
        Add a coordinate inversion of the wavefront, for instance
        a flip in the sign of the X and Y axes due to passage through a focus.

        Parameters
        -----------
        index : int
            Index into the optical system's planes for where to add the new optic. Defaults to
            appending the optic to the end of the plane list.

        Returns
        -------
        poppy.CoordinateInversion
            The inversion added to the optical system
        """
        optic = CoordinateInversion(*args, **kwargs)
        return self._add_plane(optic, index=index, logstring="coordinate inversion plane")

    def add_detector(self, pixelscale, oversample=None, index=None, **kwargs):
        """ Add a Detector object to an optical system.
        By default, use the same oversampling as the rest of the optical system,
        but the user can override to a different value if desired by setting `oversample`.


        Other arguments are passed to the init method for Detector().

        Parameters
        ----------
        pixelscale : float
            Pixel scale in arcsec/pixel
        oversample : int, optional
            Oversampling factor for *this detector*, relative to hardware pixel size.
            Optionally distinct from the default oversampling parameter of the OpticalSystem.
        index : int
            Index into the optical system's planes for where to add the new optic. Defaults to
            appending the optic to the end of the plane list.


        Returns
        -------
        poppy.Detector
            The detector added to the optical system

        """

        if oversample is None:
            oversample = self.oversample
        optic = Detector(pixelscale, oversample=oversample, **kwargs)

        return self._add_plane(optic, index=index,
                               logstring="detector with pixelscale={} and oversampling={}".format(
                               pixelscale,
                               oversample))

    def describe(self):
        """ Print out a string table describing all planes in an optical system"""
        print(str(self)+"\n\t"+ "\n\t".join([str(p) for p in self.planes]))

    def __getitem__(self, num):
        return self.planes[num]

    # methods for dealing with wavefronts:
    @utils.quantity_input(wavelength=u.meter)
    def input_wavefront(self, wavelength=2e-6*u.meter):
        """Create a Wavefront object suitable for sending through a given optical system, based on
        the size of the first optical plane, assumed to be a pupil.

        If the first optical element is an Analytic pupil (i.e. has no pixel scale) then
        the default size is set by the `npix` parameter to __init__ of this class,
        which itself has a default value of 1024.

        Uses self.source_offset to assign an off-axis tilt, if requested.

        The convention here is that the desired source position is specified with
        respect to the **final focal plane** of the optical system. If there are any
        intervening coordinate transformation planes, this function attempts to take
        them into account when setting the tilt of the input wavefront. This is
        subtle trickery and may not work properly in all instances.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters

        Returns
        -------
        wavefront : poppy.Wavefront instance
            A wavefront appropriate for passing through this optical system.

        """

        # somewhat complicated logic here for historical reasons.
        # if we have a first optical plane, check and see if it specifies the entrance sampling.

        npix=None
        diam=None
        if len(self.planes) > 0:
            if self.planes[0].shape is not None:
                npix=self.planes[0].shape[0]
            if hasattr(self.planes[0], 'pupil_diam') and self.planes[0].pupil_diam is not None:
                diam = self.planes[0].pupil_diam

        # if sampling is still undefined, fall back to what is set for this optical system itself
        if npix is None:
            npix=self.npix if self.npix is not None else 1024
        if diam is None:
            diam = self.pupil_diameter if self.pupil_diameter is not None else 1

        # if the diameter was specified as an astropy.Quantity, cast it to just a scalar in meters
        if isinstance(diam, u.Quantity):
            diam = diam.to(u.m).value

        inwave = Wavefront(wavelength=wavelength, npix=npix,
                           diam=diam, oversample=self.oversample)
        _log.debug("Creating input wavefront with wavelength={}, npix={:d}, pixel scale={:.3g} meters/pixel".format(
            wavelength, npix, diam/npix))

        if np.abs(self.source_offset_r) > 0:
            # Add a tilt to the input wavefront.
            # First we must work out the handedness of the input pupil relative to the
            # final image plane.  This is needed to apply (to the input pupil) shifts
            # with the correct handedness to get the desired motion in the final plane.
            sign_x = 1
            sign_y = 1
            rotation_angle = 0
            if len(self.planes) > 0:
                for plane in self.planes:
                    if isinstance(plane, CoordinateInversion):
                        if plane.axis == 'x' or plane.axis=='both':
                            sign_x *= -1
                        if plane.axis == 'y' or plane.axis=='both':
                            sign_y *= -1
                    elif isinstance(plane, Rotation):
                        rotation_angle += plane.angle*sign_x*sign_y

            # now we must also work out the rotation

            # convert to offset X,Y in arcsec using the usual astronomical angle convention
            offset_x = sign_x * self.source_offset_r *-np.sin((self.source_offset_theta-rotation_angle)*np.pi/180)
            offset_y = sign_y * self.source_offset_r * np.cos((self.source_offset_theta-rotation_angle)*np.pi/180)
            inwave.tilt(Xangle=offset_x, Yangle=offset_y)
            _log.debug("Tilted input wavefront by theta_X=%f, theta_Y=%f arcsec. (signs=%d, %d; theta offset=%f) " % (offset_x, offset_y, sign_x, sign_y, rotation_angle))
        return inwave

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self, wavelength=2e-6*u.meter, normalize='first',
                       retain_intermediates=False, display_intermediates=False):
        """Propagate a monochromatic wavefront through the optical system. Called from within `calc_psf`.
        Returns a tuple with a `fits.HDUList` object and a list of intermediate `Wavefront`s (empty if
        `retain_intermediates=False`).

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        normalize : string, {'first', 'last'}
            how to normalize the wavefront?
            * 'first' = set total flux = 1 after the first optic, presumably a pupil
            * 'last' = set total flux = 1 after the entire optical system.
            * 'exit_pupil' = set total flux = 1 at the last pupil of the optical system.
            * 'first=2' = set total flux = 2 after the first optic (used for debugging only)
        display_intermediates : bool
            Should intermediate steps in the calculation be displayed on screen? Default: False.
        retain_intermediates : bool
            Should intermediate steps in the calculation be retained? Default: False.
            If True, the second return value of the method will be a list of `poppy.Wavefront` objects
            representing intermediate optical planes from the calculation.

        Returns
        -------
        final_wf : fits.HDUList
            The final result of the monochromatic propagation as a FITS HDUList
        intermediate_wfs : list
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False.)
        """

        if conf.enable_speed_tests:
            t_start = time.time()
        if self.verbose:
            _log.info(" Propagating wavelength = {0:g}".format(wavelength))
        wavefront = self.input_wavefront(wavelength)

        intermediate_wfs = []

        # note: 0 is 'before first optical plane; 1 = 'after first plane and before second plane' and so on
        current_plane_index = 0
        for optic in self.planes:
            # The actual propagation:
            wavefront.propagate_to(optic)
            wavefront *= optic
            current_plane_index += 1

            # Normalize if appropriate:
            if normalize.lower() == 'first' and current_plane_index==1:  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower() == 'first=2' and current_plane_index==1:  # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower() == 'exit_pupil': # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(np.asarray(
                    [p.planetype is PlaneType.pupil for p in self.planes]))[0].max() +1
                if current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug("normalizing at exit pupil (plane {0}) to 1.0 total intensity".format(
                        current_plane_index))
            elif normalize.lower() == 'last' and current_plane_index == len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")


            # Optional outputs:
            if conf.enable_flux_tests: _log.debug("  Flux === " + str(wavefront.total_intensity))

            if retain_intermediates: # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if conf.enable_speed_tests: t0 = time.time()
                #title = None if current_plane_index > 1 else "propagating $\lambda=$ {0:.3f}".format(
                    #wavelength.to(u.micron))
                if hasattr(optic, 'wavefront_display_hint'):
                    display_what = optic.wavefront_display_hint
                else:
                    display_what='best'
                if hasattr(optic, 'wavefront_display_vmax_hint'):
                    display_vmax = optic.wavefront_display_vmax_hint
                else:
                    display_vmax=None
                if hasattr(optic, 'wavefront_display_vmin_hint'):
                    display_vmin = optic.wavefront_display_vmin_hint
                else:
                    display_vmin=None


                ax = wavefront.display(what=display_what,nrows=len(self.planes),row=current_plane_index,
                                       colorbar=False, vmax=display_vmax, vmin=display_vmin)
                if hasattr(optic,'display_annotate'):
                    optic.display_annotate(optic, ax)  # atypical calling convention needed empirically

                if conf.enable_speed_tests:
                    t1 = time.time()
                    _log.debug("\tTIME %f s\t for displaying the wavefront." % (t1-t0))

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop-t_start))

        return wavefront.as_fits(), intermediate_wfs

    @utils.quantity_input(wavelength=u.meter)
    def calc_psf(self, wavelength=1e-6*u.meter, weight=None, save_intermediates=False, save_intermediates_what='all',
                display=False, return_intermediates=False, source=None, normalize='first', display_intermediates=False):
        """Calculate a PSF, either multi-wavelength or monochromatic.

        The wavelength coverage computed will be:
        - multi-wavelength PSF over some weighted sum of wavelengths (if you provide a `source` argument)
        - monochromatic (if you provide just a `wavelength` argument)

        Parameters
        ----------
        wavelength : float or Astropy.Quantity, optional
            wavelength in meters, or some other length unit if specified as an astropy.Quantity. Either
            scalar for monochromatic calculation or list or ndarray for multiwavelength calculation.
        weight : float, optional
            weight by which to multiply each wavelength. Must have same length as
            wavelength parameter. Defaults to 1s if not specified.
        save_intermediates : bool, optional
            whether to output intermediate optical planes to disk. Default is False
        save_intermediate_what : string, optional
            What to save - phase, intensity, amplitude, complex, parts, all. Default is all.
        return_intermediates: bool, optional
            return intermediate wavefronts as well as PSF?
        source : dict
            a dict containing 'wavelengths' and 'weights' list.
        normalize : string, optional
            How to normalize the PSF. See the documentation for propagate_mono() for details.
        display : bool, optional
            whether to plot the results when finished or not.
        display_intermediates: bool, optional
            Display intermediate optical planes? Default is False. This option is incompatible with
            parallel calculations using `multiprocessing`. (If calculating in parallel, it will have no effect.)

        Returns
        -------
        outfits :
            a fits.HDUList
        intermediate_wfs : list of `poppy.Wavefront` objects (optional)
            Only returned if `return_intermediates` is specified.
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
        """

        tstart = time.time()
        if source is not None:
            wavelength = source['wavelengths']
            weight = source['weights']

            # Make sure the wavelength is unit-y
            if not isinstance(wavelength, u.Quantity):
                wavelength = np.asarray(wavelength) * u.meter

        # ensure wavelength is a quantity which is iterable:
        # (the check for a quantity of type length is applied in the decorator)
        if np.isscalar(wavelength.value):
            wavelength = np.asarray([wavelength.value], dtype=float)*wavelength.unit

        if weight is None:
            weight = [1.0] * len(wavelength)

        if len(tuple(wavelength)) != len(tuple(weight)):
            raise ValueError("Input source has different number of weights and wavelengths...")

        # loop over wavelengths
        if self.verbose: _log.info("Calculating PSF with %d wavelengths" % (len(wavelength)))
        outFITS = None
        intermediate_wfs = None
        if save_intermediates or return_intermediates:
            _log.info("User requested saving intermediate wavefronts in call to poppy.calc_psf")
            retain_intermediates = True
        else:
            retain_intermediates = False

        normwts =  np.asarray(weight, dtype=float)
        normwts /= normwts.sum()

        _USE_FFTW = (conf.use_fftw and _FFTW_AVAILABLE)
        if _USE_FFTW:
            utils.fftw_load_wisdom()

        if conf.use_multiprocessing and len(wavelength) > 1: ######### Parallellized computation ############
            # Avoid a Mac OS incompatibility that can lead to hard-to-reproduce crashes.
            # see issues #23 and #176
            import sys
            import platform
            if ( (sys.version_info < (3,4,0)) and platform.system()=='Darwin' and
                    (('extra_link_args' in np.__config__.blas_opt_info) and
                    '-Wl,Accelerate' in np.__config__.blas_opt_info['extra_link_args'])):
                    _log.error("Multiprocessing not compatible with Apple Accelerate library on Python < 3.4")
                    _log.error(" See https://github.com/mperrin/poppy/issues/23 ")
                    _log.error(" Either disable multiprocessing, or recompile your numpy without Accelerate.")
                    raise NotImplementedError("Multiprocessing not compatible with Apple Accelerate framework.")

            if _USE_FFTW:
                _log.warn('IMPORTANT WARNING: Python multiprocessing and fftw3 do not appear to play well together. '
                          'This may crash intermittently')
                _log.warn('   We suggest you set   poppy.conf.use_fftw to False   if you want to use multiprocessing().')
            if display:
                _log.warn('Display during calculations is not supported for multiprocessing mode. '
                          'Please set poppy.conf.use_multiprocessing.set(False) if you want to use display=True.')
                _log.warn('(Plot the returned PSF with poppy.utils.display_PSF.)')

            if return_intermediates:
                _log.warn('Memory usage warning: When preserving intermediate optical planes in multiprocessing mode, '
                          'memory usage scales with the number of planes times the number of wavelengths. Disable '
                          'use_multiprocessing if you are running out of memory.')
            if save_intermediates:
                _log.warn('Saving intermediate steps does not take advantage of multiprocess parallelism. '
                          'Set save_intermediates=False for improved speed.')

            # do *NOT* just blindly try to create as many processes as one has CPUs, or one per wavelength either
            # This is a memory-intensive task so that can end up swapping to disk and thrashing IO
            nproc = conf.n_processes if conf.n_processes > 1 \
                                     else utils.estimate_optimal_nprocesses(self, nwavelengths=len(wavelength))
            nproc = min(nproc, len(wavelength)) # never try more processes than wavelengths.
            # be sure to cast nproc to int below; will fail if given a float even if of integer value

            if sys.version_info < (3, 4, 0):
                pool = multiprocessing.Pool(int(nproc))
            else:
                # Use new forkserver for more robustness;
                # Resolves https://github.com/mperrin/poppy/issues/23
                ctx = multiprocessing.get_context('forkserver')
                pool =ctx.Pool(int(nproc))

            # build a single iterable containing the required function arguments
            _log.info("Beginning multiprocessor job using {0} processes".format(nproc))
            worker_arguments = [(self, wlen, retain_intermediates, normalize, _USE_FFTW)
                                for wlen in wavelength]
            results = pool.map(_wrap_propagate_for_multiprocessing, worker_arguments)
            _log.info("Finished multiprocessor job")
            pool.close()

            # Sum all the results up into one array, using the weights
            outFITS, intermediate_wfs = results[0]
            outFITS[0].data *= normwts[0]
            for idx, wavefront in enumerate(intermediate_wfs):
                 intermediate_wfs[idx] *= normwts[0]
            _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                0, len(tuple(wavelength)), wavelength[0]))
            for i in range(1, len(normwts)):
                mono_psf, mono_intermediate_wfs = results[i]
                wave_weight = normwts[i]
                _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                    i, len(tuple(wavelength)), wavelength[i]))
                outFITS[0].data += mono_psf[0].data * wave_weight
                for idx, wavefront in enumerate(mono_intermediate_wfs):
                    intermediate_wfs[idx] += wavefront * wave_weight
            outFITS[0].header.add_history("Multiwavelength PSF calc using {} processes completed.".format(nproc) )

        else:  ########## single-threaded computations (may still use multi cores if FFTW enabled ######
            if display:
                plt.clf()
            for wlen, wave_weight in zip(wavelength, normwts):
                mono_psf, mono_intermediate_wfs = self.propagate_mono(
                    wlen,
                    retain_intermediates=retain_intermediates,
                    display_intermediates=display_intermediates,
                    normalize=normalize
                )

                if outFITS is None:
                    # for the first wavelength processed, set up the arrays where we accumulate the output
                    outFITS = mono_psf
                    outFITS[0].data *= wave_weight
                    intermediate_wfs = mono_intermediate_wfs
                    for wavefront in intermediate_wfs:
                        wavefront *= wave_weight  # modifies Wavefront in-place
                else:
                    # for subsequent wavelengths, scale and add the data to the existing arrays
                    outFITS[0].data += mono_psf[0].data * wave_weight
                    for idx, wavefront in enumerate(mono_intermediate_wfs):
                        intermediate_wfs[idx] += wavefront * wave_weight

            if display:
                # Add final intensity panel to intermediate WF plot
                cmap = getattr(matplotlib.cm, conf.cmap_sequential)
                cmap.set_bad('0.3')
                halffov_x =outFITS[0].header['PIXELSCL']*outFITS[0].data.shape[1]/2
                halffov_y =outFITS[0].header['PIXELSCL']*outFITS[0].data.shape[0]/2
                extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
                unit="arcsec"
                vmax = outFITS[0].data.max()
                vmin = vmax / 1e4
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)  #vmin=1e-8,vmax=1e-1)
                plt.xlabel(unit)

                utils.imshow_with_mouseover(outFITS[0].data, extent=extent, norm=norm, cmap=cmap,
                                            origin='lower')

        if save_intermediates:
            _log.info('Saving intermediate wavefronts:')
            for idx, wavefront in enumerate(intermediate_wfs):
                filename = 'wavefront_plane_{:03d}.fits'.format(idx)
                wavefront.writeto(filename, what=save_intermediates_what)
                _log.info('  saved {} to {} ({} / {})'.format(save_intermediates_what, filename,
                                                              idx, len(intermediate_wfs)))

        tstop = time.time()
        tdelta = tstop-tstart
        _log.info("  Calculation completed in {0:.3f} s".format(tdelta))
        outFITS[0].header.add_history("Calculation completed in {0:.3f} seconds".format(tdelta))

        if _USE_FFTW and conf.autosave_fftw_wisdom:
            utils.fftw_save_wisdom()

        # TODO update FITS header for oversampling here if detector is different from regular?
        waves = np.asarray(wavelength)
        wts = np.asarray(weight)
        mnwave = (waves*wts).sum() / wts.sum()
        outFITS[0].header['WAVELEN'] = ( mnwave, 'Weighted mean wavelength in meters')
        outFITS[0].header['NWAVES'] = (waves.size, 'Number of wavelengths used in calculation')
        for i in range(waves.size):
            outFITS[0].header['WAVE'+str(i)] = ( waves[i], "Wavelength "+str(i))
            outFITS[0].header['WGHT'+str(i)] = ( wts[i], "Wavelength weight "+str(i))
        ffttype = "pyFFTW" if _USE_FFTW else "numpy.fft"
        outFITS[0].header['FFTTYPE'] = (ffttype, 'Algorithm for FFTs: numpy or fftw')
        outFITS[0].header['NORMALIZ'] = (normalize, 'PSF normalization method')

        if self.verbose:
            _log.info("PSF Calculation completed.")
        if return_intermediates:
            return outFITS, intermediate_wfs
        else:
            return outFITS

    def display(self, **kwargs):
        """ Display all elements in an optical system on screen.

        Any extra arguments are passed to the `optic.display()` methods of each element.

        """

        planes_to_display = [p for p in self.planes if (not isinstance(p, Detector) and not p._suppress_display)]
        nplanes = len(planes_to_display)
        for i, plane in enumerate(planes_to_display):
            _log.info("Displaying plane {0:s} in row {1:d} of {2:d}".format(plane.name, i+1, nplanes))
            plane.display(nrows=nplanes, row=i+1, **kwargs)

    def _propagation_info(self):
        """ Provide some summary information on the optical propagation calculations that
        would be done for a given optical system

        Right now this mostly is checking whether a given propagation makes use of FFTs or not,
        since the padding for oversampled FFTS majorly affects the max memory used for multiprocessing
        estimation """

        steps = []
        for i, p in enumerate(self.planes):
            if i == 0: continue  # no propagation needed for first plane
            if p.planetype == _ROTATION:  steps.append('rotation')
            elif self.planes[i-1].planetype == _PUPIL and p.planetype == _DETECTOR: steps.append('MFT')
            elif self.planes[i-1].planetype == _PUPIL and p.planetype == _IMAGE:
                  if i > 1 and steps[-1] =='MFT': steps.append('invMFT')
                  else: steps.append('FFT')
            elif self.planes[i-1].planetype == _IMAGE and p.planetype == _DETECTOR: steps.append('resample')
            else: steps.append('FFT')


        output_shape = [a * self.planes[-1].oversample for a in self.planes[-1].shape]
        output_size = output_shape[0]*output_shape[1]

        return {'steps': steps, 'output_shape': output_shape, 'output_size':output_size}

    # back compatible aliases for PEP8 compliant names;
    # these old versions will be deprecated in a future release.
    addPupil = add_pupil
    addImage = add_image
    addRotation = add_rotation
    addDetector = add_detector
    inputWavefront = input_wavefront
    calcPSF = calc_psf


class SemiAnalyticCoronagraph(OpticalSystem):
    """ A subclass of OpticalSystem that implements a specialized propagation
    algorithm for coronagraphs whose occulting mask has limited and small support in
    the image plane. Algorithm from Soummer et al. (2007)

    The way to use this class is to build an OpticalSystem class the usual way, and then
    cast it to a SemiAnalyticCoronagraph, and then you can just call calc_psf on that in the
    usual fashion.

    Parameters
    -----------
    ExistingOpticalSystem : OpticalSystem
        An optical system which can be converted into a SemiAnalyticCoronagraph. This
        means it must have exactly 4 planes, in order Pupil, Image, Pupil, Detector.
    oversample : int
        Oversampling factor in intermediate image plane. Default is 8
    occulter_box : float
        half size of field of view region entirely including the occulter, in arcseconds. Default 1.0
        This can be a tuple or list to specify a rectangular region [deltaY,deltaX] if desired.


    Notes
    ------

    Note that this algorithm is only appropriate for certain types of Fourier transform,
    namely those using occulters limited to a sub-region of the image plane.
    It is certainly appropriate for TFI, and probably the right choice for NIRCam as well, but
    is of no use for MIRI's FQPMs.



    """

    def __init__(self, ExistingOpticalSystem, oversample=8, occulter_box = 1.0):
        from . import optics

        if len(ExistingOpticalSystem.planes) != 4:
            raise ValueError("Input optical system must have exactly 4 planes to be convertible into a SemiAnalyticCoronagraph")
        self.name = "SemiAnalyticCoronagraph for "+ExistingOpticalSystem.name
        self.verbose = ExistingOpticalSystem.verbose
        self.source_offset_r = ExistingOpticalSystem.source_offset_r
        self.source_offset_theta = ExistingOpticalSystem.source_offset_theta
        self.planes = ExistingOpticalSystem.planes
        self.npix = ExistingOpticalSystem.npix
        self.pupil_diameter = ExistingOpticalSystem.pupil_diameter

        # SemiAnalyticCoronagraphs have some fixed planes, so give them reasonable names.
        self.inputpupil = self.planes[0]
        self.occulter = self.planes[1]
        self.lyotplane = self.planes[2]
        self.detector = self.planes[3]

        self.mask_function = optics.InverseTransmission(self.occulter)

        for i, typecode in enumerate([_PUPIL, _IMAGE, _PUPIL, _DETECTOR]):
            if not self.planes[i].planetype == typecode:
                raise ValueError("Plane {0:d} is not of the right type for a semianalytic coronagraph calculation: should be {1:s} but is {2:s}.".format(i, typecode, self.planes[i].planetype))

        self.oversample = oversample

        #if hasattr(occulter_box, '__getitem__'):
        if not np.isscalar(occulter_box):
            occulter_box = np.array(occulter_box) # cast to numpy array so the multiplication by 2 just below will work
        self.occulter_box = occulter_box

        self.occulter_det = Detector(self.detector.pixelscale/self.oversample,
                                     fov_arcsec = self.occulter_box*2, name='Oversampled Occulter Plane')

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self, wavelength=2e-6*u.meter, normalize='first',
                       retain_intermediates=False, display_intermediates=False):
        """Propagate a monochromatic wavefront through the optical system. Called from within `calc_psf`.
        Returns a tuple with a `fits.HDUList` object and a list of intermediate `Wavefront`s (empty if
        `retain_intermediates=False`).

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        normalize : string, {'first', 'last'}
            how to normalize the wavefront?
            * 'first' = set total flux = 1 after the first optic, presumably a pupil
            * 'last' = set total flux = 1 after the entire optical system.
        display_intermediates : bool
            Should intermediate steps in the calculation be displayed on screen? Default: False.
        retain_intermediates : bool
            Should intermediate steps in the calculation be retained? Default: False.
            If True, the second return value of the method will be a list of `poppy.Wavefront` objects
            representing intermediate optical planes from the calculation.

        Returns
        -------
        final_wf : fits.HDUList
            The final result of the monochromatic propagation as a FITS HDUList
        intermediate_wfs : list
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False.)
        """
        if conf.enable_speed_tests:
           t_start = time.time()
        if self.verbose:
           _log.info(" Propagating wavelength = {0:g} meters using "
                     "Fast Semi-Analytic Coronagraph method".format(wavelength))
        wavefront = self.inputWavefront(wavelength)
        current_plane_index = 0

        intermediate_wfs = []

        wavefront.history.append("Propagating using Fast Semi-Analytic Method")
        wavefront.history.append(" for Coronagraphy (See Soummer et al. 2007).")


        #------- differences from regular propagation begin here --------------
        wavefront *= self.inputpupil
        current_plane_index += 1

        if normalize.lower() == 'first':
            wavefront.normalize()
        if retain_intermediates:
            intermediate_wfs.append(wavefront.copy())

        if display_intermediates:
            nrows = 6
            wavefront.display(what='best',nrows=nrows,row=1, colorbar=False,
                    title="propagating $\lambda=$ {:.3f}".format(wavelength.to(u.micron)))


        # determine FOV region bounding the image plane occulting stop.
        # determine number of pixels across that to use ("N_B")
        # calculate the MFT to the N_B x N_B occulting region.
        wavefront_cor = wavefront.copy()
        wavefront_cor.propagate_to(self.occulter_det)
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_cor.copy())

        if display_intermediates:
            wavefront_cor.display(what='best',nrows=nrows,row=2, colorbar=False)

        # Multiply that by M(r) =  1 - the occulting plane mask function
        wavefront_cor *= self.mask_function
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_cor.copy())

        if display_intermediates:
            wavefront_cor.display(what='best',nrows=nrows,row=3, colorbar=False)

        # calculate the MFT from that small region back to the full Lyot plane

        wavefront_lyot = wavefront_cor.copy()
        wavefront_lyot.propagate_to(self.lyotplane)
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_lyot.copy())

        if display_intermediates:
            wavefront_lyot.display(what='best',nrows=nrows,row=4, colorbar=False)

        # combine that with the original pupil function
        wavefront_combined = wavefront + (-1)*wavefront_lyot
        wavefront_combined *= self.lyotplane
        wavefront_combined.location = 'after combined Lyot pupil'
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_combined.copy())

        if display_intermediates:
            wavefront_combined.display(what='best',nrows=nrows,row=5, colorbar=False)

        # propagate to the real detector in the final image plane.
        wavefront_combined.propagate_to(self.detector)
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_combined.copy())

        if display_intermediates:
            wavefront_combined.display(what='best',nrows=nrows,row=6, colorbar=False)
            #suptitle.remove() #  does not work due to some matplotlib limitation, so work arount:
            #plt.suptitle.set_text('') # clean up before next iteration to avoid ugly overwriting

        #------- differences from regular propagation end here --------------

        # prepare output arrays
        if normalize.lower()=='last':
                wavefront_combined.normalize()

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop-t_start))

        return wavefront_combined.as_fits(), intermediate_wfs


class MatrixFTCoronagraph(OpticalSystem):
    """ A subclass of OpticalSystem that implements a specialized propagation
    algorithm for coronagraphs which are most efficiently modeled by
    matrix Fourier transforms, and in which the semi-analytical/Babinet
    superposition approach does not apply.

    The way to use this class is to build an OpticalSystem class the usual way, and then
    cast it to a MatrixFTCoronagraph, and then you can just call `calc_psf` on that in the
    usual fashion.

    Parameters
    -----------
    ExistingOpticalSystem : OpticalSystem
        An optical system which can be converted into a SemiAnalyticCoronagraph. This
        means it must have exactly 4 planes, in order Pupil, Image, Pupil, Detector.
    oversample : int
        Oversampling factor in intermediate image plane. Default is 4
    occulter_box : float
        half size of field of view region entirely including the occulter, in arcseconds. Default 1.0
        This can be a tuple or list to specify a rectangular region [deltaY,deltaX] if desired.


    Notes
    ------

    This subclass is best suited for a coronagraph design in which the region
    transmitted by the focal plane mask is bounded and small, thereby offering a
    large speed gain over FFT propagation. In particular, the shaped pupil Lyot
    coronagraphs in the baseline WFIRST CGI design, which use a diaphragm-type
    focal plane mask, can benefit highly.

    """

    def __init__(self, ExistingOpticalSystem, oversample=4, occulter_box = 1.0):
        from . import optics

        if len(ExistingOpticalSystem.planes) < 4:
            raise ValueError("Input optical system must have at least 4 planes to be convertible into a MatrixFTCoronagraph")
        self.name = "MatrixFTCoronagraph for "+ExistingOpticalSystem.name
        self.verbose = ExistingOpticalSystem.verbose
        self.source_offset_r = ExistingOpticalSystem.source_offset_r
        self.source_offset_theta = ExistingOpticalSystem.source_offset_theta
        self.planes = ExistingOpticalSystem.planes
        self.npix = ExistingOpticalSystem.npix
        self.pupil_diameter = ExistingOpticalSystem.pupil_diameter

        self.oversample = oversample

        #if hasattr(occulter_box, '__getitem__'):
        if not np.isscalar(occulter_box):
            occulter_box = np.array(occulter_box) # cast to numpy array so the multiplication by 2 just below will work
        self.occulter_box = occulter_box

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self, wavelength=1e-6, normalize='first',
                       retain_intermediates=False, display_intermediates=False):
        """Propagate a monochromatic wavefront through the optical system using matrix FTs. Called from
        within `calc_psf`. Returns a tuple with a `fits.HDUList` object and a list of intermediate `Wavefront`s
        (empty if `retain_intermediates=False`).

        We use the Detector subclass of OpticalElement as the destination in the first
        pupil-to-image propagation, to force the propagation method to switch to the
        matrix FT. Otherwise it would default to FFT.

        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        normalize : string, {'first', 'last'}
            how to normalize the wavefront?
            * 'first' = set total flux = 1 after the first optic, presumably a pupil
            * 'last' = set total flux = 1 after the entire optical system.
        display_intermediates : bool
            Should intermediate steps in the calculation be displayed on screen? Default: False.
        retain_intermediates : bool
            Should intermediate steps in the calculation be retained? Default: False.
            If True, the second return value of the method will be a list of `poppy.Wavefront` objects
            representing intermediate optical planes from the calculation.

        Returns
        -------
        final_wf : fits.HDUList
            The final result of the monochromatic propagation as a FITS HDUList
        intermediate_wfs : list
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False.)
        """

        if conf.enable_speed_tests:
           t_start = time.time()
        if self.verbose:
           _log.info(" Propagating wavelength = {0:g} meters using "
                     "Matrix FTs".format(wavelength))
        wavefront = self.inputWavefront(wavelength)

        intermediate_wfs = []

        # note: 0 is 'before first optical plane; 1 = 'after first plane and before second plane' and so on
        current_plane_index = 0
        for optic in self.planes:
            # The actual propagation:
            if optic.planetype == _IMAGE:
                if len(optic.amplitude.shape) == 2: # Match detector object to the loaded FPM transmission array
                    metadet = Detector(optic.pixelscale, fov_pixels = optic.amplitude.shape[0],
                                       name='Oversampled Occulter Plane')
                else:
                    metadet_pixelscale = ((wavelength / self.planes[0].pupil_diam).decompose()
                                          * u.radian).to(u.arcsec) / self.oversample / 2 / u.pixel
                    metadet = Detector(metadet_pixelscale, fov_arcsec = self.occulter_box*2,
                                       name='Oversampled Occulter Plane')
                wavefront.propagate_to(metadet)
            else:
                wavefront.propagate_to(optic)
            wavefront *= optic
            current_plane_index += 1

            # Normalize if appropriate:
            if normalize.lower() == 'first' and current_plane_index == 1 :  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower() == 'first=2' and current_plane_index == 1 : # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower() == 'exit_pupil': # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(np.asarray([p.planetype is PlaneType.pupil for p in self.planes]))[0].max() +1
                if current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug("normalizing at exit pupil (plane {0}) to 1.0 total intensity".format(current_plane_index))
            elif normalize.lower() == 'last' and current_plane_index == len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")

            # Optional outputs:
            if conf.enable_flux_tests: _log.debug("  Flux === "+str(wavefront.total_intensity))

            if retain_intermediates: # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if conf.enable_speed_tests:
                    t0 = time.time()

                if current_plane_index > 1:
                    title = None
                else:
                    title = "propagating $\lambda=$ {:.3} $\mu$m".format(wavelength.to(u.micron).value)

                wavefront.display(
                    what='best',
                    nrows=len(self.planes),
                    row=current_plane_index,
                    colorbar=False,
                    title=title
                )

                if conf.enable_speed_tests:
                    t1 = time.time()
                    _log.debug("\tTIME %f s\t for displaying the wavefront." % (t1-t0))

        # prepare output arrays
        if normalize.lower()=='last':
                wavefront.normalize()

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop-t_start))

        return wavefront.as_fits(), intermediate_wfs


#------ core Optical Element Classes ------
class OpticalElement(object):
    """ Base class for all optical elements, whether from FITS files or analytic functions.

    If instantiated on its own, this just produces a null optical element (empty space,
    i.e. an identity function on transmitted wavefronts.) Use one of the many subclasses to
    create a nontrivial optic.

    The OpticalElement class follows the behavoior of the Wavefront class, using units
    of meters/pixel in pupil space and arcsec/pixel in image space.

    The internal implementation of this class represents an optic with an array
    for the electric field amplitude transmissivity (or reflectivity), plus an
    array for the optical path difference in units of meters. This
    representation was chosen since most typical optics of interest will have
    wavefront error properties that are independent of wavelength. Subclasses
    particularly the AnalyticOpticalElements extend this paradigm with optics
    that have wavelength-dependent properties.

    The get_phasor() function is used to obtain the complex phasor for any desired
    wavelength based on the amplitude and opd arrays. Those can individually be
    obtained from the get_transmission() and get_opd() functions.

    Parameters
    ----------
    name : string
        descriptive name for optic
    verbose : bool
        whether to be more verbose in log outputs while computing
    planetype : int
        either poppy._IMAGE or poppy._PUPIL
    oversample : int
        how much to oversample beyond Nyquist.
    interp_order : int
        the order (0 to 5) of the spline interpolation used if the optic is resized.
    """
    #pixelscale = None
    #"float attribute. Pixelscale in arcsec or meters per pixel. Will be 'None' for null or analytic optics."


    def __init__(self, name="unnamed optic", verbose=True, planetype=PlaneType.unspecified, oversample=1, interp_order=3):

        self.name = name
        """ string. Descriptive Name of this optic"""
        self.verbose=verbose

        self.planetype = planetype      # pupil or image
        self.oversample = oversample    # oversampling factor, none by default
        self.ispadded = False           # are we padded w/ zeros for oversampling the FFT?
        self._suppress_display=False    # should we avoid displaying this optic on screen? (useful for 'virtual' optics like FQPM aligner)

        self.amplitude = np.asarray([1.])
        self.opd = np.asarray([0.])
        self.pixelscale = None
        self.interp_order=interp_order

    def get_transmission(self, wave):
        """ Return the electric field amplitude transmission, given a wavelength.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        Returns
        --------
        ndarray giving electric field amplitude transmission between 0 - 1.0

        """
        return self.amplitude

    def get_opd(self, wave):
        """ Return the optical path difference, given a wavelength.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        Returns
        --------
        ndarray giving OPD in meters

        """
        return self.opd

    def get_phasor(self,wave):
        """ Compute a complex phasor from an OPD, given a wavelength.

        The returned value should be the complex phasor array as appropriate for
        multiplying by the wavefront amplitude.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        """
        #_log.info("Pixelscales for %s: wave %f, optic  %f" % (self.name, wave.pixelscale, self.pixelscale))

        if isinstance(wave, Wavefront):
            wavelength=wave.wavelength
        else:
            wavelength=wave
        scale = 2. * np.pi / wavelength.to(u.meter).value

        # set the self.phasor attribute:
        # first check whether we need to interpolate to do this.
        float_tolerance = 0.001  #how big of a relative scale mismatch before resampling?
        if self.pixelscale is not None and hasattr(wave,'pixelscale') and abs(wave.pixelscale -self.pixelscale)/self.pixelscale >= float_tolerance:
            _log.debug("Pixelscales: wave {}, optic {}" .format(wave.pixelscale, self.pixelscale))
            #raise ValueError("Non-matching pixel scale for wavefront and optic! Need to add interpolation / ing ")
            if hasattr(self,'_resampled_scale') and abs(self._resampled_scale-wave.pixelscale)/self._resampled_scale >= float_tolerance:
                # we already did this same resampling, so just re-use it!
                self.phasor = self._resampled_amplitude * np.exp (1.j * self._resampled_opd * scale)
            else:
                #raise NotImplementedError("Need to implement resampling.")
                zoom=(self.pixelscale/wave.pixelscale).decompose().value
                resampled_opd = scipy.ndimage.interpolation.zoom(self.opd, zoom,
                        output=self.opd.dtype, order=self.interp_order)
                resampled_amplitude = scipy.ndimage.interpolation.zoom(self.amplitude,zoom,output=self.amplitude.dtype,order=self.interp_order)
                _log.debug("resampled optic to match wavefront via spline interpolation by a zoom factor of %.3g"%(zoom))
                _log.debug("resampled optic shape: {}   wavefront shape: {}".format(resampled_amplitude.shape, wave.shape))

                lx,ly=resampled_amplitude.shape
                #crop down to match size of wavefront:
                lx_w,ly_w = wave.amplitude.shape
                border_x = np.abs(np.floor((lx-lx_w)/2))
                border_y = np.abs(np.floor((ly-ly_w)/2))
                if (self.pixelscale*self.amplitude.shape[0] < wave.pixelscale*wave.amplitude.shape[0]) or (self.pixelscale*self.amplitude.shape[1] < wave.pixelscale*wave.amplitude.shape[0]):
                    #raise ValueError("Optic is smaller than input wavefront")
                    _log.warn("Optic"+str(np.shape(resampled_opd))+" is smaller than input wavefront"+str([lx_w,ly_w])+", will attempt to zero-pad the rescaled array")
                    self._resampled_opd = np.zeros([lx_w,ly_w])
                    self._resampled_amplitude = np.zeros([lx_w,ly_w])

                    self._resampled_opd[border_x:border_x+resampled_opd.shape[0],border_y:border_y+resampled_opd.shape[1]] = resampled_opd
                    self._resampled_amplitude[border_x:border_x+resampled_opd.shape[0],border_y:border_y+resampled_opd.shape[1]]=resampled_amplitude
                    _log.debug("padded an optic with a %i x %i border to optic to match the wavefront"%(border_x,border_y))

                else:
                    self._resampled_opd = resampled_opd[border_x:border_x+lx_w,border_y:border_y+ly_w]
                    self._resampled_amplitude = resampled_amplitude[border_x:border_x+lx_w,border_y:border_y+ly_w]
                    _log.debug("trimmed a border of %i x %i pixels from optic to match the wavefront"%(border_x,border_y))

                self.phasor = self._resampled_amplitude * np.exp (1.j * self._resampled_opd * scale)

        else:
            # compute the phasor directly, without any need to rescale.
            self.phasor = self.get_transmission(wave) * np.exp (1.j * self.get_opd(wave)* scale)



        # check whether we need to pad before returning or not.
        # note: do not pad the phasor if it's just a scalar!
        if self.planetype == _PUPIL and wave.ispadded and self.phasor.size !=1:
            # old version: pad to a fixed oversampling. All FITS arrays in an OpticalSystem must be the same size
            #return padToOversample(self.phasor, wave.oversample)

            # new version: pad to match the wavefront sampling, from whatever sized array we started with. Allows more
            # flexibility for differently sized FITS arrays, so long as they all have the same pixel scale as checked above!
            return utils.pad_to_size(self.phasor, wave.shape)
        else:
            return self.phasor

    def getPhasor(self,wave):
        warnings.warn("getPhasor is deprecated; use get_phasor instead", DeprecationWarning)
        return self.get_phasor(wave)


    @utils.quantity_input(opd_vmax=u.meter)
    def display(self, nrows=1, row=1, what='intensity', crosshairs=True, ax=None, colorbar=True,
                colorbar_orientation=None, title=None, opd_vmax=0.5e-6*u.meter):
        """Display plots showing an optic's transmission and OPD.

        Parameters
        ----------
        what : str
            What to display: 'intensity', 'amplitude', 'phase', 'opd',
            or 'both' (meaning intensity and OPD in two subplots)
        ax : matplotlib.Axes instance
            Axes to display into
        nrows, row : integers
            number of rows and row index for subplot display
        crosshairs : bool
            Display crosshairs indicating the center?
        colorbar : bool
            Show colorbar?
        colorbar_orientation : bool
            Desired orientation, horizontal or vertical?
            Default is horizontal if only 1 row of plots, else vertical
        opd_vmax : float
            Max absolute value for OPD image display, in meters.
        title : string
            Plot label
        """
        if colorbar_orientation is None:
            colorbar_orientation = "horizontal" if nrows == 1 else 'vertical'

        if self.planetype is _PUPIL:
            cmap_amp = getattr(matplotlib.cm, conf.cmap_pupil_intensity)
        else:
            cmap_amp = getattr(matplotlib.cm, conf.cmap_sequential)
        cmap_amp.set_bad('0.0')
        cmap_opd = getattr(matplotlib.cm, conf.cmap_diverging)
        cmap_opd.set_bad('0.3')
        norm_amp = matplotlib.colors.Normalize(vmin=0, vmax=1)

        opd_vmax_m = opd_vmax.to(u.meter).value
        norm_opd = matplotlib.colors.Normalize(vmin=-opd_vmax_m, vmax=opd_vmax_m)

        # TODO infer correct units from pixelscale's units?
        units = "[arcsec]" if self.planetype == _IMAGE else "[meters]"
        if nrows > 1:
            # for display inside an optical system, we repurpose the units display to label the plane
            units = self.name + "\n" + units
            # and wrap long lines if necessary
            if len(units)>20:
                units = "\n".join(textwrap.wrap(units,20))

        if self.pixelscale is not None:
            # TODO handle units better here for pupil vs. image planes? meters/pix vs arcsec/pix
            halfsize = self.pixelscale.value * self.amplitude.shape[0] / 2
            _log.debug("Display pixel scale = {} ".format(self.pixelscale))
        else:
            # TODO not sure this code path ever gets used - since pixelscale is set temporarily
            # in AnalyticOptic.display
            _log.debug("No defined pixel scale - this must be an analytic optic")
            halfsize = 1.0
        extent = [-halfsize, halfsize, -halfsize, halfsize]

        ampl = self.amplitude
        opd = self.opd.copy()
        opd[np.where(self.amplitude == 0)] = np.nan

        if what == 'both':
            # recursion!
            if ax is None:
                ax = plt.subplot(nrows, 2, row * 2 - 1)
            self.display(what='intensity', ax=ax, crosshairs=crosshairs, colorbar=colorbar,
                         colorbar_orientation=colorbar_orientation, title=None, opd_vmax=opd_vmax,
                         nrows=nrows)
            ax2 = plt.subplot(nrows, 2, row * 2)
            self.display(what='opd', ax=ax2, crosshairs=crosshairs, colorbar=colorbar,
                         colorbar_orientation=colorbar_orientation, title=None, opd_vmax=opd_vmax,
                         nrows=nrows)
            ax2.set_ylabel('') # suppress redundant label which duplicates the intensity plot's label
            return ax, ax2
        elif what == 'amplitude':
            plot_array = ampl
            title = 'Transmissivity'
            cb_label = 'Fraction'
            cb_values = [0, 0.25, 0.5, 0.75, 1.0]
            cmap = cmap_amp
            norm = norm_amp
        elif what == 'intensity':
            plot_array = ampl ** 2
            title = "Transmittance"
            cb_label = 'Fraction'
            cb_values = [0, 0.25, 0.5, 0.75, 1.0]
            cmap = cmap_amp
            norm = norm_amp
        elif what == 'phase':
            warnings.warn("displaying 'phase' has been deprecated. Use what='opd' instead.", category=DeprecationWarning)
            plot_array = opd
            title = "OPD"
            cb_label = 'waves'
            cb_values = np.array([-1, -0.5, 0, 0.5, 1]) * opd_vmax_m
            cmap = cmap_opd
            norm = norm_opd
        elif what == 'opd':
            plot_array = opd
            title = "OPD"
            cb_label = 'meters'
            cb_values = np.array([-1, -0.5, 0, 0.5, 1]) * opd_vmax_m
            cmap = cmap_opd
            norm = norm_opd
        else:
            raise ValueError("Invalid value for 'what' parameter")

        # now we plot whichever was chosen...
        if ax is None:
            if nrows > 1:
                ax = plt.subplot(nrows, 2, row * 2 - 1)
            else:
                ax = plt.subplot(111)
        utils.imshow_with_mouseover(plot_array, ax=ax, extent=extent, cmap=cmap, norm=norm,
                                    origin='lower')
        if nrows == 1:
            plt.title(title + " for " + self.name)
        plt.ylabel(units)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        if colorbar:
            cb = plt.colorbar(ax.images[0], orientation=colorbar_orientation, ticks=cb_values)
            cb.set_label(cb_label)
        if crosshairs:
            ax.axhline(0, ls=":", color='k')
            ax.axvline(0, ls=":", color='k')

        if hasattr(self,'display_annotate'):
            self.display_annotate(self, ax)  # atypical calling convention needed empirically
                                         # since Python doesn't seem to automatically pass
                                         # self as first argument for functions added at
                                         # run time as attributes?
        return ax

    def __str__(self):
        if self.planetype == _PUPIL:
            return "Pupil plane: %s " % (self.name)
        elif self.planetype == _IMAGE:
            desc = "({}x{} pixels, scale={} arcsec/pixel)".format(self.shape[0], self.shape[0], self.pixelscale) if self.pixelscale is not None else "(Analytic)"
            return "Image plane: %s %s" % (self.name, desc)
        else:
            return "Optic: "+self.name

    @property
    def shape(self):
        """ Return shape of the OpticalElement, as a tuple """
        if hasattr(self, 'amplitude'):
            return self.amplitude.shape
        else: return None


class FITSOpticalElement(OpticalElement):
    """ Defines an arbitrary optic, based on amplitude transmission and/or OPD FITS files.

    This optic could be a pupil or field stop, an aberrated mirror, a phase mask, etc.
    The FITSOpticalElement class follows the behavior of the Wavefront class, using units
    of meters/pixel in pupil space and arcsec/pixel in image space.

    The interface is **very** flexible.  You can define a FITSOpticalElement either from

    * a single FITS file giving the amplitude transmission (in which case phase is zero)
    * a single FITS file giving the OPD (in which case transmission is 1 everywhere)
    * two FITS files specifying both transmission and OPD.

    The FITS file argument(s) can be supplied either as

        1. a string giving the path to a file on disk,
        2. a FITS HDUlist object, or
        3. in the case of OPDs, a tuple consisting of a path to a datacube and an integer index of a slice in that datacube.

    A better interface for slice selection in datacubes is the transmission_index and opd_index keyword parameters listed below,
    but the tuple interface is retained for back compatibility with existing code.


    Parameters
    ----------
    name : string
        descriptive name for optic
    transmission, opd : string or fits HDUList
        Either FITS filenames *or* actual fits.HDUList objects for the transmission (from 0-1) and opd (in meters)
    transmission_slice, opd_slice : integers, optional
        If either transmission or OPD files are datacubes, you can specify the slice index using this argument.
    opdunits : string
        units for the OPD file. Default is 'meters'. can be 'meter', 'meters', 'micron(s)', 'nanometer(s)', or their SI abbreviations.
        If this keyword is not set explicitly, the BUNIT keyword in the FITS header will be checked.
    planetype : int
        either _IMAGE or _PUPIL
    oversample : int
        how much to oversample beyond Nyquist.
    flip_x, flip_y : bool
        Should the FITS file be inverted in either of these axes after being loaded? Useful for
        matching coordinate system orientations.  If a flip is specified, it takes place prior to any
        shift or rotation operations.
    shift : tuple of floats, optional
        2-tuple containing X and Y fractional shifts for the pupil. These shifts are implemented by rounding them
        to the nearest integer pixel, and doing integer pixel shifts on the data array, without interpolation.
        If a shift is specified, it takes place prior to any rotation operations.
    rotation : float
        Rotation for that optic, in degrees counterclockwise. This is implemented using spline interpolation via
        the scipy.ndimage.interpolation.rotate function.
    pixelscale : optical str or float
        By default, poppy will attempt to determine the appropriate pixel scale by examining the FITS header,
        checking keywords "PUPLSCAL" and 'PIXSCALE' for pupil and image planes respectively. If you would like to
        override and use a different keyword, provide that as a string here. Alternatively, you can just set a
        floating point value directly too (in meters/pixel or arcsec/pixel, respectively, for pupil or image planes).
    transmission_index, opd_index : ints, optional
        If the input transmission or OPD files are datacubes, provide a scalar index here for which cube
        slice should be used.


    *NOTE:* All mask files must be *squares*.

    Also, please note that the adopted convention is for the spectral throughput (transmission) to be given
    in appropriate units for acting on the *amplitude* of the electric field. Thus for example an optic with
    a uniform transmission of 0.5 will reduce the electric field amplitude to 0.5 relative to the input,
    and thus reduce the total power to 0.25. This distinction only matters in the case of semitransparent
    (grayscale) masks.



    """

    def __init__(self, name="unnamed optic", transmission=None, opd=None, opdunits=None,
            shift=None, rotation=None, pixelscale=None, planetype=None,
            transmission_index=None, opd_index=None,
            flip_x=False,flip_y=False,
            **kwargs):

        OpticalElement.__init__(self,name=name, **kwargs)
        self.opd_file = None
        self.amplitude_file = None
        self.amplitude_header = None
        self.opd_header = None

        self.planetype=planetype


        _log.debug("Trans: "+str(transmission))
        _log.debug("OPD: "+str(opd))

        #---- Load amplitude transmission file. ---
        if opd is None and transmission is None:   # no input files, so just make a scalar
            _log.warn("No input files specified. You should set transmission=filename or opd=filename.")
            _log.warn("Creating a null optical element. Are you sure that's what you want to do?")
            self.amplitude = np.asarray([1.])
            self.opd = np.asarray([0.])
            self.pixelscale = None
            self.name = "-empty-"
        else:
            # load transmission file.
            if transmission is not None:
                if isinstance(transmission,six.string_types):
                    self.amplitude_file = transmission
                    self.amplitude, self.amplitude_header = fits.getdata(self.amplitude_file, header=True)
                    if self.name=='unnamed optic': self.name='Optic from '+self.amplitude_file
                    _log.info(self.name+": Loaded amplitude transmission from "+self.amplitude_file)
                elif isinstance(transmission,fits.HDUList):
                    self.amplitude_file='supplied as fits.HDUList object'
                    self.amplitude = transmission[0].data.copy()
                    self.amplitude_header = transmission[0].header.copy()
                    if self.name=='unnamed optic': self.name='Optic from fits.HDUList object'
                    _log.info(self.name+": Loaded amplitude transmission from supplied fits.HDUList object")
                else:
                    raise TypeError('Not sure how to use a transmission parameter of type '+str(type(transmission)))

                # check for datacube?
                if len(self.amplitude.shape) > 2:
                    if transmission_index is None:
                        _log.info("The supplied pupil amplitude is a datacube but no slice was specified. Defaulting to use slice 0.")
                        transmission_index=0
                    self.amplitude_slice_index = transmission_index
                    self.amplitude = self.amplitude[self.amplitude_slice_index, :,:]
                    _log.debug(" Datacube detected, using slice ={0}".format(self.amplitude_slice_index))
            else:
                _log.debug("No transmission supplied - will assume uniform throughput = 1 ")
                # if transmission is none, wait until after OPD is loaded, below, and then create a matching
                # amplitude array uniformly filled with 1s.


            #---- Load OPD file. ---
            if opd is None:
                # if only amplitude set, create an array of 0s with same size.
                self.opd = np.zeros(self.amplitude.shape)
                opdunits = 'meter' # doesn't matter, it's all zeros, but this will indicate no need to rescale below.

            elif isinstance(opd, fits.HDUList):
                # load from fits HDUList
                self.opd_file='supplied as fits.HDUList object'
                self.opd = opd[0].data.copy()
                self.opd_header = opd[0].header.copy()
                if self.name=='unnamed optic': self.name='OPD from supplied fits.HDUList object'
                _log.info(self.name+": Loaded OPD from supplied fits.HDUList object")
            elif isinstance(opd, six.string_types):
                # load from regular FITS filename
                self.opd_file=opd
                self.opd, self.opd_header = fits.getdata(self.opd_file, header=True)
                if self.name=='unnamed optic': self.name='OPD from '+self.opd_file
                _log.info(self.name+": Loaded OPD from "+self.opd_file)

            elif len(opd) ==2 and isinstance(opd[0], six.string_types) :
                # if OPD is specified as a 2-element iterable, treat the first element as the filename and 2nd as the slice of a cube.
                self.opd_file = opd[0]
                self.opd_slice = opd[1]
                self.opd, self.opd_header = fits.getdata(self.opd_file, header=True)
                self.opd = self.opd[self.opd_slice, :,:]
                if self.name=='unnamed optic': self.name='OPD from %s, plane %d' % (self.opd_file, self.opd_slice)
                _log.info(self.name+": Loaded OPD from  %s, plane %d" % (self.opd_file, self.opd_slice) )
            else:
                raise TypeError('Not sure how to use an OPD parameter of type '+str(type(transmission)))

            # check for datacube?
            if len(self.opd.shape) > 2:
                if opd_index is None:
                    _log.info("The supplied pupil OPD is a datacube but no slice was specified. Defaulting to use slice 0.")
                    transmission_index=0
                self.opd_slice_index = transmission_index
                self.opd = self.opd[self.opd_slice_index, :,:]
                _log.debug(" Datacube detected, using slice ={0}".format(self.opd_slice_index))


            if transmission is None:
                _log.info("No info supplied on amplitude transmission; assuming uniform throughput = 1")
                self.amplitude = np.ones(self.opd.shape)

            if opdunits is None:
                try:
                    opdunits = self.opd_header['BUNIT']
                except KeyError:
                    _log.error("No opdunit keyword supplied, and BUNIT keyword not found in header. Cannot determine OPD units")
                    raise Exception("No opdunit keyword supplied, and BUNIT keyword not found in header. Cannot determine OPD units.")

            # normalize and drop any trailing 's'
            opdunits = opdunits.lower()
            if opdunits.endswith('s'):
                opdunits = opdunits[:-1]

            # rescale OPD to meters if necessary
            if opdunits in ('meter', 'm'):
                pass
            elif opdunits in ('micron', 'um', 'micrometer'):
                self.opd *= 1e-6
            elif opdunits in ('nanometer', 'nm'):
                self.opd *= 1e-9

            if len (self.opd.shape) != 2 or self.opd.shape[0] != self.opd.shape[1]:
                _log.debug('OPD shape: '+str(self.opd.shape))
                raise ValueError("OPD image must be 2-D and square")

            if len (self.amplitude.shape) != 2 or self.amplitude.shape[0] != self.amplitude.shape[1]:
                raise ValueError("Pupil amplitude image must be 2-D and square")


            assert self.amplitude.shape == self.opd.shape, "Amplitude and OPD FITS file shapes are incompatible."
            assert self.amplitude.shape[0] == self.amplitude.shape[1], "Amplitude and OPD FITS files must be square."

            # ---- transformation: inversion ----
            # if an inversion is specified and we're not a null (scalar) opticm then do the inversion:
            if flip_y and len(self.amplitude.shape) ==2:
                self.amplitude = self.amplitude[::-1]
                self.opd = self.opd[::-1]
                _log.debug("Inverted optic in the Y axis")
            if flip_x and len(self.amplitude.shape) ==2:
                self.amplitude = self.amplitude[:,::-1]
                self.opd = self.opd[:,::-1]
                _log.debug("Inverted optic in the X axis")


            # ---- transformation: shift ----
            # if a shift is specified and we're NOT a null (scalar) optic, then do the shift:
            if shift is not None and len(self.amplitude.shape) ==2:
                if abs(shift[0]) > 0.5 or abs(shift[1])> 0.5:
                    raise ValueError("""You have asked for an implausibly large shift. Remember, shifts should be specified as
                      decimal values between -0.5 and 0.5, a fraction of the total optic diameter. """)
                rolly = int(np.round(self.amplitude.shape[0] * shift[1])) #remember Y,X order for shape, but X,Y order for shift
                rollx = int(np.round(self.amplitude.shape[1] * shift[0]))
                _log.info("Requested optic shift of (%6.3f, %6.3f) %%" % (shift))
                _log.info("Actual shift applied   = (%6.3f, %6.3f) %%" % (rollx*1.0/self.amplitude.shape[1], rolly *1.0/ self.amplitude.shape[0]))
                self._shift = (rollx*1.0/self.amplitude.shape[1], rolly *1.0/ self.amplitude.shape[0])

                self.amplitude = scipy.ndimage.shift(self.amplitude, (rolly, rollx))
                self.opd       = scipy.ndimage.shift(self.opd,       (rolly, rollx))

            # ---- transformation: rotation ----
            # Likewise, if a rotation is specified and we're NOT a null (scalar) optic, then do the rotation:
            if rotation is not None and len(self.amplitude.shape) ==2:

                # do rotation with interpolation, but try to clean up some of the artifacts afterwards.
                # this is imperfect at best, of course...

                self.amplitude = scipy.ndimage.interpolation.rotate(self.amplitude, rotation, reshape=False).clip(min=0,max=1.0)
                wnoise = np.where(( self.amplitude < 1e-3) & (self.amplitude > 0))
                self.amplitude[wnoise] = 0
                self.opd       = scipy.ndimage.interpolation.rotate(self.opd,       rotation, reshape=False)
                _log.info("  Rotated optic by %f degrees counter clockwise." % rotation)
                #fits.PrimaryHDU(self.amplitude).writeto("test_rotated_amp.fits", clobber=True)
                #fits.PrimaryHDU(self.opd).writeto("test_rotated_opt.fits", clobber=True)
                self._rotation = rotation


            # Determine the pixel scale for this image.
            _MISSING_PIXELSCALE_MSG = ("No FITS header keyword for pixel scale found "
                                       "(tried: {}). Supply pixelscale as a float in "
                                       "meters/px or arcsec/px, or as a string specifying which "
                                       "header keyword to use.")

            def _find_pixelscale_in_headers(keywords, headers):
                """
                Loops through provided possible FITS header keywords and a list of FITS
                header objects (may contain Nones), returning the first
                (keyword, header value) pair found
                """
                for keyword in keywords:
                    for header in headers:
                        if header is not None and keyword in header:
                            return keyword, header[keyword]
                raise LookupError(_MISSING_PIXELSCALE_MSG.format(', '.join(keywords)))

            if pixelscale is None and self.planetype is None:
                # we don't know which keywords might be present yet, so check for both keywords
                # in both header objects (at least one must be non-None at this point!)
                _log.debug("  Looking for 'PUPLSCAL' or 'PIXSCALE' in FITS headers to set "
                           "pixel scale")
                keyword, self.pixelscale = _find_pixelscale_in_headers(
                    ('PUPLSCAL', 'PIXSCALE'),
                    (self.amplitude_header, self.opd_header)
                )
                if keyword == 'PUPLSCAL':
                    self.planetype = _PUPIL
                else:
                    self.planetype = _IMAGE
            elif pixelscale is None and self.planetype == _IMAGE:
                # the planetype tells us which header keyword to check when a keyword is
                # not provided (PIXSCALE for image planes)...
                _, self.pixelscale = _find_pixelscale_in_headers(
                    ('PIXSCALE',),
                    (self.amplitude_header, self.opd_header)
                )
            elif pixelscale is None and self.planetype == _PUPIL:
                # ... likewise for pupil planes
                _, self.pixelscale = _find_pixelscale_in_headers(
                    ('PUPLSCAL',),
                    (self.amplitude_header, self.opd_header)
                )
            elif isinstance(pixelscale, six.string_types):
                # If provided as a keyword string, check for it using the same helper function
                _log.debug("  Getting pixel scale from FITS keyword:" + pixelscale)
                _, self.pixelscale = _find_pixelscale_in_headers(
                    (pixelscale,),
                    (self.opd_header, self.amplitude_header)
                )
            else:
                # pixelscale had better be a floating point value here.
                try:
                    _log.debug("  Getting pixel scale from user-provided float value: " +
                               str(pixelscale))
                    self.pixelscale = float(pixelscale)
                except ValueError:
                    raise ValueError("pixelscale=%s is neither a FITS keyword string "
                                     "nor a floating point value." % str(pixelscale))
            # now turn the pixel scale into a Quantity
            if self.planetype == _IMAGE:
                self.pixelscale *= u.arcsec/u.pixel
            else: # pupil or any other types of plane
                self.pixelscale *= u.meter/u.pixel

    @property
    def pupil_diam(self):
        return self.pixelscale * (self.amplitude.shape[0]*u.pixel)
    "Diameter of the pupil (if this is a pupil plane optic)"


class CoordinateTransform(OpticalElement):
    """ Performs a coordinate transformation (rotation or axes inversion
    in the optical train.

    This is not an actual optic itself but a placeholder to indicate
    when a coordinate transform should take place.

    You should generally not need to use this class or its subclasses directly;
    rather use the OpticalSystem add_rotation or add_inversion functions to
    insert these as needed into optical systems.

    Parameters
    ----------
    hide : bool
        Should this optic be displayed or hidden when showing the
        planes of an OpticalSystem?


    """
    def __init__(self, name='Coordinate transform', hide=False, **kwargs):
        OpticalElement.__init__(self, name=name, **kwargs)
        self._suppress_display=hide

    def get_phasor(self,wave):
        return 1.0  #no change in wavefront
        # returning this is necessary to allow the multiplication in propagate_mono to be OK

    def display(self, nrows=1, row=1, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplot(nrows, 2, row*2-1)
        plt.text(0.3,0.3,self.name)
        return ax


class Rotation(CoordinateTransform):
    """ Performs a rotation of the axes in the optical train.

    This is not an actual optic itself, of course, but can be used to model
    a rotated optic by appling a Rotation before and/or after light is incident
    on that optic.


    This is basically a placeholder to indicate the need for a rotation at a
    given part of the optical train. The actual rotation computation is performed
    in the Wavefront object's propagation routines.


    Parameters
    ----------
    angle : float
        Rotation angle, counterclockwise. By default in degrees.
    units : 'degrees' or 'radians'
        Units for the rotation angle.
    hide : bool
        Should this optic be displayed or hidden when showing the
        planes of an OpticalSystem?

    """
    def __init__(self, angle=0.0, units='degrees', hide=False, **kwargs):
        if units == 'radians':
            angle*= np.pi/180
        elif units =='degrees':
            pass
        else:
            raise ValueError("Unknown value for units='%s'. Must be degrees or radians." % units)
        self.angle = angle

        CoordinateTransform.__init__(self, name= "Rotation by %.2f degrees" % angle,
                planetype=PlaneType.rotation, hide=hide, **kwargs)

    def __str__(self):
        return "Rotation by %f degrees counter clockwise" % self.angle


class CoordinateInversion(CoordinateTransform):
    """ Coordinate axis inversion indicator.

    The actual inversion happens in Wavefront.propagate_to

    Parameters
    ------------
    axes : string
        either 'both', 'x', or 'y', for which axes to invert
    hide : bool
        Should this optic be displayed or hidden when showing the
        planes of an OpticalSystem?

    """
    def __init__(self, name='Coordinate inversion', axis='both', hide=False, **kwargs):
        self.axis=axis.lower()
        CoordinateTransform.__init__(self, name=name,
                planetype=PlaneType.inversion, hide=hide, **kwargs)

    def __str__(self):
        return "Coordinate Inversion in {} axis".format(self.axis)


#------ Detector ------

class Detector(OpticalElement):
    """ A Detector is a specialized type of OpticalElement that forces a wavefront
    onto a specific fixed pixelization of an Image plane.

    This class is in effect just a metadata container for the desired sampling;
    all the machinery for transformation of a wavefront to that sampling happens
    within Wavefront.

    Note that this is *not* in any way a representation of real noisy detectors;
    no model for read noise, imperfect sensitivity, etc is included whatsoever.



    Parameters
    ----------
    name : string
        Descriptive name
    pixelscale : float or astropy.units.Quantity
        Pixel scale in arcsec/pixel, or other angular unit if specified as a Quantity.
    fov_pixels, fov_arcsec : float or astropy.units.Quantity
        The field of view may be specified either in arcseconds or by a number
        of pixels. Either is acceptable and the pixel scale is used to convert
        as needed. You may specify a non-square FOV by providing two elements in
        an iterable.  Note that this follows the usual Python convention of
        ordering axes (Y,X), so put your desired Y axis size first.
    oversample : int
        Oversampling factor beyond the detector pixel scale
    offset : tuple (X,Y)
        Offset for the detector center relative to a hypothetical off-axis PSF.
        Specifying this lets you pick a different sub-region for the detector
        to compute, if for some reason you are computing a small subarray
        around an off-axis source. (Has not been tested!)

    """
    @utils.quantity_input(pixelscale=u.arcsec/u.pixel, fov_pixels=u.pixel, fov_arcsec=u.arcsec)
    def __init__(self, pixelscale=1*(u.arcsec/u.pixel), fov_pixels=None, fov_arcsec=None, oversample=1, name="Detector", offset=None, **kwargs):
        OpticalElement.__init__(self,name=name, planetype=_DETECTOR, **kwargs)
        self.pixelscale = pixelscale
        self.oversample = oversample

        if fov_pixels is None and fov_arcsec is None:
            raise ValueError("Either fov_pixels or fov_arcsec must be specified!")
        elif fov_pixels is not None:
            self.fov_pixels = np.round(fov_pixels)
            self.fov_arcsec = self.fov_pixels * self.pixelscale
        else:
            # set field of view to closest value possible to requested,
            # consistent with having an integer number of pixels
            self.fov_pixels = np.round( (fov_arcsec.to(u.arcsec) / self.pixelscale).to(u.pixel) )
            self.fov_arcsec = self.fov_pixels * self.pixelscale
        if np.any(self.fov_pixels <= 0):
            raise ValueError("FOV in pixels must be a positive quantity. Invalid: "+str(self.fov_pixels))

        if offset is not None:
            try:
                self.det_offset = np.asarray(offset)[0:2]
            except IndexError:
                raise ValueError("The offset parameter must be a 2-element iterable")

        self.amplitude = 1
        self.opd = 0

    @property
    def shape(self):
        fpix = self.fov_pixels.to(u.pixel).value
        # have to cast back to int since Quantities are all float internally
        return (int(fpix), int(fpix)) if np.isscalar(fpix) else fpix.astype(int)[0:2]

    def __str__(self):
        return "Detector plane: {} ({}x{} pixels, {})".format(self.name, self.shape[1], self.shape[0], self.pixelscale)


