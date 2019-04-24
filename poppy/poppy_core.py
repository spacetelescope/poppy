import multiprocessing
import copy
import time
import enum
import warnings
import textwrap
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
import matplotlib

import astropy.io.fits as fits
import astropy.units as u

from .matrixDFT import MatrixFourierTransform
from . import utils
from . import conf
from . import accel_math
from .accel_math import _float, _complex

if accel_math._USE_NUMEXPR:
    import numexpr as ne

import logging

_log = logging.getLogger('poppy')

__all__ = ['Wavefront', 'OpticalSystem', 'CompoundOpticalSystem',
           'OpticalElement', 'ArrayOpticalElement', 'FITSOpticalElement', 'Rotation', 'Detector']


# internal constants for types of plane
class PlaneType(enum.Enum):
    unspecified = 0
    pupil = 1  # pupil plane
    image = 2  # image plane
    detector = 3
    rotation = 4  # coordinate system rotation
    intermediate = 5  # arbitrary plane between pupil and image
    inversion = 6  # coordinate system inversion (flip axes, e.g. like going through focus)


_PUPIL = PlaneType.pupil
_IMAGE = PlaneType.image
_DETECTOR = PlaneType.detector  # specialized type of image plane
_ROTATION = PlaneType.rotation  # not a real optic, just a coordinate transform
_INTERMED = PlaneType.intermediate  # for Fresnel propagation

_RADIANStoARCSEC = 180. * 60 * 60 / np.pi


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
    optical_system, wavelength, retain_intermediates, retain_final, normalize, usefftwflag = args
    conf.use_fftw = usefftwflag  # passed in from parent process

    # we're in a different Python interpreter process so we
    # need to load the wisdom here too
    if conf.use_fftw and accel_math._FFTW_AVAILABLE:
        utils._loaded_fftw_wisdom = False
        utils.fftw_load_wisdom()

    return optical_system.propagate_mono(wavelength,
                                         retain_intermediates=retain_intermediates,
                                         retain_final=retain_final,
                                         normalize=normalize)


class BaseWavefront(ABC):
    """ Abstract base class for wavefronts.
    In general you should not need to use this class directly; use either
    Wavefront or FresnelWavefront child classes for most purposes.

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
        For PlaneType.image PLANE wavefronts, use this pixel scale.
    oversample : int, optional
        how much to oversample by in FFTs. Default is 2.
        Note that final propagations to Detectors use a different algorithm
        and, optionally, a separate oversampling factor.
    dtype : numpy.dtype, optional
        default is double complex.

    """

    @utils.quantity_input(wavelength=u.meter, diam=u.meter)
    def __init__(self, wavelength=1e-6 * u.meter, npix=1024, dtype=None, diam=1.0 * u.meter,
                 oversample=2):

        self.oversample = oversample

        self.wavelength = wavelength  # Wavelength in meters (or other unit if specified)

        self.diam = diam  # array size in meters
        self.pixelscale = None
        "Pixel scale, in arcsec/pixel or meters/pixel depending on plane type"

        self.pixelscale = self.diam / (npix * u.pixel)
        self.planetype = PlaneType.pupil  # assume we begin at an entrance pupil

        self._image_centered = 'array_center'  # one of 'array_center', 'pixel', 'corner'
        # This records where the coordinate origin is
        # in image planes, and depends on how the image
        # plane was produced (e.g. FFT implies pixel)
        "Are FT'ed image planes centered on a pixel or on a corner between pixels? "

        if dtype is None:
            dtype = _complex()
        self.wavefront = np.ones((npix, npix), dtype=dtype)  # the actual complex wavefront array
        self.ispadded = False  # is the wavefront padded for oversampling?
        self.history = []  # List of strings giving a descriptive history of actions
                           # performed on the wavefront. Saved to FITS headers.
        self.history.append("Created wavefront: wavelength={0:.4g}, diam={1:.4g}".format(self.wavelength, self.diam))
        self.history.append(" using array size %s" % (self.wavefront.shape,))
        self.location = 'Entrance Pupil'    # Descriptive string for where a wavefront is instantaneously located.
                                            # Used mostly for titling displayed plots.

        self.current_plane_index = 0  # For tracking stages in a calculation

        accel_math.update_math_settings()                   # ensure optimal propagation based on user settings

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
        sqrt_ti = np.sqrt(self.total_intensity)
        if sqrt_ti == 0:
            _log.warning("Total intensity is zero when trying to normalize the wavefront. Cannot normalize.")
        else:
            self.wavefront /= sqrt_ti

    def __imul__(self, optic):
        """Multiply a Wavefront by an OpticalElement or scalar"""
        if isinstance(optic, CoordinateTransform):
            return self  # a coord transform doesn't actually affect the wavefront via multiplication,
            # but instead via forcing a call to rotate() or invert() in propagate_to...
        elif np.isscalar(optic):
            self.wavefront *= optic  # it's just a scalar
            self.history.append("Multiplied WF by scalar value " + str(optic))
            return self
        elif not isinstance(optic, OpticalElement):
            raise ValueError('Wavefronts can only be *= multiplied by OpticalElements or scalar values')

        if isinstance(optic, Detector):
            # detectors don't modify a wavefront, but we do update the label
            self.location = 'at ' + optic.name
            return self

        phasor = optic.get_phasor(self)

        if not np.isscalar(phasor) and phasor.size > 1:
            assert self.wavefront.shape == phasor.shape, "Phasor shape {} does not match wavefront shape {}".format(
                phasor.shape, self.wavefront.shape)

        self.wavefront *= phasor
        msg = "  Multiplied WF by phasor for " + str(optic)
        _log.debug(msg)
        self.history.append(msg)
        self.location = 'after ' + optic.name
        return self

    def __mul__(self, optic):
        """ Multiply a wavefront by an OpticalElement or scalar """
        new = self.copy()
        new *= optic
        return new

    __rmul__ = __mul__  # either way works.

    def __iadd__(self, wave):
        """Add another wavefront to this one"""
        if not isinstance(wave, BaseWavefront):
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
            what kind of data to write. Must be one of 'all', 'parts', 'intensity',
            'phase' or 'complex'.  The default is to write intensity.
            'all' means write a file containing intensity, amplitude, and phase
            in a data cube of shape (3, N, N).  'parts' omits intensity and
            produces a (2, N, N) array with amplitude and phase.  'intensity'
            and 'phase' write out 2D arrays with the corresponding values.
        includepadding : bool
            include any "padding" region, if present, in the returned FITS file?
        """

        def get_unpadded(attribute_array):
            if self.planetype == PlaneType.pupil and self.ispadded and not includepadding:
                return utils.removePadding(attribute_array.copy(), self.oversample)
            else:
                return attribute_array.copy()

        if what.lower() == 'all':
            intens = get_unpadded(self.intensity)
            outarr = np.zeros((3, intens.shape[0], intens.shape[1]))
            outarr[0, :, :] = intens
            outarr[1, :, :] = get_unpadded(self.amplitude)
            outarr[2, :, :] = get_unpadded(self.phase)
            outfits = fits.HDUList(fits.PrimaryHDU(outarr))
            outfits[0].header['PLANE1'] = 'Wavefront Intensity'
            outfits[0].header['PLANE2'] = 'Wavefront Amplitude'
            outfits[0].header['PLANE3'] = 'Wavefront Phase'
        elif what.lower() == 'parts':
            amp = get_unpadded(self.amplitude)
            outarr = np.zeros((2, amp.shape[0], amp.shape[1]))
            outarr[0, :, :] = amp
            outarr[1, :, :] = get_unpadded(self.phase)
            outfits = fits.HDUList(fits.PrimaryHDU(outarr))
            outfits[0].header['PLANE1'] = 'Wavefront Amplitude'
            outfits[0].header['PLANE2'] = 'Wavefront Phase'
        elif what.lower() == 'intensity':
            outfits = fits.HDUList(fits.PrimaryHDU(get_unpadded(self.intensity)))
            outfits[0].header['PLANE1'] = 'Wavefront Intensity'
        elif what.lower() == 'phase':
            outfits = fits.HDUList(fits.PrimaryHDU(get_unpadded(self.phase)))
            outfits[0].header['PLANE1'] = 'Phase'
        elif what.lower() == 'complex':
            real = get_unpadded(self.wavefront.real)
            outarr = np.zeros((2, real.shape[0], real.shape[1]))
            outarr[0, :, :] = real
            outarr[1, :, :] = get_unpadded(self.wavefront.imag)
            outfits = fits.HDUList(fits.PrimaryHDU(outarr))
            outfits[0].header['PLANE1'] = 'Real part of complex wavefront'
            outfits[0].header['PLANE2'] = 'Imaginary part of complex wavefront'
        else:
            raise ValueError("Unknown string for what to return: " + what)

        outfits[0].header['WAVELEN'] = (self.wavelength.to(u.meter).value, 'Wavelength in meters')
        outfits[0].header['DIFFLMT'] = ((self.wavelength / self.diam * u.radian).to(u.arcsec).value,
                                        'Diffraction limit lambda/D in arcsec')
        outfits[0].header['OVERSAMP'] = (self.oversample, 'Oversampling factor for FFTs in computation')
        outfits[0].header['DET_SAMP'] = (self.oversample, 'Oversampling factor for MFT to detector plane')
        if self.planetype == PlaneType.image:
            outfits[0].header['PIXELSCL'] = (self.pixelscale.to(u.arcsec / u.pixel).value,
                                             'Scale in arcsec/pix (after oversampling)')
            fov_arcsec = self.fov.to(u.arcsec).value
            if np.isscalar(fov_arcsec):
                outfits[0].header['FOV'] = (fov_arcsec, 'Field of view in arcsec (full array)')
            else:
                outfits[0].header['FOV_X'] = (fov_arcsec[1], 'Field of view in arcsec (full array), X direction')
                outfits[0].header['FOV_Y'] = (fov_arcsec[0], 'Field of view in arcsec (full array), Y direction')
                outfits[0].header['PIXUNIT'] = 'arcsecond'

        else:
            outfits[0].header['PIXELSCL'] = (self.pixelscale.to(u.meter / u.pixel).value, 'Pixel scale in meters/pixel')
            outfits[0].header['DIAM'] = (self.diam.to(u.meter).value, 'Pupil diameter in meters (not incl padding)')
            outfits[0].header['PIXUNIT'] = 'meter'
        for h in self.history:
            outfits[0].header.add_history(h)

        return outfits

    def writeto(self, filename, overwrite=True, **kwargs):
        """Write a wavefront to a FITS file.

        Parameters
        -----------
        filename : string
            filename to use
        what : string
            what to write. Must be one of 'parts', 'intensity', 'complex'
        overwrite : bool, optional
            overwhat existing? default is True

        Returns
        -------
        outfile: file on disk
            The output is written to disk.

        """
        self.as_fits(**kwargs).writeto(filename, overwrite=overwrite)
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
            Which row to display this one in? If set to None, use the
            wavefront's self.current_plane_index
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
            scale = 'log' if self.planetype == PlaneType.image else 'linear'

        if row is None:
            row = self.current_plane_index

        intens = self.intensity.copy()

        # make a version of the phase where we try to mask out
        # areas with particularly low intensity
        phase = self.phase.copy()
        mean_intens = np.mean(intens[intens != 0])
        phase[np.where(intens < mean_intens / 100)] = np.nan
        amp = self.amplitude

        y, x = self.coordinates()
        if self.planetype == PlaneType.pupil and self.ispadded and not showpadding:
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

        halfpix = self.pixelscale.value * 0.5
        extent = [x.min() - halfpix, x.max() + halfpix, y.min() - halfpix, y.max() + halfpix]

        if use_angular_coordinates is None:
            use_angular_coordinates = self.planetype == PlaneType.image

        unit = 'arcsec' if use_angular_coordinates else 'm'

        # implement semi-intellegent selection of what to display, if the user wants
        if what == 'best':
            if self.planetype == PlaneType.image:
                what = 'intensity'  # always show intensity for image planes
            elif phase[np.where(np.isfinite(phase))].sum() == 0:
                what = 'intensity'  # for perfect pupils
            # FIXME re-implement this in some better way that doesn't depend on
            # optic positioning in the plot grid!
            # elif int(row) > 2:
            # what = 'intensity'  # show intensity for coronagraphic downstream propagation.
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
                vmin = vmax * 1e-6
            norm_inten = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            cmap_inten = getattr(matplotlib.cm, conf.cmap_sequential)
            cmap_inten.set_bad(cmap_inten(0))
        cmap_phase = getattr(matplotlib.cm, conf.cmap_diverging)
        cmap_phase.set_bad('0.3')
        norm_phase = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        def wrap_lines_title(title):
            # Helper fn to add line breaks in plot titles,
            # tweaked to put in particular places for aesthetics
            for prep in ['after', 'before']:
                if prep in title:
                    part1, part2 = title.split(prep)
                    return part1 + prep + "\n" + "\n".join(textwrap.wrap(part2, 30))
            return "\n".join(textwrap.wrap(title, 30))

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
        elif what == 'amplitude':
            if ax is None:
                ax = plt.subplot(nr, nc, int(row))

            utils.imshow_with_mouseover(
                amp,
                ax=ax,
                extent=extent,
                norm=norm_inten,
                cmap=cmap_inten,
                origin='lower'
            )
            if title is None:
                title = wrap_lines_title("Amplitude " + self.location)
            ax.set_title(title)
            ax.set_xlabel(unit)
            if colorbar:
                plt.colorbar(ax.images[0], orientation='vertical', shrink=0.8)
            plot_axes = [ax]
            to_return = ax
        else:
            raise ValueError("Invalid value for what to display; must be: "
                             "'intensity', 'amplitude', 'phase', or 'both'.")

        # now apply axes cropping and/or overplots, if requested.
        for ax in plot_axes:
            if crosshairs:
                ax.axhline(0, ls=":", color='white')
                ax.axvline(0, ls=":", color='white')

            if use_angular_coordinates:
                if imagecrop is None:
                    imagecrop = conf.default_image_display_fov

            if imagecrop is not None:
                cropsize_x = min((imagecrop / 2, intens.shape[1] / 2. * self.pixelscale.value))
                cropsize_y = min((imagecrop / 2, intens.shape[0] / 2. * self.pixelscale.value))
                ax.set_xbound(-cropsize_x, cropsize_x)
                ax.set_ybound(-cropsize_y, cropsize_y)

            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

        plt.draw()
        return to_return

    def _display_after_optic(self, optic, default_nplanes=2, **kwargs):
        """ Convenience function for displaying a wavefront during propagations.

        Checks for hint information attached to either the wavefront or the
        current optic, and uses that to configure the plot as desired.
        Called from within the various propagate() functions.

        Parameters
        ----------
        optic : OpticalElement instance
            An optic that might have display hint information attached
        default_nplanes :
            How many rows to use for the display, if this is not
            already annotated onto this wavefront object itself.

        Returns the plot axes instance.
        """
        display_what = getattr(optic, 'wavefront_display_hint', 'best')
        display_vmax = getattr(optic, 'wavefront_display_vmax_hint', None)
        display_vmin = getattr(optic, 'wavefront_display_vmin_hint', None)
        display_crop = getattr(optic, 'wavefront_display_imagecrop', None)
        display_nrows = getattr(self, '_display_hint_expected_nplanes', default_nplanes)

        ax = self.display(what=display_what,
                          row=None,
                          nrows=display_nrows,
                          colorbar=False,
                          vmax=display_vmax, vmin=display_vmin,
                          imagecrop=display_crop,
                          **kwargs)
        if hasattr(optic, 'display_annotate'):
            optic.display_annotate(optic, ax)  # atypical calling convention needed empirically

        return ax

    # add convenient properties for intensity, phase, amplitude, total_flux
    @property
    def amplitude(self):
        """Electric field amplitude of the wavefront """
        return np.abs(self.wavefront)

    @property
    def intensity(self):
        """Electric field intensity of the wavefront (i.e. field amplitude squared)"""
        if accel_math._USE_NUMEXPR:
            w = self.wavefront
            return ne.evaluate("real(abs(w))**2")
        else:
            return np.abs(self.wavefront) ** 2

    @property
    def phase(self):
        """Phase of the wavefront, in radians"""
        return np.angle(self.wavefront)

    @property
    def shape(self):
        """ Shape of the wavefront array"""
        return self.wavefront.shape

    @property
    def dtype(self):
        """ Numpy Data type """
        return self.wavefront.dtype

    @property
    def total_intensity(self):
        """Integrated intensity over the entire spatial/angular extent of the wavefront"""
        return self.intensity.sum()

    # methods for wavefront propagation:
    @abstractmethod
    def propagate_to(self, optic):
        """ Placeholder for wavefront propagation.
        To be implemented by subclasses
        """
        pass

    def _resample_wavefront_pixelscale(self, detector):
        """ Resample a wavefront to a desired detector sampling.

        The interpolation is done via the scipy.ndimage.zoom function, by default
        using cubic interpolation.  If you wish a different order of interpolation,
        set the `.interp_order` attribute of the detector instance.

        Parameters
        ----------
        detector : Detector class instance
            Detector that defines the desired pixel scale

        Returns
        -------
        The wavefront object is modified to have the appropriate pixel scale and spatial extent.

        """
        import scipy.interpolate

        pixscale_ratio = (self.pixelscale / detector.pixelscale).decompose().value
        _log.info("Resampling wavefront to detector with {} pixels and {}. Zoom factor is {:.5f}".format(
            detector.shape, detector.pixelscale, pixscale_ratio))

        _log.debug("Wavefront pixel scale:        {:.3f}".format(self.pixelscale.to(detector.pixelscale.unit)))
        _log.debug("Desired detector pixel scale: {:.3f}".format(detector.pixelscale))
        _log.debug("Wavefront FOV:        {} pixels, {:.3f}".format(self.shape,
                                                                    self.shape[0]*u.pixel*self.pixelscale.to(
                                                                    detector.pixelscale.unit)))
        _log.debug("Desired detector FOV: {} pixels, {:.3f}".format(detector.shape,
                                                                    detector.shape[0]*u.pixel*detector.pixelscale))

        def make_axis(npix, step):
            """ Helper function to make coordinate axis for interpolation """
            return step * np.arange(-npix // 2, npix // 2, dtype=np.float64)

        # Provide 2-pixel margin around image to reduce interpolation errors at edge, but also make
        # sure that image is centered properly after it gets cropped down to detector size
        margin = 2
        crop_shape = [margin + shape for shape in self.wavefront.shape]

        # Crop wavefront down to detector size + margin- don't waste computation interpolating
        # parts of plane that get cropped out later anyways
        cropped_wf = utils.pad_or_crop_to_shape(self.wavefront, crop_shape)

        # Input and output axes for interpolation.  The interpolated wavefront will be evaluated
        # directly onto the detector axis, so don't need to crop afterwards.
        x_in = make_axis(crop_shape[0], self.pixelscale.to(u.m/u.pix).value)
        y_in = make_axis(crop_shape[1], self.pixelscale.to(u.m/u.pix).value)
        x_out = make_axis(detector.shape[0], detector.pixelscale.to(u.m/u.pix).value)
        y_out = make_axis(detector.shape[1], detector.pixelscale.to(u.m/u.pix).value)

        def interpolator(arr):
            """
            Bind arguments to scipy's RectBivariateSpline function.
            For data on a regular 2D grid, RectBivariateSpline is more efficient than interp2d.
            """
            return scipy.interpolate.RectBivariateSpline(
                x_in, y_in, arr, kx=detector.interp_order, ky=detector.interp_order)

        # Interpolate real and imaginary parts separately
        real_resampled = interpolator(cropped_wf.real)(x_out, y_out)
        imag_resampled = interpolator(cropped_wf.imag)(x_out, y_out)
        new_wf = real_resampled + 1j * imag_resampled

        # enforce conservation of energy:
        new_wf *= 1. / pixscale_ratio

        self.ispadded = False   # if a pupil detector, avoid auto-cropping padded pixels on output
        self.wavefront = new_wf
        self.pixelscale = detector.pixelscale

    @utils.quantity_input(Xangle=u.arcsec, Yangle=u.arcsec)
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
        if self.planetype == PlaneType.image:
            raise NotImplementedError("Are you sure you want to tilt a wavefront in an _IMAGE plane?")

        if np.abs(Xangle) > 0 or np.abs(Yangle) > 0:
            xangle_rad = Xangle.to(u.radian).value
            yangle_rad = Yangle.to(u.radian).value

            if isinstance(self.pixelscale, u.Quantity):
                pixelscale = self.pixelscale.to(u.m / u.pixel).value
            else:
                pixelscale = self.pixelscale

            npix = self.wavefront.shape[0]
            V, U = np.indices(self.wavefront.shape, dtype=_float())
            V -= (npix - 1) / 2.0
            V *= pixelscale
            U -= (npix - 1) / 2.0
            U *= pixelscale

            tiltphasor = np.exp(2.0j * np.pi * (U * xangle_rad + V * yangle_rad) / self.wavelength.to(u.meter).value)
            self.wavefront *= tiltphasor
            self.history.append("Tilted wavefront by "
                                "X={:2.2}, Y={:2.2} arcsec".format(Xangle, Yangle))

        else:
            _log.warning("Wavefront.tilt() called, but requested tilt was zero. No change.")

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
        self.wavefront = rot_real + 1.j * rot_imag

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

    @abstractmethod
    def coordinates(self):
        """ Return Y, X coordinates for this wavefront, in the manner of numpy.indices()
        """
        pass


class Wavefront(BaseWavefront):
    """ Wavefront in the Fraunhofer approximation: a monochromatic wavefront that
    can be transformed between pupil and image planes only, not to intermediate planes

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
        For PlaneType.image PLANE wavefronts, use this pixel scale.
    oversample : int, optional
        how much to oversample by in FFTs. Default is 2.
        Note that final propagations to Detectors use a different algorithm
        and, optionally, a separate oversampling factor.
    dtype : numpy.dtype, optional
        default is double complex.

    """

    @utils.quantity_input(wavelength=u.meter, diam=u.meter, pixelscale=u.arcsec / u.pixel)
    def __init__(self, wavelength=1e-6 * u.meter, npix=1024, dtype=None, diam=8.0 * u.meter,
                 oversample=2, pixelscale=None):
        super(Wavefront, self).__init__(wavelength=wavelength,
                                        npix=npix,
                                        dtype=dtype,
                                        diam=diam,
                                        oversample=oversample)

        if pixelscale is None:
            self.pixelscale = self.diam / (npix * u.pixel)  # scale in meters/pix or arcsec/pix, as appropriate
            self.planetype = PlaneType.pupil  # are we at image or pupil?
        else:
            self.pixelscale = pixelscale  # scale in meters/pix or arcsec/pix, as appropriate
            self.planetype = PlaneType.image

        self._last_transform_type = None  # later used to track MFT vs FFT pixel coord centering in coordinates()

        self.fov = None  # Field of view in arcsec. Applies to an image plane only.

    def propagate_to(self, optic):
        """Propagates a wavefront object to the next optic in the list.
        Modifies this wavefront object itself.

        Transformations between pupil and detector planes use MFT or inverse MFT.
        Transformations between pupil and other (non-detector) image planes use FFT or inverse FFT, unless
        explicitly tagged to use MFT via a propagation hint.
        Transformations from any frame through a rotation or coordinate transform plane simply transform the
        wavefront accordingly.

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.
        """
        if self.planetype == optic.planetype:
            if isinstance(optic, Detector):
                _log.debug("  Resampling wavefront to match detector pixellation.")
                self._resample_wavefront_pixelscale(optic)
            else:
                _log.debug("  Wavefront and optic %s already at same plane type, no propagation needed." % optic.name)
            self.current_plane_index += 1
            return
        else:
            msg = "  Propagating wavefront to %s. " % str(optic)
            _log.debug(msg)
            self.history.append(msg)

        if optic.planetype == PlaneType.rotation:  # rotate
            self.rotate(optic.angle)
            self.location = 'after ' + optic.name
        elif optic.planetype == PlaneType.inversion:  # invert coordinates
            self.invert(axis=optic.axis)
            self.location = 'after ' + optic.name
        elif ((optic.planetype == PlaneType.detector or getattr(optic, 'propagation_hint', None) == 'MFT')
                and self.planetype == PlaneType.pupil):  # from pupil to detector in image plane: use MFT
            self._propagate_mft(optic)
            self.location = 'before ' + optic.name
        elif (optic.planetype == PlaneType.pupil and self.planetype == PlaneType.image and
                self._last_transform_type == 'MFT'):
            # inverse MFT detector to pupil
            # n.b. transforming PlaneType.pupil -> PlaneType.detector results in self.planetype == PlaneType.image
            # while setting _last_transform_type to MFT
            self._propagate_mft_inverse(optic)
            self.location = 'before ' + optic.name
        elif self.planetype == PlaneType.image and optic.planetype == PlaneType.detector:
            raise NotImplementedError('image plane directly to detector propagation (resampling!) not implemented yet')
        else:
            self._propagate_fft(optic)  # FFT pupil to image or image to pupil
            self.location = 'before ' + optic.name

        self.current_plane_index += 1

    def _propagate_fft(self, optic):
        """ Propagate from pupil to image or vice versa using a padded FFT

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.

        """
        if self.oversample > 1 and not self.ispadded:  # add padding for oversampling, if necessary
            assert self.oversample == optic.oversample
            self.wavefront = utils.pad_to_oversample(self.wavefront, self.oversample)
            self.ispadded = True
            if optic.verbose:
                _log.debug("    Padded WF array for oversampling by %dx" % self.oversample)
            self.history.append("    Padded WF array for oversampling by %dx" % self.oversample)

        # Set up for computation - figure out direction & normalization
        if self.planetype == PlaneType.pupil and optic.planetype == PlaneType.image:
            fft_forward = True

            # (pre-)update state:
            self.planetype = PlaneType.image
            self.pixelscale = (self.wavelength / self.diam * u.radian / self.oversample).to(u.arcsec) / u.pixel
            self.fov = self.wavefront.shape[0] * u.pixel * self.pixelscale
            self.history.append('   FFT {},  to IMAGE plane  scale={:.4f}'.format(self.wavefront.shape, self.pixelscale))

        elif self.planetype == PlaneType.image and optic.planetype == PlaneType.pupil:
            fft_forward = False

            # (pre-)update state:
            self.planetype = PlaneType.pupil
            self.pixelscale = self.diam * self.oversample / (self.wavefront.shape[0] * u.pixel)
            self.history.append('   FFT {},  to PUPIL scale={:.4f}'.format(self.wavefront.shape, self.pixelscale))

        # do FFT
        if conf.enable_flux_tests: _log.debug("\tPre-FFT total intensity: " + str(self.total_intensity))
        if conf.enable_speed_tests: t0 = time.time()

        self.wavefront = accel_math.fft_2d(self.wavefront, forward=fft_forward)

        if fft_forward:
            # FFT produces pixel-centered images by default, unless the _image_centered param
            # has already been set by an FQPM_FFT_aligner class
            if self._image_centered != 'corner':
                self._image_centered = 'pixel'

        self._last_transform_type = 'FFT'

        if conf.enable_speed_tests:
            t1 = time.time()
            _log.debug("\tTIME %f s\t for the FFT" % (t1 - t0))

        if conf.enable_flux_tests:
            _log.debug("\tPost-FFT total intensity: " + str(self.total_intensity))

    def _propagate_mft(self, det):
        """ Compute from pupil to an image using the Soummer et al. 2007 MFT algorithm

        Parameters
        -----------
        det : OpticalElement, must be of type DETECTOR
            The target optical plane to propagate to."""

        assert self.planetype == PlaneType.pupil
        assert (det.planetype == PlaneType.detector or
                getattr(det, 'propagation_hint', None) == 'MFT')

        if self.ispadded:
            # pupil plane is padded - trim that out since it's not needed
            self.wavefront = utils.removePadding(self.wavefront, self.oversample)
            self.ispadded = False
        self._preMFT_pupil_shape = self.wavefront.shape  # save for possible inverseMFT
        self._preMFT_pupil_pixelscale = self.pixelscale  # save for possible inverseMFT

        # the arguments for the matrixDFT are
        # - wavefront (assumed to fill the input array)
        # - focal plane size in lambda/D units
        # - number of pixels on a side in focal plane array.

        # extract everything from Quantities to regular scalars here
        lam_d = ((self.wavelength / self.diam) * u.radian).to(u.arcsec).value

        det_fov_lam_d = det.fov_arcsec.to(u.arcsec).value / lam_d
        det_calc_size_pixels = det.fov_pixels.to(u.pixel).value * det.oversample

        mft = MatrixFourierTransform(centering='ADJUSTABLE', verbose=False)
        if not np.isscalar(det_fov_lam_d):  # hasattr(det_fov_lam_d,'__len__'):
            msg = '    Propagating w/ MFT: {:.4f}     fov=[{:.3f},{:.3f}] lam/D    npix={} x {}'.format(
                det.pixelscale / det.oversample, det_fov_lam_d[0], det_fov_lam_d[1],
                det_calc_size_pixels[0], det_calc_size_pixels[1])
        else:
            msg = '    Propagating w/ MFT: {:.4f}     fov={:.3f} lam/D    npix={:d}'.format(
                det.pixelscale / det.oversample, det_fov_lam_d, int(det_calc_size_pixels))
        _log.debug(msg)
        self.history.append(msg)
        det_offset = det.det_offset if hasattr(det, 'det_offset') else (0, 0)

        _log.debug('      MFT method = ' + mft.centering)

        # det_offset controls how to shift the PSF.
        # it gives the coordinates (X, Y) relative to the exact center of the array
        # for the location of the phase center of a converging perfect spherical wavefront.
        # This is where a perfect PSF would be centered. Of course any tilts, comas, etc, from the OPD
        # will probably shift it off elsewhere for an entirely different reason, too.
        self.wavefront = mft.perform(self.wavefront, det_fov_lam_d, det_calc_size_pixels, offset=det_offset)
        _log.debug("     Result wavefront: at={0} shape={1} ".format(
            self.location, str(self.shape)))
        self._last_transform_type = 'MFT'

        self.planetype = PlaneType.image
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

        assert self.planetype == PlaneType.image
        assert pupil.planetype == PlaneType.pupil

        # the arguments for the matrixDFT are
        # - wavefront (assumed to fill the input array)
        # - focal plane size in lambda/D units
        # - number of pixels on a side in focal plane array.

        # Try to transform to whatever the intrinsic scale of the next pupil is.
        # but if this ends up being a scalar (meaning it is an AnalyticOptic) then
        # just go back to our own prior shape and pixel scale.
        if pupil_npix is None:
            if pupil.shape is not None and pupil.shape[0] != 1:
                # Use next optic's shape, extent, and pixelscale to define the target sampling
                pupil_npix = pupil.shape[0]
                next_pupil_diam = pupil.shape[0]*pupil.pixelscale*u.pixel
                _log.debug("Got post-invMFT pupil npix from next optic: {} pix, {} diam".format(pupil_npix, next_pupil_diam))
            else:
                # Use the prior pupil's shape, extent, and pixelscale to define the target sampling
                pupil_npix = self._preMFT_pupil_shape[0]
                next_pupil_diam = self.diam
                _log.debug("Got post-invMFT pupil npix from pre-MFT pupil: {} pix, {} diam ".format(pupil_npix, self.diam))

        # extract everything from Quantities to regular scalars here
        lam_d = (self.wavelength / next_pupil_diam * u.radian).to(u.arcsec).value
        det_fov_lam_d = self.fov.to(u.arcsec).value / lam_d

        mft = MatrixFourierTransform(centering='ADJUSTABLE', verbose=False)

        # these can be either scalar or 2-element lists/tuples/ndarrays
        msg_pixscale = ('{0:.4f}'.format(self.pixelscale) if np.isscalar(self.pixelscale.value) else
                        '{0:.4f} x {1:.4f} arcsec/pix'.format(self.pixelscale.value[0], self.pixelscale.value[1]))
        msg_det_fov = ('{0:.4f} lam/D'.format(det_fov_lam_d) if np.isscalar(det_fov_lam_d) else
                       '{0:.4f} x {1:.4f}  lam/D'.format(det_fov_lam_d[0], det_fov_lam_d[1]))

        msg = '    Propagating w/ InvMFT:  scale={0}    fov={1}    npix={2:d} x {2:d}'.format(
            msg_pixscale, msg_det_fov, pupil_npix)
        _log.debug(msg)
        self.history.append(msg)
        # det_offset = (0,0)  # det_offset not supported for InvMFT (yet...)

        self.wavefront = mft.inverse(self.wavefront, det_fov_lam_d, pupil_npix)
        self._last_transform_type = 'InvMFT'

        self.planetype = PlaneType.pupil
        self.pixelscale = next_pupil_diam / self.wavefront.shape[0] / u.pixel
        self.diam = next_pupil_diam

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
        y, x = np.indices(shape, dtype=_float())
        pixelscale_mpix = pixelscale.to(u.meter / u.pixel).value if isinstance(pixelscale, u.Quantity) else pixelscale
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
        y, x = np.indices(shape, dtype=_float())
        pixelscale_arcsecperpix = pixelscale.to(u.arcsec / u.pixel).value
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

        if self.planetype == PlaneType.pupil:
            return type(self).pupil_coordinates(self.shape, self.pixelscale)
        elif self.planetype == PlaneType.image:
            return Wavefront.image_coordinates(self.shape, self.pixelscale,
                                               self._last_transform_type, self._image_centered)
        else:
            raise RuntimeError("Unknown plane type (should be pupil or image!)")

    @classmethod
    def from_fresnel_wavefront(cls, fresnel_wavefront, verbose=False):
        """Convert a Fresnel type wavefront object to a Fraunhofer one

        Note, this function implicitly assumes this wavefront is at a
        pupil plane, so the resulting Fraunhofer wavefront will have
        pixelscale in meters/pix rather than arcsec/pix.

        Parameters
        ----------
        fresnel_wavefront : Wavefront
            The (Fresnel-type) wavefront to be converted.

        """
        # Generate a Fraunhofer wavefront with the same sampling
        wf = fresnel_wavefront
        beam_diam = (wf.wavefront.shape[0]//wf.oversample) * wf.pixelscale*u.pixel
        new_wf = Wavefront(diam=beam_diam,
                           npix=wf.shape[0]//wf.oversample,
                           oversample=wf.oversample,
                           wavelength=wf.wavelength)
        if verbose:
            print(wf.pixelscale, new_wf.pixelscale, new_wf.shape)
        # Deal with metadata
        new_wf.history = wf.history.copy()
        new_wf.history.append("Converted to Fraunhofer propagation")
        new_wf.history.append("  Fraunhofer array pixel scale = {:.4g}, oversample = {}".format(new_wf.pixelscale, new_wf.oversample))
        # Copy over the contents of the array
        new_wf.wavefront = utils.pad_or_crop_to_shape(wf.wavefront, new_wf.shape)
        # Copy over misc internal info
        if hasattr(wf, '_display_hint_expected_nplanes'):
            new_wf._display_hint_expected_nplanes = wf._display_hint_expected_nplanes
        new_wf.current_plane_index = wf.current_plane_index
        new_wf.location = wf.location

        return new_wf

# ------ core Optical System classes -------


class BaseOpticalSystem(ABC):
    """ Abstract Base class for optical systems

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
                 npix=None, pupil_diameter=None):
        self.name = name
        self.verbose = verbose
        self.planes = []  # List of OpticalElements
        self.oversample = oversample
        self.npix = npix
        self.pupil_diameter = pupil_diameter

        self.source_offset_r = 0  # = np.zeros((2))     # off-axis tilt of the source, in ARCSEC
        self.source_offset_theta = 0  # in degrees CCW

        self.intermediate_wfs = None  #
        if self.verbose:
            _log.info("Initialized OpticalSystem: " + self.name)

    def __getitem__(self, num):
        return self.planes[num]

    def __len__(self):
        return len(self.planes)

    def _add_plane(self, optic, index=None, logstring=""):
        """ utility helper function for adding a generic plane """
        if index is None:
            self.planes.append(optic)
        else:
            self.planes.insert(index, optic)
        if self.verbose: _log.info("Added {}: {}".format(logstring, optic.name))
        return optic

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
            Pixel scale in arcsec/pixel (or m/pixel for Fresnel optical systems)
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
            oversample = getattr(self, 'oversample', 1)
            # assume oversample is 1 if not present as an attribute; needed for
            # compatibility use in subclass FresnelOpticalSystem.
        optic = Detector(pixelscale, oversample=oversample, **kwargs)

        return self._add_plane(optic, index=index,
                               logstring="detector with pixelscale={} and oversampling={}".format(
                                   pixelscale,
                                   oversample))

    @abstractmethod
    def propagate(self, wavefront):
        """Propagate a wavefront through this system
        and return the output wavefront."""
        pass

    @abstractmethod
    @utils.quantity_input(wavelength=u.meter)
    def input_wavefront(self, wavelength=1e-6*u.meter):
        """Create an input wavefront suitable for propagation"""
        pass

    @utils.quantity_input(wavelength=u.meter)
    def calc_psf(self, wavelength=1e-6,
                 weight=None,
                 save_intermediates=False,
                 save_intermediates_what='all',
                 display=False,
                 return_intermediates=False,
                 return_final=False,
                 source=None,
                 normalize='first',
                 display_intermediates=False):
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
            wavelength = np.asarray([wavelength.value], dtype=_float()) * wavelength.unit

        if weight is None:
            weight = [1.0] * len(wavelength)

        if len(tuple(wavelength)) != len(tuple(weight)):
            raise ValueError("Input source has different number of weights and wavelengths...")

        # loop over wavelengths
        if self.verbose:
            _log.info("Calculating PSF with %d wavelengths" % (len(wavelength)))
        outfits = None
        intermediate_wfs = None
        if save_intermediates or return_intermediates:
            _log.info("User requested saving intermediate wavefronts in call to poppy.calc_psf")
            retain_intermediates = True
        else:
            retain_intermediates = False

        normwts = np.asarray(weight, dtype=_float())
        normwts /= normwts.sum()

        _USE_FFTW = (conf.use_fftw and accel_math._FFTW_AVAILABLE)
        if _USE_FFTW:
            utils.fftw_load_wisdom()

        if conf.use_multiprocessing and len(wavelength) > 1:  # ######## Parallellized computation ############
            # Avoid a Mac OS incompatibility that can lead to hard-to-reproduce crashes.
            # see issues #23 and #176

            if _USE_FFTW:
                _log.warning('IMPORTANT WARNING: Python multiprocessing and fftw3 do not appear to play well together. '
                             'This may crash intermittently')
                _log.warning('   We suggest you set poppy.conf.use_fftw to False if you want to use multiprocessing().')
            if display:
                _log.warning('Display during calculations is not supported for multiprocessing mode. '
                             'Please set poppy.conf.use_multiprocessing = False if you want to use display=True.')
                _log.warning('(Plot the returned PSF with poppy.utils.display_psf.)')

            if return_intermediates:
                _log.warning('Memory usage warning: When preserving intermediate  planes in multiprocessing mode, '
                             'memory usage scales with the number of planes times number of wavelengths. Disable '
                             'use_multiprocessing if you are running out of memory.')
            if save_intermediates:
                _log.warning('Saving intermediate steps does not take advantage of multiprocess parallelism. '
                             'Set save_intermediates=False for improved speed.')

            # do *NOT* just blindly try to create as many processes as one has CPUs, or one per wavelength either
            # This is a memory-intensive task so that can end up swapping to disk and thrashing IO
            nproc = conf.n_processes if conf.n_processes > 1 \
                else utils.estimate_optimal_nprocesses(self, nwavelengths=len(wavelength))
            nproc = min(nproc, len(wavelength))  # never try more processes than wavelengths.
            # be sure to cast nproc to int below; will fail if given a float even if of integer value

            # Use forkserver method (requires Python >= 3.4) for more robustness, instead of just Pool
            # Resolves https://github.com/mperrin/poppy/issues/23
            ctx = multiprocessing.get_context('forkserver')
            pool = ctx.Pool(int(nproc))

            # build a single iterable containing the required function arguments
            _log.info("Beginning multiprocessor job using {0} processes".format(nproc))
            worker_arguments = [(self, wlen, retain_intermediates, return_final, normalize, _USE_FFTW)
                                for wlen in wavelength]
            results = pool.map(_wrap_propagate_for_multiprocessing, worker_arguments)
            _log.info("Finished multiprocessor job")
            pool.close()

            # Sum all the results up into one array, using the weights
            outfits, intermediate_wfs = results[0]
            outfits[0].data *= normwts[0]
            for idx, wavefront in enumerate(intermediate_wfs):
                intermediate_wfs[idx] *= normwts[0]
            _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                0, len(tuple(wavelength)), wavelength[0]))
            for i in range(1, len(normwts)):
                mono_psf, mono_intermediate_wfs = results[i]
                wave_weight = normwts[i]
                _log.info("got results for wavelength channel {} / {} ({:g} meters)".format(
                    i, len(tuple(wavelength)), wavelength[i]))
                outfits[0].data += mono_psf[0].data * wave_weight
                for idx, wavefront in enumerate(mono_intermediate_wfs):
                    intermediate_wfs[idx] += wavefront * wave_weight
            outfits[0].header.add_history("Multiwavelength PSF calc using {} processes completed.".format(nproc))

        else:  # ######### single-threaded computations (may still use multi cores if FFTW enabled ######
            if display:
                plt.clf()
            for wlen, wave_weight in zip(wavelength, normwts):
                mono_psf, mono_intermediate_wfs = self.propagate_mono(
                    wlen,
                    retain_intermediates=retain_intermediates,
                    retain_final=return_final,
                    display_intermediates=display_intermediates,
                    normalize=normalize
                )

                if outfits is None:
                    # for the first wavelength processed, set up the arrays where we accumulate the output
                    outfits = mono_psf
                    outfits[0].data *= wave_weight
                    intermediate_wfs = mono_intermediate_wfs
                    for wavefront in intermediate_wfs:
                        wavefront *= wave_weight  # modifies Wavefront in-place
                else:
                    # for subsequent wavelengths, scale and add the data to the existing arrays
                    outfits[0].data += mono_psf[0].data * wave_weight
                    for idx, wavefront in enumerate(mono_intermediate_wfs):
                        intermediate_wfs[idx] += wavefront * wave_weight

            # Display WF if requested.
            #  Note - don't need to display here if we are showing all steps already
            if display and not display_intermediates:
                cmap = getattr(matplotlib.cm, conf.cmap_sequential)
                cmap.set_bad('0.3')
                halffov_x = outfits[0].header['PIXELSCL'] * outfits[0].data.shape[1] / 2
                halffov_y = outfits[0].header['PIXELSCL'] * outfits[0].data.shape[0] / 2
                extent = [-halffov_x, halffov_x, -halffov_y, halffov_y]
                unit = "arcsec"
                vmax = outfits[0].data.max()
                vmin = vmax / 1e4
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)  # vmin=1e-8,vmax=1e-1)
                plt.xlabel(unit)

                utils.imshow_with_mouseover(outfits[0].data, extent=extent, norm=norm, cmap=cmap,
                                            origin='lower')

        if save_intermediates:
            _log.info('Saving intermediate wavefronts:')
            for idx, wavefront in enumerate(intermediate_wfs):
                filename = 'wavefront_plane_{:03d}.fits'.format(idx)
                wavefront.writeto(filename, what=save_intermediates_what)
                _log.info('  saved {} to {} ({} / {})'.format(save_intermediates_what, filename,
                                                              idx, len(intermediate_wfs)))

        tstop = time.time()
        tdelta = tstop - tstart
        _log.info("  Calculation completed in {0:.3f} s".format(tdelta))
        outfits[0].header.add_history("Calculation completed in {0:.3f} seconds".format(tdelta))

        if _USE_FFTW and conf.autosave_fftw_wisdom:
            utils.fftw_save_wisdom()

        # TODO update FITS header for oversampling here if detector is different from regular?
        waves = np.asarray(wavelength)
        wts = np.asarray(weight)
        mnwave = (waves * wts).sum() / wts.sum()
        outfits[0].header['WAVELEN'] = (mnwave, 'Weighted mean wavelength in meters')
        outfits[0].header['NWAVES'] = (waves.size, 'Number of wavelengths used in calculation')
        for i in range(waves.size):
            outfits[0].header['WAVE' + str(i)] = (waves[i], "Wavelength " + str(i))
            outfits[0].header['WGHT' + str(i)] = (wts[i], "Wavelength weight " + str(i))
        ffttype = "pyFFTW" if _USE_FFTW else "numpy.fft"
        outfits[0].header['FFTTYPE'] = (ffttype, 'Algorithm for FFTs: numpy or fftw')
        outfits[0].header['NORMALIZ'] = (normalize, 'PSF normalization method')

        if self.verbose:
            _log.info("PSF Calculation completed.")

        if return_intermediates | return_final:
            return outfits, intermediate_wfs

        else:
            return outfits

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self,
                       wavelength=1e-6 * u.meter,
                       normalize='first',
                       retain_intermediates=False,
                       retain_final=False,
                       display_intermediates=False):
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
        retain_final : bool
            Should the final complex wavefront be retained? Default: False.
            If True, the second return value of the method will be a single element list
            (for consistency with retain intermediates) containing a `poppy.Wavefront` object
            representing the final optical plane from the calculation.
            Overridden by retain_intermediates.

        Returns
        -------
        final_wf : fits.HDUList
            The final result of the monochromatic propagation as a FITS HDUList
        intermediate_wfs : list
            A list of `poppy.Wavefront` objects representing the wavefront at intermediate optical planes.
            The 0th item is "before first optical plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False and singular if retain_final is True.)
        """

        if conf.enable_speed_tests:
            t_start = time.time()
        if self.verbose:
            _log.info(" Propagating wavelength = {0:g}".format(wavelength))
        wavefront = self.input_wavefront(wavelength)

        kwargs = {'normalize': normalize,
                  'display_intermediates': display_intermediates,
                  'return_intermediates': retain_intermediates}

        # Is there a more elegant way to handle optional return quantities?
        # without making them mandatory.
        if retain_intermediates:
            wavefront, intermediate_wfs = self.propagate(wavefront, **kwargs)
        else:
            wavefront = self.propagate(wavefront, **kwargs)
            intermediate_wfs = []

        if (not retain_intermediates) & retain_final:  # return the full complex wavefront of the last plane.
            intermediate_wfs = [wavefront]

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop - t_start))

        return wavefront.as_fits(), intermediate_wfs

    def display(self, **kwargs):
        """ Display all elements in an optical system on screen.

        Any extra arguments are passed to the `optic.display()` methods of each element.

        """

        planes_to_display = [p for p in self.planes if (not isinstance(p, Detector) and not p._suppress_display)]
        nplanes = len(planes_to_display)
        for i, plane in enumerate(planes_to_display):
            _log.info("Displaying plane {0:s} in row {1:d} of {2:d}".format(plane.name, i + 1, nplanes))
            plane.display(nrows=nplanes, row=i + 1, **kwargs)


class OpticalSystem(BaseOpticalSystem):
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

    # Methods for adding or manipulating optical planes:

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
            warnings.warn("The function argument to add_pupil is deprecated. Please provide an Optic object instead.",
                          DeprecationWarning)
        if optic is None and function is not None:
            # ease of use: 'function' input and providing 'optic' parameter as a string are synonymous.
            optic = function

        if isinstance(optic, OpticalElement):
            # OpticalElement object provided.
            # We can use it directly, but make sure the plane type is set.
            optic.planetype = PlaneType.pupil
        elif isinstance(optic, str):
            # convenience code to instantiate objects from a string name.
            raise NotImplementedError('Setting optics based on strings is now deprecated.')
        elif optic is None and len(kwargs) > 0:  # create image from files specified in kwargs
            # create image from files specified in kwargs
            optic = FITSOpticalElement(planetype=PlaneType.pupil, oversample=self.oversample, **kwargs)
        elif optic is None and len(kwargs) == 0:  # create empty optic.
            from . import optics
            optic = optics.ScalarTransmission()  # placeholder optic, transmission=100%
            optic.planetype = PlaneType.pupil
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

        if isinstance(optic, str):
            function = optic
            optic = None

        if optic is None:
            from . import optics
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
            elif len(kwargs) > 0:  # create image from files specified in kwargs
                fn = FITSOpticalElement
            else:
                fn = optics.ScalarTransmission  # placeholder optic, transmission=100%

            optic = fn(oversample=self.oversample, **kwargs)
            optic.planetype = PlaneType.image
        else:
            optic.planetype = PlaneType.image
            optic.oversample = self.oversample  # these need to match...

        return self._add_plane(optic, index=index, logstring="image plane")

    def describe(self):
        """ Print out a string table describing all planes in an optical system"""
        print(str(self) + "\n\t" + "\n\t".join([str(p) for p in self.planes]))

    # methods for dealing with wavefronts:
    @utils.quantity_input(wavelength=u.meter)
    def input_wavefront(self, wavelength=1e-6 * u.meter):
        """Create a Wavefront object suitable for sending through a given optical system, based on
        the size of the first optical plane, assumed to be a pupil.

        Defining this needs both a number of pixels (npix) and physical size (diam) to set the sampling.

        If this OpticalSystem has a provided `npix` attribute that is not None, use that to
        set the input wavefront size. Otherwise, check if the first optical element has a defined sampling.
        If not, default to 1024 pixels.

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

        # somewhat complicated logic here  (Revised version after #246)
        # First check if the sampling parameters are explicitly defined for the OpticalSystem:
        npix = getattr(self, 'npix', None)
        diam = getattr(self, 'pupil_diameter', None)

        # If not then if we have a first optical plane, check and see if it specifies the entrance sampling:
        if len(self.planes) > 0:
            if npix is None:
                if self.planes[0].shape is not None:
                    npix = self.planes[0].shape[0]
            if diam is None:
                if hasattr(self.planes[0], 'pupil_diam') and self.planes[0].pupil_diam is not None:
                    diam = self.planes[0].pupil_diam

        if npix is None:
            _log.info("You did not define npix either on the OpticalSystem or its first optic; defaulting to 1024 pixels.")
            npix = 1024
        if diam is None:
            raise RuntimeError("You must define an entrance pupil diameter either on the OpticalSystem or its first optic.")

        # if the diameter was specified as an astropy.Quantity, cast it to just a scalar in meters
        if isinstance(diam, u.Quantity):
            diam = diam.to(u.m).value

        inwave = Wavefront(wavelength=wavelength, npix=npix,
                           diam=diam, oversample=self.oversample)
        _log.debug("Creating input wavefront with wavelength={}, npix={:d}, diam={:.3g}, pixel scale={:.3g} meters/pixel".format(
            wavelength, npix, diam, diam / npix))

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
                        if plane.axis == 'x' or plane.axis == 'both':
                            sign_x *= -1
                        if plane.axis == 'y' or plane.axis == 'both':
                            sign_y *= -1
                    elif isinstance(plane, Rotation):
                        rotation_angle += plane.angle * sign_x * sign_y

            # now we must also work out the rotation

            # convert to offset X,Y in arcsec using the usual astronomical angle convention
            angle = (self.source_offset_theta - rotation_angle) * np.pi / 180
            offset_x = sign_x * self.source_offset_r * -np.sin(angle)
            offset_y = sign_y * self.source_offset_r * np.cos(angle)
            inwave.tilt(Xangle=offset_x, Yangle=offset_y)
            _log.debug("Tilted input wavefront by theta_X=%f, theta_Y=%f arcsec. (signs=%d, %d; theta offset=%f) " % (
                       offset_x, offset_y, sign_x, sign_y, rotation_angle))

        inwave._display_hint_expected_nplanes = len(self)  # For display of intermediate steps nicely
        return inwave

    def propagate(self,
                  wavefront,
                  normalize='none',
                  return_intermediates=False,
                  display_intermediates=False):
        """ Core low-level routine for propagating a wavefront through an optical system

        This is a **linear operator** that acts on an input complex wavefront to give an
        output complex wavefront.

        Parameters
        ----------
        wavefront : Wavefront instance
            Wavefront to propagate through this optical system
        normalize : string
            How to normalize the wavefront?
            * 'first' = set total flux = 1 after the first optic, presumably a pupil
            * 'last' = set total flux = 1 after the entire optical system.
            * 'exit_pupil' = set total flux = 1 at the last pupil of the optical system.
        display_intermediates : bool
            Should intermediate steps in the calculation be displayed on screen? Default: False.
        return_intermediates : bool
            Should intermediate steps in the calculation be returned? Default: False.
            If True, the second return value of the method will be a list of `poppy.Wavefront` objects
            representing intermediate optical planes from the calculation.

        Returns a wavefront, and optionally also the intermediate wavefronts after
        each step of propagation.

        """

        if not isinstance(wavefront, Wavefront):
            raise ValueError("First argument to propagate must be a Wavefront.")

        intermediate_wfs = []

        # note: 0 is 'before first optical plane; 1 = 'after first plane and before second plane' and so on
        for optic in self.planes:
            # The actual propagation:
            wavefront.propagate_to(optic)
            wavefront *= optic

            # Normalize if appropriate:
            if normalize.lower() == 'first' and wavefront.current_plane_index == 1:  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower() == 'first=2' and wavefront.current_plane_index == 1:
                # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower() == 'exit_pupil':  # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(np.asarray(
                    [p.planetype is PlaneType.pupil for p in self.planes]))[0].max() + 1
                if wavefront.current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug("normalizing at exit pupil (plane {0}) to 1.0 total intensity".format(
                        wavefront.current_plane_index))
            elif normalize.lower() == 'last' and wavefront.current_plane_index == len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")

            # Optional outputs:
            if conf.enable_flux_tests:
                _log.debug("  Flux === " + str(wavefront.total_intensity))

            if return_intermediates:  # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())
            if display_intermediates:
                wavefront._display_after_optic(optic, default_nplanes=len(self))

        if return_intermediates:
            return wavefront, intermediate_wfs
        else:
            return wavefront

    def _propagation_info(self):
        """ Provide some summary information on the optical propagation calculations that
        would be done for a given optical system

        Right now this mostly is checking whether a given propagation makes use of FFTs or not,
        since the padding for oversampled FFTS majorly affects the max memory used for multiprocessing
        estimation """

        steps = []
        for i, p in enumerate(self.planes):
            if i == 0:
                continue  # no propagation needed for first plane
            if p.planetype == PlaneType.rotation:
                steps.append('rotation')
            elif self.planes[i - 1].planetype == PlaneType.pupil and p.planetype == PlaneType.detector:
                steps.append('MFT')
            elif self.planes[i - 1].planetype == PlaneType.pupil and p.planetype == PlaneType.image:
                if i > 1 and steps[-1] == 'MFT':
                    steps.append('invMFT')
                else:
                    steps.append('FFT')
            elif self.planes[i - 1].planetype == PlaneType.image and p.planetype == PlaneType.detector:
                steps.append('resample')
            else:
                steps.append('FFT')

        output_shape = [a * self.planes[-1].oversample for a in self.planes[-1].shape]
        output_size = output_shape[0] * output_shape[1]

        return {'steps': steps, 'output_shape': output_shape, 'output_size': output_size}


class CompoundOpticalSystem(OpticalSystem):
    """ A concatenation of two or more optical systems,
    acting as a single larger optical system.

    This can be used to combine together multiple existing
    OpticalSystem instances, including mixed lists of both
    Fraunhofer and Fresnel type systems.
    """

    def __init__(self, optsyslist=None, name=None, **kwargs):
        """ Create combined optical system,

        Parameters
        ----------
        optsyslist : List of OpticalSystem and/or FresnelOpticalSystem instances.

        """
        # validate the input optical systems make sense
        if optsyslist is None:
            raise ValueError("Missing required optsyslist argument to CompoundOpticalSystem")
        elif len(optsyslist) == 0:
            raise ValueError("The provided optsyslist argument is an empty list. Must contain at least 1 optical system.")
        for item in optsyslist:
            if not isinstance(item, BaseOpticalSystem):
                raise ValueError("All items in the optical system list must be OpticalSystem instances, not "+repr(item))

        if name is None:
            name = "CompoundOpticalSystem containing {} systems".format(len(optsyslist))
        super(CompoundOpticalSystem, self).__init__(name=name,  **kwargs)

        self.optsyslist = optsyslist

    def _add_plane(self, *args, **kwargs):
        raise RuntimeError("Adding individual optical elements is disallowed for CompoundOpticalSystems."
                           " Add to an OpticalSystem instead.")

    def __len__(self):
        # The length of a compound optical system is the sum of the lengths of the individual systems
        return np.sum([len(optsys) for optsys in self.optsyslist])

    @utils.quantity_input(wavelength=u.meter)
    def input_wavefront(self, wavelength=1e-6 * u.meter):
        """ Create input wavefront for a CompoundOpticalSystem

        Input wavefronts for a compound system are defined by the first OpticalSystem in the list.
        We tweak the _display_hint_expected_planes to reflect the full compound system however.

        """
        inwave = self.optsyslist[0].input_wavefront(wavelength)
        inwave._display_hint_expected_nplanes = len(self)     # For displaying a multi-step calculation nicely
        return inwave

    def propagate(self,
                  wavefront,
                  normalize='none',
                  return_intermediates=False,
                  display_intermediates=False):
        """ Core low-level routine for propagating a wavefront through an optical system

        See docstring of OpticalSystem.propagate for details

        """
        from poppy.fresnel import FresnelOpticalSystem, FresnelWavefront

        if return_intermediates:
            intermediate_wfs = []

        # helper function for logging:
        def loghistory(wavefront, msg):
            _log.debug(msg)
            wavefront.history.append(msg)

        for i, optsys in enumerate(self.optsyslist):
            # If necessary, convert wavefront type.
            if (isinstance(optsys, FresnelOpticalSystem) and
               not isinstance(wavefront, FresnelWavefront)):
                wavefront = FresnelWavefront.from_wavefront(wavefront)
                loghistory(wavefront, "CompoundOpticalSystem: Converted wavefront to Fresnel type")
            elif (not isinstance(optsys, FresnelOpticalSystem) and
                  isinstance(wavefront, FresnelWavefront)):
                wavefront = Wavefront.from_fresnel_wavefront(wavefront)
                loghistory(wavefront, "CompoundOpticalSystem: Converted wavefront to Fraunhofer type")

            # Propagate
            loghistory(wavefront, "CompoundOpticalSystem: Propagating through system {}: {}".format(i+1, optsys.name))
            retval = optsys.propagate(wavefront,
                                      normalize=normalize,
                                      return_intermediates=return_intermediates,
                                      display_intermediates=display_intermediates)

            # Deal with returned item(s) as appropriate
            if return_intermediates:
                wavefront, intermediate_wfs_i = retval
                intermediate_wfs += intermediate_wfs_i
            else:
                wavefront = retval

        if return_intermediates:
            return wavefront, intermediate_wfs
        else:
            return wavefront

    @property
    def planes(self):
        """ A merged list containing all the planes in all the included optical systems """
        out = []
        [out.extend(osys.planes) for osys in self.optsyslist]
        return out

    @planes.setter
    def planes(self, value):
        # needed for compatibility with superclass init
        pass

# ------ core Optical Element Classes ------


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
        either poppy.PlaneType.image or poppy.PlaneType.pupil
    oversample : int
        how much to oversample beyond Nyquist.
    interp_order : int
        the order (0 to 5) of the spline interpolation used if the optic is resized.
    """

    def __init__(self, name="unnamed optic", verbose=True, planetype=PlaneType.unspecified,
                 oversample=1, interp_order=3):

        self.name = name
        """ string. Descriptive Name of this optic"""
        self.verbose = verbose

        self.planetype = planetype  # pupil or image
        self.oversample = oversample  # oversampling factor, none by default
        self.ispadded = False  # are we padded w/ zeros for oversampling the FFT?
        self._suppress_display = False  # should we avoid displaying this optic on screen?
                                        # (useful for 'virtual' optics like FQPM aligner)

        self.amplitude = np.asarray([1.])
        self.opd = np.asarray([0.])
        self.pixelscale = None
        self.interp_order = interp_order

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

    def get_phasor(self, wave):
        """ Compute a complex phasor from an OPD, given a wavelength.

        The returned value should be the complex phasor array as appropriate for
        multiplying by the wavefront amplitude.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        """

        if isinstance(wave, BaseWavefront):
            wavelength = wave.wavelength
        else:
            wavelength = wave
        scale = 2. * np.pi / wavelength.to(u.meter).value

        # set the self.phasor attribute:
        # first check whether we need to interpolate to do this.
        float_tolerance = 0.001  # how big of a relative scale mismatch before resampling?
        if self.pixelscale is not None and hasattr(wave, 'pixelscale') and abs(
                wave.pixelscale - self.pixelscale) / self.pixelscale >= float_tolerance:
            _log.debug("Non-matching pixel scales for wavefront and optic. Need to interpolate. "
                       "Pixelscales: wave {}, optic {}".format(wave.pixelscale, self.pixelscale))
            if hasattr(self, '_resampled_scale') and abs(
                    self._resampled_scale - wave.pixelscale) / self._resampled_scale >= float_tolerance:
                # we already did this same resampling, so just re-use it!
                self.phasor = self._resampled_amplitude * np.exp(1.j * self._resampled_opd * scale)
            else:
                # raise NotImplementedError("Need to implement resampling.")
                zoom = (self.pixelscale / wave.pixelscale).decompose().value
                original_opd = self.get_opd(wave)
                resampled_opd = scipy.ndimage.interpolation.zoom(original_opd, zoom,
                                                                 output=original_opd.dtype,
                                                                 order=self.interp_order)
                original_amplitude = self.get_transmission(wave)
                resampled_amplitude = scipy.ndimage.interpolation.zoom(original_amplitude, zoom,
                                                                       output=original_amplitude.dtype,
                                                                       order=self.interp_order)
                _log.debug("resampled optic to match wavefront via spline interpolation by a" +
                           " zoom factor of {:.3g}".format(zoom))
                _log.debug("resampled optic shape: {}   wavefront shape: {}".format(resampled_amplitude.shape,
                                                                                    wave.shape))

                lx, ly = resampled_amplitude.shape
                # crop down to match size of wavefront:
                lx_w, ly_w = wave.amplitude.shape
                border_x = np.abs(lx - lx_w) // 2
                border_y = np.abs(ly - ly_w) // 2
                if (self.pixelscale * self.amplitude.shape[0] < wave.pixelscale * wave.amplitude.shape[0]) or (
                        self.pixelscale * self.amplitude.shape[1] < wave.pixelscale * wave.amplitude.shape[0]):
                    _log.warning("After resampling, optic phasor shape " + str(np.shape(resampled_opd)) +
                                 " is smaller than input wavefront " + str(
                                 (lx_w, ly_w)) + "; will zero-pad the rescaled array.")
                    self._resampled_opd = np.zeros([lx_w, ly_w])
                    self._resampled_amplitude = np.zeros([lx_w, ly_w])

                    self._resampled_opd[border_x:border_x + resampled_opd.shape[0],
                                        border_y:border_y + resampled_opd.shape[1]] = resampled_opd
                    self._resampled_amplitude[border_x:border_x + resampled_opd.shape[0],
                                              border_y:border_y + resampled_opd.shape[1]] = resampled_amplitude
                    _log.debug("padded an optic with a {:d} x {:d} border to "
                               "optic to match the wavefront".format(border_x, border_y))

                else:
                    self._resampled_opd = resampled_opd[border_x:border_x + lx_w, border_y:border_y + ly_w]
                    self._resampled_amplitude = resampled_amplitude[border_x:border_x + lx_w, border_y:border_y + ly_w]
                    _log.debug("trimmed a border of {:d} x {:d} pixels from "
                               "optic to match the wavefront".format(border_x, border_y))

                self.phasor = self._resampled_amplitude * np.exp(1.j * self._resampled_opd * scale)

        else:
            # compute the phasor directly, without any need to rescale.
            if accel_math._USE_NUMEXPR:
                trans = self.get_transmission(wave)
                opd = self.get_opd(wave)
                self.phasor = ne.evaluate("trans * exp(1.j * opd * scale)")
            else:
                self.phasor = self.get_transmission(wave) * np.exp(1.j * self.get_opd(wave) * scale)

        # check whether we need to pad or crop the array before returning or not.
        # note: do not pad the phasor if it's just a scalar!
        if self.phasor.size != 1 and self.phasor.shape != wave.shape:
            # pad to match the wavefront sampling, from whatever sized array we started with.
            # Allows more flexibility for differently sized FITS arrays, so long as they all have the
            # same pixel scale as checked above!
            return utils.pad_or_crop_to_shape(self.phasor, wave.shape)
        else:
            return self.phasor

    @utils.quantity_input(opd_vmax=u.meter, wavelength=u.meter)
    def display(self, nrows=1, row=1, what='intensity', crosshairs=False, ax=None, colorbar=True,
                colorbar_orientation=None, title=None, opd_vmax=0.5e-6 * u.meter, wavelength=1e-6 * u.meter):
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
        wavelength : float, default 1 micron
            For optics with wavelength-dependent behavior, evaluate at this
            wavelength for display.
        """
        if colorbar_orientation is None:
            colorbar_orientation = "horizontal" if nrows == 1 else 'vertical'

        if self.planetype is PlaneType.pupil:
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
        units = "[arcsec]" if self.planetype == PlaneType.image else "[meters]"
        if nrows > 1:
            # for display inside an optical system, we repurpose the units display to label the plane
            units = self.name + "\n" + units
            # and wrap long lines if necessary
            if len(units) > 20:
                units = "\n".join(textwrap.wrap(units, 20))

        if self.pixelscale is not None:
            if self.pixelscale.decompose().unit == u.m / u.pix:
                halfsize = self.pixelscale.to(u.m / u.pix).value * self.amplitude.shape[0] / 2
            elif self.pixelscale.decompose().unit == u.radian / u.pix:
                halfsize = self.pixelscale.to(u.arcsec / u.pix).value * self.amplitude.shape[0] / 2
            else:
                halfsize = self.pixelscale.value * self.amplitude.shape[0] / 2
                _log.warning("Using pixelscale value without conversion, units not recognized.")
            _log.debug("Display pixel scale = {} ".format(self.pixelscale))
        else:
            # TODO not sure this code path ever gets used - since pixelscale is set temporarily
            # in AnalyticOptic.display
            _log.debug("No defined pixel scale - this must be an analytic optic")
            halfsize = 1.0
        extent = [-halfsize, halfsize, -halfsize, halfsize]

        temp_wavefront = Wavefront(wavelength)
        ampl = self.get_transmission(temp_wavefront)
        opd = self.get_opd(temp_wavefront)
        opd[np.where(ampl == 0)] = np.nan

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
            ax2.set_ylabel('')  # suppress redundant label which duplicates the intensity plot's label
            if title is not None:
                plt.suptitle(title)
            return ax, ax2
        elif what == 'amplitude':
            plot_array = ampl
            default_title = 'Transmissivity'
            cb_label = 'Fraction'
            cb_values = [0, 0.25, 0.5, 0.75, 1.0]
            cmap = cmap_amp
            norm = norm_amp
        elif what == 'intensity':
            plot_array = ampl ** 2
            default_title = "Transmittance"
            cb_label = 'Fraction'
            cb_values = [0, 0.25, 0.5, 0.75, 1.0]
            cmap = cmap_amp
            norm = norm_amp
        elif what == 'phase':
            warnings.warn("displaying 'phase' has been deprecated. Use what='opd' instead.",
                          category=DeprecationWarning)
            plot_array = opd
            default_title = "OPD"
            cb_label = 'waves'
            cb_values = np.array([-1, -0.5, 0, 0.5, 1]) * opd_vmax_m
            cmap = cmap_opd
            norm = norm_opd
        elif what == 'opd':
            plot_array = opd
            default_title = "OPD"
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
            if title is None:
                title = default_title + " for " + self.name
            plt.title(title)
        plt.ylabel(units)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        if colorbar:
            cb = plt.colorbar(ax.images[0], orientation=colorbar_orientation, ticks=cb_values)
            cb.set_label(cb_label)
        if crosshairs:
            ax.axhline(0, ls=":", color='k')
            ax.axvline(0, ls=":", color='k')

        if hasattr(self, 'display_annotate'):
            self.display_annotate(self, ax)  # atypical calling convention needed empirically
            # since Python doesn't seem to automatically pass
            # self as first argument for functions added at
            # run time as attributes?
        return ax

    def __str__(self):
        if self.planetype == PlaneType.pupil:
            return "Pupil plane: {} ".format(self.name)
        elif self.planetype == PlaneType.image:
            desc = "({}x{} pixels, scale={} arcsec/pixel)".format(self.shape[0], self.shape[0],
                                                                  self.pixelscale) if \
                                                                  self.pixelscale is not None else "(Analytic)"
            return "Image plane: %s %s" % (self.name, desc)
        else:
            return "Optic: " + self.name

    @property
    def shape(self):
        """ Return shape of the OpticalElement, as a tuple """
        if hasattr(self, 'amplitude'):
            return self.amplitude.shape
        else:
            return None


class ArrayOpticalElement(OpticalElement):
    """ Defines an arbitrary optic, based on amplitude transmission and/or OPD given as numpy arrays.

    This is a very lightweight wrapper for the base OpticalElement class, which just provides some
    additional convenience features in the initializer..
    """

    def __init__(self, opd=None, transmission=None, pixelscale=None, **kwargs):
        super(ArrayOpticalElement, self).__init__(**kwargs)
        if opd is not None:
            self.opd = opd
        if transmission is not None:
            self.amplitude = transmission
            if opd is None:
                self.opd = np.zeros_like(transmission)

        if pixelscale is not None:
            self.pixelscale = pixelscale


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
        3. in the case of OPDs, a tuple consisting of a path to a datacube and an integer index of
           a slice in that datacube.

    A better interface for slice selection in datacubes is the transmission_index
    and opd_index keyword parameters listed below, but the tuple interface is
    retained for back compatibility with existing code.


    Parameters
    ----------
    name : string
        descriptive name for optic
    transmission, opd : string or fits HDUList
        Either FITS filenames *or* actual fits.HDUList objects for the
        transmission (from 0-1) and opd (in meters)
    transmission_slice, opd_slice : integers, optional
        If either transmission or OPD files are datacubes, you can specify the
        slice index using this argument.
    opdunits : string
        units for the OPD file. Default is 'meters'. can be 'meter', 'meters',
        'micron(s)', 'nanometer(s)', or their SI abbreviations. If this keyword
        is not set explicitly, the BUNIT keyword in the FITS header will be checked.
    planetype : int
        either PlaneType.image or PlaneType.pupil
    oversample : int
        how much to oversample beyond Nyquist.
    flip_x, flip_y : bool
        Should the FITS file be inverted in either of these axes after being
        loaded? Useful for matching coordinate system orientations.  If a flip
        is specified, it takes place prior to any shift or rotation operations.
    shift : tuple of floats, optional
        2-tuple containing X and Y fractional shifts for the pupil. These shifts
        are implemented by rounding them to the nearest integer pixel, and doing
        integer pixel shifts on the data array, without interpolation. If a
        shift is specified, it takes place after any rotation operations.
    shift_x, shift_y : floats, optional
        Alternate way of specifying shifts, given in meters of shift per each axis.
        This is consistent with how AnalyticOpticalElement classes specify shifts.
        If a shift is specified, it takes place after any rotation operations.
        If both shift and shift_x/shift_y are specified, an error is raised.
    rotation : float
        Rotation for that optic, in degrees counterclockwise. This is
        implemented using spline interpolation via the
        scipy.ndimage.interpolation.rotate function.
    pixelscale : optical str or float
        By default, poppy will attempt to determine the appropriate pixel scale
        by examining the FITS header, checking keywords "PUPLSCAL" and 'PIXSCALE'
        for pupil and image planes respectively. If you would like to override
        and use a different keyword, provide that as a string here. Alternatively,
        you can just set a floating point value directly too (in meters/pixel
        or arcsec/pixel, respectively, for pupil or image planes).
    transmission_index, opd_index : ints, optional
        If the input transmission or OPD files are datacubes, provide a scalar
        index here for which cube slice should be used.


    *NOTE:* All mask files must be *squares*.

    Also, please note that the adopted convention is for the spectral throughput
    (transmission) to be given in appropriate units for acting on the *amplitude*
    of the electric field. Thus for example an optic with a uniform transmission
    of 0.5 will reduce the electric field amplitude to 0.5 relative to the input,
    and thus reduce the total power to 0.25. This distinction only matters in the
    case of semitransparent (grayscale) masks.



    """

    def __init__(self, name="unnamed optic", transmission=None, opd=None, opdunits=None,
                 rotation=None, pixelscale=None, planetype=None,
                 transmission_index=None, opd_index=None,
                 shift=None, shift_x=None, shift_y=None,
                 flip_x=False, flip_y=False,
                 **kwargs):

        OpticalElement.__init__(self, name=name, **kwargs)
        self.opd_file = None
        self.amplitude_file = None
        self.amplitude_header = None
        self.opd_header = None
        self._opd_in_radians = False
        self.planetype = planetype

        _log.debug("Trans: " + str(transmission))
        _log.debug("OPD: " + str(opd))

        # ---- Load amplitude transmission file. ---
        if opd is None and transmission is None:  # no input files, so just make a scalar
            _log.warning("No input files specified. You should set transmission=filename or opd=filename.")
            _log.warning("Creating a null optical element. Are you sure that's what you want to do?")
            self.amplitude = np.asarray([1.])
            self.opd = np.asarray([0.])
            self.pixelscale = None
            self.name = "-empty-"
        else:
            # load transmission file.
            if transmission is not None:
                if isinstance(transmission, str):
                    self.amplitude_file = transmission
                    self.amplitude, self.amplitude_header = fits.getdata(self.amplitude_file, header=True)
                    self.amplitude = self.amplitude.astype('=f8')  # ensure native byte order, see #213
                    if self.name == 'unnamed optic':
                        self.name = 'Optic from ' + self.amplitude_file
                    _log.info(self.name + ": Loaded amplitude transmission from " + self.amplitude_file)
                elif isinstance(transmission, fits.HDUList):
                    self.amplitude_file = 'supplied as fits.HDUList object'
                    self.amplitude = transmission[0].data.astype('=f8')  # ensure native byte order, see #213
                    self.amplitude_header = transmission[0].header.copy()
                    if self.name == 'unnamed optic':
                        self.name = 'Optic from fits.HDUList object'
                    _log.info(self.name + ": Loaded amplitude transmission from supplied fits.HDUList object")
                else:
                    raise TypeError('Not sure how to use a transmission parameter of type ' + str(type(transmission)))

                # check for datacube?
                if len(self.amplitude.shape) > 2:
                    if transmission_index is None:
                        _log.info("The supplied pupil amplitude is a datacube but no slice was specified. "
                                  "Defaulting to use slice 0.")
                        transmission_index = 0
                    self.amplitude_slice_index = transmission_index
                    self.amplitude = self.amplitude[self.amplitude_slice_index, :, :]
                    _log.debug(" Datacube detected, using slice ={0}".format(self.amplitude_slice_index))
            else:
                _log.debug("No transmission supplied - will assume uniform throughput = 1 ")
                # if transmission is none, wait until after OPD is loaded, below, and then create a matching
                # amplitude array uniformly filled with 1s.

            # ---- Load OPD file. ---
            if opd is None:
                # if only amplitude set, create an array of 0s with same size.
                self.opd = np.zeros(self.amplitude.shape)
                opdunits = 'meter'  # doesn't matter, it's all zeros, but this will indicate no need to rescale below.

            elif isinstance(opd, fits.HDUList):
                # load from fits HDUList
                self.opd_file = 'supplied as fits.HDUList object'
                self.opd = opd[0].data.astype('=f8')
                self.opd_header = opd[0].header.copy()
                if self.name == 'unnamed optic':
                    self.name = 'OPD from supplied fits.HDUList object'
                _log.info(self.name + ": Loaded OPD from supplied fits.HDUList object")
            elif isinstance(opd, str):
                # load from regular FITS filename
                self.opd_file = opd
                self.opd, self.opd_header = fits.getdata(self.opd_file, header=True)
                self.opd = self.opd.astype('=f8')
                if self.name == 'unnamed optic': self.name = 'OPD from ' + self.opd_file
                _log.info(self.name + ": Loaded OPD from " + self.opd_file)

            elif len(opd) == 2 and isinstance(opd[0], str):
                # if OPD is specified as a 2-element iterable, treat the first element as the filename
                # and 2nd as the slice of a cube.
                self.opd_file = opd[0]
                self.opd_slice = opd[1]
                self.opd, self.opd_header = fits.getdata(self.opd_file, header=True)
                self.opd = self.opd.astype('=f8')
                self.opd = self.opd[self.opd_slice, :, :]
                if self.name == 'unnamed optic':
                    self.name = 'OPD from %s, plane %d' % (self.opd_file, self.opd_slice)
                _log.info(self.name + ": Loaded OPD from  %s, plane %d" % (self.opd_file, self.opd_slice))
            else:
                raise TypeError('Not sure how to use an OPD parameter of type ' + str(type(transmission)))

            # check for datacube?
            if len(self.opd.shape) > 2:
                if opd_index is None:
                    _log.info("The supplied pupil OPD is a datacube but no slice was specified. "
                              "Defaulting to use slice 0.")
                    opd_index = 0
                self.opd_slice = opd_index
                self.opd = self.opd[self.opd_slice, :, :]
                _log.debug(" Datacube detected, using slice ={0}".format(self.opd_slice))

            if transmission is None:
                _log.info("No info supplied on amplitude transmission; assuming uniform throughput = 1")
                self.amplitude = np.ones(self.opd.shape)

            if opdunits is None:
                try:
                    opdunits = self.opd_header['BUNIT']
                except KeyError:
                    _log.error("No opdunits keyword supplied, and BUNIT keyword not found in header. "
                               "Cannot determine OPD units")
                    raise Exception("No opdunit keyword supplied, and BUNIT keyword not found in header. "
                                    "Cannot determine OPD units.")

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
            elif opdunits == 'radian':
                self._opd_in_radians = True
            else:
                raise ValueError(
                    "Got opdunits (or BUNIT header keyword) {}. Valid options "
                    "are meter, micron, nanometer, or radian.".format(repr(opdunits))
                )

            if self.opd_header is not None and not self._opd_in_radians:
                self.opd_header['BUNIT'] = 'meter'

            if len(self.opd.shape) != 2 or self.opd.shape[0] != self.opd.shape[1]:
                _log.debug('OPD shape: ' + str(self.opd.shape))
                raise ValueError("OPD image must be 2-D and square")

            if len(self.amplitude.shape) != 2 or self.amplitude.shape[0] != self.amplitude.shape[1]:
                raise ValueError("Pupil amplitude image must be 2-D and square")

            assert self.amplitude.shape == self.opd.shape, "Amplitude and OPD FITS file shapes are incompatible."
            assert self.amplitude.shape[0] == self.amplitude.shape[1], "Amplitude and OPD FITS files must be square."

            # ---- transformation: inversion ----
            # if an inversion is specified and we're not a null (scalar) opticm then do the inversion:
            if flip_y and len(self.amplitude.shape) == 2:
                self.amplitude = self.amplitude[::-1]
                self.opd = self.opd[::-1]
                _log.debug("Inverted optic in the Y axis")
            if flip_x and len(self.amplitude.shape) == 2:
                self.amplitude = self.amplitude[:, ::-1]
                self.opd = self.opd[:, ::-1]
                _log.debug("Inverted optic in the X axis")

            # ---- transformation: rotation ----
            # If a rotation is specified and we're NOT a null (scalar) optic, then do the rotation:
            if rotation is not None and len(self.amplitude.shape) == 2:
                # do rotation with interpolation, but try to clean up some of the artifacts afterwards.
                # this is imperfect at best, of course...
                self.amplitude = scipy.ndimage.interpolation.rotate(self.amplitude, -rotation,  # negative = CCW
                                                                    reshape=False).clip(min=0, max=1.0)
                wnoise = np.where((self.amplitude < 1e-3) & (self.amplitude > 0))
                self.amplitude[wnoise] = 0
                self.opd = scipy.ndimage.interpolation.rotate(self.opd, -rotation, reshape=False)  # negative = CCW
                _log.info("  Rotated optic by %f degrees counter clockwise." % rotation)
                self._rotation = rotation

            # ---- Determine the pixel scale for this image. ----
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

            # The following logic is convoluted for historical back compatibility.
            # All new files should use PIXELSCL. But we still allow reading in
            # older files with PIXSCALE or PUPLSCAL.
            # This code can probably be simplified.

            if pixelscale is None and self.planetype is None:
                # we don't know which keywords might be present yet, so check for both keywords
                # in both header objects (at least one must be non-None at this point!)
                _log.debug("  Looking for 'PUPLSCAL' or 'PIXSCALE' or 'PIXELSCL' in FITS headers to set "
                           "pixel scale")
                keyword, self.pixelscale = _find_pixelscale_in_headers(
                    ('PUPLSCAL', 'PIXSCALE', 'PIXELSCL'),
                    (self.amplitude_header, self.opd_header)
                )
                if keyword == 'PUPLSCAL':
                    self.planetype = PlaneType.pupil
                else:
                    self.planetype = PlaneType.image
            elif pixelscale is None and self.planetype == PlaneType.image:
                # the planetype tells us which header keyword to check when a keyword is
                # not provided (PIXSCALE for image planes)...
                _, self.pixelscale = _find_pixelscale_in_headers(
                    ('PIXELSCL', 'PIXSCALE'),
                    (self.amplitude_header, self.opd_header)
                )
            elif pixelscale is None and (self.planetype == PlaneType.pupil or self.planetype == _INTERMED):
                # ... likewise for pupil planes
                _, self.pixelscale = _find_pixelscale_in_headers(
                    ('PIXELSCL', 'PUPLSCAL',),
                    (self.amplitude_header, self.opd_header)
                )
            elif isinstance(pixelscale, str):
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
            if self.planetype == PlaneType.image:
                self.pixelscale *= u.arcsec / u.pixel
            else:  # pupil or any other types of plane
                self.pixelscale *= u.meter / u.pixel

            # ---- transformation: shift ----
            # if a shift is specified and we're NOT a null (scalar) optic, then do the shift
            # This has to happen after the pixelscale has been determined, for the shift_x/shift_y path.
            if shift is not None and (shift_x is not None or shift_y is not None):
                raise RuntimeError("You cannot specify both the shift and shift_x/shift_y parameters simultaneously.")
            elif ((shift is not None) or (shift_x is not None or shift_y is not None)) and len(self.amplitude.shape) == 2:
                if shift_x is not None or shift_y is not None:
                    # determine shift using the shift_x and shift_y parameters
                    if shift_x is None: shift_x = 0
                    if shift_y is None: shift_y = 0
                    rollx = int(shift_x/self.pixelscale.to(u.m/u.pixel).value)
                    rolly = int(shift_y/self.pixelscale.to(u.m/u.pixel).value)
                    _log.info("Requested optic shift of ({:6.3f}, {:6.3f}) meters".format(shift_x, shift_y))
                    _log.info("Actual shift applied  = ({:6.3f}, {:6.3f}) pixels".format(rollx, rolly))

                elif shift is not None:
                    # determine shift using the shift tuple
                    if abs(shift[0]) > 0.5 or abs(shift[1]) > 0.5:
                        raise ValueError("You have asked for an implausibly large shift. Remember, "
                                         "shifts should be specified as decimal values between -0.5 and 0.5, "
                                         "a fraction of the total optic diameter. ")
                    rolly = int(np.round(self.amplitude.shape[0] * shift[1]))   # remember Y,X order for shape,
                                                                                # but X,Y order for shift
                    rollx = int(np.round(self.amplitude.shape[1] * shift[0]))
                    _log.info("Requested optic shift of ({:6.3f}, {:6.3f}) fraction of pupil ".format(*shift))
                    _log.info("Actual shift applied   = (%6.3f, %6.3f) " % (
                              rollx * 1.0 / self.amplitude.shape[1], rolly * 1.0 / self.amplitude.shape[0]))
                    self._shift = (rollx * 1.0 / self.amplitude.shape[1], rolly * 1.0 / self.amplitude.shape[0])

                self.amplitude = scipy.ndimage.shift(self.amplitude, (rolly, rollx))
                self.opd = scipy.ndimage.shift(self.opd, (rolly, rollx))

    @property
    def pupil_diam(self):
        """Diameter of the pupil (if this is a pupil plane optic)"""
        return self.pixelscale * (self.amplitude.shape[0] * u.pixel)

    def get_opd(self, wave):
        """ Return the optical path difference, given a wavelength.

        When the OPD map is defined in terms of wavelength-independent
        phase, as in the case of the vector apodizing phase plate
        coronagraph of Snik et al. (Proc. SPIE, 2012), it is converted
        to optical path difference in meters at the given wavelength for
        consistency with the rest of POPPY.

        Parameters
        ----------
        wave : float or obj
            either a scalar wavelength or a Wavefront object

        Returns
        --------
        ndarray giving OPD in meters

        """
        if isinstance(wave, BaseWavefront):
            wavelength = wave.wavelength
        else:
            wavelength = wave
        if self._opd_in_radians:
            return self.opd * wavelength.to(u.m).value / (2 * np.pi)
        return self.opd


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
        self._suppress_display = hide

    def get_phasor(self, wave):
        return 1.0  # no change in wavefront
        # returning this is necessary to allow the multiplication in propagate_mono to be OK

    def display(self, nrows=1, row=1, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplot(nrows, 2, row * 2 - 1)
        plt.text(0.3, 0.3, self.name)
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
            angle *= np.pi / 180
        elif units == 'degrees':
            pass
        else:
            raise ValueError("Unknown value for units='%s'. Must be degrees or radians." % units)
        self.angle = angle

        CoordinateTransform.__init__(self, name="Rotation by %.2f degrees" % angle,
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
        self.axis = axis.lower()
        CoordinateTransform.__init__(self, name=name,
                                     planetype=PlaneType.inversion, hide=hide, **kwargs)

    def __str__(self):
        return "Coordinate Inversion in {} axis".format(self.axis)


# ------ Detector ------

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
        Pixel scale, either in angular units such as arcsec/pixel, or
        (for Fresnel optical systems only) in physical units such as micron/pixel.
        Units should be specified as astropy Quantities. If pixelscale is given as
        a float without an explicit unit, it will be interpreted as in arcsec/pixel.
        Note, this value may be further subdivided by specifying the oversample
        parameter > 1.
    fov_pixels, fov_arcsec : float or astropy.units.Quantity
        The field of view may be specified either in arcseconds or by a number
        of pixels. Either is acceptable and the pixel scale is used to convert
        as needed. You may specify a non-square FOV by providing two elements in
        an iterable.  Note that this follows the usual Python convention of
        ordering axes (Y,X), so put your desired Y axis size first.
        For Fresnel optical systems, if specifying pixelscale in microns/pixel then
        you must specify fov_pixels rather than fov_arcsec.
    oversample : int
        Oversampling factor beyond the detector pixel scale. The returned array will
        have sampling that much finer than the specified pixelscale.
    offset : tuple (X,Y)
        Offset for the detector center relative to a hypothetical off-axis PSF.
        Specifying this lets you pick a different sub-region for the detector
        to compute, if for some reason you are computing a small subarray
        around an off-axis source. (Has not been tested!)

    """

    # Note, pixelscale argument is intentionally not included in the quantity_input decorator; that is
    # specially handled. See the _handle_pixelscale_units_flexibly method
    @utils.quantity_input(fov_pixels=u.pixel, fov_arcsec=u.arcsec)
    def __init__(self, pixelscale=1 * (u.arcsec / u.pixel), fov_pixels=None, fov_arcsec=None, oversample=1,
                 name="Detector", offset=None,
                 **kwargs):
        OpticalElement.__init__(self, name=name, planetype=PlaneType.detector, **kwargs)
        self.pixelscale = self._handle_pixelscale_units_flexibly(pixelscale, fov_pixels)
        self.oversample = oversample

        if fov_pixels is None and fov_arcsec is None:
            raise ValueError("Either fov_pixels or fov_arcsec must be specified!")
        elif fov_pixels is not None:
            self.fov_pixels = np.round(fov_pixels)
            self.fov_arcsec = self.fov_pixels * self.pixelscale
        else:
            # set field of view to closest value possible to requested,
            # consistent with having an integer number of pixels
            self.fov_pixels = np.round((fov_arcsec.to(u.arcsec) / self.pixelscale).to(u.pixel))
            self.fov_arcsec = self.fov_pixels * self.pixelscale
        if np.any(self.fov_pixels <= 0):
            raise ValueError("FOV in pixels must be a positive quantity. Invalid: " + str(self.fov_pixels))

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
        return "Detector plane: {} ({}x{} pixels, {:.3f})".format(self.name, self.shape[1], self.shape[0], self.pixelscale)

    @staticmethod
    def _handle_pixelscale_units_flexibly(pixelscale, fov_pixels):
        """ The unit conventions for pixelscale are tricky; deal with that.
        For historical reasons and API simplicity, the Detector class can be
        used with pixels in angular units (arcsec/pixel) or physical units (micron/pixel).
        The regular @utils.quantity_input decorator won't support that, so we handle it here.
        """
        # This code is adapted from utils.BackCompatibleQuantityInput

        arcsec_per_pixel = u.arcsec/u.pixel
        micron_per_pixel = u.micron/u.pixel

        # Case 1: pixelscale given without units. Treat it as angular units in arcsec/pixel
        if not isinstance(pixelscale, u.Quantity):
            try:
                new_pixelscale = pixelscale * arcsec_per_pixel
            except (ValueError, TypeError):
                raise ValueError("Argument '{0}' to function '{1}'"
                                 " must be a number (not '{3}'), and convertable to"
                                 " units='{2}'.".format('pixelscale', 'Detector.__init__',
                                                        arcsec_per_pixel, pixelscale))

        # Case 2: pixelscale compatible with angular units. Treat it as such.
        elif pixelscale.unit.is_equivalent(arcsec_per_pixel):
            new_pixelscale = pixelscale

        # Case 3: pixelscale compatible with physical units. Treat it as such. Also, in
        # this case, the user *must* specify a value for fov_pixels.
        elif pixelscale.unit.is_equivalent(micron_per_pixel):
            new_pixelscale = pixelscale
            if fov_pixels is None:
                raise ValueError("If you specify the detector pixelscale in microns/pixel or "
                                 "other linear units (not angular), then you must specify the "
                                 "field of view via fov_pixels=<some integer>.")

        # Case 4: some other units. Raise an error.
        else:
            raise ValueError("Argument '{0}' to function '{1}'"
                             " must be a number (not '{2}'), and convertable to"
                             " units=arcsec/pixel or micron/pixel.".format('pixelscale',
                                                                           'Detector.__init__',
                                                                           pixelscale))

        return new_pixelscale
