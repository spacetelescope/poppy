import getpass
import os
import platform
import re
import time
import astropy.io.fits as fits
import astropy.units as units
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage

try:
    import synphot
    _HAS_SYNPHOT = True
except ImportError:
    synphot = None
    _HAS_SYNPHOT = False

from . import poppy_core
from . import optics
from . import utils
from . import conf

import logging

_log = logging.getLogger('poppy')

__all__ = ['Instrument']


class Instrument(object):
    """ A generic astronomical instrument, composed of
        (1) an optical system implemented using POPPY, optionally with several configurations such as
            selectable image plane or pupil plane stops, and
        (2) some defined spectral bandpass(es) such as selectable filters, implemented using synphot.

    This provides the capability to model both the optical and spectral responses of a given system.
    PSFs may be calculated for given source
    spectral energy distributions and output as FITS files, with substantial flexibility.

    It also provides capabilities for modeling some PSF effects not due to wavefront aberrations, for instance
    blurring caused by pointing jitter.


    This is a base class for Instrument functionality - you cannot easily use this directly, but
    rather should subclass it for your particular instrument of interest.   Some of the complexity of this class
    is due to splitting up functionality into many separate routines to allow users to subclass just the relevant
    portions for a given task. There's a fair amount of functionality here but the learning curve is steeper than
    elsewhere in POPPY.

    You will at a minimum want to override the following class methods:

        * get_optical_system
        * _get_filter_list
        * _get_default_nlambda
        * _get_default_fov
        * _get_fits_header

    For more complicated systems you may also want to override:

        * _validate_config
        * _get_synphot_bandpass
        * _apply_jitter
    """

    name = "Instrument"
    pupil = None
    "Aperture for this optical system. May be a FITS filename, FITS HDUList object, or poppy.OpticalElement"
    pupilopd = None
    """Pupil OPD for this optical system. May be a FITS filename, or FITS HDUList.
    If the file contains a datacube, you may set this to a tuple (filename, slice) to select a given slice, or else
    the first slice will be used."""
    options = {}
    """
    A dictionary capable of storing other arbitrary options, for extensibility. The following are all optional, and
    may or may not be meaningful depending on which instrument is selected.

    Parameters
    ----------
    source_offset_r : float
        Radial offset of the target from the center, in arcseconds
    source_offset_theta : float
        Position angle for that offset
    pupil_shift_x, pupil_shift_y : float
        Relative shift of a coronagraphic pupil in X and Y, expressed as a decimal between 0.0-1.0
        Note that shifting an array too much will wrap around to the other side unphysically, but
        for reasonable values of shift this is a non-issue.
    jitter : string "gaussian" or None
        Type of jitter model to apply. Currently only convolution with a Gaussian kernel of specified
        width `jitter_sigma` is implemented. (default: None)
    jitter_sigma : float
        Width of the jitter kernel in arcseconds per axis (default: 0.007 arcsec)
    parity : string "even" or "odd"
        You may wish to ensure that the output PSF grid has either an odd or even number of pixels.
        Setting this option will force that to be the case by increasing npix by one if necessary.

    """
    filter_list = None
    """List of available filter names for this instrument"""
    pixelscale = 0.025
    """Detector pixel scale, in arcseconds/pixel (default: 0.025)"""

    def __init__(self, name="", *args, **kwargs):
        self.name = name
        self.pupil = optics.CircularAperture(*args, **kwargs)
        self.pupilopd = None
        self.options = {}
        self.filter_list, self._synphot_bandpasses = self._get_filter_list()  # List of available filter names

        # create private instance variables. These will be
        # wrapped just below to create properties with validation.
        self._filter = None
        self._rotation = None
        # for caching synphot results.
        self._spectra_cache = {}
        self.filter = self.filter_list[0]

        self.optsys = None  # instance attribute for Optical System

    def __str__(self):
        return "Instrument name=" + self.name

    # create properties with error checking
    @property
    def filter(self):
        """Currently selected filter name (e.g. F200W)"""
        return self._filter

    @filter.setter
    def filter(self, value):
        value = value.upper()  # force to uppercase
        if value not in self.filter_list:
            raise ValueError("Instrument %s doesn't have a filter called %s." % (self.name, value))
        self._filter = value

    # ----- actual optical calculations follow here -----
    def calc_psf(self, outfile=None, source=None, nlambda=None, monochromatic=None,
                 fov_arcsec=None, fov_pixels=None, oversample=None, detector_oversample=None, fft_oversample=None,
                 overwrite=True, display=False, save_intermediates=False, return_intermediates=False,
                 normalize='first'):
        """ Compute a PSF.
        The result can either be written to disk (set outfile="filename") or else will be returned as
        a FITS HDUlist object.


        Output sampling may be specified in one of two ways:

        1) Set `oversample=<number>`. This will use that oversampling factor beyond detector pixels
           for output images, and beyond Nyquist sampling for any FFTs to prior optical planes.
        2) set `detector_oversample=<number>` and `fft_oversample=<other_number>`. This syntax lets
           you specify distinct oversampling factors for intermediate and final planes.

        By default, both oversampling factors are set equal to 2.

        Notes
        -----
        More advanced PSF computation options (pupil shifts, source positions, jitter, ...)
        may be set by configuring the `.options` dictionary attribute of this class.

        Parameters
        ----------
        source : synphot.spectrum.SourceSpectrum or dict
            specification of source input spectrum. Default is a 5700 K sunlike star.
        nlambda : int
            How many wavelengths to model for broadband?
            The default depends on how wide the filter is: (5,3,1) for types (W,M,N) respectively
        monochromatic : float, optional
            Setting this to a wavelength value (in meters) will compute a monochromatic PSF at that
            wavelength, overriding filter and nlambda settings.
        fov_arcsec : float
            field of view in arcsec. Default=5
        fov_pixels : int
            field of view in pixels. This is an alternative to fov_arcsec.
        outfile : string
            Filename to write. If None, then result is returned as an HDUList
        oversample, detector_oversample, fft_oversample : int
            How much to oversample. Default=4. By default the same factor is used for final output
            pixels and intermediate optical planes, but you may optionally use different factors
            if so desired.
        overwrite : bool
            overwrite output FITS file if it already exists?
        display : bool
            Whether to display the PSF when done or not.
        save_intermediates, return_intermediates : bool
            Options for saving to disk or returning to the calling function the intermediate optical planes during
            the propagation. This is useful if you want to e.g. examine the intensity in the Lyot plane for a
            coronagraphic propagation.
        normalize : string
            Desired normalization for output PSFs. See doc string for OpticalSystem.calc_psf. Default is
            to normalize the entrance pupil to have integrated total intensity = 1.

        Returns
        -------
        outfits : fits.HDUList
            The output PSF is returned as a fits.HDUlist object.
            If `outfile` is set to a valid filename, the output is also written to that file.


        """
        local_options = self.options.copy()  # all local state should be stored in a dict, for
        # ease of handing off to the various subroutines of
        # calc_psf. Don't just modify the global self.options
        # structure since that would pollute it with temporary
        # state as well as persistent state.
        local_options['monochromatic'] = monochromatic

        # ----- choose # of wavelengths intelligently. Do this first before generating the source spectrum weighting.
        if nlambda is None or nlambda == 0:
            nlambda = self._get_default_nlambda(self.filter)
        local_options['nlambda'] = nlambda

        # ----- calculate field of view depending on supplied parameters
        if fov_arcsec is None and fov_pixels is None:  # pick decent defaults.
            fov_arcsec = self._get_default_fov()
        if fov_pixels is not None:
            if np.isscalar(fov_pixels):
                fov_spec = 'pixels = %d' % fov_pixels
            else:
                fov_spec = 'pixels = (%d, %d)' % (fov_pixels[0], fov_pixels[1])
            local_options['fov_pixels'] = fov_pixels
        elif fov_arcsec is not None:
            if np.isscalar(fov_arcsec):
                fov_spec = 'arcsec = %f' % fov_arcsec
            else:
                fov_spec = 'arcsec = (%.3f, %.3f)' % (fov_arcsec[0], fov_arcsec[1])
            local_options['fov_arcsec'] = fov_arcsec
        local_options['fov_spec'] = fov_spec

        # ---- Implement the semi-convoluted logic for the oversampling options. See docstring above
        if oversample is not None and detector_oversample is not None and fft_oversample is not None:
            # all options set, contradictorily -> complain!
            raise ValueError(
                "You cannot specify simultaneously the oversample= option with the detector_oversample " +
                "and fft_oversample options. Pick one or the other!")
        elif oversample is None and detector_oversample is None and fft_oversample is None:
            # nothing set -> set oversample = 4
            oversample = 4
        if detector_oversample is None:
            detector_oversample = oversample
        if fft_oversample is None:
            fft_oversample = oversample
        local_options['detector_oversample'] = detector_oversample
        local_options['fft_oversample'] = fft_oversample

        # ----- compute weights for each wavelength based on source spectrum
        wavelens, weights = self._get_weights(source=source, nlambda=local_options['nlambda'],
                                              monochromatic=local_options['monochromatic'])

        # Validate that the calculation we're about to do makes sense with this instrument config
        self._validate_config(wavelengths=wavelens)
        poppy_core._log.info(
            "PSF calc using fov_%s, oversample = %d, number of wavelengths = %d" % (
                local_options['fov_spec'], local_options['detector_oversample'], len(wavelens)
            )
        )

        # ---- now at last, actually do the PSF calc:
        #  instantiate an optical system using the current parameters
        self.optsys = self._get_optical_system(fov_arcsec=fov_arcsec, fov_pixels=fov_pixels,
                                               fft_oversample=fft_oversample, detector_oversample=detector_oversample,
                                               options=local_options)
        self._check_for_aliasing(wavelens)
        # and use it to compute the PSF (the real work happens here, in code in poppy.py)
        result = self.optsys.calc_psf(wavelens, weights, display_intermediates=display, display=display,
                                      save_intermediates=save_intermediates, return_intermediates=return_intermediates,
                                      normalize=normalize)

        if return_intermediates:  # this implies we got handed back a tuple, so split it apart
            result, intermediates = result

        self._apply_jitter(result,
                           local_options)  # will immediately return if there is no jitter parameter in local_options

        self._get_fits_header(result, local_options)

        self._calc_psf_format_output(result, local_options)

        if display:
            f = plt.gcf()
            plt.suptitle("%s, filter= %s" % (self.name, self.filter), size='xx-large')

            if monochromatic is not None:
                labeltext = "Monochromatic calculation at {:.3f} um".format(monochromatic * 1e6)
            else:
                labeltext = "Calculation with %d wavelengths (%g - %g um)" % (
                    nlambda, wavelens[0] * 1e6, wavelens[-1] * 1e6)
            plt.text(0.99, 0.04, labeltext,
                     transform=f.transFigure, horizontalalignment='right')

        if outfile is not None:
            result[0].header["FILENAME"] = (os.path.basename(outfile), "Name of this file")
            result.writeto(outfile, overwrite=overwrite)
            poppy_core._log.info("Saved result to " + outfile)

        if return_intermediates:
            return result, intermediates
        else:
            return result

    def calc_datacube(self, wavelengths, *args, **kwargs):
        """Calculate a spectral datacube of PSFs

        Parameters
        -----------
        wavelengths : iterable of floats
            List or ndarray or tuple of floating point wavelengths in meters, such as
            you would supply in a call to calc_psf via the "monochromatic" option
        """

        # Allow up to 10,000 wavelength slices. The number matters because FITS
        # header keys can only have up to 8 characters. Backward-compatible.
        nwavelengths = len(wavelengths)
        if nwavelengths < 100:
            label_wl = lambda i: 'WAVELN{:02d}'.format(i)
        elif nwavelengths < 10000:
            label_wl = lambda i: 'WVLN{:04d}'.format(i)
        else:
            raise ValueError("Maximum number of wavelengths exceeded. "
                             "Cannot be more than 10,000.")

        # Set up cube and initialize structure based on PSF at first wavelength
        poppy_core._log.info("Starting multiwavelength data cube calculation.")
        psf = self.calc_psf(*args, monochromatic=wavelengths[0], **kwargs)
        from copy import deepcopy
        cube = deepcopy(psf)
        for ext in range(len(psf)):
            cube[ext].data = np.zeros((nwavelengths, psf[ext].data.shape[0], psf[ext].data.shape[1]))
            cube[ext].data[0] = psf[ext].data
            cube[ext].header[label_wl(0)] = wavelengths[0]

        # iterate rest of wavelengths
        for i in range(1, nwavelengths):
            wl = wavelengths[i]
            psf = self.calc_psf(*args, monochromatic=wl, **kwargs)
            for ext in range(len(psf)):
                cube[ext].data[i] = psf[ext].data
                cube[ext].header[label_wl(i)] = wl
                cube[ext].header.add_history("--- Cube Plane {} ---".format(i))
                for h in psf[ext].header['HISTORY']:
                    cube[ext].header.add_history(h)

        cube[0].header['NWAVES'] = nwavelengths
        return cube

    def _calc_psf_format_output(self, result, options):
        """ Apply desired formatting to output file:
                 - rebin to detector pixel scale if desired
                 - set up FITS extensions if desired
                 - output either the oversampled, rebinned, or both
        Which image(s) get output depends on the value of the options['output_mode']
        parameter. It may be set to 'Oversampled image' to output just the oversampled image,
        'Detector sampled image' to output just the image binned down onto detector pixels, or
        'Both as FITS extensions' to output the oversampled image as primary HDU and the
        rebinned image as the first image extension. For convenience, the option can be set
        to just 'oversampled', 'detector', or 'both'.

        Modifies the 'result' HDUList object.

        """

        output_mode = options.get('output_mode', 'Both as FITS extensions')
        detector_oversample = options.get('detector_oversample', 1)

        if (output_mode == 'Oversampled image') or ('oversampled' in output_mode.lower()):
            # we just want to output the oversampled image as
            # the primary HDU. Nothing special needs to be done.
            poppy_core._log.info(" Returning only the oversampled data. Oversampled by {}".format(detector_oversample))
            return

        elif (output_mode == 'Detector sampled image') or ('detector' in output_mode.lower()):
            # output only the detector sampled image as primary HDU.
            # need to downsample it and replace the existing primary HDU
            if options['detector_oversample'] > 1:
                poppy_core._log.info(" Downsampling to detector pixel scale, by {}".format(detector_oversample))
                for ext in range(len(result)):
                    result[ext].data = utils.rebin_array(result[ext].data,
                                                         rc=(detector_oversample, detector_oversample))
            else:
                poppy_core._log.info(" Result already at detector pixel scale; no downsampling needed.")

            for ext in np.arange(len(result)):
                result[ext].header['OVERSAMP'] = (1, 'These data are rebinned to detector pixels')
                result[ext].header['CALCSAMP'] = (detector_oversample, 'This much oversampling used in calculation')
                result[ext].header['PIXELSCL'] *= detector_oversample
                result[ext].header['EXTNAME'] = result[ext].header['EXTNAME'].replace("OVER", "DET_")
            return

        elif (output_mode == 'Both as FITS extensions') or ('both' in output_mode.lower()):
            # return the downsampled image in the first image extension
            # keep the oversampled image in the primary HDU.
            # create the image extension even if we're already at 1x sampling, for consistency
            poppy_core._log.info(" Adding extension with image downsampled to detector pixel scale.")

            hdu = fits.HDUList()  # append to new hdulist object to preserve the order
            for ext in np.arange(len(result)):
                rebinned_result = result[ext].copy()
                if options['detector_oversample'] > 1:
                    poppy_core._log.info(" Downsampling to detector pixel scale, by {}".format(detector_oversample))
                    rebinned_result.data = utils.rebin_array(rebinned_result.data,
                                                             rc=(detector_oversample, detector_oversample))

                rebinned_result.header['OVERSAMP'] = (1, 'These data are rebinned to detector pixels')
                rebinned_result.header['CALCSAMP'] = (detector_oversample, 'This much oversampling used in calculation')
                rebinned_result.header['PIXELSCL'] *= detector_oversample
                rebinned_result.header['EXTNAME'] = rebinned_result.header['EXTNAME'].replace("OVER", "DET_")

                hdu.append(result[ext])
                hdu.append(rebinned_result)

            # Create enough new extensions to append all psfs to them
            [result.append(fits.ImageHDU()) for i in np.arange(len(hdu) - len(result))]
            for ext in np.arange(len(hdu)): result[ext] = hdu[ext]

            return

    def _get_fits_header(self, result, options):
        """ Set instrument-specific FITS header keywords

        Parameters:
            result : fits.HDUList object
                The HDUList containing the image to be output.
            options : dict
                A dictionary containing options

        This function will modify the primary header of the result HDUlist.
        """

        try:
            from .version import version as __version__
        except ImportError:
            __version__ = ''

        # ---  update FITS header, display, and output.
        if isinstance(self.pupil, str):
            pupilstr = os.path.basename(self.pupil)
        elif isinstance(self.pupil, fits.HDUList):
            pupilstr = 'pupil from supplied FITS HDUList object'
        elif isinstance(self.pupil, poppy_core.OpticalElement):
            pupilstr = 'pupil from supplied OpticalElement: ' + str(self.pupil)
        result[0].header['PUPILINT'] = (pupilstr, 'Pupil aperture intensity source')

        if self.pupilopd is None:
            opdstring = "NONE - perfect telescope! "
            opdfile = 'None'
            opdslice = 0
        elif isinstance(self.pupilopd, str):
            opdstring = os.path.basename(self.pupilopd)
            opdfile = os.path.basename(self.pupilopd)
            opdslice = 0  # default slice
        elif isinstance(self.pupilopd, fits.HDUList):
            opdstring = 'OPD from supplied FITS HDUlist object'
            if isinstance(self.pupilopd.filename(), str):
                opdfile = os.path.basename(self.pupilopd.filename())
            else:
                opdfile = 'None'
            opdslice = 0
        elif isinstance(self.pupilopd, poppy_core.OpticalElement):
            opdstring = 'OPD from supplied OpticalElement: ' + str(self.pupilopd)
            opdfile = str(self.pupilopd)
            opdslice = 0
        else:  # tuple?
            opdstring = "%s slice %d" % (os.path.basename(self.pupilopd[0]), self.pupilopd[1])
            opdfile = os.path.basename(self.pupilopd[0])
            opdslice = self.pupilopd[1]
        result[0].header['PUPILOPD'] = (opdstring, 'Pupil OPD source')
        result[0].header['OPD_FILE'] = (opdfile, 'Pupil OPD file name')
        result[0].header['OPDSLICE'] = (opdslice, 'Pupil OPD slice number, if file is a datacube')

        result[0].header['INSTRUME'] = (self.name, 'Instrument')
        result[0].header['FILTER'] = (self.filter, 'Filter name')
        result[0].header['EXTNAME'] = ('OVERSAMP', 'This extension is oversampled.')
        result[0].header.add_history('Created by POPPY version ' + __version__)

        if 'fft_oversample' in options:
            result[0].header['OVERSAMP'] = (options['fft_oversample'], 'Oversampling factor for FFTs in computation')
        if 'detector_oversample' in options:
            result[0].header['DET_SAMP'] = (
                options['detector_oversample'], 'Oversampling factor for MFT to detector plane')

        (year, month, day, hour, minute, second, weekday, doy, dst) = time.gmtime()
        result[0].header["DATE"] = (
            "%4d-%02d-%02dT%02d:%02d:%02d" % (year, month, day, hour, minute, second), "Date of calculation")
        # get username and hostname in a cross-platform way
        username = getpass.getuser()
        hostname = platform.node()
        result[0].header["AUTHOR"] = ("%s@%s" % (username, hostname), "username@host for calculation")

    def _validate_config(self, wavelengths=None):
        """Determine if a provided instrument configuration is valid.

        Wavelengths to be propagated in the calculation are passed in as the `wavelengths`
        keyword argument.

        Subclasses should raise an exception if the configuration is invalid/unachievable.
        """
        pass

    def get_optical_system(self, fft_oversample=2, detector_oversample=None, fov_arcsec=2, fov_pixels=None,
                            options=None):
        """ Return an OpticalSystem instance corresponding to the instrument as currently configured.

        When creating such an OpticalSystem, you must specify the parameters needed to define the
        desired sampling, specifically the oversampling and field of view.


        Parameters
        ----------

        fft_oversample : int
            Oversampling factor for intermediate plane calculations. Default is 2
        detector_oversample: int, optional
            By default the detector oversampling is equal to the intermediate calculation oversampling.
            If you wish to use a different value for the detector, set this parameter.
            Note that if you just want images at detector pixel resolution you will achieve higher fidelity
            by still using some oversampling (i.e. *not* setting `oversample_detector=1`) and instead rebinning
            down the oversampled data.
        fov_pixels : float
            Field of view in pixels. Overrides fov_arcsec if both set.
        fov_arcsec : float
            Field of view, in arcseconds. Default is 2
        options : dict
            Other arbitrary options for optical system creation


        Returns
        -------
        osys : poppy.OpticalSystem
            an optical system instance representing the desired configuration.

        """

        poppy_core._log.info("Creating optical system model:")

        if detector_oversample is None:
            detector_oversample = fft_oversample
        if options is None:
            options = dict()

        poppy_core._log.debug("Oversample: %d  %d " % (fft_oversample, detector_oversample))
        optsys = poppy_core.OpticalSystem(name=self.name, oversample=fft_oversample)

        if 'source_offset_x' in options or 'source_offset_y' in options:
            if 'source_offset_r' in options:
                raise ValueError("Cannot set source offset using source_offset_x and source_offset_y" +
                                 " at the same time as source_offset_r")
            offx = options.get('source_offset_x', 0)
            offy = options.get('source_offset_y', 0)
            optsys.source_offset_r = np.sqrt(offx ** 2 + offy ** 2)
            optsys.source_offset_theta = np.rad2deg(np.arctan2(-offx, offy))
            _log.debug("Source offset from X,Y = ({}, {}) is (r,theta) = {},{}".format(
                offx, offy, optsys.source_offset_r, optsys.source_offset_theta))
        else:
            if 'source_offset_r' in options:
                optsys.source_offset_r = options['source_offset_r']
            if 'source_offset_theta' in options:
                optsys.source_offset_theta = options['source_offset_theta']
            _log.debug("Source offset is (r,theta) = {},{}".format(
                optsys.source_offset_r, optsys.source_offset_theta))

        # ---- set pupil intensity
        pupil_optic = None  # no optic yet defined
        if isinstance(self.pupil, poppy_core.OpticalElement):  # do we already have an object?
            pupil_optic = self.pupil
            full_pupil_path = None
        elif isinstance(self.pupil, str):  # simple filename
            if os.path.exists(self.pupil):
                full_pupil_path = self.pupil
            else:
                raise IOError("File not found: " + self.pupil)
        elif isinstance(self.pupil, fits.HDUList):  # pupil supplied as FITS HDUList object
            full_pupil_path = self.pupil
        else:
            raise TypeError("Not sure what to do with a pupil of that type:" + str(type(self.pupil)))

        # ---- set pupil OPD
        if isinstance(self.pupilopd, str):  # simple filename
            full_opd_path = self.pupilopd if os.path.exists(self.pupilopd) else os.path.join(self._datapath, "OPD",
                                                                                             self.pupilopd)
        elif hasattr(self.pupilopd, '__getitem__') and isinstance(self.pupilopd[0],
                                                                  str):  # tuple with filename and slice
            full_opd_path = (
                self.pupilopd[0] if os.path.exists(self.pupilopd[0]) else os.path.join(self._datapath, "OPD",
                                                                                       self.pupilopd[0]),
                self.pupilopd[1])
        elif isinstance(self.pupilopd, fits.HDUList):  # OPD supplied as FITS HDUList object
            full_opd_path = self.pupilopd  # not a path per se but this works correctly to pass it to poppy
        elif self.pupilopd is None:
            full_opd_path = None
        else:
            raise TypeError("Not sure what to do with a pupilopd of that type:" + str(type(self.pupilopd)))

        # ---- apply pupil intensity and OPD to the optical model
        optsys.add_pupil(name='Entrance Pupil', optic=pupil_optic, transmission=full_pupil_path, opd=full_opd_path,
                        rotation=self._rotation)

        # Allow instrument subclass to add field-dependent aberrations
        aberration_optic = self._get_aberrations()
        if aberration_optic is not None:
            optsys.add_pupil(aberration_optic)

        # --- add the detector element.
        if fov_pixels is None:
            fov_pixels = np.round(fov_arcsec / self.pixelscale)
            if 'parity' in self.options:
                if self.options['parity'].lower() == 'odd' and np.remainder(fov_pixels, 2) == 0:
                    fov_pixels += 1
                if self.options['parity'].lower() == 'even' and np.remainder(fov_pixels, 2) == 1:
                    fov_pixels += 1

        optsys.add_detector(self.pixelscale, fov_pixels=fov_pixels, oversample=detector_oversample,
                           name=self.name + " detector")

        return optsys

    def _get_optical_system(self, *args, **kwargs):
        """ Return an OpticalSystem instance corresponding to the instrument as currently configured.

        """
        # Note, this has historically been an internal private API function (starting with an underscore)
        # As of version 0.9 it is promoted to a public part of the API for the Instrument class and subclasses.
        # Here we ensure the prior version works, back compatibly.
        import warnings
        warnings.warn("_get_optical_system is deprecated; use get_optical_system (without leading underscore) instead.",
                      DeprecationWarning)
        return self.get_optical_system(*args, **kwargs)

    def _check_for_aliasing(self, wavelengths):
        """ Check for spatial frequency aliasing and warn if the
        user is requesting a FOV which is larger than supported based on
        the available pupil resolution in the optical system entrance pupil.
        If the requested FOV of the output PSF exceeds that which is Nyquist
        sampled in the entrance pupil, raise a warning to the user.

        The check implemented here is fairly simple, designed to catch the most
        common cases, and makes assumptions about the optical system which are
        not necessarily true in all cases, specifically that it starts with a
        pupil plane with fixed spatial resolution and ends with a detector
        plane. If either of those assumptions is violated, this check is skipped.

        See https://github.com/mperrin/poppy/issues/135 and
        https://github.com/mperrin/poppy/issues/180 for more background on the
        relevant Fourier optics.
        """
        # Note this must be called after self.optsys is defined in calc_psf()

        # compute spatial sampling in the entrance pupil
        if not hasattr(self.optsys.planes[0], 'pixelscale') or self.optsys.planes[0].pixelscale is None:
            return  # analytic entrance pupil, no sampling limitations.
        if not isinstance(self.optsys.planes[-1], poppy_core.Detector):
            return  # optical system doesn't end on some fixed sampling detector, not sure how to check sampling limit

        # determine the spatial frequency which is Nyquist sampled by the input pupil.
        # convert this to units of cycles per meter and make it not a Quantity
        sf = (1. / (self.optsys.planes[0].pixelscale * 2 * units.pixel)).to(1. / units.meter).value

        det_fov_arcsec = self.optsys.planes[-1].fov_arcsec.to(units.arcsec).value
        if np.isscalar(det_fov_arcsec):  # FOV can be scalar (square) or rectangular
            det_fov_arcsec = (det_fov_arcsec, det_fov_arcsec)

        # determine the angular scale that corresponds to for the given wavelength
        for wl in wavelengths:
            critical_angle_arcsec = wl * sf * poppy_core._RADIANStoARCSEC
            if (critical_angle_arcsec < det_fov_arcsec[0] / 2) or (critical_angle_arcsec < det_fov_arcsec[1] / 2):
                import warnings
                warnings.warn((
                        "For wavelength {:.3f} microns, a FOV of {:.3f} * {:.3f} arcsec exceeds the maximum " +
                        " spatial frequency well sampled by the input pupil. Your computed PSF will suffer from " +
                        "aliasing for angles beyond {:.3f} arcsec radius.").format(
                    wl * 1e6, det_fov_arcsec[0], det_fov_arcsec[1], critical_angle_arcsec))

    def _get_aberrations(self):
        """Incorporate a pupil-plane optic that represents optical aberrations
        (e.g. field-dependence as an OPD map). Subclasses should override this method.
        (If no aberration optic should be applied, None should be returned.)

        Returns
        -------
        aberration_optic : poppy.OpticalElement subclass or None
            Optional. Will be added to the optical system immediately after the
            entrance pupil (and any pupil OPD map).
        """
        return None

    def _apply_jitter(self, result, local_options=None):
        """ Modify a PSF to account for the blurring effects of image jitter.
        Parameter arguments are taken from the options dictionary.

        Parameters
        -----------
        result : fits.HDUList
            HDU list containing a point spread function
        local_options : dict, optional
            Options dictionary. If not present, options will be taken from self.options.

        The key configuration argument is options['jitter'] which defines the type of jitter.
        If this is the string 'gaussian', then a Gaussian blurring kernel will be applied, the
        amount of the blur is taken from the options['jitter_sigma'] value (arcsec per axis).

        Other types of jitter are not yet implemented.

        The image in the 'result' HDUlist will be modified by this function.
        """
        if local_options is None:
            local_options = self.options
        if 'jitter' not in local_options:
            result[0].header['JITRTYPE'] = ('None', 'Type of jitter applied')
            return

        if conf.enable_speed_tests: t0 = time.time()  # pragma: no cover

        poppy_core._log.info("Calculating jitter using " + str(local_options['jitter']))

        if local_options['jitter'] is None:
            return
        elif local_options['jitter'].lower() == 'gaussian':
            import scipy.ndimage

            sigma = local_options.get('jitter_sigma')
            if sigma is None:
                poppy_core._log.warning(
                    "Gaussian jitter model requested, but no width for jitter distribution specified. " +
                    "Assuming jitter_sigma = 0.007 arcsec per axis by default")
                sigma = 0.007

            # that will be in arcseconds, we need to convert to pixels:

            poppy_core._log.info("Jitter: Convolving with Gaussian with sigma={0:.3f} arcsec".format(sigma))
            out = scipy.ndimage.gaussian_filter(result[0].data, sigma / result[0].header['PIXELSCL'])
            peak = result[0].data.max()
            newpeak = out.max()
            strehl = newpeak / peak  # not really the whole Strehl ratio, just the part due to jitter

            poppy_core._log.info("        resulting image peak drops to {0:.3f} of its previous value".format(strehl))
            result[0].header['JITRTYPE'] = ('Gaussian convolution', 'Type of jitter applied')
            result[0].header['JITRSIGM'] = (sigma, 'Gaussian sigma for jitter, per axis [arcsec]')
            result[0].header['JITRSTRL'] = (strehl, 'Strehl reduction from jitter ')

            result[0].data = out
        else:
            raise ValueError('Unknown jitter option value: ' + local_options['jitter'])

        if conf.enable_speed_tests: # pragma: no cover
            t1 = time.time()
            _log.debug("\tTIME %f s\t for jitter model" % (t1 - t0))


    #####################################################
    # Display routines

    def display(self):
        """Display the currently configured optical system on screen"""
        # if coronagraphy is set, then we have to temporarily disable
        # semi-analytic coronagraphic mode to get a regular displayable optical system
        try:
            old_no_sam = self.options['no_sam']
            self.options['no_sam'] = True
        except KeyError:
            old_no_sam = None
        # Trigger config validation to update any optical planes
        # (specifically auto-selected pupils based on filter selection)
        wavelengths, _ = self._get_weights(nlambda=1)
        self._validate_config(wavelengths=wavelengths)
        optsys = self._get_optical_system()
        optsys.display(what='both')
        if old_no_sam is not None:
            self.options['no_sam'] = old_no_sam

    #####################################################
    #
    # Synthetic Photometry related methods
    #
    def _get_spec_cache_key(self, source, nlambda):
        """ return key for the cache of precomputed spectral weightings.
        This is a separate function so the TFI subclass can override it.
        """
        name = source.meta.get('name')
        if not name:
            name = source.meta['expr']
        return self.filter, name, nlambda

    def _get_synphot_bandpass(self, filtername):
        """ Return a synphot.spectrum.SpectralElement object for the given desired band.

        By subclassing this, you can define whatever custom bandpasses are appropriate for your instrument

        Parameters
        ----------
        filtername : str
            String name of the filter that you are interested in

        Returns
        --------
        a synphot.spectrum.ObservationSpectralElement object for that filter.

        """
        if not _HAS_SYNPHOT:
            raise RuntimeError("synphot not found")

        bpname = self._synphot_bandpasses[filtername]

        try:
            band = synphot.spectrum.SpectralElement.from_filter(bpname)
        except Exception:
            raise LookupError("Don't know how to compute bandpass for a filter named " + bpname)

        return band

    def _get_default_nlambda(self, filtername):
        """ Return the default # of wavelengths to be used for calculation by a given filter """
        return 10

    def _get_default_fov(self):
        """ Return default FOV in arcseconds """
        return 5

    def _get_filter_list(self):
        """ Returns a list of allowable filters, and the corresponding synphot obsmode
        for each.

        If you need to define bandpasses that are not already available in synphot, consider subclassing
        _getSynphotBandpass instead to create a synphot spectrum based on data read from disk, etc.

        Returns
        --------
        filterlist : list
            List of string filter names
        bandpasslist : dict
            dictionary of string names for use by synphot

        This could probably be folded into one using an OrderdDict. FIXME do that later

        """

        filterlist = ['U', 'B', 'V', 'R', 'I']
        bandpasslist = {
            'U': 'johnson_u',
            'B': 'johnson_b',
            'V': 'johnson_v',
            'R': 'johnson_r',
            'I': 'johnson_i',
        }

        return filterlist, bandpasslist

    # def _getJitterKernel(self, type='Gaussian', sigma=10):

    def _get_weights(self, source=None, nlambda=None, monochromatic=None, verbose=False):
        """ Return the set of discrete wavelengths, and weights for each wavelength,
        that should be used for a PSF calculation.

        Uses synphot (if installed), otherwise assumes simple-minded flat spectrum

        """
        if nlambda is None or nlambda == 0:
            nlambda = self._get_default_nlambda(self.filter)

        if monochromatic is not None:
            poppy_core._log.info("Monochromatic calculation requested.")
            monochromatic_wavelen_meters = monochromatic.to_value(units.meter) if isinstance(monochromatic, units.Quantity) else monochromatic
            return (np.asarray([monochromatic_wavelen_meters]), np.asarray([1]))

        elif _HAS_SYNPHOT and (isinstance(source, synphot.SourceSpectrum) or source is None):
            """ Given a synphot.SourceSpectrum object, perform synthetic photometry for
            nlambda bins spanning the wavelength range of interest.

            Because this calculation is kind of slow, cache results for reuse in the frequent
            case where one is computing many PSFs for the same spectral source.
            """
            from synphot import SpectralElement, Observation
            from synphot.models import Box1D, BlackBodyNorm1D, Empirical1D

            poppy_core._log.debug(
                "Calculating spectral weights using synphot, nlambda=%d, source=%s" % (nlambda, str(source)))
            if source is None:
                source = synphot.SourceSpectrum(BlackBodyNorm1D, temperature=5700 * units.K)
                poppy_core._log.info("No source spectrum supplied, therefore defaulting to 5700 K blackbody")
            poppy_core._log.debug("Computing spectral weights for source = " + str(source))

            try:
                key = self._get_spec_cache_key(source, nlambda)
                if key in self._spectra_cache:
                    poppy_core._log.debug("Previously computed spectral weights found in cache, just reusing those")
                    return self._spectra_cache[key]
            except KeyError:
                pass  # in case sourcespectrum lacks a name element so the above lookup fails - just do the below calc.

            poppy_core._log.info("Computing wavelength weights using synthetic photometry for %s..." % self.filter)
            band = self._get_synphot_bandpass(self.filter)
            band_wave = band.waveset
            band_thru = band(band_wave)

            # Update source to ensure that it covers the entire filter
            if band_wave.value.min() < source.waveset.value.min() or \
                    band_wave.value.max() > source.waveset.value.max():
                source_meta = source.meta
                wave, wave_str = synphot.utils.generate_wavelengths(band_wave.value.min(), band_wave.value.max(),
                                                                    wave_unit=units.angstrom, log=False)
                source = synphot.SourceSpectrum(Empirical1D, points=wave, lookup_table=source(wave))
                source.meta.update(source_meta)

            # choose reasonable min and max wavelengths
            w_above10 = (band_thru > 0.10 * band_thru.max())

            minwave = band_wave[w_above10].min()
            maxwave = band_wave[w_above10].max()
            poppy_core._log.debug("Min, max wavelengths = %f, %f" % (
                minwave.to_value(units.micron), maxwave.to_value(units.micron)))

            wave_bin_edges = np.linspace(minwave, maxwave, nlambda + 1)
            wavesteps = (wave_bin_edges[:-1] + wave_bin_edges[1:]) / 2
            deltawave = wave_bin_edges[1] - wave_bin_edges[0]
            area = 1 * (units.m * units.m)
            effstims = []

            for wave in wavesteps:
                poppy_core._log.debug(
                    f"Integrating across band centered at {wave.to(units.micron):.2f} "
                    f"with width {deltawave.to(units.micron):.2f}")
                box = SpectralElement(Box1D, amplitude=1, x_0=wave, width=deltawave) * band
                if box.tpeak() == 0:
                    # watch out for pathological cases with no overlap (happens with MIRI FND at high nlambda)
                    result = 0.0
                else:
                    binset = np.linspace(wave - deltawave, wave + deltawave,
                                         30)  # what wavelens to use when integrating across the sub-band?
                    binset = binset[binset >= 0]  # remove any negative values
                    result = Observation(source, box, binset=binset).effstim('count', area=area)
                effstims.append(result)

            effstims = units.Quantity(effstims)
            effstims /= effstims.sum()  # Normalized count rate is unitless
            wave_m = wavesteps.to_value(units.m)  # convert to meters

            newsource = (wave_m, effstims.to_value())
            if verbose:
                _log.info(" Wavelengths and weights computed from synphot: " + str(newsource))
            self._spectra_cache[self._get_spec_cache_key(source, nlambda)] = newsource
            return newsource
        elif isinstance(source, dict) and ('wavelengths' in source) and ('weights' in source):
            # Allow providing directly a set of specific weights and wavelengths, as in poppy.calc_psf source option #2
            return source['wavelengths'], source['weights']
        elif isinstance(source, tuple) and len(source) == 2:
            # Allow user to provide directly a tuple, as in poppy.calc_psf source option #3
            return source

        else:  # Fallback simple code for if we don't have synphot.
            poppy_core._log.warning(
                "synphot unavailable (or invalid source supplied)! Assuming flat # of counts versus wavelength.")
            # compute a source spectrum weighted by the desired filter curves.
            # The existing FITS files all have wavelength in ANGSTROMS since that is the synphot convention...
            filterfile = self._filters[self.filter].filename
            filterheader = fits.getheader(filterfile, 1)
            filterdata = fits.getdata(filterfile, 1)
            try:
                wavelengths = filterdata.WAVELENGTH.astype('=f8')
                throughputs = filterdata.THROUGHPUT.astype('=f8')
            except AttributeError:
                raise ValueError(
                    "The supplied file, {0}, does not appear to be a FITS table with WAVELENGTH and " +
                    "THROUGHPUT columns.".format(filterfile))
            if 'WAVEUNIT' in filterheader:
                waveunit = filterheader['WAVEUNIT'].lower()
                if re.match(r'[Aa]ngstroms?', waveunit) is None:
                    raise ValueError(
                        "The supplied file, {0}, has WAVEUNIT='{1}'. Only WAVEUNIT = Angstrom supported " +
                        "when synphot is not installed.".format(filterfile, waveunit))
            else:
                waveunit = 'Angstrom'
                poppy_core._log.warning(
                    "CAUTION: no WAVEUNIT keyword found in filter file {0}. Assuming = {1} by default".format(
                        filterfile, waveunit))

            poppy_core._log.warning(
                "CAUTION: Just interpolating rather than integrating filter profile, over {0} steps".format(nlambda))
            wavelengths = wavelengths * units.Unit(waveunit)
            lrange = wavelengths[throughputs > 0.4].to_value(units.m)  # convert from Angstroms to Meters
            # get evenly spaced points within the range of allowed lambdas, centered on each bin
            lambd = np.linspace(np.min(lrange), np.max(lrange), nlambda, endpoint=False) + (
                    np.max(lrange) - np.min(lrange)) / (2 * nlambda)
            filter_fn = scipy.interpolate.interp1d(wavelengths.to_value(units.m), throughputs, kind='cubic',
                                                   bounds_error=False)
            weights = filter_fn(lambd)
            return lambd, weights
