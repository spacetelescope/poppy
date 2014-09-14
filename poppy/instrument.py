from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.ndimage
import matplotlib
import astropy.io.fits as fits

from . import poppy_core
from . import utils


__all__ = ['Instrument']

class Instrument(object):
    """ A generic astronomical instrument, composed of 
        (1) an optical system implemented using POPPY, optionally with several configurations such as
            selectable image plane or pupil plane stops, and
        (2) some defined spectral bandpass(es) such as selectable filters, implemented using pysynphot. 

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

        _getOpticalSystem
        _getFilterList
        _getDefaultNLambda
        _getDefaultFOV
        _getFITSHeader

    For more complicated systems you may also want to override: 
        _validateConfig
        _getSynphotBandpass
        _applyJitter


    """
    def __init__(self, name="", *args, **kwargs):
        self.name=name
        self.pupil = poppy_core.CircularAperture( *args, **kwargs)
        "Aperture for this optical system. May be a FITS filename, FITS HDUList object, or poppy.OpticalElement"
        self.pupilopd = None   # This can optionally be set to a tuple indicating (filename, slice in datacube)
        """Pupil OPD for this optical system. May be a FITS filename, or FITS HDUList. 
        If the file contains a datacube, you may set this to a tuple (filename, slice) to select a given slice, or else
        the first slice will be used."""


        self.options = {} # dict for storing other arbitrary options. 
        """ A dictionary capable of storing other arbitrary options, for extensibility. The following are all optional, and
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
        rebin : bool
            For output files, write an additional FITS extension including a version of the output array 
            rebinned down to the actual detector pixel scale?
        jitter : string
            Type of jitter model to apply. Currently the only
        parity : string "even" or "odd"
            You may wish to ensure that the output PSF grid has either an odd or even number of pixels.
            Setting this option will force that to be the case by increasing npix by one if necessary.

        """

        self.filter_list, self._synphot_bandpasses = self._getFilterList() # List of available filter names


        #create private instance variables. These will be
        # wrapped just below to create properties with validation.
        self._filter=None
        self._rotation = None

        self.pixelscale = 0.025
        """ Detector pixel scale, in arcseconds/pixel """

        self._spectra_cache = {}  # for caching pysynphot results.


        self.filter = self.filter_list[0]


    def __str__(self):
        return "Instrument name="+self.name

    # create properties with error checking
    @property
    def filter(self):
        'Currently selected filter name (e.g. "F200W")'
        return self._filter
    @filter.setter
    def filter(self, value):
        value = value.upper() # force to uppercase
        if value not in self.filter_list:
            raise ValueError("Instrument %s doesn't have a filter called %s." % (self.name, value))
        self._filter = value
        self._validate_config()


    #----- actual optical calculations follow here -----
    def calcPSF(self, outfile=None, source=None, nlambda=None, monochromatic=None ,
            fov_arcsec=None, fov_pixels=None,  oversample=None, detector_oversample=None, fft_oversample=None, rebin=True,
            clobber=True, display=False, save_intermediates=False, return_intermediates=False):
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
        source : pysynphot.SourceSpectrum or dict
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
        rebin : bool, optional
            If set, the output file will contain a FITS image extension containing the PSF rebinned
            onto the actual detector pixel scale. Thus, setting oversample=<N> and rebin=True is
            the proper way to obtain high-fidelity PSFs computed on the detector scale. Default is True.
        clobber : bool
            overwrite output FITS file if it already exists?
        display : bool
            Whether to display the PSF when done or not.
        save_intermediates, return_intermediates : bool
            Options for saving to disk or returning to the calling function the intermediate optical planes during the propagation. 
            This is useful if you want to e.g. examine the intensity in the Lyot plane for a coronagraphic propagation.

        Returns
        -------
        outfits : fits.HDUList
            The output PSF is returned as a fits.HDUlist object.
            If `outfile` is set to a valid filename, the output is also written to that file.


        """
        local_options = self.options  # all local state should be stored in a dict, for
                                      # ease of handing off to the various subroutines of
                                      # calcPSF. Don't just modify the global self.options
                                      # structure since that would pollute it with temporary
                                      # state as well as persistent state. 
        local_options['monochromatic'] = monochromatic

        #----- choose # of wavelengths intelligently. Do this first before generating the source spectrum weighting.
        if nlambda is None or nlambda==0:
            nlambda = self._getDefaultNLambda(self.filter)
        local_options['nlambda'] = nlambda


        #----- calculate field of view depending on supplied parameters
        if fov_arcsec is None and fov_pixels is None:  #pick decent defaults.
            fov_arcsec=self._getDefaultFOV()
        if fov_pixels is not None:
            local_options['fov_spec'] = 'pixels = %d' % fov_pixels
            local_options['fov_pixels'] = fov_pixels
        elif fov_arcsec is not None:
            local_options['fov_spec'] = 'arcsec = %f' % fov_arcsec
            local_options['fov_arcsec'] = fov_arcsec


        #---- Implement the semi-convoluted logic for the oversampling options. See docstring above
        if oversample is not None and detector_oversample is not None and fft_oversample is not None:
            # all options set, contradictorily -> complain!
            raise ValueError("You cannot specify simultaneously the oversample= option with the detector_oversample and fft_oversample options. Pick one or the other!")
        elif oversample is None and detector_oversample is None and fft_oversample is None:
            # nothing set -> set oversample = 4
            oversample = 4
        if detector_oversample is None: detector_oversample = oversample
        if fft_oversample is None: fft_oversample = oversample
        local_options['detector_oversample']=detector_oversample
        local_options['fft_oversample']=fft_oversample

        poppy_core._log.info("PSF calc using fov_%s, oversample = %d, nlambda = %d" % (local_options['fov_spec'], local_options['detector_oversample'], local_options['nlambda']) )

        #----- compute weights for each wavelength based on source spectrum
        wavelens, weights = self._getWeights(source=source, nlambda=local_options['nlambda'], monochromatic=local_options['monochromatic'])


        #---- now at last, actually do the PSF calc:
        #  instantiate an optical system using the current parameters
        self.optsys = self._getOpticalSystem(fov_arcsec=fov_arcsec, fov_pixels=fov_pixels,
            fft_oversample=fft_oversample, detector_oversample=detector_oversample, 
            options=local_options)
        # and use it to compute the PSF (the real work happens here, in code in poppy.py)
        result = self.optsys.calcPSF(wavelens, weights, display_intermediates=display, display=display, save_intermediates=save_intermediates, return_intermediates=return_intermediates)

        if return_intermediates: # this implies we got handed back a tuple, so split it apart
            result, intermediates = result

        self._applyJitter(result, local_options)  # will immediately return if there is no jitter parameter in local_options


        self._getFITSHeader(result, local_options) 

        self._calcPSF_format_output(result, local_options)


        if display:
            f = plt.gcf()
            plt.suptitle( "%s, filter= %s" % (self.name, self.filter), size='xx-large')
            plt.text( 0.99, 0.04, "Calculation with %d wavelengths (%g - %g um)" % (nlambda, wavelens[0]*1e6, wavelens[-1]*1e6), 
                    transform=f.transFigure, horizontalalignment='right')

        if outfile is not None:
            result[0].header["FILENAME"] = ( os.path.basename(outfile), "Name of this file")
            result.writeto(outfile, clobber=clobber)
            poppy_core._log.info("Saved result to "+outfile)

        if return_intermediates:
            return result, intermediates
        else:
            return result

    def _calcPSF_format_output(self, result, options):
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
        output_mode = options.get('output_mode','Both as FITS extensions')
        detector_oversample = options.get('detector_oversample',1)

        if (output_mode == 'Oversampled image') or ('oversampled' in output_mode.lower()):
            # we just want to output the oversampled image as
            # the primary HDU. Nothing special needs to be done.
            return
        elif (output_mode == 'Detector sampled image') or ('detector' in output_mode.lower()):
            # output only the detector sampled image as primary HDU.
            # need to downsample it and replace the existing primary HDU
            poppy_core._log.info(" Downsampling to detector pixel scale.")
            if options['detector_oversample'] > 1:
                result[0].data = utils.rebin_array(result[0].data, 
                        rc=(detector_oversample, detector_oversample))
            result[0].header['OVERSAMP'] = ( 1, 'These data are rebinned to detector pixels')
            result[0].header['CALCSAMP'] = ( detector_oversample, 'This much oversampling used in calculation')
            result[0].header['EXTNAME'] = ( 'DET_SAMP')
            result[0].header['PIXELSCL'] *= detector_oversample
            return
        elif (output_mode == 'Both as FITS extensions') or ('both' in output_mode.lower()):
            # return the downsampled image in the first image extension
            # keep the oversampled image in the primary HDU.
            # create the image extension even if we're already at 1x sampling, for consistency
            poppy_core._log.info(" Downsampling to detector pixel scale.")
            rebinned_result = result[0].copy()
            if options['detector_oversample'] > 1:
                rebinned_result.data = utils.rebin_array(rebinned_result.data,
                        rc=(detector_oversample, detector_oversample))
            rebinned_result.header['OVERSAMP'] = ( 1, 'These data are rebinned to detector pixels')
            rebinned_result.header['CALCSAMP'] = ( detector_oversample, 'This much oversampling used in calculation')
            rebinned_result.header['EXTNAME'] =  'DET_SAMP'
            rebinned_result.header['PIXELSCL'] *= detector_oversample
            result.append(rebinned_result)
            return







    def _getFITSHeader(self, result, options):
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

        #---  update FITS header, display, and output.
        if isinstance( self.pupil, basestring):
            pupilstr= os.path.basename(self.pupil)
        elif isinstance( self.pupil, fits.HDUList):
            pupilstr= 'pupil from supplied FITS HDUList object'
        elif isinstance( self.pupil, poppy_core.OpticalElement):
            pupilstr = 'pupil from supplied OpticalElement: '+str(self.pupil)
        result[0].header['PUPILINT'] = ( pupilstr, 'Pupil aperture intensity source')

        if self.pupilopd is None:
            opdstring = "NONE - perfect telescope! "
        elif isinstance( self.pupilopd, basestring):
            opdstring = os.path.basename(self.pupilopd)
        elif isinstance( self.pupilopd, fits.HDUList):
            opdstring = 'OPD from supplied FITS HDUlist object'
        else: # tuple?
            opdstring =  "%s slice %d" % (os.path.basename(self.pupilopd[0]), self.pupilopd[1])
        result[0].header['PUPILOPD'] = ( opdstring,  'Pupil wavefront OPD source')

        result[0].header['INSTRUME'] = ( self.name, 'Instrument')
        result[0].header['FILTER'] = ( self.filter, 'Filter name')
        result[0].header['EXTNAME'] = ( 'OVERSAMP')
        result[0].header.add_history('Created by POPPY version '+__version__)

        if 'fft_oversample' in options.keys():
            result[0].header['OVERSAMP'] = ( options['fft_oversample'], 'Oversampling factor for FFTs in computation')
        if 'detector_oversample' in options.keys():
            result[0].header['DET_SAMP'] = ( options['detector_oversample'], 'Oversampling factor for MFT to detector plane')

        (year, month, day, hour, minute, second, weekday, DOY, DST) =  time.gmtime()
        result[0].header["DATE"] = ( "%4d-%02d-%02dT%02d:%02d:%02d" % (year, month, day, hour, minute, second), "Date of calculation")
        result[0].header["AUTHOR"] = ( "%s@%s" % (os.getenv('USER'), os.getenv('HOST')), "username@host for calculation")

    def _validate_config(self):
        """ Determine if a provided instrument configuration is valid.

        Should raise an exception if the configuration is invalid/unachievable.
        """
        pass


    def _getOpticalSystem(self,fft_oversample=2, detector_oversample = None, fov_arcsec=2, fov_pixels=None, options=dict()):
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

        self._validate_config()

        poppy_core._log.info("Creating optical system model:")


        if detector_oversample is None: detector_oversample = fft_oversample

        poppy_core._log.debug("Oversample: %d  %d " % (fft_oversample, detector_oversample))
        optsys = poppy_core.OpticalSystem(name=self.name, oversample=fft_oversample)
        if 'source_offset_r' in options.keys(): optsys.source_offset_r = options['source_offset_r']
        if 'source_offset_theta' in options.keys(): optsys.source_offset_theta = options['source_offset_theta']


        #---- set pupil intensity
        pupil_optic=None # no optic yet defined
        if isinstance(self.pupil, poppy_core.OpticalElement): # do we already have an object?
            pupil_optic = self.pupil
            full_pupil_path = None
        elif isinstance(self.pupil, str): # simple filename
            if os.path.exists( self.pupil) :
                full_pupil_path = self.pupil 
            else: raise IOError("File not found: "+full_pupil_path)
        elif isinstance(self.pupil, fits.HDUList): # pupil supplied as FITS HDUList object
            full_pupil_path = self.pupil
        else: 
            raise TypeError("Not sure what to do with a pupil of that type:"+str(type(self.pupil)))

        #---- set pupil OPD
        if isinstance(self.pupilopd, str):  # simple filename
            full_opd_path = self.pupilopd if os.path.exists( self.pupilopd) else os.path.join(self._datapath, "OPD",self.pupilopd)
        elif hasattr(self.pupilopd, '__getitem__') and isinstance(self.pupilopd[0], basestring): # tuple with filename and slice
            full_opd_path =  (self.pupilopd[0] if os.path.exists( self.pupilopd[0]) else os.path.join(self._datapath, "OPD",self.pupilopd[0]), self.pupilopd[1])
        elif isinstance(self.pupilopd, fits.HDUList): # OPD supplied as FITS HDUList object
            full_opd_path = self.pupilopd # not a path per se but this works correctly to pass it to poppy
        elif self.pupilopd is None: 
            full_opd_path = None
        else:
            raise TypeError("Not sure what to do with a pupilopd of that type:"+str(type(self.pupilopd)))



        #---- apply pupil intensity and OPD to the optical model
        optsys.addPupil(name='Entrance Pupil', optic=pupil_optic, transmission=full_pupil_path, opd=full_opd_path, opdunits='micron', rotation=self._rotation)


        #--- add the detector element. 
        if fov_pixels is None:
            fov_pixels = np.round(fov_arcsec/self.pixelscale)
            if 'parity' in self.options.keys():
                if self.options['parity'].lower() == 'odd'  and np.remainder(fov_pixels,2)==0: fov_pixels +=1
                if self.options['parity'].lower() == 'even' and np.remainder(fov_pixels,2)==1: fov_pixels +=1

        optsys.addDetector(self.pixelscale, fov_pixels = fov_pixels, oversample = detector_oversample, name=self.name+" detector")

        return optsys

    def _applyJitter(self, result, local_options=None):
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
        amount of the blur is taken from the options['jitter_sigma'] value.

        Other types of jitter are not yet implemented. 

        The image in the 'result' HDUlist will be modified by this function.
        """
        if local_options is None: local_options = self.options
        if 'jitter' not in local_options.keys(): return

        poppy_core._log.info("Calculating jitter using "+str(local_options['jitter']) )


        if local_options['jitter'] is None:
            return
        elif 'gauss' in local_options['jitter'].lower():
            import scipy.ndimage

            try:
                sigma = local_options['jitter_sigma']
            except:
                poppy_core._log.warn("Gaussian jitter model requested, but no width for jitter distribution specified. Assuming jitter_sigma = 0.007 arcsec by default")
                sigma = 0.007

            # that will be in arcseconds, we need to convert to pixels:
            
            poppy_core._log.info("Jitter: Convolving with Gaussian with sigma=%.2f arcsec" % sigma)
            out = scipy.ndimage.gaussian_filter(result[0].data, sigma/self.pixelscale)
            peak = result[0].data.max()
            newpeak = out.max()
            strehl   = newpeak/peak # not really the whole Strehl ratio, just the part due to jitter

            poppy_core._log.info("        resulting image peak drops to %.3f of its previous value" % strehl)
            result[0].header['JITRTYPE'] = ( 'Gaussian convolution', 'Type of jitter applied')
            result[0].header['JITRSIGM'] = ( sigma, 'Gaussian sigma for jitter [arcsec]')
            result[0].header['JITRSTRL'] = ( strehl, 'Image peak reduction due to jitter')

            result[0].data = out
        else:
            raise ValueError('Unknown jitter option value: '+local_options['jitter'])





    #####################################################
    # Display routines

    def display(self):
        """Display the currently configured optical system on screen """
        #if coronagraphy is set, then we have to temporarily disable semi-analytic coronagraphic mode
        # to get a regular displayable optical system
        try:
            old_no_sam = self.options['no_sam']
            self.options['no_sam'] = True
        except:
            old_no_sam = None
        
        optsys = self._getOpticalSystem()
        optsys.display(what='both')
        if old_no_sam is not None: self.options['no_sam'] = old_no_sam

    #####################################################
    #
    # Synthetic Photometry related methods
    #
    def _getSpecCacheKey(self, source, nlambda):
        """ return key for the cache of precomputed spectral weightings.
        This is a separate function so the TFI subclass can override it.
        """
        return (self.filter, source.name, nlambda)

    def _getSynphotBandpass(self, filtername):
        """ Return a pysynphot.ObsBandpass object for the given desired band. 

        By subclassing this, you can define whatever custom bandpasses are appropriate for your instrument

        Parameters
        ----------
        filtername : str
            String name of the filter that you are interested in

        Returns
        --------
        a pysynphot.ObsBandpass object for that filter. 

        """

        if filtername.lower().startswith('f'):
            # attempt to treat it as an HST filter name?
            bpname = ('wfc3,uvis1,%s'%(filtername)).lower()
        else:
            bpname=filtername

        try:
            band = pysynphot.ObsBandpass( bpname)
        except:
            raise LookupError("Don't know how to compute pysynphot.ObsBandpass for a filter named "+filtername)

        return band

    def _getDefaultNLambda(self, filtername):
        """ Return the default # of wavelengths to be used for calculation by a given filter """
        return 10

    def _getDefaultFOV(self):
        """ Return default FOV in arcseconds """
        return 5

    def _getFilterList(self):
        """ Returns a list of allowable filters, and the corresponding pysynphot ObsBandpass strings
        for each. 

        If you need to define bandpasses that are not already available in pysynphot, consider subclassing
        _getSynphotBandpass instead to create a pysynphot spectrum based on data read from disk, etc.

        Returns
        --------
        filterlist : list
            List of string filter names
        bandpasslist : dict
            dictionary of string names for use by pysynphot

        This could probably be folded into one using an OrderdDict. FIXME do that later

        """

        filterlist =  ['V','R','I', 'F606W']
        bandpasslist = { 'V':"V",'R':"R",'I':"I", "F606W":'acs,wfc,f606w'}

        return (filterlist, bandpasslist)


    #def _getJitterKernel(self, type='Gaussian', sigma=10):

    def _getWeights(self, source=None, nlambda=5, monochromatic=None, verbose=False):
        """ Return the set of discrete wavelengths, and weights for each wavelength,
        that should be used for a PSF calculation.

        Uses pysynphot (if installed), otherwise assumes simple-minded flat spectrum

        """
        try:
            import pysynphot
            _HAS_PYSYNPHOT = True
        except:
            _HAS_PYSYNPHOT = False


        if monochromatic is not None:
            poppy_core._log.info(" monochromatic calculation requested.")
            return (np.asarray([monochromatic]),  np.asarray([1]) )

        elif _HAS_PYSYNPHOT and (isinstance(source, pysynphot.spectrum.SourceSpectrum)  or source is None):
            """ Given a pysynphot.SourceSpectrum object, perform synthetic photometry for
            nlambda bins spanning the wavelength range of interest.

            Because this calculation is kind of slow, cache results for reuse in the frequent
            case where one is computing many PSFs for the same spectral source.
            """
            poppy_core._log.debug("Calculating spectral weights using pysynphot, nlambda=%d, source=%s" % (nlambda, str(source)))
            if source is None:
                try:
                    source = pysynphot.Icat('ck04models',5700,0.0,2.0)
                except:
                    poppy_core._log.error("Could not load Castelli & Kurucz stellar model from disk; falling back to 5700 K blackbody")
                    source = pysynphot.BlackBody(5700)
            poppy_core._log.debug("Computing spectral weights for source = "+str(source))

            try:
                key = self._getSpecCacheKey(source, nlambda)
                if key in self._spectra_cache.keys():
                    poppy_core._log.debug("Previously computed spectral weights found in cache, just reusing those")
                    return self._spectra_cache[keys]
            except:
                pass  # in case sourcespectrum lacks a name element so the above lookup fails - just do the below calc.

            poppy_core._log.info("Computing wavelength weights using synthetic photometry for %s..." % self.filter)
            band = self._getSynphotBandpass(self.filter)
            # choose reasonable min and max wavelengths
            w_above10 = np.where(band.throughput > 0.10*band.throughput.max())

            minwave = band.wave[w_above10].min()
            maxwave = band.wave[w_above10].max()
            poppy_core._log.debug("Min, max wavelengths = %f, %f" % (minwave/1e4, maxwave/1e4))
            # special case: ignore red leak for MIRI F560W, which has a negligible effect in practice
            # this is lousy test data rather than a bad filter?
            if self.filter == 'F560W':
                poppy_core._log.debug("Special case: setting max wavelength to 6.38 um to ignore red leak")
                maxwave = 63800.0
            elif self.filter == 'F1280W':
                poppy_core._log.debug("Special case: setting max wavelength to 14.32 um to ignore red leak")
                maxwave = 143200.0

            wave_bin_edges =  np.linspace(minwave,maxwave,nlambda+1)
            wavesteps = (wave_bin_edges[:-1] +  wave_bin_edges[1:])/2
            deltawave = wave_bin_edges[1]-wave_bin_edges[0]
            effstims = []

            for wave in wavesteps:
                poppy_core._log.debug("Integrating across band centered at %.2f microns with width %.2f" % (wave/1e4,deltawave/1e4))
                box = pysynphot.Box(wave, deltawave) * band
                if box.throughput.max() == 0:  # watch out for pathological cases with no overlap (happens with MIRI FND at high nlambda)
                    result = 0.0
                else:
                    binset =  np.linspace(wave-deltawave, wave+deltawave, 30)  # what wavelens to use when integrating across the sub-band?
                    result = pysynphot.Observation(source, box, binset=binset).effstim('counts')
                effstims.append(result)

            effstims = np.array(effstims)
            effstims /= effstims.sum()
            wave_m =  band.waveunits.Convert(wavesteps,'m') # convert to meters

            newsource = (wave_m, effstims)
            if verbose: _log.info( " Wavelengths and weights computed from pysynphot: "+str( newsource))
            self._spectra_cache[ self._getSpecCacheKey(source,nlambda)] = newsource
            return newsource

        else:  #Fallback simple code for if we don't have pysynphot.
            poppy_core._log.warning("Pysynphot unavailable (or invalid source supplied)!   Assuming flat # of counts versus wavelength.")
            # compute a source spectrum weighted by the desired filter curves.
            # TBD this will eventually use pysynphot, so don't write anything fancy for now!
            wf = np.where(np.asarray(self.filter_list) == self.filter)[0]
            # The existing FITS files all have wavelength in ANGSTROMS since that is the pysynphot convention...
            #filterdata = atpy.Table(self._filter_files[wf], type='fits')
            filterfits = fits.open(self._filter_files[wf])
            filterdata = filterfits[1].data 
            try:
                f1 = filterdata.WAVELENGTH
                d2 = filterdata.THROUGHPUT
            except:
                raise ValueError("The supplied file, %s, does not appear to be a FITS table with WAVELENGTH and THROUGHPUT columns." % self._filter_files[wf] )
            if 'WAVEUNIT' in  filterfits[1].header.keys():
                waveunit  = filterfits[1].header['WAVEUNIT']
            else:
                poppy_core._log.warn("CAUTION: no WAVEUNIT keyword found in filter file {0}. Assuming = Angstroms by default".format(filterfits.filename()))
                waveunit = 'Angstrom'
            if waveunit != 'Angstrom': raise ValueError("The supplied file, %s, does not have WAVEUNIT = Angstrom as expected." % self._filter_files[wf] )
            poppy_core._log.warn("CAUTION: Just interpolating rather than integrating filter profile, over %d steps" % nlambda)
            wtrans = np.where(filterdata.THROUGHPUT > 0.4)
            if self.filter == 'FND':  # special case MIRI's ND filter since it is < 0.1% everywhere...
                wtrans = np.where(  ( filterdata.THROUGHPUT > 0.0005)  & (filterdata.WAVELENGTH > 7e-6*1e10) & (filterdata.WAVELENGTH < 26e-6*1e10 ))
            lrange = filterdata.WAVELENGTH[wtrans] *1e-10  # convert from Angstroms to Meters
            lambd = np.linspace(np.min(lrange), np.max(lrange), nlambda)
            filter_fn = scipy.interpolate.interp1d(filterdata.WAVELENGTH*1e-10, filterdata.THROUGHPUT,kind='cubic', bounds_error=False)
            weights = filter_fn(lambd)
            return (lambd,weights)
            #source = {'wavelengths': lambd, 'weights': weights}



