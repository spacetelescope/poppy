# Specialized optical system propagators
# In particular for efficient modeling of astronomical coronagraphs

from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
import time
import logging
import astropy.units as u

from . import poppy_core
from . import utils
from . import conf

_log = logging.getLogger('poppy')


class SemiAnalyticCoronagraph(poppy_core.OpticalSystem):
    """ A subclass of OpticalSystem that implements a specialized propagation
    algorithm for coronagraphs whose occulting mask has limited and small support in
    the image plane. Algorithm from Soummer et al. (2007)

    The way to use this class is to build an OpticalSystem class the usual way, and then
    cast it to a SemiAnalyticCoronagraph, and then you can just call calc_psf on that in the
    usual fashion.

    Parameters
    -----------
    existing_optical_system : OpticalSystem
        An optical system which can be converted into a SemiAnalyticCoronagraph.
    oversample : int
        Oversampling factor in intermediate image plane. Default is 8
    occulter_box : float
        half size of field of view region entirely including the occulter, in arcseconds. Default 1.0
        This can be a tuple or list to specify a rectangular region [deltaY,deltaX] if desired.
    fpm_index : int
        Which plane in the provided existing optical system is the focal plane mask? Note, the
        Lyot plane must be the plane immediately after this.


    Notes
    ------

    Note that this algorithm is only appropriate for certain types of Fourier transform,
    namely those using occulters limited to a sub-region of the image plane.

    See also MatrixFTCoronagraph.

    """

    def __init__(self, existing_optical_system, oversample=8, occulter_box=1.0,
                 fpm_index=1, **kwargs):
        from . import optics
        super(SemiAnalyticCoronagraph, self).__init__(**kwargs)

        self.name = "SemiAnalyticCoronagraph for " + existing_optical_system.name
        self.verbose = existing_optical_system.verbose
        self.source_offset_r = existing_optical_system.source_offset_r
        self.source_offset_theta = existing_optical_system.source_offset_theta
        self.planes = existing_optical_system.planes
        self.npix = existing_optical_system.npix
        self.pupil_diameter = existing_optical_system.pupil_diameter

        # SemiAnalyticCoronagraphs have some mandatory planes, so give them reasonable names:
        self.fpm_index = fpm_index
        self.occulter = self.planes[fpm_index]
        self.lyotplane = self.planes[fpm_index + 1]
        self.detector = self.planes[-1]

        self.mask_function = optics.InverseTransmission(self.occulter)

        pt = poppy_core.PlaneType
        for label, plane, typecode in zip(["Occulter (plane {})".format(fpm_index),
                                           "Lyot (plane {})".format(fpm_index + 1),
                                           "Detector (last plane)"],
                                          [self.occulter, self.lyotplane, self.detector],
                                          [pt.image, pt.pupil, pt.detector]):
            if not plane.planetype == typecode:
                raise ValueError("Plane {0} is not of the right type for a semianalytic \
                        coronagraph calculation: should be {1:s} but is {2:s}.".format(label,
                                                                                       typecode, plane.planetype))

        self.oversample = oversample

        if not np.isscalar(occulter_box):
            occulter_box = np.array(occulter_box)  # cast to numpy array so the multiplication by 2
                                                  # just below will work
        self.occulter_box = occulter_box

        self.occulter_highres = poppy_core.Detector(self.detector.pixelscale / self.oversample,
                                                    fov_arcsec=self.occulter_box * 2,
                                                    name='Oversampled Occulter Plane')

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self, wavelength=2e-6 * u.meter,
                       normalize='first',
                       retain_final=False,
                       retain_intermediates=False,
                       display_intermediates=False):
        """Propagate a monochromatic wavefront through the optical system. Called from within `calc_psf`.
        Returns a tuple with a `fits.HDUList` object and a list of intermediate `Wavefront`s (empty if
        `retain_intermediates=False`).

        *** This is a modified/custom version of the code in OpticalSystem.propagate_mono() ***

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
            The 0th item is "before first plane", 1st is "after first plane and before second plane", and so on.
            (n.b. This will be empty if `retain_intermediates` is False and singular if retain_final is True.)
        """
        if conf.enable_speed_tests:
            t_start = time.time()
        if self.verbose:
            _log.info(" Propagating wavelength = {0:g} meters using "
                      "Fast Semi-Analytic Coronagraph method".format(wavelength))
        wavefront = self.input_wavefront(wavelength)

        intermediate_wfs = []

        wavefront.history.append("Propagating using Fast Semi-Analytic Method")
        wavefront.history.append(" for Coronagraphy (See Soummer et al. 2007).")

        # note: 0 is 'before first optical plane; 1 = 'after first plane and before second plane' and so on
        current_plane_index = 0

        # ------- differences from regular propagation begin here --------------

        nrows = len(self.planes) + 2  # there are some extra display planes
        if (normalize.lower() != 'first') and (normalize.lower() != 'last'):
            raise NotImplementedError("Only normalizations 'first' or 'last' are implemented for SAMC")

        # SAMC step 1: Propagate up until just before the FPM in the regular manner
        for optic in self.planes[0:self.fpm_index]:
            # The actual propagation:
            wavefront.propagate_to(optic)
            wavefront *= optic
            current_plane_index += 1

            # Normalize if appropriate:
            if normalize.lower() == 'first' and current_plane_index == 1:  # set entrance plane to 1.
                wavefront.normalize()

            if retain_intermediates:  # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if hasattr(optic, 'wavefront_display_hint'):
                    display_what = optic.wavefront_display_hint
                else:
                    display_what = 'best'
                if hasattr(optic, 'wavefront_display_vmax_hint'):
                    display_vmax = optic.wavefront_display_vmax_hint
                else:
                    display_vmax = None
                if hasattr(optic, 'wavefront_display_vmin_hint'):
                    display_vmin = optic.wavefront_display_vmin_hint
                else:
                    display_vmin = None

                wavefront.display(what=display_what, nrows=nrows, row=current_plane_index,
                                  colorbar=False, vmax=display_vmax, vmin=display_vmin)

                # nrows = 6
                # wavefront.display(what='best',nrows=nrows,row=1, colorbar=False,
                # title="propagating $\lambda=$ {:.3f}".format(wavelength.to(u.micron)))

        # SAMC step 2: propagate to detector via MFT at high res.

        # determine FOV region bounding the image plane occulting stop.
        # determine number of pixels across that to use ("N_B")
        # calculate the MFT to the N_B x N_B occulting region.
        wavefront_cor = wavefront.copy()
        wavefront_cor.propagate_to(self.occulter_highres)  # This will be an MFT propagation
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_cor.copy())

        if display_intermediates:  # Display prior to the occulter
            wavefront_cor.display(what='best', nrows=nrows, row=current_plane_index, colorbar=False)

        # Multiply that by M(r) =  1 - the occulting plane mask function
        wavefront_cor *= self.mask_function
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_cor.copy())

        if display_intermediates:  # Display after the occulter (EXTRA PLANE)
            wavefront_cor.display(what='intensity', nrows=nrows, row=current_plane_index, colorbar=False)

        # SAMC step 3:
        # calculate the MFT from that small region back to the full Lyot plane, and
        # subtract that from the original electric field at the prior pupil

        wavefront_lyot = wavefront_cor.copy()
        wavefront_lyot.propagate_to(self.lyotplane)
        current_plane_index += 1
        if retain_intermediates:
            intermediate_wfs.append(wavefront_lyot.copy())

        # combine that with the original pupil function
        wavefront_combined = wavefront + (-1) * wavefront_lyot
        wavefront_combined.location = 'recombined Lyot pupil'

        wavefront = wavefront_combined

        if display_intermediates:  # Display back at Lyot (EXTRA PLANE)
            wavefront.display(what='intensity', nrows=nrows, row=current_plane_index, colorbar=False)

        # SAMC step 4: propagate through the rest of the optical system
        for optic in self.planes[self.fpm_index + 1:]:
            # The actual propagation:
            wavefront.propagate_to(optic)
            wavefront *= optic
            current_plane_index += 1

            if retain_intermediates:  # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if hasattr(optic, 'wavefront_display_hint'):
                    display_what = optic.wavefront_display_hint
                else:
                    display_what = 'best'
                if hasattr(optic, 'wavefront_display_vmax_hint'):
                    display_vmax = optic.wavefront_display_vmax_hint
                else:
                    display_vmax = None
                if hasattr(optic, 'wavefront_display_vmin_hint'):
                    display_vmin = optic.wavefront_display_vmin_hint
                else:
                    display_vmin = None

                wavefront.display(what=display_what, nrows=nrows, row=current_plane_index,
                                  colorbar=False, vmax=display_vmax, vmin=display_vmin)

        # ------- differences from regular propagation end here --------------

        # prepare output arrays
        if normalize.lower() == 'last':
            wavefront.normalize()

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop - t_start))

        if (not retain_intermediates) & retain_final:  # return the full complex wavefront of the last plane.
            intermediate_wfs = [wavefront]

        return wavefront.as_fits(), intermediate_wfs


class MatrixFTCoronagraph(poppy_core.OpticalSystem):
    """ A subclass of OpticalSystem that implements a specialized propagation
    algorithm for coronagraphs which are most efficiently modeled by
    matrix Fourier transforms, and in which the semi-analytical/Babinet
    superposition approach does not apply.

    The way to use this class is to build an OpticalSystem class the usual way, and then
    cast it to a MatrixFTCoronagraph, and then you can just call `calc_psf` on that in the
    usual fashion.

    Parameters
    -----------
    existing_optical_system : OpticalSystem
        An optical system which can be converted into a SemiAnalyticCoronagraph
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

    def __init__(self, existing_optical_system, oversample=4, occulter_box=1.0,
                 **kwargs):
        super(MatrixFTCoronagraph, self).__init__(**kwargs)

        if len(existing_optical_system.planes) < 4:
            raise ValueError("Input optical system must have at least 4 planes "
                             "to be convertible into a MatrixFTCoronagraph")
        self.name = "MatrixFTCoronagraph for " + existing_optical_system.name
        self.verbose = existing_optical_system.verbose
        self.source_offset_r = existing_optical_system.source_offset_r
        self.source_offset_theta = existing_optical_system.source_offset_theta
        self.planes = existing_optical_system.planes
        self.npix = existing_optical_system.npix
        self.pupil_diameter = existing_optical_system.pupil_diameter

        self.oversample = oversample

        # if hasattr(occulter_box, '__getitem__'):
        if not np.isscalar(occulter_box):
            occulter_box = np.array(occulter_box)  # cast to numpy array so the multiplication by 2
                                                   # just below will work
        self.occulter_box = occulter_box

    @utils.quantity_input(wavelength=u.meter)
    def propagate_mono(self, wavelength=1e-6*u.meter,
                       normalize='first',
                       retain_intermediates=False,
                       retain_final=False,
                       display_intermediates=False):
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
            The 0th item is "before first plane", 1st is "after first plane and before second plane", and so on.
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
            if optic.planetype == poppy_core.PlaneType.image:
                if len(optic.amplitude.shape) == 2:  # Match detector object to the loaded FPM transmission array
                    metadet = poppy_core.Detector(optic.pixelscale, fov_pixels=optic.amplitude.shape[0],
                                                  name='Oversampled Occulter Plane')
                else:
                    metadet_pixelscale = ((wavelength / self.planes[0].pupil_diam).decompose()
                                          * u.radian).to(u.arcsec) / self.oversample / 2 / u.pixel
                    metadet = poppy_core.Detector(metadet_pixelscale, fov_arcsec=self.occulter_box * 2,
                                                  name='Oversampled Occulter Plane')
                wavefront.propagate_to(metadet)
            else:
                wavefront.propagate_to(optic)
            wavefront *= optic
            current_plane_index += 1

            # Normalize if appropriate:
            if normalize.lower() == 'first' and current_plane_index == 1:  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower() == 'first=2' and current_plane_index == 1:
                # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower() == 'exit_pupil':  # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(np.asarray([p.planetype is poppy_core.PlaneType.pupil
                                                              for p in self.planes]))[0].max() + 1
                if current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug("normalizing at exit pupil (plane {0}) "
                               "to 1.0 total intensity".format(current_plane_index))
            elif normalize.lower() == 'last' and current_plane_index == len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")

            # Optional outputs:
            if conf.enable_flux_tests: _log.debug("  Flux === " + str(wavefront.total_intensity))

            if retain_intermediates:  # save intermediate wavefront, summed for polychromatic if needed
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
                    _log.debug("\tTIME %f s\t for displaying the wavefront." % (t1 - t0))

        # prepare output arrays
        if normalize.lower() == 'last':
            wavefront.normalize()

        if conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop - t_start))

        if (not retain_intermediates) & retain_final:  # return the full complex wavefront of the last plane.
            intermediate_wfs = [wavefront]
        return wavefront.as_fits(), intermediate_wfs
