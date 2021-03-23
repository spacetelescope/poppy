# Code for modeling deformable mirrors
# By Neil Zimmerman based on Marshall's dms.py in the gpipsfs repo

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
import scipy.signal
import astropy.io.fits as fits
import astropy.units as u
from abc import ABC, abstractmethod

from . import utils, accel_math, poppy_core, optics

import logging

_log = logging.getLogger('poppy')

if accel_math._USE_NUMEXPR:
    import numexpr as ne

__all__ = ['ContinuousDeformableMirror', 'HexSegmentedDeformableMirror', 'CircularSegmentedDeformableMirror']


# noinspection PyUnresolvedReferences
class ContinuousDeformableMirror(optics.AnalyticOpticalElement):
    # noinspection PyUnresolvedReferences
    """ Generic deformable mirror, of the continuous face sheet variety.

            Parameters
            ----------
            dm_shape : tuple with 2 elements
                Number of actuators across the clear aperture in each dimension
            actuator_spacing : float or astropy Quantity with dimension length
                Spacing between adjacent actuators as seen in that plane
            influence_func : Influence function filename. Optional. If not supplied,
                a Gaussian model approximatly representative of an influence function
                for a Boston MEMS DMs will be used. This parameter can let you provide
                a more detailed model for your particular mirror.
            radius : float or Quantity with dimension length
                radius of the clear aperture of the DM. Note, the reflective clear aperture
                may potentially be larger than the controllable active aperture, depending
                on the details of your particular hardware. This parameter does not affect
                any of the actuator models, only the reflective area for wavefront amplitude.
            flip_x, flip_y : Bool
                Flip the orientation of the X or Y axes of the DM actuators. Useful if your
                device is not oriented such that the origin is at lower left as seen in the
                pupil.
            include_factor_of_two : Bool
                include the factor of two due to reflection in the OPD function (optional, default False).
                If this is set False (default), actuator commands are interpreted as being in units of
                desired wavefront error directly; the returned WFE will be directly proportional to the requested
                values (convolved with the actuator response function etc).
                If this is set to True, then the actuator commands are interpreted as being in physical surface
                units, and the WFE is therefore a factor of two larger. The returned WFE will be twice the
                amplitude of the requested values (convolved with the actuator response function etc.)

            Additionally, the standard parameters for shift_x and shift_y can be accepted and
            will be handled by the **kwargs mechanism. Note, rotation is not yet supported for DMs,
            and shifts are currently rounded to integer pixels in the sampled wavefront, rather
            than being applied as floating point values.  Shifts are specified in physical units
            of meters transverse motion in the DM plane.

            Note:  Keeping track of actuator locations and spacing is subtle. This note
            assumes you are familiar with 'counting fenceposts' and off-by-one errors.
            This class follows the convention adopted by Boston Micromachines:
            If there are N actuators across a given distance, there are N-1 spaces between
            actuators across that distance (plus half an actuator space border on the outside,
            which is still controlled by the outermost actuators.). The controlled portion of
            the pupil thus has diameter  = (N-1)*actuator_spacing.
            However, for display purposes etc it is often useful if the displayed pupil extends
            over the full N actuators, so we set the `pupil_diam` attribute to N*actuator_spacing,
            or the reflective radius, whichever is larger.

            """

    @utils.quantity_input(actuator_spacing=u.meter, radius=u.meter)

    def __init__(self, dm_shape=(10, 10), actuator_spacing=None,
                 influence_func=None, name='DM',
                 include_actuator_print_through=False,
                 actuator_print_through_file=None,
                 actuator_mask_file=None,
                 radius=1.0 * u.meter,
                 flip_x=False, flip_y=False,
                 include_factor_of_two = False,
                 **kwargs
                 ):

        optics.AnalyticOpticalElement.__init__(self, planetype=poppy_core.PlaneType.pupil, **kwargs)
        self._dm_shape = dm_shape  # number of actuators
        self.name = name
        self._surface = np.zeros(dm_shape)  # array for the DM surface OPD, in meters
        self.numacross = dm_shape[0]  # number of actuators across diameter of
            # the optic's cleared aperture (may be less than full diameter of array)
        self.flip_x = flip_x
        self.flip_y = flip_y

        # What is the total reflective area that passes light onwards?
        self.radius_reflective = radius
        self._aperture = optics.CircularAperture(radius=radius, **kwargs)

        # What is the active area and spacing between actuators?
        if actuator_spacing is None:
            self.actuator_spacing = 2*radius / (self.numacross - 1)  # distance between actuators,
                                                                     # projected onto the primary
        else:
            self.actuator_spacing = actuator_spacing
        self.radius_active = self.actuator_spacing*(self.numacross-1)/2  # Boston Micromachines convention;
                                                                         # active area is 'inside the fenceposts'
        self.pupil_center = (dm_shape[0] - 1.) / 2  # center of clear aperture in actuator units

        # the poppy-standard attribute 'pupil_diam' is used for default display or input wavefront sizes
        self.pupil_diam = max(np.max(dm_shape) * self.actuator_spacing, self.radius_reflective*2)  # see note above

        self.include_actuator_print_through = include_actuator_print_through

        # are some actuators masked out/inactive (i.e. circular factor Boston DMs have inactive corners)
        self.include_actuator_mask = actuator_mask_file is not None
        if self.include_actuator_mask:
            self.actuator_mask_file = actuator_mask_file
            self.actuator_mask = fits.getdata(self.actuator_mask_file)

        if self.include_actuator_print_through:
            self._load_actuator_surface_file(actuator_print_through_file)

        self.include_factor_of_two = include_factor_of_two

        if isinstance(influence_func, str):
            self.influence_type = "from file"
            self._load_influence_fn(filename=influence_func)
        elif isinstance(influence_func, fits.HDUList):
            self.influence_type = "from file"
            self._load_influence_fn(hdulist=influence_func)
        elif influence_func is None:
            self.influence_type = "default Gaussian"
        else:
            raise TypeError('Not sure how to use an influence function parameter of type ' + str(type(influence_func)))


    def _load_influence_fn(self, filename=None, hdulist=None):
        """ Load and verify an influence function provided by FITS file or HDUlist """
        import copy

        if filename is None and hdulist is not None:
            # supplied as HDUlist
            hdulist = copy.deepcopy(hdulist)  # don't modify the one provided as argument.
            self.influence_func_file = 'supplied as fits.HDUList object'
            _log.info("Loaded influence function from supplied fits.HDUList object")
        elif filename is not None and hdulist is None:
            # supplied as filename
            # with fits.open(filename) as filehandle:
            # hdulist = copy.copy(filehandle)
            hdulist = fits.open(filename)
            self.influence_func_file = filename
            _log.info("Loaded influence function from " + self.influence_func_file+" for "+self.name)
        else:
            raise RuntimeError("must supply exactly one of the filename and hdulist arguments.")

        self.influence_func = hdulist[0].data.copy()
        self.influence_header = hdulist[0].header.copy()
        if len(self.influence_func.shape) != 2:
            raise RuntimeError("Influence function file must contain a 2D array.")

        try:
            self.influence_func_sampling = float(self.influence_header['SAMPLING'])
        except KeyError:
            raise RuntimeError("Influence function file must have a SAMPLING keyword giving # pixels per actuator.")

        hdulist.close()

    def _get_rescaled_influence_func(self, pixelscale):
        """ Return the influence function, rescaled onto the appropriate pixel scale for
        the wavefront array."""
        # self.influence_func contains the 2D influence function array.
        # self.influence_func_sampling records how many pixels, in the provided array, represents 1 actuator spacing.
        # How many pixels in the output array between actuators?
        act_space_m = self.actuator_spacing.to(u.meter).value
        act_space_pix = act_space_m / pixelscale.to(u.meter / u.pixel).value
        scale = act_space_pix / self.influence_func_sampling

        # suppress irrelevant scipy warning here
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return scipy.ndimage.zoom(self.influence_func, scale)

    def _get_rescaled_actuator_surface(self, pixelscale):
        """ Return the actuator surface print-through, rescaled onto the
        appropriate pixel scale for the wavefront array."""
        # self.actuator_surface contains the 2D influence function array,
        # for one single pixel.
        # How many pixels in the output array between actuators?
        act_space_pix = (self.actuator_spacing / pixelscale).to(u.pixel).value
        scale = act_space_pix / self.actuator_surface.shape[0]
        # suppress irrelevant scipy warning here
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return scipy.ndimage.zoom(self.actuator_surface, scale)

    def _load_actuator_surface_file(self, filename=None):
        """ Load an array representing the actuator surface print-through

        TODO header keyword checking etc!
        """
        if filename is None:
            raise ValueError("Must supply actuator_print_through_file parameter to include actuator print through.")
        self.actuator_surface = fits.getdata(filename)

    @property
    def dm_shape(self):
        """ DM actuator geometry - i.e. how many actuators per axis """
        return self._dm_shape

    @property
    def surface(self):
        """ The commanded surface shape of the deformable mirror, in
        **meters**.
        This is the input to the DM. See the .opd property for the output.
        """
        return self._surface

    @utils.quantity_input(new_surface=u.meter)
    def set_surface(self, new_surface):
        """ Set the entire surface shape of the DM.

        Parameters
        -------------
        new_surface : 2d ndarray, or scalar
            Desired DM surface OPD, in meters by default, or use
            the astropy units system to specify a different unit
        """
        if np.isscalar(new_surface.value):
            self._surface[:] = new_surface.to(u.meter).value
        else:
            assert new_surface.shape == self._surface.shape, "Supplied surface shape doesn't match DM. Must be {}".format(self._surface.shape)
            self._surface[:] = np.asarray(new_surface.to(u.meter).value, dtype=float)

    @utils.quantity_input(new_value=u.meter)
    def set_actuator(self, actx, acty, new_value):
        """ Set an individual actuator of the DM.
        Parameters
        -------------
        actx, acty : integers
            Coordinates of the actuator you wish to control
        new_value : float
            Desired surface height for that actuator, in meters
            by default or use astropy Units to specify another unit if desired.
        Example
        -----------
        dm.set_actuator(12, 22, 123.4*u.nm)
        """

        if self.include_actuator_mask:
            if not self.actuator_mask[acty, actx]:
                raise RuntimeError("Actuator ({}, {}) is masked out for that DM.".format(actx, acty))

        if actx < 0 or actx > self.dm_shape[1] - 1:
            raise ValueError("X axis coordinate is out of range")
        if acty < 0 or acty > self.dm_shape[0] - 1:
            raise ValueError("Y axis coordinate is out of range")

        self._surface[acty, actx] = new_value.to(u.meter).value

    def flatten(self):
        """Flatten the DM by setting all actuators to zero piston"""
        self._surface[:] = 0

    def get_act_coordinates(self, one_d=False, include_transformations=False):
        """ Y and X coordinates for the actuators

        Parameters
        ------------
        one_d : bool
            Return 1-dimensional arrays of coordinates per axis?
            Default is to return 2D arrays with same shape as full array.
        include_transformations : bool
            Return *apparent* coordinates after rotations, etc of the DM.

        Returns
        -------
        y_act, x_act : float ndarrays
            actuator coordinates, in units of meters
        """

        act_space_m = self.actuator_spacing.to(u.meter).value
        y_act = (np.arange(self.dm_shape[0]) - self.pupil_center) * act_space_m
        x_act = (np.arange(self.dm_shape[1]) - self.pupil_center) * act_space_m

        if not one_d:  # convert to 2D
            y_act.shape = (self.dm_shape[0], 1)
            y_act = y_act * np.ones((1, self.dm_shape[1]))

            x_act.shape = (1, self.dm_shape[1])
            x_act = x_act * np.ones((self.dm_shape[0], 1))

        if include_transformations:
            # Repeat the same transformations here as applied in AnalyticOpticalElement.get_coordinates()
            # But with opposite sense, since there the transformation is on the coordinate system and here
            # it is on the optic
            if hasattr(self, "rotation"):
                angle = -np.deg2rad(self.rotation)
                xp = np.cos(angle) * x_act + np.sin(angle) * y_act
                yp = -np.sin(angle) * x_act + np.cos(angle) * y_act
                x_act = xp
                y_act = yp
            if getattr(self, 'inclination_x', 0) != 0:
                y_act *= np.cos(np.deg2rad(self.inclination_x))
            if getattr(self, 'inclination_y', 0) != 0:
                x_act *= np.cos(np.deg2rad(self.inclination_y))

        return y_act, x_act

    def get_opd(self, wave):
        """ Return the surface shape OPD for the optic.
        Interpolates from the current optic surface state onto the
        desired coordinates for the wave.
        """

        if self.influence_type == 'from file':
            interpolated_surface = self._get_surface_via_convolution(wave)
        else:
            # the following could be replaced with a higher fidelity model if needed
            interpolated_surface = self._get_surface_via_gaussian_influence_functions(wave)

        if self.include_actuator_print_through:
            interpolated_surface += self._get_actuator_print_through(wave)

        if hasattr(self, 'shift_x') or hasattr(self, 'shift_y'):
            # Apply shifts here, if necessary. Doing it this way lets the same shift code apply
            # across all of the above 3 potential ingredients into the OPD, potentially at some
            # small cost in accuracy rather than shifting each individuall at a subpixel level.

            # suppress irrelevant scipy warning from ndzoom calls
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                if (hasattr(self, 'shift_x') and self.shift_x !=0):
                    pixscale_m = wave.pixelscale.to(u.m/u.pixel).value
                    shift_x_pix = int(np.round(self.shift_x /pixscale_m))
                    interpolated_surface = np.roll(interpolated_surface, shift_x_pix, axis=1)

                if (hasattr(self, 'shift_y') and self.shift_y !=0):
                    pixscale_m = wave.pixelscale.to(u.m/u.pixel).value
                    shift_y_pix = int(np.round(self.shift_y /pixscale_m))
                    interpolated_surface = np.roll(interpolated_surface, shift_y_pix, axis=0)
        
        # account for DM being reflective (optional, governed by include_factor_of_two parameter)
        coefficient = 2 if self.include_factor_of_two else 1

        return coefficient*interpolated_surface # note optional *2 coefficient to account for DM being reflective surface

    def _get_surface_arrays_with_orientation(self):
        """ Return representations of the DM actuators and masks
        possibly with flips horizontally or vertically
        """
        surface = self._surface
        if self.flip_x:
            surface = np.fliplr(surface)
        if self.flip_y:
            surface = np.flipud(surface)

        if self.include_actuator_mask:
            act_mask = self.actuator_mask
            if self.flip_x:
                act_mask = np.fliplr(act_mask)
            if self.flip_y:
                act_mask = np.flipud(act_mask)
        else:
            act_mask = None

        return surface, act_mask

    def _get_surface_via_gaussian_influence_functions(self, wave):
        """ Infer a finely-sampled surface from simple Gaussian influence functions centered on
        each actuator.

        This default representation has 15% crosstalk between adjacent actuators, typical of
        Boston MEMS continuous DMs.

        Work in progress, oversimplified, not a high fidelity representation of the true influence function

        See also self._get_surface_via_convolution
        """
        y, x = self.get_coordinates(wave)
        y_act, x_act = self.get_act_coordinates(one_d=True, include_transformations=False)
        # In this case we don't need to transform the actuator coordinates, since we evaluate
        # them as Gaussian functions relative to the y and x arrays that already include any
        # coordinate transforms present for this optic.

        interpolated_surface = np.zeros(wave.shape)

        crosstalk = 0.15  # amount of crosstalk on advancent actuator
        sigma = self.actuator_spacing.to(u.meter).value / np.sqrt((-np.log(crosstalk)))

        pixelscale = x[0, 1] - x[0, 0]  # scale of x,y

        # check for flips
        surface, act_mask = self._get_surface_arrays_with_orientation()

        for yi, yc in enumerate(y_act):
            for xi, xc in enumerate(x_act):
                if surface[yi, xi] == 0 or (self.include_actuator_mask and act_mask[yi,xi]==0):
                    continue

                # 2d Gaussian
                if accel_math._USE_NUMEXPR:
                    roversigma2 = ne.evaluate("((x - xc)**2 + (y-yc)**2)/sigma**2")
                else:
                    roversigma2 = ((x - xc) ** 2 + (y - yc) ** 2) / sigma ** 2

                interpolated_surface += surface[yi, xi] * accel_math._exp(-roversigma2)

        return interpolated_surface

    def _get_surface_via_convolution(self, wave):
        """ Infer the physical DM surface by convolving the actuator
            "picket fence" trace with the influence function.

            This version uses an influence function read from a file on disk
        """
        # Determine the center indices of the actuators in wavefront space,
        # if not already established.
        if not hasattr(self, '_act_ind_flat') or True:
            self._setup_actuator_indices(wave)

        # check for flips
        surface, act_mask = self._get_surface_arrays_with_orientation()

        if self.include_actuator_mask:
            target_val = (surface * act_mask).ravel()
        else:
            target_val = surface.ravel()

        # Compute the 'surface trace', i.e the values for each actuator, projected
        # into the appropriate locations on the detector. For each actuator, we
        # weight the surface value across a 2x2 square of pixels to account for subpixel
        # positions of the actuators.

        # First, determine DM actuator coordinates in fractional pixels:
        # Since we are working in units of square pixels here, we need to include
        # any coordinate transformations onto the DM actuator coordinates before that.
        dm_act_m = np.stack(self.get_act_coordinates(include_transformations=True))
        center = (np.asarray(wave.shape)-1)/2  # need to be careful here re exact wave center
        center.shape=(2,1,1)
        dm_act_pix = dm_act_m / wave.pixelscale.to(u.m/u.pixel).value + center

        # Then iterate over a 2x2 square of pixels, weighting linearly between adjacent pixels
        # based on the subpixel offset for each actuator
        fracpart, intpart = np.modf(dm_act_pix)
        for ix in (0,1):
            for iy in (0,1):
                xweight = fracpart[1] if ix==1 else (1-fracpart[1])
                yweight = fracpart[0] if iy==1 else (1-fracpart[0])
                try:
                    self._surface_trace_flat[self._act_ind_flat[0] + ix + iy*wave.shape[0]] = (xweight*yweight).flat*target_val
                except:
                    pass # Ignore any actuators outside the FoV


        # Now we can convolve with the influence function to get the full continuous surface.
        influence_rescaled = self._get_rescaled_influence_func(wave.pixelscale)
        dm_surface = scipy.signal.fftconvolve(self._surface_trace_flat.reshape(wave.shape),
                                              influence_rescaled, mode='same')

        return dm_surface

    def _setup_actuator_indices(self, wave):
        # This attribute will hold the 1D, flattened indices of each of the
        # actuators, in the larger wave array.
        # FIXME this will need to become smarter about when to regenerate these.
        # for cases with different wave samplings.
        # For now will just regenerate this every time. Slower but strict.
        y_wave, x_wave = self.get_coordinates(wave)

        N_act = self.numacross
        center = (np.asarray(wave.shape)-1)/2  # need to be careful here re exact wave center

        if getattr(self, 'rotation', 0)==0:
            # The DM is not rotated, so each row or column has a consistent X or Y coordinate
            # This simplifies the math.
            y_act, x_act = self.get_act_coordinates(one_d=True, include_transformations=True)
        else:
            # The DM is rotated, so each actuator has its own unique X and Y coordinate.
            y_act, x_act = self.get_act_coordinates(one_d=False, include_transformations=True)

        # Find integer pixel to the left & down (i.e. floor) for each actuator.
        # This sets us up for the subpixel offsets inside
        # _get_surface_via_convolution()
        dm_x_act_pix = x_act / wave.pixelscale.to(u.m/u.pixel).value + center[1]
        dm_y_act_pix = y_act / wave.pixelscale.to(u.m/u.pixel).value + center[0]
        x_wave_ind_act = np.asarray(np.floor(dm_x_act_pix), dtype=int)
        y_wave_ind_act = np.asarray(np.floor(dm_y_act_pix), dtype=int)

        if getattr(self, 'rotation', 0)==0:
            act_trace_row = np.zeros((1, wave.shape[1]), dtype='bool')
            act_trace_col = np.zeros((wave.shape[0], 1), dtype='bool')
            act_trace_row[0, x_wave_ind_act] = 1
            act_trace_col[y_wave_ind_act, 0] = 1
            act_trace_2d = act_trace_col * act_trace_row
        else:
            self._tmp = (y_wave, x_wave, y_act, x_act, wave )
            # Depending on the amount of rotation, some actuators may have rotated outside of the wavefront array
            acts_in_pupil = ((0 < x_wave_ind_act) & (x_wave_ind_act < wave.shape[1]) &
                             (0 < y_wave_ind_act) & (y_wave_ind_act < wave.shape[0]))
            act_trace_2d = np.zeros(wave.shape, dtype='bool')
            for y, x in zip(y_wave_ind_act[acts_in_pupil], x_wave_ind_act[acts_in_pupil]):
                act_trace_2d[y, x] = 1

        self._act_trace_2d = act_trace_2d

        act_trace_flat = act_trace_2d.ravel()
        self._act_ind_flat = np.nonzero(act_trace_flat)  # 1-d indices of actuator centers in wavefront space
        if self._act_ind_flat[0].shape[0] < N_act**2:
            raise RuntimeError("The specified sampling is too small a region to include all the DM actuators")
        self._surface_trace_flat = np.zeros(act_trace_flat.shape)  # flattened representation of DM surface trace

    def _get_actuator_print_through(self, wave):
        """ DM surface print through. This function currently hardcoded for Boston MEMS.
        TODO - write something more generalized. """

        # Determine the center indices of the actuators in wavefront space,
        # if not already established.
        if not hasattr(self, '_act_ind_flat') or True:
            self._setup_actuator_indices(wave)

        # Set physical DM surface trace -
        # this is constant for the surface print through, for all actuators that are present.
        if self.include_actuator_mask:
            target_val = self.actuator_mask.ravel()
        else:
            target_val = 1

        self._surface_trace_flat[self._act_ind_flat] = target_val

        actuator_rescaled = self._get_rescaled_actuator_surface(wave.pixelscale)
        dm_surface = scipy.signal.fftconvolve(self._surface_trace_flat.reshape(wave.shape),
                                              actuator_rescaled, mode='same')

        return dm_surface

    def get_transmission(self, wave):
        # Pass through transformations of this optic to the aperture sub-optic, if needed
        if hasattr(self, 'shift_x'):
            self._aperture.shift_x = self.shift_x
        if hasattr(self, 'shift_y'):
            self._aperture.shift_y = self.shift_y
        return self._aperture.get_transmission(wave)

    def display(self, annotate=False, grid=False, what='opd', crosshairs=False, *args, **kwargs):
        """Display an Analytic optic by first computing it onto a grid.
        Parameters
        ----------
        wavelength : float
            Wavelength to evaluate this optic's properties at
        npix : int
            Number of pixels to use when sampling the optical element.
        what : str
            What to display: 'intensity', 'surface' or 'phase', or 'both'
        ax : matplotlib.Axes instance
            Axes to display into
        nrows, row : integers
            # of rows and row index for subplot display
        crosshairs : bool
            Display crosshairs indicating the center?
        colorbar : bool
            Show colorbar?
        colorbar_orientation : bool
            Desired orientation, horizontal or vertical?
            Default is horizontal if only 1 row of plots, else vertical
        opd_vmax : float
            Max value for OPD image display, in meters.
        title : string
            Plot label
        annotate : bool
            Draw annotations on plot
        grid : bool
            Show the grid for the DM actuator spacing
        """

        kwargs['crosshairs'] = crosshairs
        kwargs['what'] = what
        returnvalue = optics.AnalyticOpticalElement.display(self, *args, **kwargs)

        # Annotations not yet smart enough to deal with two panels
        if what != 'both':
            if annotate:
                self.annotate()
            if grid:
                self.annotate_grid()
        return returnvalue

    def display_actuators(self, annotate=False, grid=True, what='opd', crosshairs=False, *args, **kwargs):
        """ Display the optical surface, viewed as discrete actuators

        Parameters
        ------------
        annotate : bool
            Annotate coordinates and types of actuators on the display? Default false.
        grid : bool
            Annotate grid of actuators on the display?  Default true.
        what : string
            What to display: 'intensity' transmission, 'opd', or 'both'
        crosshairs : bool
            Draw crosshairs on plot to indicate the origin.
        """

        # call parent class display, setting parameters to display at actuator grid resolution
        returnvalue = super().display(what=what, crosshairs=crosshairs,
                npix = self.dm_shape[0],
                grid_size = self.dm_shape[0]*self.actuator_spacing.to(u.m).value,
                **kwargs)

        if annotate: self.annotate()
        if grid: self.annotate_grid()
        return returnvalue

    def annotate(self, marker='+', **kwargs):
        """ Overplot actuator coordinates on some already-existing pupil display
        """
        yc, xc = self.get_act_coordinates()
        ax = plt.gca()

        # jump through some hoops to avoid autoscaling the X,Y coords
        # of the prior plot here, but retain the autoscale state
        autoscale_state = (ax._autoscaleXon, ax._autoscaleYon)
        ax.autoscale(False)
        plt.scatter(xc, yc, marker=marker, **kwargs)
        ax._autoscaleXon, ax._autoscaleYon = autoscale_state

    def annotate_grid(self, linestyle=":", color="black", **kwargs):
        import matplotlib

        y_act, x_act = self.get_act_coordinates(one_d=True)

        ax = plt.gca()
        # jump through some hoops to avoid autoscaling the X,Y coords
        # of the prior plot here, but retain the autoscale state
        autoscale_state = (ax._autoscaleXon, ax._autoscaleYon)
        ax.autoscale(False)

        act_space_m = self.actuator_spacing.to(u.meter).value
        for x in x_act:
            plt.axvline(x + (act_space_m / 2), linestyle=linestyle, color=color)
        for y in y_act:
            plt.axhline(y + (act_space_m / 2), linestyle=linestyle, color=color)

        ap_radius = self.radius_reflective.to(u.m).value
        aperture = matplotlib.patches.Circle((0,0), radius=ap_radius, fill=False,
                color='red')
        ax.add_patch(aperture)

        active_radius = self.radius_active.to(u.m).value
        aperture = matplotlib.patches.Circle((0,0), radius=active_radius, fill=False,
                color='blue')
        ax.add_patch(aperture)

        ax._autoscaleXon, ax._autoscaleYon = autoscale_state

    def display_influence_fn(self):
        if self.influence_type == 'from file':
            act_space_m = self.actuator_spacing.to(u.meter).value
            r = np.linspace(0, 4 * act_space_m, 50)
            crosstalk = 0.15  # amount of crosstalk on advancent actuator
            sigma = act_space_m / np.sqrt((-np.log(crosstalk)))
            plt.plot(r, np.exp(- (r / sigma) ** 2))
            plt.xlabel('Separation [m]')
            for i in range(4):
                plt.axvline(act_space_m * i, ls=":", color='black')
            plt.ylabel('Actuator Influence')
            plt.title("Gaussian influence function with 15% crosstalk")
        else:
            raise NotImplementedError("Display of influence functions from files not yet written.")



class SegmentedDeformableMirror(ABC):
    """ Abstract class for segmented DMs.
    See below for subclasses for hexagonal and circular apertures.
    """
    def __init__(self, rings=1, include_factor_of_two=False):
        self._surface = np.zeros((self._n_aper_inside_ring(rings + 1), 3))

        # see _setup_arrays for the following
        self._last_npix = np.nan
        self._last_pixelscale = np.nan * u.meter / u.pixel

        self.include_factor_of_two = include_factor_of_two

    @property
    def dm_shape(self):
        """ DM actuator geometry - i.e. how many actuators """
        return len(self.segmentlist)

    @property
    def surface(self):
        """ The surface shape of the deformable mirror, in
        **meters**. This is the commanded shape, input to the DM.
        See the .opd property for the output. """
        return self._surface

    def flatten(self):
        """Flatten the DM by setting all actuators to zero piston"""
        self._surface[:] = 0

    @utils.quantity_input(piston=u.meter, tip=u.radian, tilt=u.radian)
    def set_actuator(self, segnum, piston, tip, tilt):
        """ Set an individual actuator of the DM.
        Parameters
        -------------
        segnum : integer
            Index of the actuator you wish to control
        piston, tip, tilt : floats or astropy Quantities
            Piston (in meters or other length units) and tip and tilt
            (in radians or other angular units)
        """

        if segnum not in self.segmentlist:
            raise ValueError("Segment {} is not present for this DM instance.".format(segnum))
        self._surface[segnum] = [piston.to(u.meter).value,
                                 tip.to(u.radian).value,
                                 tilt.to(u.radian).value]

    def _setup_arrays(self, npix, pixelscale, wave=None):
        """ Set up the arrays to compute an OPD into.
        This is relatively slow, but we only need to do this once for
        each size of input array. A simple caching mechanism avoids
        unnecessary recomputations.

        """
        # Don't recompute if values unchanged.
        if (npix == self._last_npix) and (pixelscale == self._last_pixelscale):
            return
        else:
            self._last_npix = npix
            self._last_pixelscale = pixelscale

        self._seg_mask = np.zeros((npix, npix))
        self._seg_x = np.zeros((npix, npix))
        self._seg_y = np.zeros((npix, npix))
        self._seg_indices = dict()

        self.transmission = np.zeros((npix, npix))
        for i in self.segmentlist:
            self._one_aperture(wave, i, value=i + 1)
        self._seg_mask = self.transmission
        self._transmission = np.asarray(self._seg_mask != 0, dtype=float)

        y, x = self.get_coordinates((wave))

        for i in self.segmentlist:
            wseg = np.where(self._seg_mask == i+1)
            self._seg_indices[i] = wseg
            ceny, cenx = self._aper_center(i)
            self._seg_x[wseg] = x[wseg] - cenx
            self._seg_y[wseg] = y[wseg] - ceny

    def get_opd(self, wave):
        """ Return OPD  - Faster version with caching"""
        self._setup_arrays(wave.shape[0], wave.pixelscale, wave=wave)

        self.opd = np.zeros(wave.shape)
        for i in self.segmentlist:
            wseg = self._seg_indices[i]
            self.opd[wseg] = (self._surface[i, 0] +
                              self._surface[i, 1] * self._seg_x[wseg] +
                              self._surface[i, 2] * self._seg_y[wseg])

        # account for DM being reflective (optional, governed by include_factor_of_two parameter)
        if self.include_factor_of_two:
            self.opd *= 2

        return self.opd

    def get_transmission(self, wave):
        """ Return transmission - Faster version with caching"""
        self._setup_arrays(wave.shape[0], wave.pixelscale, wave=wave)
        return self._transmission


# note, must inherit first from SegmentedDeformableMirror to get correct method resolution order
class HexSegmentedDeformableMirror(SegmentedDeformableMirror, optics.MultiHexagonAperture, ):
    """ Hexagonally segmented DM. Each actuator is controllable in piston, tip, and tilt

            Parameters
            ----------
            rings, flattoflat, gap, center : various
                All keywords for defining the segmented aperture geometry are inherited from
                the MultiHexagonAperture class. See that class for details.

             include_factor_of_two : Bool
                include the factor of two due to reflection in the OPD function (optional, default False).
                If this is set False (default), actuator commands are interpreted as being in units of
                desired wavefront error directly; the returned WFE will be directly proportional to the requested
                values (convolved with the actuator response function etc).
                If this is set to True, then the actuator commands are interpreted as being in physical surface
                units, and the WFE is therefore a factor of two larger. The returned WFE will be twice the
                amplitude of the requested values (convolved with the actuator response function etc.)
    """

    def __init__(self, rings=3, flattoflat=1.0 * u.m, gap=0.01 * u.m,
                 name='HexDM', center=True, include_factor_of_two=False, **kwargs):
        optics.MultiHexagonAperture.__init__(self, name=name, rings=rings, flattoflat=flattoflat,
                                             gap=gap, center=center, **kwargs)
        SegmentedDeformableMirror.__init__(self, rings=rings, include_factor_of_two=include_factor_of_two)



# note, must inherit first from SegmentedDeformableMirror to get correct method resolution order
class CircularSegmentedDeformableMirror(SegmentedDeformableMirror, optics.MultiCircularAperture):
    """ Circularly segmented DM. Each actuator is controllable in piston, tip, and tilt (and any zernike term)

            Parameters
            ----------
            rings, segment_radius, gap, center : various
                All keywords for defining the segmented aperture geometry are inherited from
                the MultiCircularperture class. See that class for details.

             include_factor_of_two : Bool
                include the factor of two due to reflection in the OPD function (optional, default False).
                If this is set False (default), actuator commands are interpreted as being in units of
                desired wavefront error directly; the returned WFE will be directly proportional to the requested
                values (convolved with the actuator response function etc).
                If this is set to True, then the actuator commands are interpreted as being in physical surface
                units, and the WFE is therefore a factor of two larger. The returned WFE will be twice the
                amplitude of the requested values (convolved with the actuator response function etc.)
    """
    
    def __init__(self, rings=1, segment_radius=1.0 * u.m, gap=0.01 * u.m,
                 name='CircSegDM', center=True, include_factor_of_two=False, **kwargs):
        #FIXME ? using grey pixel does not work. something in the geometry module generate a true divide error
        optics.MultiCircularAperture.__init__(self, name=name, rings=rings, segment_radius=segment_radius,
                                              gap=gap, center=center, gray_pixel = False, **kwargs)
        SegmentedDeformableMirror.__init__(self, rings=rings, include_factor_of_two=include_factor_of_two)
