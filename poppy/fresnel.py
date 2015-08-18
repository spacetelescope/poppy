from __future__ import division

import poppy
import numpy as np
import matplotlib.pyplot as plt

import astropy.io.fits as fits
import astropy.units as u

from . import utils

import logging
_log = logging.getLogger('poppy')

try:
    from IPython.core.debugger import Tracer; stop = Tracer()
except:
    pass

from poppy.poppy_core import PlaneType, _PUPIL, _IMAGE, _DETECTOR, _ROTATION, _INTERMED, _FFTW_AVAILABLE



class QuadPhase(poppy.AnalyticOpticalElement):
    '''
    Quadratic phase factor,  q(z)
    suitable for representing a radially-dependent wavefront curvature.

    Parameters
    -----------------
    z : float or astropy.Quantity of type length
        radius of curvature
    planetype : poppy.PlaneType constant
        plane type
    name : string
        Descriptive string name
    reference_wavelength : float
        wavelength
    units : astropy.unit of type length
        Unit to apply to reference wavelength; default is meter


    References
    -------------------
    Lawrence eq. 88

    '''
    def __init__(self,
                 z,     #FIXME consider renaming fl? z seems ambiguous with distance.
                 planetype = _INTERMED,
                 name = 'Quadratic Wavefront Curvature Operator',
                 reference_wavelength = 2e-6,
                 units=u.m,
                 **kwargs):
        poppy.AnalyticOpticalElement.__init__(self,name=name, planetype=planetype, **kwargs)
        self.z=z
        self.reference_wavelength = reference_wavelength*units

        if  isinstance(z,u.quantity.Quantity):
            self.z_m = (z).to(u.m) #convert to meters.
        else:
            _log.debug("Assuming meters, phase (%.3g) has no units for Optic: "%(z)+self.name)
            self.z_m=z*u.m

    def getPhasor(self, wave):
        """ return complex phasor for the quadratic phase

        Parameters
        ----------
        wave : obj
            a Fresnel Wavefront object
        """

        if not isinstance(wave,Wavefront):
            raise TypeError("Must supply a Fresnel Wavefront")
        #y, x = wave.coordinates()
        y, x = wave._fft_coordinates()
        rsqd = (x**2+y**2)*u.m**2
        _log.debug("Applying spherical phase curvature ={0:0.2e}".format(self.z_m))
        _log.debug("Applying spherical lens phase ={0:0.2e}".format(1.0/self.z_m))
        _log.debug("max_rsqd ={0:0.2e}".format(np.max(rsqd)))


        k = 2* np.pi/self.reference_wavelength
        lens_phasor = np.exp(1.j * k * rsqd/(2.0*self.z_m))
        return lens_phasor


class GaussianLens(QuadPhase):
    '''
    Gaussian Lens

    Thin wrapper for QuadPhase

    Parameters
    -----------------
    f_lens : float or astropy.Quantity of type length
        Focal length of this lens
    name : string
        Descriptive string name
    planetype : poppy.PlaneType constant
        plane type
    reference_wavelength : float
        wavelength
    units : astropy.unit of type length
        Unit to apply to reference_wavelength; defaults to meters


    '''
    def __init__(self,
                 f_lens,
                 planetype = _INTERMED,
                 name = 'Gaussian Lens',
                 reference_wavelength = 2e-6,
                 units=u.m,
                 **kwargs):
        QuadPhase.__init__(self,
                 f_lens,
                 planetype =planetype,
                 name = name,
                 reference_wavelength = reference_wavelength,
                 units=units,
                 **kwargs)
        if  isinstance(f_lens,u.quantity.Quantity):
            self.fl = (f_lens).to(u.m) #convert to meters.
        else:
            _log.warn("Assuming meters, focal length (%.3g) has no units for Optic: "%(f_lens)+self.name)
            self.fl=f_lens*u.m
        _log.debug("Initialized: "+self.name+", fl ={0:0.2e}".format(self.fl))

    def __str__(self):
        return "Lens: {0}, with focal length {1}".format(self.name, self.fl)


class Wavefront(poppy.Wavefront):
    @u.quantity_input(beam_radius=u.m)
    def __init__(self,
                 beam_radius,
                 units=u.m,
                 force_fresnel=True,
                 rayl_factor=2.0,
                 oversample=2,
                 **kwds):
        '''
        Wavefront for Fresnel diffraction calculation.

        This class inherits from and extends the Fraunhofer-domain
        poppy.Wavefront class.


        Parameters:
        --------------------
        beam_radius : astropy.Quantity of type length
            Radius of the illuminated beam at the initial optical plane.
            I.e. this would be the pupil aperture radius in an entrance pupil.
        units :
            astropy units of input parameters
        rayl_factor:
            Threshold for considering a wave spherical.
        force_fresnel : bool
            If True then the Fresnel propagation will always be used,
            even between planes of type _PUPIL or _IMAGE
            if False the wavefront reverts to standard wavefront propagation for _PUPIL <-> _IMAGE planes
        oversample : float
            Padding factor to apply to the wavefront array, multiplying on top of the beam radius.


        References:
        -------------------
        - Lawrence, G. N. (1992), Optical Modeling, in Applied Optics and Optical Engineering., vol. XI,
            edited by R. R. Shannon and J. C. Wyant., Academic Press, New York.

        - https://en.wikipedia.org/wiki/Gaussian_beam

        - IDEX Optics and Photonics(n.d.), Gaussian Beam Optics,
            [online] Available from:
             https://marketplace.idexop.com/store/SupportDocuments/All_About_Gaussian_Beam_OpticsWEB.pdf

        - Krist, J. E. (2007), PROPER: an optical propagation library for IDL,
           vol. 6675, p. 66750P-66750P-9.
        [online] Available from: http://dx.doi.org/10.1117/12.731179

        - Andersen, T., and A. Enmark (2011), Integrated Modeling of Telescopes, Springer Science & Business Media.

        '''

        #        initialize general wavefront class first,
        #        in Python 3 this will change,
        #        https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods

        try:
            units.to(u.m)
        except (AttributeError,u.UnitsError):
            raise ValueError("The 'units' parameter must be an astropy.units.Unit representing length.")
        self.units = units
        """Astropy.units.Unit for measuring distance"""

        self.w_0 = (beam_radius).to( self.units) #convert to base units.
        """Beam waist radius at initial plane"""
        self.oversample=oversample
        """Oversampling factor for zero-padding arrays"""
        #print(oversample)
        super(Wavefront,self).__init__(diam=beam_radius.to(u.m).value*2.0, oversample=self.oversample,**kwds)

        self.z  =  0*units
        """Current wavefront coordinate along the optical axis"""
        self.z_w0 = 0*units
        """Coordinate along the optical axis of the latest beam waist"""
        self.waists_w0 = [self.w_0.value]
        """ List of beam waist radii, in series as encountered during the course of an optical propagation."""
        self.waists_z = [self.z_w0.value]
        """ List of beam waist distances along the optical axis, in series as encountered during the course of an optical propagation."""
        self.wavelen_m = self.wavelength*u.m #wavelengths should always be in meters
        """ Wavelength as an Astropy.Quantity"""
        self.spherical = False
        """is this wavefront spherical or planar?"""
        self.k = np.pi*2.0/self.wavelength
        """ Wavenumber"""
        self.force_fresnel = force_fresnel
        """Force Fresnel calculation if true, even for cases wher Fraunhofer would work"""
        self.rayl_factor= rayl_factor

        self._angular_coordinates = False
        """Should coordinates be expressed in arcseconds instead of meters at the current plane? """

        if self.oversample > 1 and not self.ispadded: #add padding for oversampling, if necessary
            self.wavefront = utils.padToOversample(self.wavefront, self.oversample)
            self.ispadded = True
            logmsg = "Padded WF array for oversampling by {0:d}, to {1}.".format(self.oversample, self.wavefront.shape)
            _log.debug(logmsg)

            self.history.append(logmsg)
        else:
            _log.debug("Skipping oversampling, oversample < 1 or already padded ")

        if self.oversample < 2:
            _log.warn("Oversampling > 2x suggested for reliable results.")

        # FIXME MP: this self.n attribute appears unnecessary?
        if self.shape[0]==self.shape[1]:
            self.n=self.shape[0]
        else:
            self.n=self.shape

        if self.planetype == _IMAGE:
            raise ValueError("Input wavefront needs to be a pupil plane in units of m/pix. Specify a diameter not a pixelscale.")

    @property
    def z_R(self):
        '''
        Rayleigh distance for the gaussian beam, based on
        current beam waist and wavelength.

        I.e. the distance along the propagation direction from the
        beam waist at which the area of the cross section has doubled.
        The depth of focus is conventionally twice this distance.
        '''

        return np.pi*self.w_0**2/(self.wavelen_m)

    @property
    def divergence(self):
        '''
        Divergence of the gaussian beam

        I.e. the angle between the optical axis and the beam radius at a large distance.
        Angle in radians.
        '''
        return 2*self.wavelen_m/(np.pi*self.w_0)

    @property
    def param_str(self):
        '''
        Formatted string of gaussian beam parameters.
        '''
        string= "w_0:{0:0.3e},".format(self.w_0)+" z_w0={0:0.3e}".format(self.z_w0) +"\n"+\
         "z={0:0.3e},".format(self.z)+" z_R={0:0.3e}".format(self.z_R)
        return string

    @property
    def waists(self):
        '''
        each [z_w_0,w_0] for each waist generated by an optic
        '''
        return np.array([self.waists_z,self.waists_w0])
        # or make this a recarray? (but changes the order compared to the above , not sure if that matters)
        #return np.array(zip(gw.waists_z, gw.waists_w0), dtype=[('z',float), ('radius', float)])

    def _fft(self):
        '''
        Apply normalized forward 2D Fast Fourier Transform to wavefront
        '''
        _USE_FFTW = (poppy.conf.use_fftw and _FFTW_AVAILABLE)

        if _USE_FFTW:
            #FFTW wisdom could be implemented here.
            # MP: not sure that anything needs manual implementation?
            #     wisdom should be already loaded during poppy.__init__
            _log.debug("   Using pyfftw")
            self.wavefront=pyfftw.interfaces.numpy_fft.fft2(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE',
                                     threads=poppy.conf.n_processes)/self.shape[0]
        else:
            _log.debug("   Using numpy FFT")
            self.wavefront=np.fft.fft2(self.wavefront)/self.shape[0]

    def _inv_fft(self):
        '''
        Apply normalized Inverse 2D Fast Fourier Transform to wavefront
        '''
        _USE_FFTW = (poppy.conf.use_fftw and _FFTW_AVAILABLE)

        if _USE_FFTW:
            #FFTW wisdom could be implemented here.
            # MP: see above comment
            _log.debug("   Using pyfftw")
            self.wavefront=pyfftw.interfaces.numpy_fft.ifft2(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE',
                                     threads=poppy.conf.n_processes)*self.shape[0]
        else:
            _log.debug("   Using numpy FFT")
            self.wavefront=np.fft.ifft2(self.wavefront)*self.shape[0]

    def R_c(self,z=None):
        '''
        The gaussian beam radius of curvature as a function of distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius of curvature at the current position.

        Returns: Astropy.units.Quantity of dimension length

        '''
        if z is None: z = self.z
        dz=(z-self.z_w0) #z relative to waist
        if dz==0:
            return np.inf * u.m
        return dz*(1+(self.z_R/dz)**2)

    def spot_radius(self,z=None):
        '''
        radius of a propagating gaussian wavefront, at a distance z

        Parameters
        -------------
        z : float, optional
            Distance along the optical axis.
            If not specified, the wavefront's current z coordinate will
            be used, returning the beam radius at the current position.

        Returns: Astropy.units.Quantity of dimension length
        '''
        if z is None: z = self.z
        return self.w_0 * np.sqrt(1.0 + ((z-self.z_w0)/self.z_R)**2 )

    def _fft_coordinates(self):
        """
        Return Y, X coordinates for this wavefront, in the manner of numpy.indices(),
        with centering consistent with the FFT convention.
        This function returns the FFT center of the array always.

        This function is intentionally distinct from the regular Wavefront.coordinates(), and behaves
        slightly differently. This is required for use in the angular spectrum propagation in the PTP and
        Direct propagations. Not intended as a general purpose replacement for coordinates() in other
        cases. See https://github.com/mperrin/poppy/issues/104 for discussion.

        Returns
        -------
        Y, X :  array_like
            Wavefront coordinates in either meters or arcseconds for pupil and image, respectively

        """
        y, x = np.indices(self.shape, dtype=float)
        y-= (self.shape[0])/2.
        x-= (self.shape[1])/2.

        xscale=self.pixelscale
        yscale=self.pixelscale

        #x *= xscale
        #y *= yscale
        return y*yscale, x*xscale

    # Override parent class method to provide one that's comparatible with
    # FFT indexing conventions. Centered one one pixel not on the middle
    # of the array.
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
        if not np.isscalar(pixelscale):
            pixel_scale_x, pixel_scale_y = pixelscale
        else:
            pixel_scale_x, pixel_scale_y = pixelscale, pixelscale

        y -= (shape[0] ) / 2.0
        x -= (shape[1] ) / 2.0

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

        raise NotImplementedError('need to calculate pixel scale from focal length')
        if not np.isscalar(pixelscale):
            pixel_scale_x, pixel_scale_y = pixelscale
        else:
            pixel_scale_x, pixel_scale_y = pixelscale, pixelscale

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

        The behavior for Fresnel wavefronts is slightly different from
        Fraunhofer wavefronts, in that the optical axis is *not* the exact
        center of an array (the corner between pixels for an even number of pixels),
        but rather is a specific pixel (e.g. pixel 512,512 for a 1024x1024 array).
        This is for consistency with the array indexing convention used in FFTs since
        this class depends on FFTs rather than the more flexible matrix DFTs for its
        propagation.

        For Fresnel wavefronts, this depends on the focal length to get the image scale right.

        Returns
        -------
        Y, X :  array_like
            Wavefront coordinates in either meters or arcseconds for pupil and image, respectively
        """

        if self.planetype == PlaneType.pupil or self.planetype == PlaneType.intermediate:
            return type(self).pupil_coordinates(self.shape, self.pixelscale)
        elif self.planetype == PlaneType.image:
            return Wavefront.image_coordinates(self.shape, self.pixelscale,
                                               self._last_transform_type, self._image_centered)
        else:
            raise RuntimeError("Unknown plane type {0} (should be pupil or image!)".format(self.planetype))



    def propagate_direct(self,z):
        '''
        Implements the direct propagation algorithm described in Andersen & Enmark (2011). Works best for far field propagation.
        Not part of the Gaussian beam propagation method.

        Parameters
        ----------
        z :  float
            the distance from the current location to propagate the beam.
        '''

        if  isinstance(z,u.quantity.Quantity):
            z_direct = (z).to(u.m).value #convert to meters.
        else:
            _log.warn("z= {0:0.2e}, has no units, assuming meters ".format(z))
            z_direct=z
        y,x = self._fft_coordinates()#*self.units
        k = np.pi*2.0/self.wavelen_m.value
        S = self.n*self.pixelscale
        _log.debug("Propagation Parameters: k={0:0.2e},".format(k)+"S={0:0.2e},".format(S)+"z={0:0.2e},".format(z_direct))

        QuadPhase_1st = np.exp(1.0j*k*(x**2+y**2)/(2*z_direct))#eq. 6.68
        QuadPhase_2nd = np.exp(1.0j*k*z_direct)/(1.0j*self.wavelength*z_direct)*np.exp(1.0j*(x**2+y**2)/(2*z_direct))#eq. 6.70

        stage1 = self.wavefront*QuadPhase_1st #eq.6.67

        result = np.fft.fftshift(forward_FFT(stage1))*self.pixelscale**2*QuadPhase_2nd  #eq.6.69 and #6.80

        result = np.fft.fftshift(result)

        self.wavefront = result
        self.history.append("Direct propagation to z= {0:0.2e}".format(z))

    def propagateTo(self, optic, distance):
        """Propagates a wavefront object to the next optic in the list, after
        some separation distance (which might be zero).
        Modifies this wavefront object itself.

        Transformations between most planes use Fresnel propagation.
        If the target plane is an image plane, the output wavefront will be set to provide its
        coordinates in arcseconds based on its focal length, but it retains its internal dimensions
        in meters for future Fresnel propagations.
        Transformations to a Detector plane are handled separately to allow adjusting the pixel scale
        to match the target scale.
        Transformations from any frame through a rotation plane simply rotate the wavefront accordingly.

        Parameters
        -----------
        optic : OpticalElement
            The optic to propagate to. Used for determining the appropriate optical plane.
        distance : astropy.Quantity of dimension length
            separation distance of this optic relative to the prior optic in the system.
        """
        msg = "  Propagating wavefront to {0} after distance {1} ".format(str(optic), distance)
        _log.debug(msg)
        self.history.append(msg)

        # Apply Fresnel propagation for the specified distance, regardless of
        # what type of plane is next
        if distance != 0*u.m:
            self.propagate_fresnel(distance)

        # Now we may do some further manipulations depending on the next plane
        self._angular_coordinates=False # by default coordinates in meters
        if optic.planetype == _ROTATION:     # rotate
            self.rotate(optic.angle)
            self.location='after '+optic.name
        elif optic.planetype == _IMAGE:
            self.location='before '+optic.name
            self._angular_coordinates=True # image planes want angular coordinates
        elif optic.planetype == _DETECTOR:
            raise NotImplemented('image plane to detector propagation (resampling!) not implemented yet')
        else:
            self.location='before '+optic.name

    def _propagate_ptp(self,dz):
        ''' Plane-to-Plane Fresnel propagation.

        This function propagates a planar wavefront some distance
        while keeping it planar, yielding a planar output wavefront.
        This is used for propagation entirely within the Rayleigh
        distance of the beam waist.


        Parameters
        ----------
        dz :  float
            the distance from the current location to propagate the beam.

        References
        ----------
        Lawrence eq. 82, 86,87
        '''

        # FIXME MP: should check here to confirm the starting wavefront
        # is indeed planar rather than spherical
        if self.spherical:
            raise RuntimeError('_propagate_ptp can only start from a planar wavefront, but was called with a spherical one.')


        if  isinstance(dz,u.quantity.Quantity):
            z_direct = (dz).to(u.m).value #convert to meters.
        else:
            _log.warn("z= {0:0.2e}, has no units, assuming meters ".format(dz))
            z_direct = dz

        if np.abs(dz) < 1*u.Angstrom:
            _log.debug("Skipping small dz = " + str(dz))
            return

        x,y = self._fft_coordinates() #meters
        rho = np.fft.fftshift((x/self.pixelscale/2.0/self.oversample)**2 + (y/self.pixelscale/2.0/self.oversample)**2)
        T = -1.0j*np.pi*self.wavelength*(z_direct)*rho #Transfer Function of diffraction propagation eq. 22, eq. 87

        self._fft()

        self.wavefront = self.wavefront*np.exp(T)#eq. 6.68

        self._inv_fft()
        self.z = self.z + dz

        self.history.append("Propagated Plane-to-Plane, dz = " + str(z_direct))

    def _propagate_wts(self,dz):
        ''' Waist-to-Spherical Fresnel propagation

        This function propagates a planar input wavefront to become a spherical wavefront.
        The starting position should be within the Rayleigh distance of the waist, and the
        ending position will be outside of that.

        Parameters:
        -----------
        dz :  float
            the distance from the current location to propagate the beam.

        References
        ----------
         Lawrence eq. 83,88
        '''
        #dz = z2-self.z
        _log.debug("Waist to Spherical propagation, dz=" + str(dz))

        # FIXME MP: check for planar input wavefront
        if self.spherical:
            raise RuntimeError('_propagate_ptp can only start from a planar wavefront, but was called with a spherical one.')

        if dz ==0:
            _log.error("Waist to Spherical propagation stopped, no change in distance.")
            return

        self *= QuadPhase(dz, reference_wavelength=self.wavelength)

        if dz > 0:
            self._fft()
        else:
            self._inv_fft()


        self.pixelscale = self.wavelength*np.abs(dz.value)/(self.n*self.pixelscale)
        self.z = self.z + dz
        self.history.append("Propagated Waist to Spherical, dz = " + str(dz))
        # FIXME MP: update self.spherical to be true here?
        self.spherical=True    # wavefront is now spherical

    def _propagate_stw(self,dz):
        '''Spherical-to-Waist Fresnel propagation

        This function propagates a spherical wavefront to become a planar wavefront.
        The starting position should be outside the Rayleigh distance of the waist,
        and the ending position will be inside of it.


        Parameters
        ----------
        dz :  float
            the distance from the current location to propagate the beam, in meters

        References
        ----------
         Lawrence eq. 89
        '''

        if not self.spherical:
            raise RuntimeError('_propagate_ptp can only start from a spherical wavefront, but was called with a planar one.')

        #dz = z2 - self.z
        _log.debug("Spherical to Waist propagation, dz="+str(dz))

        if dz ==0:
            _log.error("Spherical to Waist propagation stopped, no change in distance.")
            return

        if dz > 0:
            self._fft()
        else:
            self._inv_fft()

        #update to new pixel scale before applying curvature
        self.pixelscale = self.wavelength*np.abs(dz.value)/(self.n*self.pixelscale)
        self *= QuadPhase(dz, reference_wavelength=self.wavelength)
        self.z = self.z + dz
        self.history.append("Propagated Spherical to Waist, dz = " + str(dz))
        self.spherical=False    # wavefront is now planar

    def planar_range(self,z):
        '''
        Returns True if the input range z is within the Rayleigh range of the waist.

        Parameters:
        ----------
        z : float
            distance from the beam waist

        '''

        #if np.abs(self.z_w0 - z) < self.z_R:
        #    return True
        #else:
        #    return False
        return np.abs(self.z_w0 - z) < self.z_R

    def propagate_fresnel(self,delta_z,display_intermed=False):
        '''Top-level routine for Fresnel diffraction propagation


        Each spherical wavefront is propagated to a waist and then to the next appropriate plane
         (spherical or planar).

        Parameters
        ----------
        delta_z :  float
            the distance from the current location to propagate the beam.
        display_interm : boolean
             If True, display the complex start, intermediates waist and end surfaces.


        '''
        #self.pad_wavefront()
        z = self.z + delta_z
        if display_intermed:
            plt.figure()
            self.display('both',colorbar=True,title="Starting Surface")

        self.wavefront=np.fft.fftshift(self.wavefront)
        _log.debug("Beginning Fresnel Prop. Waist at z = "+str(self.z_w0))

        if not self.spherical:
            if self.planar_range(z):
                # Plane waves inside planar range:  use plane-to-plane
                _log.debug('  Plane to Plane Regime, dz='+str(delta_z))
                _log.debug('  Constant Pixelscale: %.2g m/pix'%self.pixelscale)
                self._propagate_ptp(delta_z)
            else:
                # Plane wave to spherical. First use PTP to the waist, then WTS to Spherical
                _log.debug('  Plane to Spherical, inside Z_R to outside Z_R')
                _log.debug('  Starting Pixelscale: %.2g m/pix'%self.pixelscale)
                self._propagate_ptp(self.z_w0 - self.z)
                if display_intermed:
                    plt.figure()
                    self.display('both',colorbar=True)
                self._propagate_wts(z-self.z_w0)
        else:
            if self.planar_range(z):
                # Spherical to plane. First use STW to the waist, then PTP to the desired plane
                _log.debug('  Spherical to Plane Regime, outside Z_R to inside Z_R')
                self._propagate_stw(self.z_w0 - self.z)
                if display_intermed:
                    plt.figure()
                    self.display('both',colorbar=True,title='Intermediate Waist')
                self._propagate_ptp(z-self.z_w0)
            else:
                #Spherical to Spherical. First STW to the waist, then WTS to the desired spherical surface
                _log.debug('  Spherical to Spherical, Outside Z_R to waist (z_w0) to outside Z_R')
                _log.debug('  Starting Pixelscale: %.2g m/pix'%self.pixelscale)
                self._propagate_stw(self.z_w0 - self.z)
                _log.debug('  Intermediate Pixelscale: %.2g m/pix'%self.pixelscale)
                self.pixelscale
                if display_intermed:
                    plt.figure()
                    self.display('both',colorbar=True,title='Intermediate Waist')
                self._propagate_wts(z-self.z_w0)
        if display_intermed:
            plt.figure()
            self.display('both',colorbar=True)

        self.wavefront = np.fft.fftshift(self.wavefront)
        self.planetype = _INTERMED
        _log.debug("------ Propagated to plane of type "+str(self.planetype)+" at z = {0:0.2e} ------".format(z))



    def __imul__(self, optic):
        "Multiply a Wavefront by an OpticalElement or scalar"
        if isinstance(optic, GaussianLens):
            # Special case: if we have a lens, call the routine for that,
            # which will modify the properties of this wavefront more fundamentally
            # than most other optics
            self.apply_optic(optic, self.z)
            return self
        else:
            # Otherwise fall back to the parent class
            return super(Wavefront,self).__imul__(optic)



    def apply_optic(self,optic,z_lens,ignore_wavefront=False):
        '''

        Adds thin lens wavefront curvature to the wavefront
        of focal length f_l and updates the
        Gaussian beam parameters of the wavefront.

        Eventually this should be called by a modified multiply function.

        Parameters
        ----------
        optic : GaussianLens
            An optic
        z_lens : float
            location of lens relative to the wavefront
        ignore_wavefront : boolean
            If True then only gaussian beam propagation parameters will be updated and the wavefront surface will not be calculated.
            Useful for quick calculations of gaussian laser beams

        '''

        #test optic and wavefront have equal oversampling
        # MP: why? the optic should adapt to whatever the input wavefront has
        #assert self.oversample == optic.oversample

        #self.pad_wavefront()
        _log.debug("------ Applying Optic: "+str(optic.name)+" ------")
        _log.debug("   wavefront oversample: {0}  optic oversample: {1}".format(self.oversample, optic.oversample))
        _log.debug("  Pre-Lens Beam Parameters: "+self.param_str)

        zl = (z_lens).to(u.m) #convert to meters.
        _log.debug(" Lens z_lens: {0}   wavefront z: {1}".format(zl, self.z))

        # MP: calculate beam radius at current surface
        new_waist = self.spot_radius(zl)
        _log.debug("  Beam radius at "+ str(optic.name)+" ={0:0.2e}".format(new_waist))

        # Is the incident beam planar or spherical?
        #is the last surface outside the rayleigh distance?
        if np.abs(self.z_w0 - self.z) > self.rayl_factor*self.z_R:
            _log.debug("spherical beam")
            _log.debug(self.param_str)
            self.spherical = True #FIXME seems like this is in the wrong place
            R_input_beam = self.z - self.z_w0
        else:
            R_input_beam = np.inf

        if self.planetype == _PUPIL or self.planetype == _IMAGE:
            #we are at a focus or pupil, so the new optic is the only curvature of the beam
            r_curve = -optic.fl
            _log.debug(" input flat wavefront and "+ str(optic.name) +" has a curvature of ={0:0.2e}".format(r_curve))

        else:
            r_curve = 1.0/(1.0/self.R_c(zl) - 1.0/optic.fl)
            _log.debug(" input curved wavefront "+str(optic.name) +" has a curvature of ={0:0.2e}".format(r_curve))

        #update the wavefront to the post-lens beam waist
        if self.R_c(zl) == optic.fl:
            _log.debug(str(optic.name) +" has a flat output wavefront")
            self.z_w0 = zl
            self.w_0 = new_waist
        else:
            self.z_w0 = -r_curve/(1.0 + (self.wavelen_m*r_curve/(np.pi*new_waist**2))**2) + zl
            self.w_0 = new_waist/np.sqrt(1.0+(np.pi*new_waist**2/(self.wavelen_m*r_curve))**2)
            _log.debug(str(optic.name) +" has a curvature of ={0:0.2e}".format(r_curve))

        _log.debug("Post Optic Parameters:"+self.param_str)

        #check that this Fresnel business is necessary.
        if (not self.force_fresnel) and (self.planetype == _PUPIL or self.planetype ==_IMAGE) \
            and (optic.planetype ==_IMAGE or optic.planetype ==_PUPIL):
            _log.debug("Simple pupil / image propagation, Fresnel unnecessary. \
                       Reverting to Fraunhofer.")
            self.propagateTo(optic)
            return

        if ignore_wavefront:
            return

        if (not self.spherical) and (np.abs(self.z_w0 - zl) < self.z_R):
            _log.debug('Near-field, Plane-to-Plane Propagation.')
            z_eff = optic.fl

        elif (not self.spherical) and (np.abs(self.z_w0 - zl) > self.z_R):
            # find the radius of curvature of the lens output beam
            # curvatures are multiplicative exponentials
            # e^(1/z) = e^(1/x)*e^(1/y) = e^(1/x+1/y) -> 1/z = 1/x + 1/y
            # z = 1/(1/x+1/y) = xy/x+y
            z_eff = 1.0/( 1.0/optic.fl+ 1.0/(self.z-self.z_w0))
            _log.debug('Inside Rayleigh distance to Outside Rayleigh distance.')

            self.spherical = True


            #optic needs new focal length:
        elif (self.spherical) and (np.abs(self.z_w0 - zl) > self.z_R):
            _log.debug('Spherical to Spherical wavefront propagation.')
            _log.debug("1/fl={0:0.4e}".format(1.0/optic.fl))
            _log.debug("1.0/(R_input_beam)={0:0.4e}".format(1.0/R_input_beam))
            _log.debug("1.0/(self.z-self.z_w0)={0:0.4e}".format(1.0/(self.z-self.z_w0)))

            if R_input_beam == 0:
                z_eff = 1.0/( 1.0/optic.fl- 1.0/(R_input_beam))
            if (zl-self.z_w0) ==0:
                z_eff = 1.0/( 1.0/optic.fl+ 1.0/(self.z-self.z_w0))
            else:
                z_eff = 1.0/( 1.0/optic.fl+ 1.0/(self.z-self.z_w0)- 1.0/(R_input_beam))


        elif (self.spherical) and (np.abs(self.z_w0 - zl) < self.z_R):
            _log.debug('Spherical to Planar.')
            z_eff=1.0/( 1.0/optic.fl - 1.0/(R_input_beam) )
            self.spherical=False

        effective_optic = QuadPhase(-z_eff, reference_wavelength=self.wavelength)
        self *= effective_optic

        self.waists_z.append(self.z_w0.value)
        self.waists_w0.append(self.w_0.value)

        #update wavefront location:
        #self.z = zl
        self.planetype = optic.planetype
        _log.debug("------ Optic: "+str(optic.name)+" applied ------")


FresnelWavefront=Wavefront # alias for now, potentially rename soon?


class FresnelOpticalSystem(poppy.OpticalSystem):
    """ Class representing a series of optical elements,
    through which light can be propagated using the Fresnel formalism.

    This is comparable to the "regular" (Fraunhofer-domain)
    OpticalSystem, but adds functionality for propagation to
    arbitrary optical planes rather than just pupil and image planes.

    Parameters
    -------------
    name : string
        descriptive name of optical system
    pupil_diameter : astropy.Quantity of dimension length
        Diameter of entrance pupil
    npix : int
        Number of pixels across the entrance pupil by default 1024
    beam_ratio : int
        Padding factor for the entrance pupil; what fraction of the array should
        correspond to the entrance pupil. Default is 0.5, which corresponds to
        Nyquist sampling (2 pixels per resolution element)
    verbose : bool
        whether to be more verbose with log output while computing
    """

    @u.quantity_input(pupil_diameter=u.m)
    def __init__(self, name="unnamed system", pupil_diameter=1*u.m,
            npix=512, beam_ratio=0.5, verbose=True):
        super(FresnelOpticalSystem, self).__init__(name=name, verbose=verbose)
        self.pupil_diameter = pupil_diameter
        self.beam_ratio = beam_ratio
        self.npix=npix

        self.distances = [] # distance along the optical axis to each successive optic

    def addPupil(self, *args, **kwargs):
        raise NotImplementedError('Use add_optic for Fresnel instead')

    def addImage(self, *args, **kwargs):
        raise NotImplementedError('Use add_optic for Fresnel instead')

    @u.quantity_input(distance=u.m)
    def add_optic(self, optic=None, distance=0.0*u.m):
        """ Add an optic to the optical system

        Parameters
        ---------------
        optic : OpticalElement instance
            Some optic
        distance : astropy.Quantity of dimension length
            separation distance of this optic relative to the prior optic in the system.
        """
        self.planes.append(optic)
        self.distances.append(distance.to(u.m))
        if self.verbose: _log.info("Added optic: {0} after separation: {1:.2e} ".format(self.planes[-1].name, distance))

        return optic

    @u.quantity_input(distance=u.m)
    def add_detector(self, pixelscale, distance=0.0*u.m, **kwargs):
        super(self,FresnelOpticalSystem).addDetector(pixelscale, **kwargs)
        self.distances.append(distance)
        if self.verbose: _log.info("Added detector: {0} after separation: {1:.2e} ".format(self.planes[-1].name, distance))

    addDetector=add_detector # for compatibility with pre-pep8 names

    def inputWavefront(self, wavelength=1e-6):
        """Create a Wavefront object suitable for sending through a given optical system.

        Uses self.source_offset to assign an off-axis tilt, if requested.
        (FIXME does not work for Fresnel yet)

        Parameters
        ----------
        wavelength : float
            Wavelength in meters

        Returns
        -------
        wavefront : poppy.Wavefront instance
            A wavefront appropriate for passing through this optical system.

        """
        inwave= FresnelWavefront(self.pupil_diameter/2, wavelength=wavelength,
            npix=self.npix, oversample=1./self.beam_ratio)
        _log.debug("Creating input wavefront with wavelength={0:e} microns, npix={1}, pixel scale={2:f} meters/pixel".format(
            wavelength*1e6, self.npix, self.pupil_diameter/self.npix))
        return inwave



    def propagate_mono(self, wavelength=2e-6, normalize='first',
                       retain_intermediates=False, display_intermediates=False):
        """Propagate a monochromatic wavefront through the optical system, via Fresnel calculations.
        Called from within `calcPSF`.
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

        if poppy.conf.enable_speed_tests:
            t_start = time.time()
        if self.verbose:
           _log.info(" Propagating wavelength = {0:g} meters".format(wavelength))
        wavefront = self.inputWavefront(wavelength)

        intermediate_wfs = []

        # note: 0 is 'before first optical plane; 1 = 'after first plane and before second plane' and so on
        current_plane_index = 0
        for optic,distance in zip(self.planes, self.distances):
            # The actual propagation:
            wavefront.propagateTo(optic, distance)
            wavefront *= optic
            current_plane_index += 1

            # Normalize if appropriate:
            if normalize.lower()=='first' and current_plane_index==1 :  # set entrance plane to 1.
                wavefront.normalize()
                _log.debug("normalizing at first plane (entrance pupil) to 1.0 total intensity")
            elif normalize.lower()=='first=2' and current_plane_index==1 : # this undocumented option is present only for testing/validation purposes
                wavefront.normalize()
                wavefront *= np.sqrt(2)
            elif normalize.lower()=='exit_pupil': # normalize the last pupil in the system to 1
                last_pupil_plane_index = np.where(np.asarray([p.planetype is PlaneType.pupil for p in self.planes]))[0].max() +1
                if current_plane_index == last_pupil_plane_index:
                    wavefront.normalize()
                    _log.debug("normalizing at exit pupil (plane {0}) to 1.0 total intensity".format(current_plane_index))
            elif normalize.lower()=='last' and current_plane_index==len(self.planes):
                wavefront.normalize()
                _log.debug("normalizing at last plane to 1.0 total intensity")


            # Optional outputs:
            if poppy.conf.enable_flux_tests: _log.debug("  Flux === "+str(wavefront.totalIntensity))

            if retain_intermediates: # save intermediate wavefront, summed for polychromatic if needed
                intermediate_wfs.append(wavefront.copy())

            if display_intermediates:
                if poppy.conf.enable_speed_tests: t0 = time.time()
                title = None if current_plane_index > 1 else "propagating $\lambda=$ %.3f $\mu$m" % (wavelength*1e6)
                wavefront.display(what='best',nrows=len(self.planes),row=current_plane_index, colorbar=False, title=title)
                #plt.title("propagating $\lambda=$ %.3f $\mu$m" % (wavelength*1e6))

                if poppy.conf.enable_speed_tests:
                    t1 = time.time()
                    _log.debug("\tTIME %f s\t for displaying the wavefront." % (t1-t0))

        if poppy.conf.enable_speed_tests:
            t_stop = time.time()
            _log.debug("\tTIME %f s\tfor propagating one wavelength" % (t_stop-t_start))

        return wavefront.asFITS(), intermediate_wfs


    def describe(self):
        """ Print out a string table describing all planes in an optical system"""
        res = (str(self)+
                "\n\tEntrance pupil diam: {0}\tnpix: {1}\tBeam ratio:{2}".format(self.pupil_diameter, self.npix, self.beam_ratio))

        for optic, distance in zip(self.planes, self.distances):
            if distance !=0: res += "\n\tPropagation distance: {0}".format(distance)
            res+= "\n\t"+str(optic)

        print(res)


