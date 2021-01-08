"""
Written by Ewan Douglas, 2018
Updated by Rachel Morgan, 2019-2020 (debugging, testing, reformatting, adding functions, updating to work with most recent POPPY)
"""

import numpy as np
import poppy
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.io.fits as fits
from .poppy_core import OpticalElement, Detector, Wavefront, PlaneType, _PUPIL, _IMAGE, _RADIANStoARCSEC
from .optics import CircularAperture
from .utils import measure_centroid

import logging
_log = logging.getLogger('poppy')


class Subapertures(poppy.OpticalElement):
    """
    Example roadmap:

    #generate wavefront
    wf = poppy.wavefront() #or fresnel wavefront

    #...various surfaces

    Subaperture(ARRAY OF OPTICS)
    #initialize this new class, where the array of optics define the subapertures (e.g. lenslets)

    Subapertures.sample_wf(wf) #this function takes the wavefront and subsamples it by the area of each optic
    Subapertures.get_wavefront_array() #returns an array of input sub-wavefronts multipled by subaperture optics
    Subapertures.get_psfs() #fraunhofer or fresnel propagation of each pupil  to the image/ waist
    image=Subapertures.get_composite_wavefont() # returns wavefront of image plane of all the spots put back together
    Subapertures.opd #single array made up of subapertures
    Subapertures.amplitude  #single array made up of subapertures
    Subapertures.getphasor #returns propagated array of spots from get_composite_wavefont
    Subapertures.get_centroids() # computes centroids of each subaperture after propagated to the detector


    Parameters
    ----------

    optic_array: numpy array of Aperture objects
        defines the subapertures
    crosstalk: boolean
        this variable sets whether light can leak from one lenslet image plane into the neighbors.
        Default is False.
    x_y_offset: tuple
        offset of central grid vertex from center of incident wavefront
    detector: Detector object
        defines a detector for the subapertures
    overwrite_inputwavefront: boolean
        option to replace the input wavefront with the subsampled wavefront array
    display_intermediates: boolean
        option to display each wavefront as it is subsampled/propagated to detector (this will take a while if there are many subapertures)
    optical_system: None, not implemented yet
    input_wavefront: None or poppy.Wavefront
        the unsampled wavefront incident on the subapertures
    """

    def __init__(self,
                 optic_array=np.array([[CircularAperture(radius=2., planetype=PlaneType.pupil),
                                        CircularAperture(radius=2., planetype=PlaneType.pupil)],
                                       [CircularAperture(radius=2., planetype=PlaneType.pupil),
                                        CircularAperture(radius=2., planetype=PlaneType.pupil)]]),
                 crosstalk=False,
                 x_y_offset=(0, 0),
                 detector=None,
                 overwrite_inputwavefront=False,
                 display_intermediates=False,
                 optical_system=None,
                 **kwargs):
        dimensions = optic_array.shape
        self.n_apertures = dimensions[0]*dimensions[1]  # number of subapertures
        self.optic_array = optic_array
        self.crosstalk = crosstalk
        if crosstalk:
            raise NotImplementedError("CROSS TALK NOT IMPLEMENTED <YET>")
        self.x_y_offset = x_y_offset
        self.amplitude = np.asarray([1.])
        self.opd = np.asarray([0.])
        self.input_wavefront = None
        self.output_wavefront = None
        if detector is None:
            self.detector = Detector(0.01, fov_pixels=128)
        else:
            self.detector = detector
        self.optical_system = optical_system
        if optical_system is not None:
            raise NotImplementedError("complete optical system after wavelets are not implemented yet")
        self.x_apertures = self.optic_array.shape[0]
        self.y_apertures = self.optic_array.shape[1]
        if self.x_apertures != self.y_apertures:
            raise ValueError("A square array of subapertures is currently required.")
        # initialize array of subsampled output wavefronts:
        self.wf_array = np.empty(self.optic_array.shape, dtype=np.object_)
        self.overwrite_inputwavefront = overwrite_inputwavefront
        self.display_intermediates = display_intermediates
        self._propagated_flag = False  # can't have propagated when initializing
        self._centroided_flag = False  # can't have centroids without propagating

        OpticalElement.__init__(self, **kwargs)
        return

    def sample_wf(self, wf):
        '''
        subsamples wavefront in order to multiply by optic array
        creates array of wavefront objects for each subaperture optic

        Parameters
        ----------
        wf: wavefront object to subsample with subapertures optic array
        '''
        # save the input wavefront
        self.input_wavefront = wf
        # create array of subsampled wavefronts
        for i in range(self.x_apertures):
            for j in range(self.y_apertures):
                opt = self.optic_array[i][j]  # get an optic

                # check for padding
                if opt is None:
                    continue

                aper_per_dim = wf.diam / (opt.pupil_diam)  # assuming squares

                self.width = opt.pupil_diam.to(u.m)/wf.pixelscale.to(u.m/u.pix)  # subaperture width in pixels
                # the generated number of subapertures might not match the input wavefront dimensions

                # want to center the subapertures on the incoming wavefront
                self.center = wf.wavefront.shape[0]/2*u.pix  # center of array
                sub_wf = wf.copy()  # new wavefront has all the previous wavefront properties except diameter
                sub_wf.diam = opt.pupil_diam  # fix this diameter to avoid scaling issues
                lower_x = int((self.center + self.width*(i) - self.width*self.x_apertures/2).value)
                lower_y = int((self.center + self.width*(j) - self.width*self.y_apertures/2).value)
                upper_x = int((self.center + self.width*(i+1) - self.width*self.x_apertures/2).value)
                upper_y = int((self.center + self.width*(j+1) - self.width*self.y_apertures/2).value)
                # sample wf:
                sub_wf.wavefront = wf.wavefront[lower_x:upper_x, lower_y:upper_y]
                # multiply by subaperture optic:
                self.wf_array[i][j] = sub_wf*opt

                if self.display_intermediates:
                    plt.figure()
                    self.wf_array[i][j].display()
        return

    @property
    def subaperture_width(self):
        # returns width in angular units
        return self.width*self.input_wavefront.pixelscale

    def get_wavefront_array(self):
        """
        returns the composite wavefront array

        """
        if self.input_wavefront is None:
            raise ValueError("No input wavefront found.")

        if self._propagated_flag:
            # recalculate dimensions if wavefront has propagated to detector
            print("Tiling propagated wavefront arrays.")
            center = self.center_out
            width = self.width_out
            # create new output wavefront
            wf = Wavefront(wavelength=self.input_wavefront.wavelength,
                           npix=int(2*self.center_out.value),
                           dtype=self.input_wavefront.wavefront.dtype,
                           pixelscale=self.detector.pixelscale,
                           oversample=self.detector.oversample)
        else:
            # otherwise keep same dimensions/properties as input wavefront
            center = self.center
            width = self.width
            wf = self.input_wavefront.copy()

        # recompile full wavefront array to display result
        for i in range(self.x_apertures):
            for j in range(self.y_apertures):
                sub_wf = self.wf_array[i][j]  # get a subaperture wavefront
                lower_x = int((center + width*(i) - width*self.x_apertures/2).value)
                lower_y = int((center + width*(j) - width*self.y_apertures/2).value)
                upper_x = int((center + width*(i+1) - width*self.x_apertures/2).value)
                upper_y = int((center + width*(j+1) - width*self.y_apertures/2).value)

                # check for padding
                if sub_wf is None:
                    wf.wavefront[lower_x:upper_x, lower_y:upper_y] = np.nan
                else:
                    wf.wavefront[lower_x:upper_x, lower_y:upper_y] = sub_wf.wavefront

        if self.overwrite_inputwavefront:
            self.input_wavefront = wf

        return wf

    def get_psfs(self):
        # propagate wavefront to detector
        if self.input_wavefront is None:
            raise ValueError("No input wavefront found.")

        for i in range(self.x_apertures):
            for j in  range(self.y_apertures):
                sub_wf = self.wf_array[i][j]
                sub_wf.propagate_to(self.detector)

                if self.display_intermediates:
                    plt.figure()
                    sub_wf.display()

        self.width_out = self.wf_array[0][0].shape[0]*u.pix  # subaperture width in pixels
        self.center_out = self.width_out*self.x_apertures/2  # center of array
        self._propagated_flag = True
        return

    def multiply_all(self, optic):
        '''
        multiply all of the wavefront arrays by an optic
        input:
        optic: Optic object
        '''
        if self.input_wavefront is None:
            raise ValueError("No input wavefront found.")
        for i in range(self.x_apertures):
            for j in range(self.y_apertures):
                self.wf_array[i][j] *= optic
        return

    def get_centroids(self,
                      cent_function=measure_centroid,
                      relativeto='origin',
                      asFITS=True,
                      **kwargs):
        """
        get centroid of intensity of each subwavefront

        inputs:
        cent_function: function names
            see available funtions in .utils (if using a different one make sure to include it at top of this file as well)
        relative to: string
            can be either 'center' or 'origin' for default centroid function
        asFITS: bool
            if True, treats subwavefronts as fits objects

        returns:
        centroid_list: np array
            list of calculated centroids in shape (2, nx, ny)
        note: if using wf_reconstruction method you need to either use 'center' here
        or provide a flat centroid list by propagating a flat wavefront through SHWFS
        to get proper wf reconstruction
        recommended to keep to defaults since 'center' does weird stuff at edges of aperture

        """
        print("Installed poppy in editable mode works!")
        _log.debug("Centroid function:"+str(cent_function))
        if self.input_wavefront is None:
            raise ValueError("No input wavefront found.")
        if not self._propagated_flag:
            _log.warn("Getting centroid without having propagated.")
        self.centroid_list = np.zeros((2,self.x_apertures, self.y_apertures))
        for i in range(self.x_apertures):
            for j in range(self.y_apertures):
                sub_wf = self.wf_array[i][j]
                if sub_wf.total_intensity == 0.0:
                    _log.warn("Setting centroid of aperture with no flux to NaN.")
                    self.centroid_list[:, i, j] = (np.nan, np.nan)
                    continue
                if asFITS:
                    intensity_array = sub_wf.as_fits()
                else:
                    intensity_array = sub_wf.intensity

                '''
                note, if fwcentroid gives index error at this step try the following steps: 
                - make sure the incident wavefront array is larger than the shack hartmann lenslet array
                - make sure the incident wavefront array is sampled appropriately considering the lenslet pitch and pixel pitch chosen
                - try adjusting the boxsize in the centroid function by adding "boxsize = #" to the call to cent_function below
                '''
                self.centroid_list[:, i, j] = cent_function(intensity_array, **kwargs, relativeto=relativeto)
        self._centroided_flag = True
        return self.centroid_list

    def _replace_subwavefronts(self, replacement_array):
        for i in range(self.x_apertures):
            for j in range(self.y_apertures):
                sub_wf = self.wf_array[i][j]  # get an subaperture wavefront
                lower_x = int((c + self.width*(i) - self.width*self.x_apertures/2).value)
                lower_y = int((c + self.width*(j) - self.width*self.y_apertures/2).value)
                upper_x = int((c + self.width*(i+1) - self.width*self.x_apertures/2).value)
                upper_y = int((c + self.width*(j+1) - self.width*self.y_apertures/2).value)
                # check for padding
                if sub_wf is None:
                    wf.wavefront[lower_x:upper_x, lower_y:upper_y] = np.nan
                else:
                    wf.wavefront[lower_x:upper_x, lower_y:upper_y] = sub_wf.wavefront
        return


class ShackHartmannWavefrontSensor(Subapertures):
    """
    Shack-Hartmann Wavefront Sensor Class
    wrapper for Subapertures class
    Parameters:
    lenslet pitch: astropy unit quantity [length]
        diameter of each lenslet (required)
    lenslet focal length: astropy unit quantity [length]
        focal length of each lenslet
    pixel pitch: astropy unit quantity [length]
        pixel pitch of detector
    n_lenslets: int
        number of lenslets per axis (assuming square lenslet array with n_lenslets**2 lenslets total)
    circular: bool
        True means supapertures are circular, False means they are square
    detector: Detector object or None
        defines plate scale, if None one will be automatically generated based on other inputs
    """

    def __init__(self, lenslet_pitch=300*u.um,
                 lenslet_fl=14.2*u.mm,
                 pixel_pitch=2.2*u.um,
                 n_lenslets=12,
                 circular=False,
                 detector=None,
                 **kwargs):

        self.lenslet_pitch = lenslet_pitch
        self.lenslet_fl = lenslet_fl
        self.pixel_pitch = pixel_pitch
        self.r_lenslet = self.lenslet_pitch/2.
        self.n_lenslets = n_lenslets

        if circular:
            aperture = poppy.CircularAperture(radius=self.lenslet_pitch/2, planetype=PlaneType.pupil)
        else:
            aperture=poppy.SquareAperture(size=self.lenslet_pitch)

        optic_array = np.array([[aperture, aperture],[aperture, aperture]])

        if detector is None:
            pixelscale = 1.0*u.rad/(lenslet_fl*u.pix/pixel_pitch)
            pix_per_lenslet = int(lenslet_pitch/pixel_pitch)
            detector = Detector(pixelscale, fov_pixels=pix_per_lenslet)

        # expand the array to make big_optic_array
        if n_lenslets % 2 != 0:
            raise ValueError("aperture replication only works for even numbers of apertures")

        big_optic_array = optic_array.repeat(n_lenslets/2., axis=0).repeat(n_lenslets/2., axis=1)

        Subapertures.__init__(self,
                              optic_array=big_optic_array,
                              detector=detector,
                              **kwargs)
        return

    def append_header(self, HDU):
        """
        adds values to output fits header to keep track of simulation parameters
        """
        HDU.header['SH_units'] = 'meters'
        HDU.header['name'] = self.name
        HDU.header['SH_pitch'] = self.lenslet_pitch.to(u.m).value
        HDU.header['SH_fl'] = self.lenslet_fl.to(u.m).value
        HDU.header['DETpitch'] = self.pixel_pitch.to(u.m/u.pix).value
        return HDU

    def calculate_centroid_requirement(self, min_WFE):
        """
        calculates centroid accuracy requirement for a given minimum wavefront error
        input:
        min_WFE: float w astropy unit
            desired minimum wavefront error

        returns:
        centroid: float
            calculated conversion from min wavefront error to centroid accuracy
        """
        if (min_WFE.decompose().unit != u.m):
            raise ValueError("minimum wavefront error must be in units of length")
        centroid = min_WFE * self.lenslet_fl/self.lenslet_pitch/self.pixel_pitch
        return centroid

    @property
    def pix_lenslet(self):
        return self.lenslet_pitch.to(u.m)/self.pixel_pitch.to(u.m)

    @property
    def max_WFE(self):
        """
        returns:
        float, maximum wavefront error detectable for ideal lenslet before the spot crosses
        into the neighboring lenslet

        """
        _log.warn("This max wavefront error ignores lenslet aberrations")
        return  1.0/self.lenslet_fl.to(u.m)*self.lenslet_pitch.to(u.m)**2/2.0

    def reconstruct_wavefront(self, flat_centroid_list):
        '''
        reconstructs wavefront using zonal reconstruction
        inputs:
        flat_centroid_list: None or array of floats, shape matches output of Subapertures.get_centroids() function (2, nx, ny)
            array of centroids from flat wavefront
            if using centroids already centered to 'center' of each subaperture, input flat_centroid_list = [None]
            recommended to repropagate a flat wavefront to avoid systematic offsets due to rounding, size/shape of real PSFs

        returns: reconstructed wavefront as an array
        '''
        if self.input_wavefront is None:
            raise ValueError("No input wavefront found.")
        if not self._propagated_flag:
            _log.warn("Trying to reconstruct wavefront without having propagated to detector. run sample_wf on wavefront and get_psfs first.")
        self.get_centroids()
        if flat_centroid_list.all() is not None:
            y_cen, x_cen = self.centroid_list
            y_cen_flat, x_cen_flat = flat_centroid_list
            x_off = np.subtract(x_cen, x_cen_flat)
            y_off = np.subtract(y_cen, y_cen_flat)
        else:
            y_off, x_off = self.centroid_list
        scale = self.pixel_pitch.to(u.m)/self.lenslet_fl.to(u.m)*u.rad/self.detector.oversample
        wf_reconstruction = self._zonal_reconstruction(x_off*scale, y_off*scale, self.lenslet_pitch)

        return wf_reconstruction

    def _zonal_reconstruction(self, centroids_x, centroids_y, subap_diameter):
        """
        Simple zonal reconstructor based on Matlab example in
        Wavefront Optics for Vision Correction.
        Dai, Guang-ming. 2008.
        Society of Photo-Optical Instrumentation Engineers.
        http://ebooks.spiedigitallibrary.org/book.aspx?doi=10.1117/3.769212.
        Modified to work with rectangular spotfields
        Modified from Greg Allan's master's thesis (Simulation and Testing of Wavefront Reconstruction Algorithms
        for the Deformable Mirror (DeMi) Cubesat. Master's Thesis, Massachusetts Institute of Techonology, 2018)

        inputs:
        centroids_x: np array
            centroid x displacement measurements
        centroids_y: np array
            centroid y displacement measurements
        subap_diameter: float with astropy unit quantity (length unit)
            diameter of subaperture (lenslets), represents distance between each centroid measurement for derivative matrix

        returns:
        w: np array with astropy unit (length)
            reconstructed wavefront array in same shape as input centroid measurement arrays,
            in same units as input centroid measurements

        The wavefront reconstruction is a solution to Cs = Ew (see section 4.3.1 of Dai 2008 for more details)
        s is vector of slope measurements (ie centroids), w is unknown wavefront vector
        C represents averaging of slope measurements, E represents derivative of the slope vector as a matrix
        """

        # reshape measured x/y centroids into a single flattened vector, flatten operates in row-major order:
        s = np.concatenate(np.nan_to_num([centroids_x.flatten().to(u.radian), centroids_y.flatten().to(u.radian)]))
        m = centroids_x.shape[0]
        n = centroids_x.shape[1]

        # define derivative matrix E (x derivative on top half, y derivative bottom half of matrix):
        ds = subap_diameter.value
        E = np.matrix(np.zeros([(m-1)*n+(n-1)*m, n*m]))
        for i in range(m):
            for j in range(n-1):
                E[i*(n-1)+j, i*n+j] = -1/ds
                E[i*(n-1)+j, i*n+j+1] = 1/ds
        for i in range(n):
            for j in range(m-1):
                E[m*(n-1)+i*(m-1)+j, i+j*n] = -1/ds
                E[m*(n-1)+i*(m-1)+j, i+(j+1)*n] = 1/ds

        # define averaging matrix C (x average top half, y average bottom half)
        C = np.matrix(np.zeros([(m-1)*n+(n-1)*m, 2*n*m]))
        for i in range(m):
            for j in range(n-1):
                C[i*(n-1)+j, i*n+j] = 0.5
                C[i*(n-1)+j, i*n+j+1] = 0.5
        for i in range(n):
            for j in range(m-1):
                C[m*(n-1)+i*(m-1)+j, n*(m+j)+i] = 0.5
                C[m*(n-1)+i*(m-1)+j, n*(m+j+1)+i] = 0.5

        C = np.matrix(C)
        s = np.matrix(s)

        # invert E
        Epinv = np.linalg.pinv(E)
        # solve equation for unknown wavefront w
        w = Epinv*C*s.T

        # return solution, reshaped match input centroid measurement array shape
        return w.reshape(m, n)*subap_diameter.unit
