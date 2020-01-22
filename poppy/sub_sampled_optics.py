"""
Written by Ewan Douglas, 2018
Developed by Rachel Morgan, 2019-2020
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

class subapertures(poppy.OpticalElement):
        """
        Example roadmap:
        
        #generate wavefront
        wf = poppy.wavefront() #or fresnel wavefront
        
        #...various surfaces
        
        subaperture(ARRAY OF OPTICS) 
        #initialize this new class, where the array of optics define the subapertures (e.g. lenslets)
        
        subapertures.sample_wf(wf) #this function takes the wavefront and subsamples it by the area of each optic
        subapertures.get_wavefront_array() #returns an array of input sub-wavefronts multipled by subaperture optics
        subapertures.get_psfs() #fraunhofer or fresnel propagation of each pupil  to the image/ waist
        image=subapertures.get_composite_wavefont() # returns wavefront of image plane of all the spots put back together
        subapertures.opd #single array made up of subapertures
        subapertures.amplitude  #single array made up of subapertures
        subapertures.getphasor #returns propagated array of spots from get_composite_wavefont
        
        
        Parameters
        ----------
        
        crosstalk: boolean
                  this variable sets whether light can leak from one lenslet image plane into the neighbors.
                  Default is False.
        x_y_offset: tuple
                  offset of central grid vertex from center of incident wavefront
        input_wavefront: None or poppy.Wavefront 
                  : the unsampled wavefront incident on the subapertures
        """
        
        def __init__(self,
                     optic_array = np.array([[CircularAperture(radius=2.,planetype=PlaneType.pupil),
                                              CircularAperture(radius=2.,planetype=PlaneType.pupil)],
                                   [CircularAperture(radius=2.,planetype=PlaneType.pupil),
                                     CircularAperture(radius=2.,planetype=PlaneType.pupil)]]),
                     crosstalk=False,
                     x_y_offset=(0,0),
                     detector=None,
                     overwrite_inputwavefront=False,
                     display_intermediates=False,
                     optical_system=None,
                     subwf_oversample=3,
                    **kwargs):
            dimensions = optic_array.shape
            self.n_apertures = dimensions[0]*dimensions[1]
            self.subwf_oversample = subwf_oversample
            self.optic_array=optic_array
            self.crosstalk=crosstalk
            if crosstalk:
                raise ValueError("CROSS TALK NOT IMPLEMENTED <YET>")
            self.x_y_offset=x_y_offset
            self.amplitude = np.asarray([1.])
            self.opd = np.asarray([0.])
            self.input_wavefront = None
            self.output_wavefront = None
            if detector == None:
                self.detector = Detector(0.01,fov_pixels=128)
            else:
                self.detector=detector
            self.optical_system = optical_system
            if  optical_system is not None:
                    raise ValueError("complete optical system after wavelets are not implemented yet")
            self.x_apertures=self.optic_array.shape[0]
            self.y_apertures=self.optic_array.shape[1]
            if self.x_apertures != self.y_apertures:
                raise ValueError("A square array of subapertures is currently required.")
            #initialize array of subsampled output wavefronts:
            self.wf_array = np.empty(self.optic_array.shape,dtype=np.object_)
            self.overwrite_inputwavefront = overwrite_inputwavefront
            self.display_intermediates = display_intermediates
            self._propagated_flag = False #can't have propagated when initializing
            self._centroided_flag = False #can't have centroids without propagating
    
            OpticalElement.__init__(self, **kwargs)
            return
                
        def sample_wf(self, wf):
            '''
            subsamples wavefront optic in order to mupliply by optic array
            creates array of wavefront objects for each subaperture optic

            Parameters
            ----------
            wf: wavefront object to subsample with subapertures optic array
            '''
            #save the input wavefront
            self.input_wavefront = wf
            #create array of subsampled wavefronts
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    opt = self.optic_array[i][j] #get an optic

                    #check for padding
                    if opt == None:
                        continue
                    
                    aper_per_dim = wf.diam /(opt.pupil_diam) #assuming squares

                    self._w = opt.pupil_diam.to(u.m)/wf.pixelscale.to(u.m/u.pix) #subaperture width in pixels 
                    #the generated number of subapertures might not match the input wavefront dimensions

                    #want to center the subapertures on the incoming wavefront
                    
                    self.c = wf.wavefront.shape[0]/2*u.pix #center of array
                    c=self.c
                    w=self._w
                    sub_wf=wf.copy() #new wavefront has all the previous wavefront properties except diameter
                    sub_wf.diam = opt.pupil_diam # fix this diameter to avoid scaling issues
                    lower_x = int((c + w*(i)  - w*self.x_apertures/2).value )
                    lower_y = int((c + w*(j)  - w*self.y_apertures/2).value )
                    upper_x = int((c + w*(i+1)  - w*self.x_apertures/2).value )
                    upper_y = int((c + w*(j+1)  - w*self.y_apertures/2).value )
                    # sample wf:
                    sub_wf.wavefront = wf.wavefront[lower_x:upper_x,lower_y:upper_y]
                    # multiply by subaperture optic:
                    self.wf_array[i][j] = sub_wf*opt

                    if self.display_intermediates:
                        plt.figure()
                        self.wf_array[i][j].display()
            return
                        
        @property
        def subaperture_width(self):
            # returns width in angular units
            return self._w*self.input_wavefront.pixelscale
        
        #return a composite wavefront if an array of output wavefronts was generated
        def get_wavefront_array(self):
            """
            
            
            """
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
                

            if self._propagated_flag:
                #recalculate dimensions
                print("Tiling propagated wavefront arrays.")
                c = self.c_out
                w = self._w_out
                #create new output wavefront
                wf = Wavefront(wavelength = self.input_wavefront.wavelength, 
                                     npix = int(2*self.c_out.value), 
                                     dtype = self.input_wavefront.wavefront.dtype, 
                                     pixelscale = self.detector.pixelscale,
                                     oversample = self.detector.oversample)
                
            else:
                c = self.c
                w = self._w
                wf = self.input_wavefront.copy()
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j] #get a subaperture wavefront
                    lower_x = int((c + w*(i)  - w*self.x_apertures/2).value)
                    lower_y = int((c + w*(j)  - w*self.y_apertures/2).value)
                    upper_x = int((c + w*(i+1)  - w*self.x_apertures/2).value)
                    upper_y = int((c + w*(j+1)  - w*self.y_apertures/2).value)

                    #check for padding

                    if sub_wf == None:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = np.nan
                    else:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = sub_wf.wavefront 
            if self.overwrite_inputwavefront:
                self.input_wavefront = wf
                
            return wf
        def get_psfs(self):
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
    
            for i in range(self.x_apertures):
                for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j]
                    sub_wf.propagate_to(self.detector)

                    if self.display_intermediates:
                        plt.figure()
                        sub_wf.display()
        
            self._w_out= self.wf_array[0][0].shape[0]*u.pix #subaperture width in pixels 
            self.c_out =  self._w_out*self.x_apertures/2 #center of array         
            self._propagated_flag = True
            return

        def multiply_all(self, optic):
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
            for i in range(self.x_apertures):
                for j in  range(self.y_apertures):
                    self.wf_array[i][j] *= optic
            return
        

        def get_centroids(self,
                          cent_function=measure_centroid,
                          relativeto = 'origin',
                          asFITS=True,
                          **kwargs):
            """
                get centroid of intensity of each subwavefront
                relative to can be either 'center' or 'origin' for default centroid function
                note, if using wf_reconstruction method you need to either use 'center' here 
                or provide a flat centroid list by propagating a flat wavefront through SHWFS 
                to get proper wf reconstruction
                recommended to keep to defaults since 'center' does weird stuff at edges of aperture

            """
            _log.debug("Centroid function:"+str(cent_function))
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
            if not self._propagated_flag:
                _log.warn("Getting centroid without having propagated.")
            self.centroid_list = np.zeros((2,self.x_apertures,self.y_apertures))
            for i in range(self.x_apertures):
                for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j]
                    if sub_wf.total_intensity == 0.0:
                        _log.warn("Setting centroid of aperture with no flux to NaN.")
                        self.centroid_list[:,i,j]=(np.nan,np.nan)
                        continue
                    if asFITS:
                        intensity_array=sub_wf.as_fits()
                    else:
                        intensity_array = sub_wf.intensity
                    self.centroid_list[:,i,j] = cent_function(intensity_array,**kwargs, relativeto = relativeto)
            self._centroided_flag = True
            return self.centroid_list

        
        def _replace_subwavefronts(self,replacement_array):
            for i in range(self.x_apertures):
                 for j in  range(self.y_apertures):
                    sub_wf = self.wf_array[i][j] #get an subaperture wavefront
                    lower_x = int((c + self._w*(i)  - self._w*self.x_apertures/2).value)
                    lower_y = int((c + self._w*(j)  - self._w*self.y_apertures/2).value)
                    upper_x = int((c + self._w*(i+1)  - self._w*self.x_apertures/2).value)
                    upper_y = int((c + self._w*(j+1)  - self._w*self.y_apertures/2).value)
                    #check for padding

                    if sub_wf == None:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = np.nan
                    else:
                        wf.wavefront[lower_x:upper_x,lower_y:upper_y] = sub_wf.wavefront 
            return 

        def reconstruct_wavefront(self, flat_centroid_list ):
            '''
            reconstruct wavefront using zonal reconstruction
            returns reconstructed wavefront as an array
            if using centroids already centered to 'center', input flat_centroid_list = [None]
            '''
            if self.input_wavefront is None:
                raise ValueError("No input wavefront found.")
            if not self._propagated_flag:
                _log.warn("Trying to reconstruct wavefront without having propagated to detector. run sample_wf on wavefront and get_psfs first.")
            self.get_centroids()
            if flat_centroid_list.all()!=None:
                y_cen, x_cen = self.centroid_list
                y_cen_flat, x_cen_flat = flat_centroid_list
                x_off = np.subtract(x_cen, x_cen_flat)
                y_off = np.subtract(y_cen, y_cen_flat)
            else:
                y_off, x_off = self.centroid_list
            scale = self.pixel_pitch.to(u.m)/self.lenslet_fl.to(u.m)*u.rad/self.detector.oversample
            wf_reconstruction = zonalReconstruction(x_off*scale, y_off*scale, self.lenslet_pitch)

            return wf_reconstruction


class SH_WFS(subapertures):
    """
        Shack-Hartmann Wavefront Sensor Class
        wrapper for subapertures class
        Parameters: 
        lenslet pitch
        lenslet focal length
        pixel pitch
        number of lenslets per axis (assuming square lenslet array with n_lenslets**2 lenslets total)
        circular aperture (True means supapertures are circular, False means they are square)
        detector: Detector object defining pixel pitch and plate scale
    """

    def __init__(self,lenslet_pitch = 300*u.um,
                             lenslet_fl = 14.2*u.mm,
                             pixel_pitch = 2.2*u.um/u.pix,
                             n_lenslets=12,
                             circular=False,
                             detector = None,
                             **kwargs
                             ):
        
        self.lenslet_pitch=lenslet_pitch
        self.lenslet_fl=lenslet_fl
        self.pixel_pitch=pixel_pitch
        self.r_lenslet=self.lenslet_pitch/2.
        self.n_lenslets = n_lenslets

        if circular:
            aperture=poppy.CircularAperture(radius = self.lenslet_pitch/2, planetype = PlaneType.pupil)
        else:
            ap_keywords={"size":self.lenslet_pitch,"planetype":PlaneType.pupil}
            aperture=poppy.SquareAperture(size = self.lenslet_pitch, planetype = PlaneType.pupil)

        optic_array = np.array([[aperture,
                                              aperture],
                                    [aperture,
                                    aperture]])
        
        #expand the array
        if n_lenslets % 2 !=0:
            raise ValueError("aperture replication only works for even numbers of apertures")
        
        big_optic_array=optic_array.repeat(n_lenslets/2.,axis=0).repeat(n_lenslets/2.,axis=1)

        subapertures.__init__(self,
                                  optic_array=big_optic_array,
                                  detector = detector,
                                  **kwargs)
        return 

    def append_header(self,HDU):
         HDU.header['SH_units']='meters'
         HDU.header['name']=self.name
         HDU.header['SH_pitch'] = self.lenslet_pitch.to(u.m).value
         HDU.header['SH_fl'] = self.lenslet_fl.to(u.m).value
         HDU.header['DETpitch'] = self.pixel_pitch.to(u.m/u.pix).value
         return HDU

    def calculate_centroid_requirement(self, min_WFE):
        """
          centroid 
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
        returns the maximum wavefront error detectable for ideal lenslet before the spot crosses
        into the neighboring lenslet

        """
        _log.warn("This max wavefront error ignores lenslet aberrations")
        return  1.0/self.lenslet_fl.to(u.m)*self.lenslet_pitch.to(u.m)**2/2.0

#helper functions for wavefront reconstruction:
def zonalReconstruction(x,y,ds):
    """
    Simple zonal reconstructor based on Matlab example in
    Wavefront Optics for Vision Correction. 
    Dai, Guang-ming. 2008. 
    Society of Photo-Optical Instrumentation Engineers. 
    http://ebooks.spiedigitallibrary.org/book.aspx?doi=10.1117/3.769212.
    Modified to work with rectangular spotfields
    From Greg Allan's master's thesis (Simulation and Testing of Wavefront Reconstruction Algorithms 
    for the Deformable Mirror (DeMi) Cubesat. Master's Thesis, Massachusetts Institute of Techonology, 2018)
    """
    #flatten operates in row-major order
    g=np.concatenate(np.nan_to_num([x.flatten().to(u.radian), y.flatten().to(u.radian)]))
    m=x.shape[0]
    n=x.shape[1]
    S=g
    
    E = getE(m,n)
    C = getC(m,n)
 
    C=np.matrix(C)
    S=np.matrix(S)

    Epinv = np.linalg.pinv(E)
    p=Epinv*C*S.T
    return np.multiply(p.reshape(m,n),ds)

def getC(m,n):
    C = np.matrix(np.zeros([(m-1)*n+(n-1)*m, 2*n*m]))

    for i in range(m):
        for j in range(n-1) :
                C[i*(n-1)+j, i*n+j]=0.5
                C[i*(n-1)+j, i*n+j+1]=0.5
    for i in range(n):
        for j in range(m-1):              
                C[m*(n-1)+i*(m-1)+j, n*(m+j)+i]=0.5
                C[m*(n-1)+i*(m-1)+j, n*(m+j+1)+i]=0.5
    return C
    
    
def getE(m,n):
    E=np.matrix(np.zeros([(m-1)*n+(n-1)*m, n*m]))
    for i in range(m):
        for j in range(n-1):
            E[i*(n-1)+j, i*n+j] = -1
            E[i*(n-1)+j, i*n+j+1] = 1
    for i in range(n):
        for j in range(m-1):
            E[m*(n-1)+i*(m-1)+j, i+j*n] = -1 
            E[m*(n-1)+i*(m-1)+j, i+(j+1)*n] = 1
    return E


