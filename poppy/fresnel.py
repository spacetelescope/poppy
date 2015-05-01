import poppy
#---- begin top of poppy_core.py
from __future__ import division
import multiprocessing
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage.interpolation
import matplotlib
import time


from matplotlib.colors import LogNorm  # for log scaling of images, with automatic colorbar support

import astropy.io.fits as fits



#from .utils import imshow_with_mouseover, estimate_optimal_nprocesses, fftw_load_wisdom, fftw_save_
from .matrixDFT import MatrixFourierTransform
from . import utils
from . import settings


import logging
_log = logging.getLogger('poppy')

try:
    from IPython.core.debugger import Tracer; stop = Tracer()
except:
    pass

# Setup infrastructure for FFTW
_FFTW_INIT = {}  # dict of array sizes for which we have already performed the required FFTW planning step
_FFTW_FLAGS = ['measure']
if settings.use_fftw():
    try:
        # try to import FFTW and use it
        import pyfftw
    except:
        # we tried but failed to import it. 
        settings.use_fftw.set(False)

# internal constants for types of plane
_PUPIL = 1
_IMAGE = 2
_DETECTOR = 3 # specialized type of image plane.
_ROTATION = 4 # not a real optic, just a coordinate transform
_typestrs = ['', 'Pupil plane', 'Image plane', 'Detector', 'Rotation']


#conversions
_RADIANStoARCSEC = 180.*60*60 / np.pi

#---end top of poppy_core.py



import logging
_log = logging.getLogger('poppy')




class curvature(poppy.AnalyticOpticalElement):
    '''
    Class
    '''
    def __init__(self, 
                 z,
                 planetype = '_INTERMED',
                 name = 'Quadratic Wavefront Curvature Operator',
                 reference_wavelength = 2e-6,
                 units=u.m,
                 **kwargs):
        poppy.AnalyticOpticalElement.__init__(self,name=name, planetype=_PUPIL, **kwargs)

        self.reference_wavelength = reference_wavelength*units
        self.name = name
        if  isinstance(z,u.quantity.Quantity):
            self.z_m = (z).to(u.m) #convert to meters.
        else:
            _log.debug("Assuming meters, phase (%.3g) has no units for Optic: "%(z)+self.name)
            self.z_m=z*u.m

    def getPhasor(self, wave):
        y, x = wave.coordinates()
        rsqd = (x**2+y**2)*u.m**2
        #quad_phase_1st= np.exp(i*k*(x**2+y**2)/(2*z))#eq. 6.68

        k=2* np.pi/self.reference_wavelength
        lens_phasor = np.exp(1.j * k * rsqd/(2.0*self.z_m))
        #stop()
        return lens_phasor
    
class Gaussian_Lens(curvature):
    '''
    Class
    '''
    def __init__(self, 
                 f_lens,
                 planetype = '_INTERMED',
                 name = 'Gaussian Lens',
                 reference_wavelength = 2e-6,
                 units=u.m,
                 **kwargs):
        curvature.__init__(self, 
                 -f_lens,
                 planetype =planetype,
                 name = name,
                 reference_wavelength = reference_wavelength,
                 units=units,
                 **kwargs)
   
        if  isinstance(f_lens,u.quantity.Quantity):
            self.fl = (f_lens).to(u.m) #convert to meters.
        else:
            _log.debug("Assuming meters, phase (%.3g) has no units for Optic: "%(z)+self.name)
            self.fl=z*u.m

  

class gaussian_wavefront(poppy.Wavefront):  
    def __init__(self, beam_radius, 
                 units=u.m, 
                 force_fresnel=True,
                 rayl_factor=2.0,
                 **kwds):
        '''
        
        Parameter:
        
        units:
        w_0
        z
        z_w0
        wavelen_m
        
        spherical:
            Indicates wavefront is spherical, default False (that is, wavefront is planar).
        force_fresnel:
            if True then the Fresnel propogation will always be used,
            even between planes of type _PUPIL or _IMAGE
            
        
        References:
        1. G.N. Lawerence, Optical Modeling. 1992. 
        2. https://en.wikipedia.org/wiki/Gaussian_beam
        
        3. IDEX Optics and Photonics(n.d.), Gaussian Beam Optics, 
        [online] Available from:
        https://marketplace.idexop.com/store/SupportDocuments/All_About_Gaussian_Beam_OpticsWEB.pdf
        
        4. Krist, J. E. (2007), PROPER: an optical propagation library for IDL, 
        vol. 6675, p. 66750P–66750P–9. 
        [online] Available from: http://dx.doi.org/10.1117/12.731179 
        (Accessed 12 September 2013)

    
        '''
        
        '''
        initialize general wavefront class first,
        in Python 3 this will change, 
        https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        '''
        super(gaussian_wavefront,self).__init__(**kwds)  
        self.units = units
        self.w_0 = (beam_radius).to( self.units) #convert to base units.
        self.z  =  0*units
        self.z_w0 = 0*units
        self.wavelen_m = self.wavelength*u.m
        self.spherical = False
        self.i = np.complex(0,1)
        self.k = np.pi*2.0/self.wavelength
        self.force_fresnel = force_fresnel
        self.rayl_factor= rayl_factor
        if self.shape[0]==self.shape[1]:
            self.n=self.shape[0]
        else:
            self.n=self.shape
        #self.wavelength = self.wavelength*u.m #breaks other parts of POPPY
        
    @property
    def z_R(self):
        '''
        The Rayleigh distance for the gaussian beam.
        '''
        
        return np.pi*self.w_0**2/(self.wavelen_m)
          
    @property
    def divergance(self):
        '''
        Divergence of the gaussian beam
        '''
        return 2*self.wavelen_m/(np.pi*self.w_0)

    def R_c(self,z):
        '''
        The gaussian beam radius of curvature as a function of distance
        '''
        dz=(z-self.z_w0) #z relative to waist
        print(dz)
        print((self.z_R/dz)**2)
        return dz*(1+(self.z_R/dz)**2)
    
    #def beam_radius(self,z):
    #    '''
    #    Diameter of the gaussian beam as a function of distance
    #    '''
    #    return self.w_0*(1.0+self.divergance*(self.z_w0-z))
    def spot_radius(self,z):
        return self.w_0 * np.sqrt(1.0 + ((z-self.z_w0)/self.z_R)**2 )

    def propogateDirect(self,z):
        #
        x,y=self.coordinates()
        k=np.pi*2.0/self.wavelength
        S=self.n*self.pixelscale
        quad_phase_1st= np.exp(i*k*(x**2+y**2)/(2*z))#eq. 6.68
        quad_phase_2nd= np.exp(i*k*z)/(i*self.wavelength*z)*np.exp(i*k*(x**2+y**2)/(2*z))#eq. 6.70

        stage1=self.wavefront*quad_phase_1st #eq.6.67

        dft=forward_FFT(stage1)

        result=np.fft.fftshift(dft*self.pixelscale**2*quad_phase_2nd) #eq.6.69 and #6.80l
        self.wavefront=result
        
        return
    
    def ptp(self,z2): 
        '''
        Lawrence eq. 82, 86,87
        '''
        self.propogateDirect(z)
    def wts(self,z2):
        '''
        Lawrence eq. 83,88
        '''
        dz = z2-self.z
        _log.debug("Waist to Spherical propagation,dz="+str(dz))

        if dz ==0:
            _log.error("Waist to Spherical propagation stopped, no change in distance.")
            return 
        
        self *= curvature(-(dz), reference_wavelength=self.wavelength)
    
        if dz > 0:
            self.wavefront = forward_FFT(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE')#, threads=multiprocessing.cpu_count())
            self.wavefront *= self.n
        else:
            self.wavefront = inverse_FFT(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE')#, threads=multiprocessing.cpu_count())
            self.wavefront *= 1.0/self.n
            
        self.pixelscale = self.wavelength*np.abs(dz.value)/(self.n*self.pixelscale)
        self.z = self.z + dz
        self.wavefront = np.fft.fftshift(self.wavefront)

    def stw(self,z2):
        '''
        Lawrence eq. 89
        '''
        '''
        Lawrence eq. 83,88
        '''
        dz = z2 - self.z
        _log.debug("Spherical to Waist propagation,dz="+str(dz))

        if dz ==0:
            _log.error("Spherical to Waist propagation stopped, no change in distance.")
            return 
           
        if dz > 0:
            self.wavefront = forward_FFT(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE')#, threads=multiprocessing.cpu_count())
            self.wavefront *= self.n
        else:
            self.wavefront = inverse_FFT(self.wavefront, overwrite_input=True,
                                     planner_effort='FFTW_MEASURE')#, threads=multiprocessing.cpu_count())
            self.wavefront *= 1.0/self.n
        
        self *= curvature(dz, reference_wavelength=self.wavelength)


        self.pixelscale = self.wavelength*np.abs(dz.value)/(self.n*self.pixelscale)
        self.z = self.z + dz
        self.wavefront = np.fft.fftshift(self.wavefront)

    def planar_range(self,z):
        print(self.z_w0,self.z,z)
        if np.abs(self.z_w0 - self.z) < self.z_R:
            return True
        else:
            return False
            
    def propogateFresnel(self,z):
        '''
        Parameters:
        z:
            the distance from the current location to propogate the beam.
        
        Description:
        Each spherical wavefront is propagated to a waist and then to the next appropriate plane 
         (spherical or planar). 
         
        '''
        if not self.spherical:
            if self.planar_range(z):
                _log.debug("waist at z="+str(self.z_w0))
                _log.debug('Plane to Plane Regime')
                self.ptp(self.z)
            else:
                _log.debug("waist at z="+str(self.z_w0))
                _log.debug('Plane to Spherical, inside Z_R to outside Z_R')
                self.ptp(self.z_w0)
                self.wts(z)
        else:
            if self.planar_range(z):
                _log.debug("waist at z="+str(self.z_w0))
                _log.debug('Spherical to Plane Regime, outside Z_R to inside Z_R')
                self.stw(self.z_w0)
                self.ptp(z)
            else:
                _log.debug("waist at z="+str(self.z_w0))
                _log.debug('Spherical to Spherical, Outside Z_R to waist (z_w0) to outside Z_R')
                _log.debug('Starting Pixelscale:%.2g'%self.pixelscale)
                self.stw(self.z_w0)
                _log.debug('Intermediate Pixelscale:%.2g'%self.pixelscale)
                self.pixelscale
                #self.wts(z)
    
    def apply_optic(self,optic,z_lens,units=u.m,ignore_wavefront=False):
        '''
        
        Adds thin lens wavefront curvature to the wavefront 
        of focal length f_l and updates the 
        Gaussian beam parameters of the wavefront.
        
        Parameters
        -------------
        optic : Gaussian_Lens
        
        f_lens : float 
             lens focal length
             
        z_lens : float 
             location of lens relative to the wavefront origin 
        '''
        zl = (z_lens).to( self.units) #convert to meters.
        new_waist = self.spot_radius(zl)
        
        #is the last surface outside the rayleigh distance?
        if np.abs(self.z_w0 - self.z) > self.rayl_factor*self.z_R:
            print("spherical",self.z_w0 ,self.z,self.z_R)
            self.spherical = True
            R_input_beam = self.z - self.z_w0

        if (self.planetype == _PUPIL or self.planetype ==_IMAGE):
            #we are at a focus or pupil, so the new optic is the only curvature of the beam
            r_curve = -optic.fl
        else:
            r_curve = 1.0/(1.0/self.R_c(zl) - 1.0/optic.fl)

        #update the wavefront to the post-lens beam waist 
        if self.R_c(zl) == optic.fl:
            _log.debug(str(optic.name) +" has a flat output wavefront")
            self.z_w0 = zl
            self.w_0 = new_waist
        else:
            self.z_w0 = -r_curve/(1.0 + (self.wavelen_m*r_curve/(np.pi*new_waist)**2)**2) + zl
            self.w_0 = new_waist/np.sqrt(1.0+(np.pi*new_waist**2/(self.wavelen_m*r_curve))**2)
        
        
        #check that this Fresnel business is necessary.
        if (not self.force_fresnel) and (self.planetype == _PUPIL or self.planetype ==_IMAGE) \
            and (optic.planetype ==_IMAGE or optic.planetype ==_PUPIL):
            _log.debug("Simple pupil / image propogation, Fresnel unneccesary. \
                       Reverting to Fraunhofer.")
            self.propagateTo(optic)
            return
    
        if ignore_wavefront:
            return
        
        if (not self.spherical) and(np.abs(self.z_w0 - zl) < self.z_R):
            print('Near-field, Plane-to-Plane Propagation.')
            z_eff = fl

        elif (not self.spherical) and (np.abs(self.z_w0 - zl) > self.z_R):
            # find the radius of curvature of the lens output beam
            # curvatures are multiplicative exponentials
            # e^(1/z) = e^(1/x)*e^(1/y) = e^(1/x+1/y) -> 1/z = 1/x + 1/y 
            # z = 1/(1/x+1/y) = xy/x+y  
            print('Inside Rayleigh distance to Outside Rayleigh distance.')
            z_eff = 1.0/( 1.0/optic.fl+ 1.0/(zl-self.z_w0))
            print(z_eff)
            self.spherical = True


            #optic needs new focal length:
        elif (self.spherical) and (np.abs(self.z_w0 - zl) > self.z_R):
            print('Spherical to Spherical wavefront propogation.')
            if R_input_beam == 0:
                z_eff = 1.0/( 1.0/optic.fl- 1.0/(R_input_beam)) 
            if (zl-self.z_w0) ==0:
                z_eff = 1.0/( 1.0/optic.fl+ 1.0/(zl-self.z_w0)) 
            else:
                z_eff = 1.0/( 1.0/optic.fl+ 1.0/(zl-self.z_w0)- 1.0/(R_input_beam)) 

            
        elif (self.spherical) and (np.abs(self.z_w0 - zl) < self.z_R):
            print('Spherical to Planar.')
            z_eff=1.0/( 1.0/optic.fl - 1.0/(R_input_beam) )
            self.spherical=False
            
        effective_optic = curvature(-(z_eff) ,reference_wavelength=self.wavelength)
        self *= effective_optic

        #update wavefront location:
        #self.z = zl
        return 
