'''
Goal: implement vector diffraction and polarization optics

To do:
* Subclass Fresnel WF, handle propagation through scalar and jones matrix elements (each prop is actually multiple props)
    * 2xYxX WF -- propagate vector WF
    * 2x2xYxX WF -- propagate tensor WF, compute intensity given stokes vector
* Implement handful of polarization optics (LP, QWP, HWP, arbitrary waveplates, jones matrix elements, VVC?)
* Add test cases


To do:
* intensity
* geometric functions: rotate, invert, etc.



Multiple ways of specifying input wavefront (but not clear that this matters for poppy approach):

    1. Vector Field
    1a. Scalar + input_polarization tuple
    1b. Vector

    2. Tensor Field
    2a. Scalar + stokes_parameters (scalar --> 2x2 unit matrix)
    2b. Tensor field + stokes_parameters

'''
import numpy as np
import astropy.units as u

from poppy.fresnel import FresnelWavefront

from . import accel_math
from .accel_math import xp, ensure_not_on_gpu

if accel_math._NUMEXPR_AVAILABLE:
    import numexpr as ne
    pi = np.pi  # needed for evaluation inside numexpr strings.


class VectorWavefront(FresnelWavefront):
    '''
    This class extends the FresnelWavefront class to handle vector diffraction.

    Parameters not in FresnelWavefront
    ----------
    stokes_vector : list-like
        4-element list of strokes parameters. If given, the VectorWavefront is a 2x2xYxX tensor. If None (default), it is 2xYxX
    input_polarization: list-like
        2-element list of input polarization. (1,0) is an x-polarized field; (0,1) is y-polarized; (1,i) is circular-polarized, etc.
        Ignored if stokes_vector is supplied.
    '''

    def __init__(self,
                 beam_radius,
                 units=u.m,
                 rayleigh_factor=2.0,
                 oversample=2,
                 stokes_vector=None,
                 input_polarization=(1,0),
                 **kwargs):
        super(VectorWavefront, self).__init__(
            beam_radius=beam_radius,
            units=units,
            rayleigh_factor=rayleigh_factor,
            oversample=oversample,
            **kwargs
        )
        # TO DO: clean up the logic of checking which is specified and handling appropriately
        self.stokes_vector = stokes_vector
        self.input_polarization = input_polarization
        self.pol_type = None

        if stokes_vector is not None: # wavefront tensor
            self.input_polarization = None
            self.pol_type = 'tensor'
            self.wavefront = self.wavefront * xp.eye(2)[:, :, xp.newaxis, xp.newaxis]
        elif input_polarization is not None: # wavefront vector
            self.pol_type = 'vector'
            self.wavefront = self.wavefront * xp.asarray(input_polarization)[:, xp.newaxis, xp.newaxis]
        else:
            raise ValueError('Either stokes_vector or input_polarization must be specified! Use Wavefront or FresnelWavefront for scalar diffraction.')
        
    @property
    def intensity(self):
        '''"""Electric field intensity of the wavefront (i.e. field amplitude squared)"""'''

        if self.pol_type == 'vector':
            if accel_math._USE_NUMEXPR:
                w = self.wavefront
                return ne.evaluate("sum(real(abs(w))**2, 0)")
            else:
                return np.abs(self.wavefront) ** 2
        elif self.pol_type == 'tensor':
            w = self.wavefront # TO DO: fix me! this is incorrect -- need to take stokes parameters into account
            return ne.evaluate("sum(real(abs(w))**2, 0)")
        else:
            raise ValueError(f'pol_type must be either "vector" or "tensor". Got {self.pol_type} instead.')