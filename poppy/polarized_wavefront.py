import numpy as np
import astropy.units as u

from poppy.poppy_core import Wavefront, BaseWavefront
from poppy.fresnel import FresnelWavefront

from . import accel_math
from .accel_math import xp

if accel_math._NUMEXPR_AVAILABLE:
    import numexpr as ne
    pi = np.pi  # needed for evaluation inside numexpr strings.

__all__ = ['PolarizedWavefront', 'PolarizedFresnelWavefront']

class BasePolarizedWavefront(BaseWavefront):
    '''
    Base class for polarized wavefronts, not intended for direct use.

    Parameters not in BaseWavefront
    ----------
    input_stokes_vector : list-like
        4-element list of stokes parameters for partially-polarized wavefronts.
        If given, the PolarizedWavefront is a 2x2xYxX tensor. If None (default), a linearly polarized
        wavefront is assumed (see input_polarization argument).
    input_polarization: list-like
        2-element list of input polarization for fully-polarized 2xYxX vector wavefronts.
        (1,0) is an x-polarized field; (0,1) is y-polarized; (1,i) is circular-polarized, etc.
        Ignored if input_stokes_vector is supplied. Default is (1,0).
    '''

    def __init__(self,
                 input_stokes_vector=None,
                 input_polarization=(1,0),
                 **kwargs):
        super(BasePolarizedWavefront, self).__init__(
            **kwargs
        )
        self.input_stokes_vector = None
        self.input_polarization = None

        if input_stokes_vector is not None: # wavefront tensor
            self.input_polarization = None
            self.pol_type = 'tensor'
            self.wavefront = self.wavefront * xp.eye(2)[:, :, xp.newaxis, xp.newaxis]
            self.input_stokes_vector = xp.asarray(input_stokes_vector)
        elif input_polarization is not None: # wavefront vector
            self.pol_type = 'vector'
            self.input_polarization = xp.asarray(input_polarization)
            self.wavefront = self.wavefront * self.input_polarization[:, xp.newaxis, xp.newaxis]
        else:
            raise ValueError('Either input_stokes_vector or input_polarization must be specified! For scalar diffraction, use Wavefront or FresnelWavefront.')

    @property
    def intensity(self):
        """Electric field intensity of the wavefront, accounting for polarization"""

        if self.pol_type == 'vector':
            if accel_math._USE_NUMEXPR:
                w = self.wavefront
                return ne.evaluate("sum(real(abs(w))**2, 0)")
            else:
                return np.sum(np.abs(self.wavefront) ** 2, axis=0)
        elif self.pol_type == 'tensor':
            return self.stokes_parameters[0] # I element of stokes parameters (I,Q,U,V)
        else:
            raise ValueError(f'pol_type must be either "vector" or "tensor". Got {self.pol_type} instead.')

    @property
    def stokes_parameters(self):
        """ Stokes parameters of the wavefront (only valid when stokes_vector is provided)"""
        if self.input_stokes_vector is None:
            raise ValueError('Stokes parameters cannot be computed unless input_stokes_vector is supplied!')
        return jones_to_stokes(self.wavefront, self.input_stokes_vector)
    
    def display_tensor(self, *args, **kwargs):
        """ Display the vector or tensor field """

        if self.pol_type == 'vector':
            nrows = 2
            indices = [0,1]
        else: # tensor
            nrows = 4
            indices = [(0,0), (0,1), (1,0), (1,1)]

        axes = []
        for n, idx in enumerate(indices):
            ax = super(BasePolarizedWavefront, self).display(
                    *args,
                    nrows=nrows,
                    row=n+1,
                    tensor_idx=idx,
                    **kwargs)
            title = ax.title
            title_text = title.get_text()
            title.set_text(str(idx))
            axes.append(ax)
        fig = ax.get_figure()
        fig.suptitle(title_text)
        return axes
    
    def _display_after_optic(self, optic, default_nplanes=2, **kwargs):
        """ Convenience function for displaying a wavefront during propagations.

        Checks for hint information attached to either the wavefront or the
        current optic, and uses that to configure the plot as desired.
        Called from within the various propagate() functions.

        This is a slight tweak of BaseWavefront._display_after_optic to
        force partially-polarized wavefronts to display Stokes parameters
        by default, regardless of what the optic wavefront_display_hint is.

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
        if self.pol_type == 'tensor':
            display_what = 'stokes'
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

        

class PolarizedWavefront(BasePolarizedWavefront, Wavefront):
    '''
    This class extends the Wavefront class to handle Fraunhofer propagation of
    fully- and partially-polarized wavefronts.

    Parameters not in Wavefront
    ----------
    input_stokes_vector : list-like
        4-element list of stokes parameters for partially-polarized wavefronts.
        If given, the PolarizedWavefront is a 2x2xYxX tensor. If None (default), a linearly polarized
        wavefront is assumed (see input_polarization argument).
    input_polarization: list-like
        2-element list of input polarization for fully-polarized 2xYxX vector wavefronts.
        (1,0) is an x-polarized field; (0,1) is y-polarized; (1,i) is circular-polarized, etc.
        Ignored if input_stokes_vector is supplied. Default is (1,0).
    '''

    def __init__(self,
                 input_stokes_vector=None,
                 input_polarization=(1,0),
                 **kwargs):
        super(PolarizedWavefront, self).__init__(
            input_stokes_vector=input_stokes_vector,
            input_polarization=input_polarization,
            **kwargs
        )

class PolarizedFresnelWavefront(BasePolarizedWavefront, FresnelWavefront):
    '''
    This class extends the FresnelWavefront class to handle Fresnel propagation of
    fully- and partially-polarized wavefronts.

    Parameters not in FresnelWavefront
    ----------
    input_stokes_vector : list-like
        4-element list of stokes parameters for partially-polarized wavefronts.
        If given, the PolarizedFresnelWavefront is a 2x2xYxX tensor. If None (default), a linearly polarized
        wavefront is assumed (see input_polarization argument).
    input_polarization: list-like
        2-element list of input polarization for fully-polarized 2xYxX vector wavefronts.
        (1,0) is an x-polarized field; (0,1) is y-polarized; (1,i) is circular-polarized, etc.
        Ignored if input_stokes_vector is supplied. Default is (1,0).
    '''

    def __init__(self,
                 beam_radius,
                 units=u.m,
                 rayleigh_factor=2.0,
                 oversample=2,
                 input_stokes_vector=None,
                 input_polarization=(1,0),
                 **kwargs):
        super(PolarizedFresnelWavefront, self).__init__(
            beam_radius=beam_radius,
            units=units,
            rayleigh_factor=rayleigh_factor,
            oversample=oversample,
            input_stokes_vector=input_stokes_vector,
            input_polarization=input_polarization,
            **kwargs
        )

def jones_to_stokes(jones_matrix, input_stokes_vector):
    """ Convert 2x2 Jones matrix to Stokes parameters
    
    Based on Eqn A4.13, Spectroscopic Ellipsometry: Principles and Applications H. Fujiwara
    """
    M = jones_to_mueller(jones_matrix)
    return xp.einsum('i...,i...', M, input_stokes_vector).real
    
def jones_to_mueller(jones_matrix):
    """ Convert 2x2 Jones matrix to Mueller matrix
    
    Based on Eqn A4.13, Spectroscopic Ellipsometry: Principles and Applications H. Fujiwara
    """
    shape = jones_matrix.shape
    # ordering convention below starts with diagonal terms
    j = xp.concatenate(xp.asanyarray([[jones_matrix[0,0], # cupy requires casting the list to a cupy array
                                       jones_matrix[1,1],
                                       jones_matrix[1,0],
                                       jones_matrix[0,1]]]),
                                       axis=0)
    jc = j.conj()
    e = j * jc

    e0, e1, e2, e3 = e
    j0, j1, j2, j3 = j
    jc0, jc1, jc2, jc3 = jc

    if accel_math._USE_NUMEXPR:
        # construct the Mueller matrix
        # row 1
        M00 = ne.evaluate('0.5*(e0 + e1 + e2 + e3)')
        M01 = ne.evaluate('0.5*(e0 - e1 - e2 + e3)')
        M02 = ne.evaluate('(j0*jc2).real + (j3*jc1).real')
        M03 = ne.evaluate('-(jc0*j2).imag - (jc3*j1).imag')
        # row 2
        M10 = ne.evaluate('0.5*(e0 - e1 + e2 - e3)')
        M11 = ne.evaluate('0.5*(e0 + e1 - e2 - e3)')
        M12 = ne.evaluate('(j0*jc2).real - (j3*jc1).real')
        M13 = ne.evaluate('-(jc0*j2).imag + (jc3*j1).imag')
        # row 3
        M20 = ne.evaluate('(j0*jc3).real + (j2*jc1).real')
        M21 = ne.evaluate('(j0*jc3).real - (j2*jc1).real')
        M22 = ne.evaluate('(j0*jc1).real + (j2*jc3).real')
        M23 = ne.evaluate('-(jc0*j1).imag + (jc2*j3).imag')
        # row 4
        M30 = ne.evaluate('(jc0*j3).imag + (jc2*j1).imag')
        M31 = ne.evaluate('(jc0*j3).imag - (jc2*j1).imag')
        M32 = ne.evaluate('(jc0*j1).imag + (jc2*j3).imag')
        M33 = ne.evaluate('(j0*jc1).real - (j2*jc3).real')
    else:
        # construct the Mueller matrix
        # row 1
        M00 =0.5*(e0 + e1 + e2 + e3)
        M01 =0.5*(e0 - e1 - e2 + e3)
        M02 = (j0*jc2).real + (j3*jc1).real
        M03 =-(jc0*j2).imag - (jc3*j1).imag
        # row 2
        M10 = 0.5*(e0 - e1 + e2 - e3)
        M11 = 0.5*(e0 + e1 - e2 - e3)
        M12 =(j0*jc2).real - (j3*jc1).real
        M13 = -(jc0*j2).imag + (jc3*j1).imag
        # row 3
        M20 = (j0*jc3).real + (j2*jc1).real
        M21 = (j0*jc3).real - (j2*jc1).real
        M22 = (j0*jc1).real + (j2*jc3).real
        M23 = -(jc0*j1).imag + (jc2*j3).imag
        # row 4
        M30 = (jc0*j3).imag + (jc2*j1).imag
        M31 = (jc0*j3).imag - (jc2*j1).imag
        M32 = (jc0*j1).imag + (jc2*j3).imag
        M33 = (j0*jc1).real - (j2*jc3).real


    M = xp.asarray([[M00, M01, M02, M03],
                    [M10, M11, M12, M13],
                    [M20, M21, M22, M23],
                    [M30, M31, M32, M33]])
    
    return M