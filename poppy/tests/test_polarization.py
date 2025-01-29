from .. import fresnel
from .. import polarized_wavefront
from poppy.poppy_core import _log, PlaneType
import poppy

from poppy.accel_math import xp   # may be numpy, or cupy on GPU
import astropy.units as u

# ---- Fresnel + Stokes (Partial Polarization) ---

def test_fresnel_stokes_linearpolarizer():
    """
    Starting with unpolarized input, propagate a PolarizedFresnelWavefront
    through a system with a linear polarizer and check that the output is
    scaled correctly and linearly polarized after propagation.
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = (1, 0, 0, 0) # unpolarized

    lp_angles = [0, xp.pi/2, xp.pi/4] # horizontal and vertical polarizations
    stokes_true = [(1,1,0,0), (1,-1,0,0), (1,0,1,0)] # linear

    for i in range(len(lp_angles)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        lp = poppy.LinearPolarizer(angle=lp_angles[i])
        osys.add_optic(circ)
        osys.add_optic(lp, distance=500*u.mm)
        
        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # test total intensity = input intensity / 2
        assert xp.isclose(0.5*wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes is linearly polarized
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

def test_fresnel_stokes_qwp():
    """
    Starting with linearly-polarized input, propagate a PolarizedFresnelWavefront
    through a system with a QWP and check that the output is
    scaled correctly and circularly polarized
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = (1, 1, 0, 0) # linear polarization input

    qwp_angles = [xp.pi/4, -xp.pi/4]
    stokes_true = [(1,0,0,1), (1,0,0,-1)] # left- and right-handed circular output

    for i in range(len(qwp_angles)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_optic(circ)
        osys.add_optic(qwp, distance=500*u.mm)

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes is circularly polarized
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

def test_fresnel_stokes_hwp():
    """
    Starting with circularly-polarized input, propagate a PolarizedFresnelWavefront
    through a system with a HWP and check that the output is scaled correctly and
    that the polarization flips handedness
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = [(1, 0, 0, 1), (1, 0, 0,-1)] # circular polarization input
    stokes_true = [(1, 0, 0, -1), (1, 0, 0, 1)] # output should flip circular handedness

    for i in range(len(stokes_true)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate()
        osys.add_optic(circ)
        osys.add_optic(hwp, distance=500*u.mm)

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # conservation of energy
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes has flipped handedness
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

# ---- Fresnel + Vector (Full Polarization) ---

def test_fresnel_vector_qwp():
    """
    Starting with linearly-polarized input, propagate a PolarizedFresnelWavefront
    through a system with a QWP and check that the output is
    scaled correctly and circularly polarized
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_vector = (1, 0) # linear polarization input

    qwp_angles = xp.pi/2 + xp.asarray([xp.pi/4, -xp.pi/4])
    #output_vector =  1/xp.sqrt(2)*xp.array([(1, 1j), (1, -1j)])
    filter_out = [
        poppy.CircularPolarizer(handedness='left'),
        poppy.CircularPolarizer(handedness='right')
        ]

    for i in range(len(qwp_angles)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_optic(circ)
        osys.add_optic(qwp, distance=500*u.mm)
        osys.add_optic(filter_out[i])

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_polarization=input_vector)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

def test_fresnel_vector_hwp():
    """
    Starting with linearly- and circularly-polarized input, propagate a
    PolarizedFresnelWavefront through a system with a HQP and check that the output is
    scaled correctly and in the expected polarization state
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    f = 1/xp.sqrt(2)
    hwp_angle = xp.pi/4
    input_vector = [(1, 0), (f, 1j*f), (f, -1j*f)]
    #output_vector = [(0, 1), (f, -1j*f), (f, 1j*f)]
    filter_out = [
        poppy.LinearPolarizer(angle=xp.pi/2.),
        poppy.CircularPolarizer(handedness='right'),
        poppy.CircularPolarizer(handedness='left')
        ]

    for i in range(len(input_vector)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate(angle=hwp_angle)
        osys.add_optic(circ)
        osys.add_optic(hwp, distance=500*u.mm)
        osys.add_optic(filter_out[i])

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_polarization=input_vector[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # test that all input field converted to field expected output polarization
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

# ---- Fraunhofer + Stokes (Partial Polarization) ---

def test_fraunhofer_stokes_linearpolarizer():
    """
    Starting with unpolarized input, propagate a PolarizedWavefront
    through a system with a linear polarizer and check that the output is
    scaled correctly and linearly polarized after propagation to focal plane.
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = (1, 0, 0, 0) # unpolarized

    lp_angles = [0, xp.pi/2, xp.pi/4] # horizontal and vertical polarizations
    stokes_true = [(1,1,0,0), (1,-1,0,0), (1,0,1,0)] # linear

    for i in range(len(lp_angles)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        lp = poppy.LinearPolarizer(angle=lp_angles[i])
        osys.add_pupil(circ)
        osys.add_pupil(lp)
        osys.add_image()
        
        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # test total output intensity = input intensity / 2
        assert xp.isclose(0.5*wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes is linearly polarized
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

def test_fraunhofer_stokes_qwp():
    """
    Starting with linearly-polarized input, propagate a PolarizedWavefront
    through a system with a QWP and check that the output is
    scaled correctly and circularly polarized at the focal plane
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = (1, 1, 0, 0) # linear polarization input

    qwp_angles = [xp.pi/4, -xp.pi/4]
    stokes_true = [(1,0,0,1), (1,0,0,-1)] # left- and right-handed circular output

    for i in range(len(qwp_angles)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_pupil(circ)
        osys.add_pupil(qwp)
        osys.add_image()

        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes is circularly polarized
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

def test_fraunhofer_stokes_hwp():
    """
    Starting with circularly-polarized input, propagate a PolarizedWavefront
    through a system with a HWP and check that the output is scaled correctly and
    that the polarization flips handedness at the focal plane
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_stokes = [(1, 0, 0, 1), (1, 0, 0,-1)] # circular polarization input
    stokes_true = [(1, 0, 0, -1), (1, 0, 0, 1)] # output should flip circular handedness

    for i in range(len(stokes_true)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate()
        osys.add_pupil(circ)
        osys.add_pupil(hwp)
        osys.add_image()

        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_stokes_vector=input_stokes[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # conservation of energy
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output stokes has flipped handedness
        stokes_out = wfs[-1].stokes_parameters
        assert xp.allclose(xp.sum(stokes_out, axis=(-2,-1)) / wfs[-1].total_intensity, stokes_true[i])

# ---- Fraunhofer + Vector (Full Polarization) ---

def test_fraunhofer_vector_qwp():
    """
    Starting with linearly-polarized input, propagate a PolarizedWavefront
    through a system with a QWP and check that the output is
    scaled correctly and circularly polarized
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    input_vector = (1, 0) # linear polarization input

    qwp_angles = xp.pi/2 + xp.asarray([xp.pi/4, -xp.pi/4])
    #output_vector =  1/xp.sqrt(2)*xp.array([(1, 1j), (1, -1j)])
    filter_out = [
        poppy.CircularPolarizer(handedness='left'),
        poppy.CircularPolarizer(handedness='right')
        ]

    for i in range(len(qwp_angles)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_pupil(circ)
        osys.add_pupil(qwp)
        osys.add_image()
        osys.add_image(filter_out[i])
        
        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_polarization=input_vector)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)


def test_fraunhofer_vector_hwp():
    """
    Starting with linearly- and circularly-polarized input, propagate a
    PolarizedWavefront through a system with a HQP and check that the output is
    scaled correctly and in the expected polarization state
    """
    npix = 128
    D = 10 * u.mm
    wavelen = 1e-6 * u.m
    f = 1/xp.sqrt(2)
    hwp_angle = xp.pi/4
    input_vector = [(1, 0), (f, 1j*f), (f, -1j*f)]
    #output_vector = [(0, 1), (f, -1j*f), (f, 1j*f)]
    filter_out = [
        poppy.LinearPolarizer(angle=xp.pi/2.),
        poppy.CircularPolarizer(handedness='right'),
        poppy.CircularPolarizer(handedness='left')
        ]

    for i in range(len(input_vector)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate(angle=hwp_angle)
        osys.add_pupil(circ)
        osys.add_pupil(hwp)
        osys.add_image()
        osys.add_image(filter_out[i])

        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_polarization=input_vector[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input field converted to expected output
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)