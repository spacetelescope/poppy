'''
Tests to write:

Vector WF
* for both PolarizedFresnelWavefront and PolarizedWavefront, input stokes/polarized and then propagate through osys with...
    * LP
    * QWP
    * HWP
(and check for each that output is as expected)

'''
from .. import fresnel
from .. import polarized_wavefront
from poppy.poppy_core import _log, PlaneType
import poppy

from poppy.accel_math import xp   # may be numpy, or cupy on GPU
import astropy.units as u


def compare_jones_vectors(j1, j2):
    """ Compare a pair of Jones vectors"""
    # relative angle between jones vectors (which is meaningless)
    dth = xp.dot(j1, xp.conj(j2) ) / xp.sqrt(xp.sum(xp.abs(j1)**2)) /  xp.sqrt(xp.sum(xp.abs(j2)**2))
    # remove it from one of the vectors
    j1 = j1 * dth.conj()

    # now subtract and compare abs(delta) to 0
    dj = j1 - j2
    dj_abs = xp.sqrt(xp.sum(xp.abs(dj)**2))
    print(j1, j2)
    assert xp.allclose(dj_abs, 0)

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
    output_vector =  1/xp.sqrt(2)*xp.array([(1, 1j), (1, -1j)])

    for i in range(len(qwp_angles)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_optic(circ)
        osys.add_optic(qwp, distance=500*u.mm)

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_polarization=input_vector)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output vector is circularly polarized
        vector_out = xp.sum(wfs[-1].wavefront, axis=(-2,-1))
        norm = xp.abs(xp.sum(wfs[0].wavefront)) # normalize to input amplitude
        compare_jones_vectors(vector_out / norm, output_vector[i])

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
    output_vector = [(0, 1), (f, -1j*f), (f, 1j*f)]

    for i in range(len(input_vector)):
        osys = fresnel.FresnelOpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate(angle=hwp_angle)
        osys.add_optic(circ)
        osys.add_optic(hwp, distance=500*u.mm)

        wf = polarized_wavefront.PolarizedFresnelWavefront(D, wavelength=wavelen,npix=npix, input_polarization=input_vector[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # conservation of energy
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output vector is polarized as expected
        vector_out = xp.sum(wfs[-1].wavefront, axis=(-2,-1))
        norm = xp.abs(xp.sum(wfs[0].wavefront)) # normalize to input amplitude
        compare_jones_vectors(vector_out / norm, output_vector[i])

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
    output_vector =  1/xp.sqrt(2)*xp.array([(1, 1j), (1, -1j)])

    for i in range(len(qwp_angles)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        qwp = poppy.QuarterWavePlate(angle=qwp_angles[i])
        osys.add_pupil(circ)
        osys.add_pupil(qwp)
        osys.add_image()

        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_polarization=input_vector)

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # all input linear should be converted to output circular
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output vector is circularly polarized
        vector_out = xp.sum(wfs[-1].wavefront, axis=(-2,-1))
        norm = xp.abs(xp.sum(wfs[0].wavefront)) # normalize to input amplitude
        compare_jones_vectors(vector_out / norm, output_vector[i])

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
    output_vector = [(0, 1), (f, -1j*f), (f, 1j*f)]

    for i in range(len(input_vector)):
        osys = poppy.OpticalSystem(npix=npix)
        circ = poppy.CircularAperture(radius=D)
        hwp = poppy.HalfWavePlate(angle=hwp_angle)
        osys.add_pupil(circ)
        osys.add_pupil(hwp)
        osys.add_image()

        wf = polarized_wavefront.PolarizedWavefront(diam=4*D, wavelength=wavelen,npix=npix, input_polarization=input_vector[i])

        psf, wfs = osys.calc_psf(inwave=wf, return_intermediates=True)

        # conservation of energy
        assert xp.isclose(wfs[0].total_intensity, wfs[-1].total_intensity)

        # test that output vector is polarized as expected
        vector_out = xp.sum(wfs[-1].wavefront, axis=(-2,-1))
        norm = xp.abs(xp.sum(wfs[0].wavefront)) # normalize to input amplitude
        compare_jones_vectors(vector_out / norm, output_vector[i])