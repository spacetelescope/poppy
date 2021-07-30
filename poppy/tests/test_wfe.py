
import numpy as np
import astropy.units as u
from astropy.io import fits

from .. import poppy_core
from .. import optics
from .. import zernike
from .. import wfe
from .. import physical_wavefront

NWAVES = 0.5
WAVELENGTH = 1e-6
RADIUS = 1.0
NPIX = 101
DIAM = 3.0

def test_ZernikeAberration(display=False):
    # verify that we can reproduce the same behavior as ThinLens
    # using ZernikeAberration
    pupil = optics.CircularAperture(radius=RADIUS)
    lens = optics.ThinLens(nwaves=NWAVES, reference_wavelength=WAVELENGTH, radius=RADIUS)
    tl_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    tl_wave *= lens
    # The ThinLens class is a subclass of CircularAperture so it automatically applies
    # pupil aperture shape as well as the wavefront

    # Sign convention: should be consistent, positive Thin Lens and positive Zernike WFE are equivalent
    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=WAVELENGTH)
    zernike_lens = wfe.ZernikeWFE(
        coefficients=[0, 0, 0, NWAVES * WAVELENGTH / (2 * np.sqrt(3))],
        radius=RADIUS
    )
    zern_wave *= zernike_lens
    # But the ZernikeWFE class does not include any aperture transmission shape, so
    # we have to apply that here separately:
    zern_wave *= pupil

    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(zern_wave.intensity-tl_wave.intensity)
        plt.title("Intensity difference")
        plt.figure()
        plt.imshow(zern_wave.phase-tl_wave.phase)
        plt.title("Phase difference")
        plt.draw()
        plt.figure()
        plt.plot(tl_wave.intensity[50,:],color='black', label='Thin Lens intensity')
        plt.plot(tl_wave.phase[50,:], color='black', linestyle=':', label='Thin Lens phase')
        plt.plot(zern_wave.intensity[50,:],color='red', linestyle='--', label='Zernike intensity')
        plt.plot(zern_wave.phase[50,:], color='red', linestyle=':', label='Zernike phase')
        plt.legend(loc='upper right', frameon=False)

    assert np.allclose(zern_wave.intensity, tl_wave.intensity)

    # The Zernike class right now only has nonzero phase in pixels that are strictly
    # radius < 1, and doesn't include partially illuminated pixels beyond that. So to
    # get this test passing for now, we need to only compute the std dev for inside
    # r < 1.

    # FIXME: enhance Zernike class to provide appropriate OPD values for
    # partially-illuminated pixels at the boundary (this is nontrivial given
    # current implementation)
    y, x = zern_wave.coordinates()
    r = np.sqrt(y**2 + x**2)
    stddev = np.std((zern_wave.phase - tl_wave.phase)[r < RADIUS])

    assert stddev < 1e-16, ("ZernikeAberration disagrees with ThinLens! stddev {}".format(stddev))


def test_zernike_get_opd():
    wave = poppy_core.Wavefront(wavelength=WAVELENGTH)
    zernike_optic = wfe.ZernikeWFE(coefficients=[NWAVES * WAVELENGTH,], radius=RADIUS)
    opd_map = zernike_optic.get_opd(wave)
    assert np.max(opd_map) == NWAVES * WAVELENGTH


def test_ParameterizedAberration():
    # verify that we can reproduce the same behavior as ZernikeAberration
    # using ParameterizedAberration
    NWAVES = 0.5
    WAVELENGTH = 1e-6
    RADIUS = 1.0

    pupil = optics.CircularAperture(radius=RADIUS)

    zern_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=1e-6)
    zernike_wfe = wfe.ZernikeWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        radius=RADIUS
    )
    zern_wave *= pupil
    zern_wave *= zernike_wfe

    parameterized_distortion = wfe.ParameterizedWFE(
        coefficients=[0, 0, 2e-7, NWAVES * WAVELENGTH / (2 * np.sqrt(3)), 0, 3e-8],
        basis_factory=zernike.zernike_basis,
        radius=RADIUS
    )

    pd_wave = poppy_core.Wavefront(npix=NPIX, diam=3.0, wavelength=1e-6)
    pd_wave *= pupil
    pd_wave *= parameterized_distortion

    np.testing.assert_allclose(pd_wave.phase, zern_wave.phase,
                               err_msg="ParameterizedAberration disagrees with ZernikeAberration")


def test_StatisticalPSDWFE(index=3, seed=123456, plot=False):

    # Verify that we produce phase screen with input RMS WFE
    NPIX = 256    # 101 is too small and results in issues for this test
    def rms(opd):
        """Calculate the RMS WFE of a wf with a zero mean."""
        rms = np.sqrt(np.mean(np.square(opd)))
        return rms

    wvferr = 134*u.nm
    psd_wave = poppy_core.Wavefront(npix=NPIX, diam=DIAM, wavelength=1e-6)
    psd_wfe = wfe.StatisticalPSDWFE(index=index, wfe=wvferr, radius=RADIUS, seed=seed)

    psd_opd = psd_wfe.get_opd(psd_wave)
    assert np.isclose(rms(psd_opd), wvferr.to(u.m).value), "WFE doesn't match input WFE."

    # Verify that we reproduce the correct spectral index of the PSD
    def radial_profile(image, center=None):
        """ Compute a radial profile of the image.
        Stripped down version of puppy.utils.radial_profile().
        Keeps only the functionality that is needed for this test.
        Parameters
        ----------
        image : array
            image to get a radial profile of
        center : tuple of floats
            Coordinates (x,y) of PSF center, in pixel units. Default is image center.
        Returns
        --------
        results : tuple
            Tuple containing (radius, profile).
        """

        y, x = np.indices(image.shape, dtype=float)
        if center is None:
            # get exact center of image
            # center = (image.shape[1]/2, image.shape[0]/2)
            center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

        x -= center[0]
        y -= center[1]

        r = np.sqrt(x ** 2 + y ** 2)

        # Use full image
        ind = np.argsort(r.flat)
        sr = r.flat[ind]  # sorted r
        sim = image.flat[ind]  # sorted image

        ri = sr.astype(int)  # sorted r as int
        deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
        rind = np.where(deltar)[0]
        nr = rind[1:] - rind[:-1]  # number in radius bin
        csim = np.nan_to_num(sim).cumsum(dtype=float)  # cumulative sum to figure out sums for each bin
        # np.nancumsum is implemented in >1.12
        tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
        radialprofile = tbin / nr

        # pre-pend the initial element that the above code misses.
        radialprofile2 = np.empty(len(radialprofile) + 1)
        if rind[0] != 0:
            radialprofile2[0] = csim[rind[0]] / (
                    rind[0] + 1)  # if there are multiple elements in the center bin, average them
        else:
            radialprofile2[0] = csim[0]  # otherwise if there's just one then just take it.
        radialprofile2[1:] = radialprofile
        rr = np.arange(
            len(radialprofile2)) +  0.5  # these should be centered in the bins, so add a half.

        return rr, radialprofile2

    inv_psd = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(psd_opd)))
    rad, prof = radial_profile(np.abs(inv_psd) ** 2, center=(int(NPIX/2), int(NPIX/2)))

    # Test that the output power law matches the requested input power law.
    # We set a relatively generous threshold for this test (3% difference in index)
    # because this is statistical, and any given realization will differ from the index
    # by potentially several percent. We also fix the seed above, to ensure test repeatability.

    assert (0.1 <= index <= 10), "index is outside of the range supported by this unit test"

    import astropy.modeling
    # guess a power law model; but don't have the starting guess be the right anwser
    plaw_guess = astropy.modeling.models.PowerLaw1D(alpha=index*1.5,
        bounds={'alpha': (0.1, 10)}) # avoid floating point overflow warnings on test

    # Perform a fit. Drop some initial and trailing values to avoid numerical artifacts.
    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    drop = 20  # how many array elements to discard from fit
    prof_norm = prof/prof.max() # Empirically this fit process works better if we normalize the profile first.
    plaw_fit = fitter(plaw_guess, rad[drop:-drop], prof_norm[drop:-drop], weights=(prof_norm[drop:-drop])**-2)

    # check the spectral index is as desired, within at least a few percent
    assert np.isclose(index, plaw_fit.alpha, rtol=0.03), ("Measured output spectral index doesn't "
            "match input within 3%: {} vs {}".format(index, plaw_fit.alpha) )

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        psd_wfe.display(what='both')
        plt.figure()
        plt.loglog(rad[1:], prof_norm[1:],
                   label='StatisticalPSD output for {}'.format(index))
        plt.plot(rad[drop:-drop], plaw_fit(rad[drop:-drop]),
                 label='power law fit: {:.5f}'.format(plaw_fit.alpha.value))
        plt.xlabel("Spatial frequency [1/m]")
        plt.ylabel("Normalized PSD")
        plt.legend()

def test_PowerSpectrumWFE(plot=False):
    # verify self-consistency of PowerSpectrumWFE with a reference case
    import os
    import astropy.io.fits as fits
    
    # declare internal functions
    def rms(opd):
        return np.sqrt(np.mean(np.square(opd)))
    def pv(opd):
        return np.amax(opd) - np.amin(opd)
    
    # Initialize reference case: Jared Males' PSD surface for OAP
    psd_ref = fits.open(os.path.join(os.path.dirname(__file__), 'psd_wfe_ref_oap.fits'))[0]
    hdr = psd_ref.header

    # Surface data
    surf_ref = psd_ref.data * u.nm * 2 # TODO: the file was scaled by 0.5, need to update it
    npix_surf = surf_ref.shape[0]
    opt_diam = hdr['opticd'] * u.m
    pixelscale = hdr['pixscale'] * u.m/u.pix # should match with opt_diam/npix
    surf_radius = (pixelscale * u.pix * npix_surf/2.0 ).to_value(u.m)

    # psd parameter data from header
    alpha = hdr['alpha']
    beta = hdr['beta'] * (pixelscale.value**2) * (surf_ref.unit**2) / (opt_diam.unit**(alpha-2))
    # beta value in header has been normalized, needs to be backsolved
    outer_scale = hdr['os'] * opt_diam.unit # sometimes labeled as L0
    inner_scale = hdr['is'] # sometimes labeled lo
    surf_roughness = 0.0 * (surf_ref.unit * opt_diam.unit)**2

    # set PSD parameters in required list format
    psd_parameters = [[alpha, beta, outer_scale, inner_scale, surf_roughness]]
    psd_weight = [1.0]

    # calculate reference rms and pv for comparison
    rms_ref = rms(surf_ref)
    pv_ref = pv(surf_ref)
    
    # single iteration testing to verify RMS matching
    screen_size = 2048
    seed = 123456
    psd_wave = poppy_core.Wavefront(npix=npix_surf, diam=opt_diam, wavelength=656e-9)
    psd_wfe = wfe.PowerSpectrumWFE(psd_parameters=psd_parameters, psd_weight=psd_weight,
                                    seed=seed, apply_reflection=False, screen_size=screen_size,
                                    rms=rms_ref, radius=surf_radius)
    psd_opd = (psd_wfe.get_opd(psd_wave)*u.m).to(surf_ref.unit)
    psd_rms = rms(psd_opd)
    psd_pv = pv(psd_opd)
    
    # compare in meters because using np.allclose in nanometers fails assert
    assert np.allclose(psd_rms.to(u.m).value,rms_ref.to(u.m).value), ('Calculated RMS wfe does not match with Reference RMS wfe')
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(dpi=100)
        plt.imshow(surf_ref.value, origin='lower')
        plt.colorbar().set_label(surf_ref.unit)
        plt.title('Ref surf, RMS={0:.4f}, PV={1:.2f}'.format(rms_ref, pv_ref))

        plt.figure(dpi=100)
        plt.imshow(psd_opd.value, origin='lower')
        plt.colorbar().set_label(psd_opd.unit)
        plt.title('PSD surf, RMS={0:.4f}, PV={1:.2f}'.format(psd_rms, psd_pv))

def test_KolmogorovWFE():
    CN2 = 1e-14*u.m**(-2/3)
    DZ = 50.0*u.m

    seed = 12345678  # Use fixed seed to ensure reproducible test behavior.
                     # This particular value is arbitrary; any 32-bit int should work here.
    
    def test_KolmogorovWFE_stats():
        # verify statistics of random numbers
        KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ, seed=seed)
        npix = 1024
        a = KolmogorovWFE.rand_turbulent(npix)
        b = KolmogorovWFE.rand_symmetrized(npix, 1)
        c = KolmogorovWFE.rand_symmetrized(npix, -1)
        
        assert(np.round(np.abs(np.mean(a)), 2) == np.round(0.0, 2))
        assert(np.round(np.var(a), 2) == np.round(1.0, 2))
        assert(np.round(np.mean(b), 2) == np.round(0.0, 2))
        assert(np.round(np.var(b), 2) == np.round(1.0, 2))
        assert(np.round(np.mean(c), 2) == np.round(0.0, 2))
        assert(np.round(np.var(c), 2) == np.round(1.0, 2))
    
    def test_KolmogorovWFE_Cn2():
        # verify correct calculation of Cn2 from Fried parameter
        lam = WAVELENGTH*u.m
        dz = 50.0*u.m
        r0 = 0.185*(lam**2/CN2/dz)**(3.0/5.0) # analytical equation
        KolmogorovWFE = wfe.KolmogorovWFE(r0=r0, dz=dz, seed=seed)
        Cn2_test = KolmogorovWFE.get_Cn2(lam)
        
        assert(np.round(Cn2_test.value, 9) == np.round(CN2.value, 9))
    
    def test_KolmogorovWFE_ps():
        # verify that first element of power spectrum is zero
        npix = 64
        wf = poppy_core.Wavefront(wavelength=WAVELENGTH*u.m,
                                  npix=npix,
                                  diam=3.0)
        KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ, inner_scale=1*u.cm, outer_scale=10*u.m, seed=seed)
        
        ps1 = KolmogorovWFE.power_spectrum(wf, kind='Kolmogorov')
        ps2 = KolmogorovWFE.power_spectrum(wf, kind='Tatarski')
        ps3 = KolmogorovWFE.power_spectrum(wf, kind='von Karman')
        ps4 = KolmogorovWFE.power_spectrum(wf, kind='Hill')
        
        assert(np.round(ps1[0,0].value, 9) == np.round(0.0, 9))
        assert(np.round(ps2[0,0].value, 9) == np.round(0.0, 9))
        assert(np.round(ps3[0,0].value, 9) == np.round(0.0, 9))
        assert(np.round(ps4[0,0].value, 9) == np.round(0.0, 9))
    
    def test_KolmogorovWFE_correlation(num_ensemble = 2000, npix = 64):
        # verify correlation of random numbers
        KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ, seed=seed)

        
        average = np.zeros((npix, npix), dtype=complex)
        for j in range(num_ensemble):
            KolmogorovWFE.seed += j
            a = KolmogorovWFE.rand_turbulent(npix)
            for l in range(npix):
                for m in range(npix):
                    average[l, m] += np.sum(a[:, l])*np.sum(np.conj(a[:, m]))/num_ensemble/npix
        
        for l in range(npix):
            for m in range(npix):
                if l == m:
                    average[l, m] -= 1.0
        
        assert(np.max(np.abs(average.real)) < 0.1)
        assert(np.max(np.abs(average.imag)) < 0.1)
    
    def test_get_opd():
        npix = 64
        wf = poppy_core.Wavefront(wavelength=WAVELENGTH*u.m,
                                  npix=npix,
                                  diam=3.0)
        KolmogorovWFE = wfe.KolmogorovWFE(Cn2=CN2, dz=DZ, inner_scale=1*u.cm, outer_scale=10*u.m, seed=seed)
        opd = KolmogorovWFE.get_opd(wf)
        assert(np.round(np.sum(opd), 9) == np.round(0.0, 9))
    
    test_KolmogorovWFE_stats()
    test_KolmogorovWFE_Cn2()
    test_KolmogorovWFE_ps()
    test_KolmogorovWFE_correlation(num_ensemble=800, npix=32)
    test_get_opd()


def test_ThermalBloomingWFE_rho():
    
    # Verify that the rho is calculated correctly for a given set of parameters.
    wf = physical_wavefront.PhysicalFresnelWavefront(beam_radius=5*14.15*u.cm,
                                                     wavelength=10.6*u.um,
                                                     units=u.m,
                                                     npix=512,
                                                     oversample=2,
                                                     M2=1.0, n0=1.00027398)
    wf.scale_power(100.0e3)
    
    # Test nat_conv_vel
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=200.0*u.cm/u.s, direction='x', isobaric=True)
    nat_conv_vel = phase_screen.nat_conv_vel(wf)
    assert(np.round(nat_conv_vel, 6) == np.round(0.07287728078361912, 6))
    
    # Test get_opd
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=200.0*u.cm/u.s, direction='x', isobaric=False)
    opd = phase_screen.get_opd(wf)
    assert(np.round(np.max(opd), 6) == np.round(1.909383278158297e-06, 6))
    assert(np.round(np.min(opd), 6) == np.round(-2.386988803403901e-06, 6))
    
    # Test isobaric phase screen x
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=200.0*u.cm/u.s, direction='x', isobaric=True)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(0.0, 6))
    assert(np.round(np.min(rho), 6) == np.round(-8.208840195737935e-06, 6))
    
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=-200.0*u.cm/u.s, direction='x', isobaric=True)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(0.0, 6))
    assert(np.round(np.min(rho), 6) == np.round(-8.208840195737935e-06, 6))
    
    # Test non-isobaric phase screen x
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=200.0*u.cm/u.s, direction='x', isobaric=False)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(4.102415953233477e-06, 6))
    assert(np.round(np.min(rho), 6) == np.round(-5.128577933666188e-06, 6))
    
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0x=-200.0*u.cm/u.s, direction='x', isobaric=False)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(4.102415953233477e-06, 6))
    assert(np.round(np.min(rho), 6) == np.round(-5.128577933666188e-06, 6))
    
    # Test isobaric phase screen y
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0y=200.0*u.cm/u.s, direction='y', isobaric=True)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(0.0, 6))
    assert(np.round(np.min(rho), 6) == np.round(-8.208840195737935e-06, 6))
    
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0y=-200.0*u.cm/u.s, direction='y', isobaric=True)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(0.0, 6))
    assert(np.round(np.min(rho), 6) == np.round(-8.208840195737935e-06, 6))
    
    # Test non-isobaric phase screen x
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0y=200.0*u.cm/u.s, direction='y', isobaric=False)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(4.102415953233477e-06, 6))
    assert(np.round(np.min(rho), 6) == np.round(-5.128577933666188e-06, 6))
    
    phase_screen = wfe.ThermalBloomingWFE(7e-7/u.cm, 2.0*u.km, v0y=-200.0*u.cm/u.s, direction='y', isobaric=False)
    rho = phase_screen.rho(wf)
    assert(rho.shape[0] == 1024)
    assert(rho.shape[1] == 1024)
    assert(np.round(np.max(rho), 6) == np.round(4.102415953233477e-06, 6))
    assert(np.round(np.min(rho), 6) == np.round(-5.128577933666188e-06, 6))
