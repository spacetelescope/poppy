import numpy as np
from poppy import poppy_core
from poppy import optics
from poppy import zernike


def test_zernikes_rms(nterms=10, size=500):
    """Verify RMS(Zernike[n,m]) == 1."""
    assert np.nanstd(zernike.zernike1(1)) == 0.0, "Zernike(j=0) has nonzero RMS"
    for j in range(2, nterms):
        n, m = zernike.noll_indices(j)
        z = zernike.zernike(n, m, npix=size)
        rms = np.nanstd(z)  # exclude masked pixels
        assert abs(1.0 - rms) < 0.001, "Zernike(j={}) has RMS value of {}".format(j, rms)


def test_ones_zernikes(nterms=10):
    """Verify the radial scaling function is correctly normalized"""
    rho = np.ones(3)
    for j in np.arange(nterms) + 1:
        n, m = zernike.noll_indices(j)
        rs = zernike.R(n, m, rho)
        print("j=%d\tZ_(%d,%d) [1] = \t %s" % (j, n, m, str(rs)))
        assert rs[0] == rs[1] == rs[2], "Radial polynomial is not radially symmetric"


def test_cached_zernike1(nterms=10):
    radius = 1.1

    osys = poppy_core.OpticalSystem()
    osys.add_pupil(optics.CircularAperture(radius=radius))
    wave = osys.input_wavefront()

    y, x = wave.coordinates()
    rho = np.sqrt(y ** 2 + x ** 2) / radius
    theta = np.arctan2(y, x)

    cached_results = []

    for j in range(1, nterms + 1):
        cached_output = zernike.cached_zernike1(j, wave.shape, wave.pixelscale, radius, outside=0.0)
        cached_results.append(cached_output)
        uncached_output = zernike.zernike1(j, rho=rho, theta=theta, outside=0.0)
        assert np.allclose(cached_output, uncached_output)

    try:
        cached_output[0, 0] = np.nan
        assert False, "Shouldn't be able to assign to a cached output array!"
    except ValueError:
        pass

    # Check that we're getting cached copies
    for j, array_ref in enumerate(cached_results, start=1):
        cached_array_ref = zernike.cached_zernike1(j, wave.shape, wave.pixelscale, radius, outside=0.0)
        assert id(array_ref) == id(cached_array_ref), "cached_zernike1 returned a new object for the same arguments"


def _test_cross_zernikes(testj=4, nterms=10, npix=500):
    """Verify the functions are orthogonal, by taking the
    integrals of a given Zernike times N other ones.

    Parameters :
    --------------
    testj : int
        Index of the Zernike polynomial to test against the others
    nterms : int
        Test that polynomial against those from 1 to this N
    npix : int
        Size of array to use for this test
    """

    zj = zernike.zernike1(testj, npix=npix)
    assert np.sum(np.isfinite(zj)) > 0, "Zernike calculation failure; all NaNs."
    zbasis = zernike.zernike_basis(nterms=nterms, npix=npix)
    for idx, z in enumerate(zbasis):
        j = idx + 1
        if j == testj or j == 1:
            continue  # discard piston term and self
        prod = z * zj
        wg = np.where(np.isfinite(prod))
        cross_sum = np.abs(prod[wg].sum())
        assert cross_sum < 1e-9, (
            "orthogonality failure, Sum[Zernike(j={}) * Zernike(j={})] = {} (> 1e-9)".format(
                j, testj, cross_sum)
        )


def test_cross_zernikes():
    """Verify orthogonality for a subset of Zernikes by taking the integral of
    that Zernike times N other ones.

    Note that the Zernikes are only strictly orthonormal over a
    fully circular aperture evauated analytically. For any discrete
    aperture the orthonormality is only approximate.
    """
    for testj in (2, 3, 4, 5, 6):
        _test_cross_zernikes(testj=testj, nterms=6)


def _test_cross_hexikes(testj=4, nterms=10, npix=500):
    """Verify the functions are orthogonal, by taking the
    integrals of a given Hexike times N other ones.

    This is a helper function for test_cross_hexike.

    Parameters :
    --------------
    testj : int
        Index of the Zernike polynomial to test against the others
    nterms : int
        Test that polynomial against those from 1 to this N
    npix : int
        Size of array to use for this test
    """

    hexike_basis = zernike.hexike_basis(nterms=nterms, npix=npix)
    test_hexike = hexike_basis[testj - 1]
    assert np.sum(np.isfinite(test_hexike)) > 0, "Hexike calculation failure; all NaNs."
    for idx, hexike_array in enumerate(hexike_basis):
        j = idx + 1
        if j == testj or j == 1:
            continue  # discard piston term and self
        prod = hexike_array * test_hexike
        wg = np.where(np.isfinite(prod))
        cross_sum = np.abs(prod[wg].sum())

        # Threshold was originally 1e-9, but we ended up getting 1.19e-9 on some machines (not always)
        # this seems acceptable, so relaxing criteria slightly
        assert cross_sum < 2e-9, (
            "orthogonality failure, Sum[Hexike(j={}) * Hexike(j={})] = {} (> 2e-9)".format(
                j, testj, cross_sum)
        )


def test_cross_hexikes():
    """Verify orthogonality for a subset of Hexikes by taking the integral of
    that Hexike times N other ones.

    Note that the Hexike are only strictly orthonormal over a
    fully hexagonal aperture evauated analytically. For any discrete
    aperture the orthonormality is only approximate.
    """
    for testj in (2, 3, 4, 5, 6):
        _test_cross_hexikes(testj=testj, nterms=6)


def test_arbitrary_basis_rms(nterms=10, size=500):
    """Verify RMS(Zernike[n,m]) == 1."""
    # choose a square aperture for the test
    square_aperture = optics.SquareAperture(size=1.).sample(npix=size,grid_size=1.1)
    square_basis = zernike.arbitrary_basis(square_aperture,nterms=nterms)

    assert np.nanstd(square_basis[0]) == 0.0, "Mode(j=0) has nonzero RMS"
    for j in range(1, nterms):
        rms = np.nanstd(square_basis[j])  # exclude masked pixels
        assert abs(1.0 - rms) < 0.001, "Mode(j={}) has RMS value of {}".format(j, rms)


def _test_cross_arbitrary_basis(testj=4, nterms=10, npix=500):
    """Verify the functions are orthogonal, by taking the
    integrals of a given mode times N other ones.

    This is a helper function for test_cross_arbitrary_basis.

    Parameters :
    --------------
    testj : int
        Index of the Zernike-like polynomial to test against the others
    nterms : int
        Test that polynomial against those from 1 to this N
    npix : int
        Size of array to use for this test
    """
    # choose a square aperture for the test
    square_aperture = optics.SquareAperture(size=1.).sample(npix=npix,grid_size=1.1)
    square_basis = zernike.arbitrary_basis(square_aperture,nterms=nterms)

    test_mode = square_basis[testj - 1]
    assert np.sum(np.isfinite(test_mode)) > 0, "Basis function calculation failure; all NaNs."
    for idx, array in enumerate(square_basis):
        j = idx + 1
        if j == testj or j == 1:
            continue  # discard piston term and self
        prod = array * test_mode
        wg = np.where(np.isfinite(prod))
        cross_sum = np.abs(prod[wg].sum())

        # Threshold was originally 1e-9, but we ended up getting 1.19e-9 on some machines (not always)
        # this seems acceptable, so relaxing criteria slightly
        assert cross_sum < 2e-9, (
            "orthogonality failure, Sum[Mode(j={}) * Mode(j={})] = {} (> 2e-9)".format(
                j, testj, cross_sum)
        )


def test_cross_arbitrary_basis():
    """Verify orthogonality for a subset of basis functions by taking the integral of
    each function times N other ones.

    Note that the Hexike are only strictly orthonormal over a
    fully hexagonal aperture evauated analytically. For any discrete
    aperture the orthonormality is only approximate.
    """
    for testj in (2, 3, 4, 5, 6):
        _test_cross_arbitrary_basis(testj=testj, nterms=6)


def test_decompose_opd(npix=512, input_coefficients=(0.1, 0.2, 0.3, 0.4, 0.5)):
    basis = zernike.zernike_basis(nterms=len(input_coefficients), npix=npix)
    for idx, coeff in enumerate(input_coefficients):
        basis[idx] *= coeff

    opd = basis.sum(axis=0)
    recovered_coeffs = zernike.decompose_opd(opd, nterms=len(input_coefficients))
    max_diff = np.max(np.abs(np.asarray(input_coefficients) - np.asarray(recovered_coeffs)))
    assert max_diff < 1e-3, "recovered coefficients from wf_expand more than 0.1% off"


    # Test the nonorthonormal version too
    # At a minimum, fitting with this variant version shouldn't be
    # worse than the regular one on a clear circular aperture.
    # We do the test in this same function for efficiency


    recovered_coeffs_v2 = zernike.decompose_opd_nonorthonormal_basis(opd, nterms=len(input_coefficients))
    max_diff_v2 = np.max(np.abs(np.asarray(input_coefficients) - np.asarray(recovered_coeffs_v2)))
    assert max_diff_v2 < 1e-3, "recovered coefficients from wf_expand more than 0.1% off"


def test_compose_opd_from_basis():
    coeffs = [0,0.1, 0.4, 2, -0.3]
    opd = zernike.compose_opd_from_basis(coeffs, npix=256)

    outcoeffs = zernike.decompose_opd(opd, nterms=len(coeffs))

    # only compare on indices 1-3 to avoid divide by zero on piston
    diffs = np.abs(np.asarray(coeffs[1:5]) - np.asarray(outcoeffs[1:5]))/np.asarray(coeffs[1:5])
    max_diff = np.max(diffs )
    assert max_diff < 2e-3, "recovered coefficients from opd_expand differ more than expected"


def test_hex_aperture():
    """ Ensure the hex aperture used for Zernikes is consistent with the regular
    poppy HexagonAperture
    """

    npix_to_try = [10, 11, 12, 13, 100, 101, 512, 513]

    for npix in npix_to_try:
        assert np.all(optics.HexagonAperture(side=1).sample(npix=npix, grid_size=2) -
                      zernike.hex_aperture( npix=npix) == 0), \
                      "hex_aperture and HexagonAperture outputs differ for npix={}".format(npix)


def test_zern_name():
    assert zernike.zern_name(3)=='Tilt Y', "Unexpected return value"
    assert zernike.zern_name(11)=='Spherical', "Unexpected return value"
    assert zernike.zern_name(20)=='Pentafoil X', "Unexpected return value"
    assert zernike.zern_name(352)=='Z352', "Unexpected return value"


def test_str_zernike():
    assert zernike.str_zernike(4,0) == 'sqrt(5)* ( 6 r^4  -6 r^2  +1 r^0  ) ', "Unexpected return value"
    assert zernike.str_zernike(5,5) == '\\sqrt{12}* ( 1 r^5  ) * \\cos(5 \\theta)', "Unexpected return value"


def test_zernike_basis_faster():
    bf = zernike.zernike_basis_faster(12, outside=0)
    bs = zernike.zernike_basis(12, outside=0)
    assert np.allclose(bf,bs), "Fast zernike basis calculation doesn't match the slow calculation"


def test_piston_basis(verbose=False):
    """ Test that we can create randomly-pistoned segments, and then
    re-determine the amounts of those pistons.

    This tests both the segment piston baseis, and the segment basis
    decomposition function.
    """

    segment_piston_basis = zernike.Segment_Piston_Basis(rings=2)

    random_pistons = np.random.randn(18)

    pistoned_opd = zernike.compose_opd_from_basis(basis=segment_piston_basis, coeffs=random_pistons, outside=0)
    aperture = segment_piston_basis.aperture()
    #aperture = np.asarray(pistoned_opd != 0, dtype=int)

    for border_pad in [None, 5]:
        results = zernike.decompose_opd_segments(pistoned_opd, basis=segment_piston_basis,
                                                 aperture=aperture, nterms=18, verbose=verbose, ignore_border=border_pad)

        if verbose:
            print(random_pistons)
            print(results)

        assert np.allclose(random_pistons, results)

    return random_pistons, results, pistoned_opd


def test_ptt_basis(verbose=False, plot=False,
                   tiptiltonly=True, pistononly=False,
                  rings=2):
    """ Test that we can create randomly-pistoned, tipped, and tilted segments, and then
    re-determine the amounts of those deviations. """

    segment_ptt_basis = zernike.Segment_PTT_Basis(rings=rings)

    # Make some random aberrations
    random_ptt = np.random.randn(segment_ptt_basis.nsegments*3)
    if tiptiltonly:
        for i in range(segment_ptt_basis.nsegments):
            random_ptt[i*3] = 0
    elif pistononly:
        for i in range(segment_ptt_basis.nsegments):
            random_ptt[i*3+1] = 0
            random_ptt[i*3+2] = 0
    else:
        # make the pistons small compared to the tips & tilts
        for i in range(segment_ptt_basis.nsegments):
            random_ptt[i*3] *= 1e-3

    # Generate an OPD with those aberrations
    ptted_opd = zernike.compose_opd_from_basis(basis=segment_ptt_basis, coeffs=random_ptt, outside=0)

    # Perform a fit to measure them
    results = zernike.decompose_opd_segments(ptted_opd,
                                             basis=segment_ptt_basis,
                                             aperture=segment_ptt_basis.aperture(),
                                             nterms=segment_ptt_basis.nsegments*3,
                                             verbose=verbose)

    # Generate another OPD to show the measurements
    ptted_v2 = zernike.compose_opd_from_basis(basis=segment_ptt_basis, coeffs=results, outside=0)

    if verbose:
        print(random_ptt)
        print(results)
    if plot:
        plt.subplot(121)
        ax = plt.imshow(ptted_opd)
        plt.title("Randomly generated OPD")
        plt.subplot(122)
        ax2 = plt.imshow(ptted_v2, norm=ax.norm)
        plt.title("Reproduced from fit coefficients")

    # adjust tolerances for the ones that are precisely zero - allow larger atol since rtol doesn't help there.
    wz = np.where(random_ptt ==0)
    wnz = np.where(random_ptt !=0)

    assert np.allclose(random_ptt[wnz], results[wnz])
    assert np.allclose(random_ptt[wz], results[wz], atol=1e-6)

    return random_ptt, results, ptted_opd, ptted_v2


def test_back_compatible_aliases():
    """ Test existence of back-compatibility alias names for several functions
    The names of these functions change in poppy 1.0, but we keep the older versions as synonyms for back-compatibility, at least for now.
    These can be removed in a future version of poppy.
    """
    assert zernike.opd_expand is zernike.decompose_opd, "Missing back compatibility alias"
    assert zernike.opd_expand_segments is zernike.decompose_opd_segments, "Missing back compatibility alias"
    assert zernike.opd_expand_nonorthonormal is zernike.decompose_opd_nonorthonormal_basis, "Missing back compatibility alias"
    assert zernike.opd_from_zernikes is zernike.compose_opd_from_basis, "Missing back compatibility alias"
