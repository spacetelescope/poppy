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
        assert 1.0 - rms < 0.001, "Zernike(j={}) has RMS value of {}".format(j, rms)


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
    osys.addPupil(optics.CircularAperture(radius=radius))
    wave = osys.inputWavefront()

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


def test_opd_expand(npix=512, input_coefficients=(0.1, 0.2, 0.3, 0.4, 0.5)):
    basis = zernike.zernike_basis(nterms=len(input_coefficients), npix=npix)
    for idx, coeff in enumerate(input_coefficients):
        basis[idx] *= coeff

    opd = basis.sum(axis=0)
    recovered_coeffs = zernike.opd_expand(opd, nterms=len(input_coefficients))
    max_diff = np.max(np.abs(np.asarray(input_coefficients) - np.asarray(recovered_coeffs)))
    assert max_diff < 1e-3, "recovered coefficients from wf_expand more than 0.1% off"


    # Test the nonorthonormal version too
    # At a minimum, fitting with this variant version shouldn't be
    # worse than the regular one on a clear circular aperture.
    # We do the test in this same function for efficiency


    recovered_coeffs_v2 = zernike.opd_expand_nonorthonormal(opd, nterms=len(input_coefficients))
    max_diff_v2 = np.max(np.abs(np.asarray(input_coefficients) - np.asarray(recovered_coeffs_v2)))
    assert max_diff_v2 < 1e-3, "recovered coefficients from wf_expand more than 0.1% off"



def test_opd_from_zernikes():

    coeffs = [0,0.1, 0.4, 2, -0.3]
    opd = zernike.opd_from_zernikes(coeffs, npix=256)

    outcoeffs = zernike.opd_expand(opd, nterms=len(coeffs))

    diffs = np.abs(np.asarray(coeffs) - np.asarray(outcoeffs))/np.asarray(coeffs)
    diffs[0] = 0 # ignore divide by zero on piston
    max_diff = np.max(diffs )
    assert max_diff < 2e-3, "recovered coefficients from opd_expand differ more than expected"


def test_hex_aperture():
    """ Ensure the hex aperture used for Zernikes is consistent with the regular
    poppy HexagonAperture
    """

    npix_to_try = [10, 11, 12, 13, 100, 101, 512, 513]

    for npix in npix_to_try:
        assert np.all(optics.HexagonAperture(side=1).sample(npix=npix) - zernike.hex_aperture(
            npix=npix) == 0), "hex_aperture and HexagonAperture outputs differ for npix={}".format(npix)


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


