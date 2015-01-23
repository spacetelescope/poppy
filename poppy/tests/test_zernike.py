import numpy as np
from poppy import poppy_core
from poppy import zernike

def test_zernikes_rms(nterms=10, size=500):
    """Verify RMS(Zernike[n,m]) == 1."""
    assert np.nanstd(zernike.zernike1(1)) == 0.0, "Zernike(j=0) has nonzero RMS"
    for j in range(2, nterms):
        n, m = zernike.noll_indices(j)
        Z = zernike.zernike(n, m, npix=size)
        rms = np.nanstd(Z)  # exclude masked pixels
        assert 1.0 - rms < 0.001, "Zernike(j={}) has RMS value of {}".format(j, rms)


def test_ones_zernikes(nterms=10):
    """Verify the radial scaling function is correctly normalized"""
    rho = np.ones(3)
    theta = np.array([0, 1, 2])
    for j in np.arange(nterms)+1:
        n, m = zernike.noll_indices(j)
        Z = zernike.zernike1(j, rho=rho, theta=theta)
        Rs = zernike.R(n, m, rho)
        print "j=%d\tZ_(%d,%d) [1] = \t %s" % (j, n, m, str(Rs))
        assert Rs[0] == Rs[1] == Rs[2], "Radial polynomial is not radially symmetric"


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

    Zj = zernike.zernike1(testj, npix=npix)
    Zbasis = zernike.zernike_basis(nterms=nterms, npix=npix)
    for idx, Z in enumerate(Zbasis):
        j = idx + 1
        if j == testj or j == 1:
            continue  # discard piston term and self
        prod = Z * Zj
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
