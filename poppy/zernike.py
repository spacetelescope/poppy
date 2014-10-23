from __future__ import division

"""
Zernike & Related Polynomials

This module implements several sets of orthonormal polynomials for
measuring and modeling wavefronts:

    * the classical Zernike polynomials, which are orthonormal over the unit circle.
    * 'Hexikes', orthonormal over the unit hexagon
    * 'jwexikes', a custom set orthonormal over a numerically supplied JWST pupil.
        (or other generalized pupil)

For definitions of Zernikes and a basic introduction to why they are a useful way to
parametrize data, see e.g.
    Hardy's 'Adaptive Optics for Astronomical Telescopes' section 3.5.1
    or even just the Wikipedia page is pretty decent.

For definition of the hexagon and JW pupil polynomials, a good reference to the
Gram-Schmidt orthonormalization process as applied to this case is
    Mahajan and Dai, 2006. Optics Letters Vol 31, 16, p 2462:
"""

import os
from math import factorial

import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

from astropy.io import fits

import logging

_log = logging.getLogger('zernike')
_log.setLevel(logging.INFO)
_log.addHandler(logging.NullHandler())

_ZCACHE = {}


def _is_odd(number):
    return number & 1


def str_zernike(n_, m_):
    """Return analytic expression for a given Zernike in LaTeX syntax"""
    m = int(np.abs(m_))
    n = int(np.abs(n_))

    terms = []
    for k in range(int((n - m) / 2) + 1):
        coef = ((-1) ** k * factorial(n - k) /
                (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
        if coef != 0:
            formatcode = "{0:d}" if k == 0 else "{0:+d}"
            terms.append((formatcode + " r^{1:d} ").format(int(coef), n - 2 * k))

    outstr = " ".join(terms)

    if m_ == 0:
        if n == 0:
            return "1"
        else:
            return "sqrt(%d)* ( %s ) " % (n + 1, outstr)
    elif m_ > 0:
        return "\sqrt{%d}* ( %s ) * \\cos(%d \\theta)" % (2 * (n + 1), outstr, m)
    else:
        return "\sqrt{%d}* ( %s ) * \\sin(%d \\theta)" % (2 * (n + 1), outstr, m)


def R(n_, m_, rho):
    """
    Zernike radial polynomial
    """

    m = int(np.abs(m_))
    n = int(np.abs(n_))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
            output += coef * rho ** (n - 2 * k)
        return output


def zernike(n, m, npix=100, r=None, theta=None, mask_outside=True, outside=np.nan):
    """ Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:
     zernike(n,m, npix)       where npix specifies a pupil diameter in pixels.
                               The returned pupil will be a circular aperture
                               with this diameter, embedded in a square array
                               of size npix*npix.
     zernike(n,m, r,theta)    Which explicitly provides the desired pupil coordinates
                               as arrays r and theta. These need not be regular or contiguous.


    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix: int
        Desired diameter for circular pupil. Only used if r and theta are not provided.
    r, theta : array_like
        Image plane coordinates. rho should be in 0<rho<1, theta should be in radians
    mask_outside : bool
        Mask out the region beyond radius 1? Default True.
    outside : float
        Value for pixels outside the circular aperture. Default is NaN, but you may also
        find it useful for this to be zero sometimes.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """

    _log.debug("Zernike(%d, %d)" % (m, n))

    if theta is None:

        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        r = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)

        aperture = np.ones(r.shape)
        if mask_outside:
            aperture[np.where(r > 1)] = outside
    else:
        if r is None:
            raise ValueError("If you provide a theta input array, you must also provide an array "
                             "r with the corresponding radii for each point.")
        # if user explicitly provides r and theta, assume they are handling the aperture.
        aperture = 1  # (for all r, theta)

    if m == 0:
        if n == 0:
            return np.ones(r.shape) * aperture
        else:
            return sqrt(n + 1) * R(n, m, r) * aperture
    elif m > 0:
        return (sqrt(2) * sqrt(n + 1)) * R(n, m, r) * np.cos(np.abs(m) * theta) * aperture
    else:
        return (sqrt(2) * sqrt(n + 1)) * R(n, m, r) * np.sin(np.abs(m) * theta) * aperture


def zernike1(j, return_indices=False, **kwargs):
    """ Return the Zernike polynomial Z_j for pupil points {r,theta}.

    For this function the desired Zernike is specified by a single index j.
    See zernike for an equivalent function in which the polynomials are
    ordered by two parameters m and n.

    Note that there are multiple contradictory conventions for labeling Zernikes
    with one single index. We follow that of Noll et al. JOSA 1976.

    Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976
    rho, theta : numpy arrays
        Image plane coordinates
    return_indices: bool
        Should this function also return (n,m)?

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta)
    n, m : int (optional)
        the n and m parameters equivalent to the supplied j.
    """
    n, m = noll_indices(j)
    if return_indices:
        return zernike(n, m, **kwargs), n, m
    else:
        return zernike(n, m, **kwargs)


def zernike_list(nterms=15, npix=512, **kwargs):
    """ Return a list of Zernicke terms from 1 to N
    each as a 2D array showing the value at each point.

    Parameters
    -----------
    nterms : int
        Number of Zernike terms to return
    npix : int
        Size of arrays on which to compute the Zernike polynomials
    """
    if ('Zern', nterms, npix) in _ZCACHE.keys():
        return _ZCACHE[('Zern', nterms, npix)]
    else:
        Z = [zernike1(j + 1, return_indices=False, npix=npix, **kwargs) for j in range(nterms)]
        _ZCACHE[('Zern', nterms, npix)] = Z
        return Z


def noll_indices(j):
    """ Convert from 1-D to 2-D indexing for zernikes or hexikes.

      Parameters
    ----------
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    """

    if j < 1:
        raise ValueError("Zernike index j must be a postitive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    _log.debug("J=%d:\t(%d, %d)" % (j, n, m))
    return n, m


def sum_zernikes(coeffs, npix=500):
    """Compute the sum of some series of zernikes

    Parameters
    ------------
    coeffs : list
        Coefficients for some number N of Zernike polynomials
    npix : int
        Size of array on which to evaluate the Zernike polynomials

    Returns the sum over i of coeffs[i] * Zernike[i]
    """
    out = np.zeros((npix, npix))
    for j in range(len(coeffs)):
        out += zernike1(j + 1, npix=npix) * coeffs[j]
    return out


def zern_name(i):
    """ Return a human-readable text name corresponding to some Zernike term

    Only works up to term 22, i.e. 5th order spherical aberration.
    """
    names = ['Null', 'Piston', 'Tilt X', 'Tilt Y', 'Focus',
             'Astigmatism 45', 'Astigmatism 0',
             'Coma Y', 'Coma X',
             'Trefoil Y', 'Trefoil X',
             'Spherical', '2nd Astig 0', '2nd Astig 45',
             'Tetrafoil 0', 'Tetrafoil 22.5',
             '2nd coma X', '2nd coma Y', '3rd Astig X', '3rd Astig Y',
             'Pentafoil X', 'Pentafoil Y', '5th order spherical']

    if i < len(names):
        return names[i]
    else:
        return "Z%d" % i


# --------------------------------------------------------------------------------
# Hexikes

def hexike():
    """ Zernike-like orthonormal polynomials defined over a hexagonal pupil

    See Mahajan and Dai, 2006. Optics Letters Vol 31, 16, p 2462:
        http://www.opticsinfobase.org/ol/abstract.cfm?uri=ol-31-16-2462
        doi: 0.1364/OL.31.002462

    """
    pass


def hexike_from_table():
    """ See tabulated matrix coefficients on Page 2 of Mahajan article """
    #TODO:jlong: make this a non-writable (constant) array at the module level
    M = np.matrix(np.zeros(11, 11))
    M[1, 1] = 1
    M[2, 2] = sqrt(6 / 5)
    M[3, 3] = M[2, 2]
    M[4, 1] = sqrt(5 / 43)
    M[4, 4] = 2 * sqrt(15 / 43)
    M[5, 5] = sqrt(10 / 7)
    M[6, 6] = M[5, 5]
    M[7, 3] = 16 * sqrt(14 / 11055)
    M[8, 2] = M[7, 3]
    M[7, 7] = 10 * sqrt(35 / 2211)
    M[8, 8] = M[7, 7]
    M[9, 9] = 2 / 3 * sqrt(5)
    M[10, 10] = 2 * sqrt(35 / 103)
    M[11, 1] = 521 * sqrt(172205)
    M[11, 4] = 88 * sqrt(15 / 214441)
    M[11, 11] = 14 * sqrt(43 / 4987)

    return M


def hex_aperture(npix=500, vertical=False):
    """ Return an aperture function for a hexagon.

    Note that the flat sides are aligned with the X direction by default.
    This is appropriate for the individual hex PMSA segments in JWST.

    Parameters
    -----------
    npix : int
        size of array to return
    vertical : bool, optional, default False
        Make flat sides parallel to the Y axis instead of the default X.
    """

    x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
    y = x
    xx, yy = np.meshgrid(x, y)
    absy = np.abs(yy)

    aperture = np.zeros((npix, npix))
    w_rect = np.where((np.abs(xx) <= 0.5) & (np.abs(yy) <= sqrt(3) / 2))
    w_left_tri = np.where((xx <= -0.5) & (xx >= -1) & (absy <= (xx + 1) * sqrt(3)))
    w_right_tri = np.where((xx >= 0.5) & (xx <= 1) & (absy <= (1 - xx) * sqrt(3)))
    aperture[w_rect] = 1
    aperture[w_left_tri] = 1
    aperture[w_right_tri] = 1

    if vertical:
        return aperture.transpose()
    else:
        return aperture


def hexike_list(nterms=11, npix=500):
    """ Return a list of hexike polynomials 1-N following the
    method of Mahajan and Dai 2006 """

    shape = (npix, npix)

    aperture = hex_aperture(npix)
    A = aperture.sum()

    # precompute zernikes
    Z = [np.zeros(shape)]
    for j in range(nterms + 1):
        Z.append(zernike1(j + 1, npix=npix, outside=0.))

    G = [np.zeros(shape), np.ones(shape)]  # array of G_i etc. intermediate fn
    H = [np.zeros(shape), np.ones(shape) * aperture]  # array of hexikes
    c = {}  # coefficients hash

    for j in np.arange(nterms - 1) + 1:  # can do one less since we already have the piston term
        _log.debug("  j = " + str(j))
        # Compute the j'th G, then H
        nextG = Z[j + 1] * aperture
        for k in np.arange(j) + 1:
            c[(j + 1, k)] = -1 / A * (Z[j + 1] * H[k] * aperture).sum()
            if c[(j + 1, k)] != 0:
                nextG += c[(j + 1, k)] * H[k]
            _log.debug("    c[%s] = %f", str((j + 1, k)), c[(j + 1, k)])

        nextH = nextG / sqrt((nextG ** 2).sum() / A)

        G.append(nextG)
        H.append(nextH)

        #TODO - contemplate whether the above algorithm is numerically stable
        # cf. modified gram-schmidt algorithm discussion on wikipedia.

    # drop the 0th null element, return the rest
    return H[1:]


#--------------------------------------------------------------------------------
# "JWST-exikes"?, "jwexikes"? Something else?

_basepath = os.path.dirname(os.path.abspath(__file__))


def derive_jwexikes(nterms=11, npix=1024, pupilfile=None):
    """ Derive a set of orthonormal polynomials over the JWST pupil, following the
    method of Mahajan and Dai 2006 """
    if pupilfile is None:
        pupilfile = _basepath + os.sep + 'JWpupil_%d.fits' % npix

    if not os.path.exists(pupilfile):
        raise IOError("Requested JWST pupil file '%s' does not exist." % pupilfile)
    aperture = np.array(fits.getdata(pupilfile), float)
    if npix != aperture.shape[0]:
        raise ValueError("For JWexikes the pupil size is fixed by the aperture "
                         "array at %d pix." % aperture.shape[0])
    A = aperture.sum()

    # precompute zernikes
    shape = (npix, npix)
    Z = [np.zeros(shape)]  #TODO:jlong: why is this a list?
    for j in range(nterms + 1):
        Z.append(zernike1(j + 1, npix=npix, outside=0., mask_outside=False))

    G = [np.zeros(shape), np.ones(shape)]  # array of G_i etc. intermediate fn
    H = [np.zeros(shape), np.ones(shape) * aperture]  # array of hexikes
    c = {}  # coefficients hash

    for j in np.arange(nterms - 1) + 1:  # can do one less since we already have the piston term
        _log.debug("  j = " + str(j))
        # Compute the j'th G, then H
        nextG = Z[j + 1] * aperture
        for k in np.arange(j) + 1:
            c[(j + 1, k)] = -1 / A * (Z[j + 1] * H[k] * aperture).sum()
            if c[(j + 1, k)] != 0:
                nextG += c[(j + 1, k)] * H[k]

            _log.debug("    c[%s] = %f", str((j + 1, k)), c[(j + 1, k)])

        #print "   nextG integral: %f" % sqrt((nextG**2).sum() / A )
        nextH = nextG / sqrt((nextG ** 2).sum() / A)

        G.append(nextG)
        H.append(nextH)

        #TODO - contemplate whether the above algorithm is numerically stable
        # cf. modified gram-schmidt algorithm discussion on wikipedia.

    # drop the 0th null element, return the rest
    return H[1:], c, aperture


def jwexike_list(nterms=15, npix=1024, **kwargs):
    """ Return a list of orthonormal polynomials 1-N over the provided pupil (e.g. JWST)

    Results are cached in memory, so there is no penalty for calling this multiple times
    in rapid succession.
    """

    if ('JW', nterms, npix) in _ZCACHE.keys():
        return _ZCACHE[('JW', nterms, npix)]
    else:
        H, x, aperture = derive_jwexikes(nterms=nterms, npix=npix, **kwargs)
        _ZCACHE[('JW', nterms, npix)] = H
        return H


#--------------------------------------------------------------------------------


def wf_expand(wavefront, aperture=None, nterms=15, kind='zernike', **kwargs):
    """ Given a wavefront, return the list of Zernike coefficients that best fit it.

    Parameters
    ----------
    aperture : 2d ndarray, optional
        ndarray giving the aperture mask to use. If not explicitly specified, all
        finite points in the wavefront array (i.e. not NaNs) are assumed to define
        the pupil aperture.
    nterms : int
        Number of terms to use. Default 15
    kind : str
        Kind of polynomial to use. Zernike, Hexike, JWexike, etc
    """

    if aperture is None:
        _log.info("No aperture supplied - using the nonzero part of the wavefront as a guess.")
        aperture = np.asarray((wavefront != 0) & np.isfinite(wavefront), dtype=np.int32)

    fns = {
        'zernike': zernike_list,
        'hexike': hexike_list,
        'jwexike': jwexike_list
    }
    if kind in fns.keys():
        basis_set = fns[kind.lower()]
        basis_set(nterms=nterms, npix=wavefront.shape[0], **kwargs)
    else:
        raise RuntimeError("Argument 'kind' must be one of: {}".format(fns.keys()))

    wgood = np.where(aperture & np.isfinite(basis_set[1]))

    ngood = (wgood[0]).size
    coeffs = [(wavefront * b)[wgood].sum() / ngood for b in basis_set]

    # normalization?

    return coeffs


def wf_generate(coeffs, npix=1024, kind='zernike', aperture=None):
    """ Generate a wavefront for a given list of Zernike coefficients
    (or Zernike-like coefficients)

    Parameters
    -------------
    coeffs : list of int
        Coefficients for the first N Zernikes (or equivalent)
    npix : int
        Size of array on which to evaluate the wavefront
    kind :  str
        Kind of polynomial to use. Zernike, Hexike, JWexike, etc

    """

    nterms = len(coeffs)

    fns = {'Z': zernike_list, 'H': hexike_list, 'J': jwexike_list}
    basis_set = fns[kind[0].upper()](nterms=nterms, npix=npix)

    out = np.zeros((npix, npix), dtype=float)
    for i in range(nterms):
        out += basis_set[i] * coeffs[i]
        #print i, out[509:511,226]

    if aperture is not None:
        wbad = np.where((aperture & np.isfinite(basis_set[0])) is False)
        out[wbad] = np.nan

    return out


def save_to_fits(type='zernike', nterms=10, npix=1024):
    """ Save a list of Zernike type terms to a FITS file
    """
    fns = {'Z': zernike_list, 'H': hexike_list, 'J': jwexike_list}
    names = {'Z': 'zernike', 'H': 'hexike', 'J': 'jwexike'}
    basis_set = fns[type[0].upper()](nterms=nterms, npix=npix)

    basis_ar = np.array(basis_set)
    outname = "%s_%d_%d.fits" % (names[type[0].upper()], npix, nterms)
    fits.PrimaryHDU(basis_ar).writeto(outname)

    print "==>> " + outname


#--------------------------------------------------------------------------------
# test routines

def test_wf_expand(npix=512, type='zernike', term=3, npixout=1024):
    """ Test the wf_expand function
    """

    if term < 1:
        raise ValueError("Zernike index must be >= 1")

    fns = {'Z': zernike_list, 'H': hexike_list, 'J': jwexike_list}
    terms = fns[type[0].upper()](term + 1, npix=npix)
    myOPD = terms[term - 1]
    if term >= 2:
        myOPD = terms[term - 1] + 0.5 * terms[term - 2]

    aperture = terms[0].astype(int)  # use piston term for aperture

    plt.subplot(121)
    plt.imshow(myOPD, vmin=-1, vmax=1)
    plt.title("Input OPD")

    coeffs = wf_expand(myOPD, aperture=aperture, type=type)
    strcoeffs = ['%.4f' % c for c in coeffs]
    print "Coeffs", strcoeffs

    new = wf_generate(coeffs, npix=npixout, type=type)
    plt.subplot(122)
    plt.imshow(new, vmin=-1, vmax=1)
    plt.title("OPD from %s fit" % type)

    print "Totals (should be equal (roughly?)): {}\t{} ".format(np.nansum(myOPD) / myOPD.size,
                                                                np.nansum(new) / new.size)


def test_plot_jwexikes(nterms=20, npix=1024, **kwargs):
    """ Test the jwexikes functions and display the results """
    plotny = int(np.floor(np.sqrt(nterms)))
    plotnx = int(nterms / plotny)

    fig = plt.gcf()
    fig.clf()

    H, c, ap = derive_jwexikes(nterms=nterms, npix=npix, **kwargs)

    wgood = np.where(ap != 0)
    ap[np.where(ap == 0)] = np.nan

    for j in np.arange(nterms):
        ax = fig.add_subplot(plotny, plotnx, j + 1, frameon=False, xticks=[], yticks=[])

        n, m = noll_indices(j + 1)

        ax.imshow(H[j] * ap, vmin=-3, vmax=3.0)
        ax.text(npix * 0.7, npix * 0.1, "$JW_%d^{%d}$" % (n, m), fontsize=20)
        print "Term %d:   std dev is %f. (should be near 1)" % (j + 1, H[j][wgood].std())

    plt.draw()


def test_plot_hexikes(nterms=20, npix=500):
    """ Test the hexikes functions and display the results """
    plotny = int(np.floor(np.sqrt(nterms)))
    plotnx = int(nterms / plotny)

    fig = plt.gcf()
    fig.clf()

    H = hexike_list(nterms=nterms, npix=npix)

    ap = hex_aperture(npix)
    wgood = np.where(ap != 0)
    ap[np.where(ap == 0)] = np.nan

    for j in np.arange(nterms):
        ax = fig.add_subplot(plotny, plotnx, j + 1, frameon=False, xticks=[], yticks=[])

        n, m = noll_indices(j + 1)

        ax.imshow(H[j] * ap, vmin=-3, vmax=3.0)
        ax.text(npix * 0.7, npix * 0.1, "$H_%d^{%d}$" % (n, m), fontsize=20)
        print "Term %d:   std dev is %f. (should be near 1)" % (j + 1, H[j][wgood].std())

    plt.draw()


def test_plot_zernikes(nterms=20, npix=500, names=False):
    """ Test the zernikes functions and display the results """
    plotny = int(np.floor(np.sqrt(nterms)))
    plotnx = int(nterms / plotny)

    fig = plt.gcf()
    fig.clf()

    ap = np.isfinite(zernike1(1, npix=npix)).astype(int)
    wgood = np.where(ap)

    for j in np.arange(nterms) + 1:
        ax = fig.add_subplot(plotny, plotnx, j, frameon=False, xticks=[], yticks=[])

        Z, n, m = zernike1(j, return_indices=True, npix=npix)

        ax.imshow(Z, vmin=-3, vmax=3.0)

        ax.text(npix * 0.7, npix * 0.1, "$Z_%d^{%d}$" % (n, m), fontsize=20)
        zl = zern_name(j) if names else "$Z%d$" % j
        ax.text(npix * 0.95, npix * 0.8, zl, fontsize=20, horizontalalignment='right')
        print "Term %d: std dev is %f. (should be near 1)" % (j, Z[wgood].std())

    plt.draw()


def test_integrate_zernikes(nterms=10, size=500):
    """Verify the functions integrate properly over the unit circle"""
    for j in np.arange(nterms) + 1:
        Z, n, m = zernike1(j, npix=size, return_indices=True)
        wg = np.where(np.isfinite(Z))
        print "j=%d\t(%d,%d)\t\\integral(Z_j) = %f" % (j, n, m, Z[wg].sum())


def test_ones_zernikes(nterms=10):
    """Verify the radial scaling function is correctly normalized"""
    rho = np.ones(3)
    theta = np.array([0, 1, 2])
    for j in np.arange(nterms) + 1:
        Z, n, m = zernike1(j, r=rho, theta=theta, return_indices=True)
        Rs = R(n, m, rho)
        print "j=%d\tZ_(%d,%d) [1] = \t %s" % (j, n, m, str(Rs))


def test_cross_zernikes(testj, nterms=10, npix=500):
    """Verify the functions are orthogonal, by taking the
    integrals of a given Zernike times N other ones.

    Note that the Zernikes are only strictly orthonormal over a
    fully circular aperture evauated analytically. For any discrete
    aperture the orthonormality is only approximate

    Parameters :
    --------------
    testj : int
        Index of the Zernike polynomial to test against the others
    nterms : int
        Test that polynomial against those from 1 to this N
    npix : int
        Size of array to use for this test

    """

    Zj = zernike1(testj, npix=npix)

    for j in np.arange(nterms) + 1:
        Z, n, m = zernike1(j, npix=npix, return_indices=True)

        prod = Z * Zj
        wg = np.where(np.isfinite(prod))
        print "integral(Z_%d * Z_%d) = %f" % (j, testj, prod[wg].sum())


def test_1d_args(nterms=10, size=256):
    plotnx = int(nterms)
    plotny = 2

    fig = plt.gcf()
    fig.clf()

    Y, X = np.indices((60, 60))
    X -= (60 - 1) / 2.0
    Y -= (60 - 1) / 2.0
    r = np.sqrt(X ** 2 + Y ** 2) / 30.0
    theta = np.arctan2(Y, X)

    for j in np.arange(nterms) + 1:
        out = np.zeros_like(X).astype(np.float64)
        ax = fig.add_subplot(plotny, plotnx, j, frameon=False, xticks=[], yticks=[])

        Z, n, m = zernike1(j, return_indices=True, npix=size)

        ax.imshow(Z, vmin=-3, vmax=3.0)
        print "j = %d\tzmin = %f\tzmax=%f" % (j, np.nanxmin(Z), np.nanmax(Z))
        ax.text(size * 0.7, size * 0.1, "$Z_%d^{%d}$" % (n, m), fontsize=20)
        ax.text(size * 0.95, size * 0.8, "$Z%d$" % j, fontsize=20, horizontalalignment='right')

        Z2 = zernike1(j, r=r.flatten(), theta=theta.flatten())
        out.flat[:] = Z2
        ax = fig.add_subplot(plotny, plotnx, j + nterms, frameon=False, xticks=[], yticks=[])
        ax.imshow(out, vmin=-3, vmax=3.0)

    plt.draw()


def test_noll_indices():
    """ Test the noll_indices function for a handful of precomputed values """
    assert noll_indices(3) == (1, -1)
    assert noll_indices(10) == (3, 3)
    assert noll_indices(13) == (4, -2)
    assert noll_indices(19) == (5, -3)


def test_str_zernike():
    """Test str_zernike """
    assert str_zernike(4, -2) == '\\sqrt{10}* ( 4 r^4  -3 r^2  ) * \sin(2 \\theta)'

#--------------------------------------------------------------------------------

if __name__ == "__main__":
    pass

    #test_plot_hexikes(25)
    # you will need a JWST pupil file to run this next one.
    #test_plot_jwexikes(25, pupilfile='JWpupil.fits')
