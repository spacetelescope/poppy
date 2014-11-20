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

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_log.addHandler(logging.NullHandler())


def _is_odd(integer):
    """Helper for testing if an integer is odd by bitwise & with 1."""
    return integer & 1


def zern_name(i):
    """Return a human-readable text name corresponding to some Zernike term as specified
    by `j`, the index

    Only works up to term 22, i.e. 5th order spherical aberration.
    """
    names = ['Null', 'Piston', 'Tilt X', 'Tilt Y',
             'Focus', 'Astigmatism 45', 'Astigmatism 0',
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


def noll_indices(j):
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

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

    _log.debug("J=%d:\t(n=%d, m=%d)" % (j, n, m))
    return n, m


def R(n_, m_, rho):
    """Compute R[n_, m_], the Zernike radial polynomial

    Parameters
    ----------
    n_, m_ : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
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


def zernike(n, m, npix=100, rho=None, theta=None, mask_outside=True,
            outside=np.nan, noll_normalize=True):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:
     zernike(n, m, npix)       where npix specifies a pupil diameter in pixels.
                               The returned pupil will be a circular aperture
                               with this diameter, embedded in a square array
                               of size npix*npix.
     zernike(n, m, rho=r, theta=theta)    Which explicitly provides the desired pupil coordinates
                               as arrays r and theta. These need not be regular or contiguous.


    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix: int
        Desired diameter for circular pupil. Only used if r and theta are not provided.
    rho, theta : array_like
        Image plane coordinates. rho should be in 0<rho<1, theta should be in radians
    mask_outside : bool
        Mask out the region beyond radius 1? Default True.
    outside : float
        Value for pixels outside the circular aperture. Default is NaN, but you may also
        find it useful for this to be zero sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernikes are normalized such that
        the integral of Z[n, m] * Z[n, m] over the unit disk is pi exactly. To omit
        the normalization constant, set this to False. Default is True.
    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        _log.warn("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    _log.debug("Zernike(n=%d, m=%d)" % (n, m))

    if theta is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    else:
        if rho is None:
            raise ValueError("If you provide a theta input array, you must also provide an array "
                             "r with the corresponding radii for each point.")

    aperture = np.ones(rho.shape)
    if mask_outside:  # TODO:jlong: this is not actually masking, it's multiplying...
        aperture[np.where(rho > 1)] = outside
    if m == 0:
        if n == 0:
            return np.ones(rho.shape) * aperture
        else:
            norm_coeff = sqrt(n + 1) if noll_normalize else 1
            return norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = sqrt(2) * sqrt(n + 1) if noll_normalize else 1
        return norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = sqrt(2) * sqrt(n + 1) if noll_normalize else 1
        return norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture


def zernike1(j, **kwargs):
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
    npix: int
        Desired diameter for circular pupil. Only used if r and theta are not provided.
    rho, theta : array_like
        Image plane coordinates. rho should be in 0<rho<1, theta should be in radians
    mask_outside : bool
        Mask out the region beyond radius 1? Default True.
    outside : float
        Value for pixels outside the circular aperture. Default is NaN, but you may also
        find it useful for this to be zero sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is modified such that
        the integral of Z[n, m] * Z[n, m] over the unit disk is pi exactly. To omit
        the normalization constant, set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta)
    """
    n, m = noll_indices(j)
    return zernike(n, m, **kwargs)


def zernike_basis(nterms=15, npix=512, rho=None, theta=None, **kwargs):
    """
    Return a cube of Zernike terms from 1 to N each as a 2D array
    showing the value at each point. (Regions outside the unit circle on which
    the Zernike is defined are initialized to zero.)

    Parameters
    -----------
    nterms : int
        Number of Zernike terms to return
    npix: int
        Desired pixel diameter for circular pupil. Only used if r and theta are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 1 at the desired pixel radius,
        `theta` should be in radians
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernikes are normalized such that
        the integral of Z[n, m] * Z[n, m] over the unit disk is pi exactly. To omit
        the normalization constant, set this to False. Default is True.
    """
    if rho is not None:
        if rho is None or theta is None:
            raise ValueError("You must supply both `theta` and `rho`, or neither.")
        npix = rho.shape[0]
        shape = rho.shape
        use_polar = True
    else:
        shape = (npix, npix)
        use_polar = False

    # pass these keyword arguments through to zernike.zernike
    kwargs['mask_outside'] = True
    kwargs['outside'] = 0.0

    zern_output = np.zeros((nterms,) + shape)
    if use_polar:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, rho=rho, theta=theta, **kwargs)
    else:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, npix=npix, **kwargs)
    return zern_output


def hex_aperture(npix=1024, rho=None, theta=None, vertical=False):
    """
    Return an aperture function for a hexagon.

    Note that the flat sides are aligned with the X direction by default.
    This is appropriate for the individual hex PMSA segments in JWST.

    Parameters
    -----------
    npix : integer
        Size, in pixels, of the aperture array. The hexagon will span
        the whole array from edge to edge in the direction aligned
        with its flat sides. (Ignored when `rho` and `theta` are
        supplied.)
    rho, theta : 2D numpy arrays
        For some square aperture, rho and theta contain each pixel's
        coordinates in polar form. The hexagon will be defined such
        that it can be circumscribed in a rho = 1 circle.
    vertical : bool
        Make flat sides parallel to the Y axis instead of the default X.
    """

    if rho is not None or theta is not None:
        if rho is None or theta is None:
            raise ValueError("You must supply both `theta` and `rho`, or neither.")
        # easier to define a hexagon in cartesian, so...
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
    else:
        x_ = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        x, y = np.meshgrid(x_, x_)

    absy = np.abs(y)

    aperture = np.zeros(x.shape)
    w_rect = np.where((np.abs(x) <= 0.5) & (np.abs(y) <= np.sqrt(3) / 2))
    w_left_tri = np.where((x <= -0.5) & (x >= -1) & (absy <= (x + 1) * np.sqrt(3)))
    w_right_tri = np.where((x >= 0.5) & (x <= 1) & (absy <= (1 - x) * np.sqrt(3)))
    aperture[w_rect] = 1
    aperture[w_left_tri] = 1
    aperture[w_right_tri] = 1

    if vertical:
        return aperture.transpose()
    else:
        return aperture


def hexike_basis(nterms=15, npix=512, rho=None, theta=None, vertical=False, **kwargs):
    """ Return a list of hexike polynomials 1-N following the
    method of Mahajan and Dai 2006 """

    if rho is not None:
        shape = rho.shape
        assert len(shape) == 2 and shape[0] == shape[1], ("only square rho and "
                                                          "theta arrays supported")
    else:
        shape = (npix, npix)

    aperture = hex_aperture(npix=npix, rho=rho, theta=theta, vertical=vertical)
    A = aperture.sum()

    # precompute zernikes
    Z = np.zeros((nterms + 1,) + shape)
    Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta)


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
