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
import matplotlib.pyplot as plt

from astropy.io import fits

import sys
if sys.version_info > (3, 2):
    from functools import lru_cache
else:
    from poppy.vendor.lru_cache import lru_cache

from poppy.poppy_core import Wavefront

import logging

__all__ = [
    'R', 'cached_zernike1', 'hex_aperture', 'hexike_basis', 'noll_indices',
    'opd_expand', 'str_zernike', 'zern_name', 'zernike', 'zernike1', 'zernike_basis'
]

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


def str_zernike(n, m):
    """Return analytic expression for a given Zernike in LaTeX syntax"""
    signed_m = int(m)
    m = int(np.abs(m))
    n = int(np.abs(n))

    terms = []
    for k in range(int((n - m) / 2) + 1):
        coef = ((-1) ** k * factorial(n - k) /
                (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
        if coef != 0:
            formatcode = "{0:d}" if k == 0 else "{0:+d}"
            terms.append((formatcode + " r^{1:d} ").format(int(coef), n - 2 * k))

    outstr = " ".join(terms)

    if m == 0:
        if n == 0:
            return "1"
        else:
            return "sqrt(%d)* ( %s ) " % (n + 1, outstr)
    elif signed_m > 0:
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
        raise ValueError("Zernike index j must be a positive integer.")

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


def R(n, m, rho):
    """Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
            output += coef * rho ** (n - 2 * k)
        return output


def zernike(n, m, npix=100, rho=None, theta=None, outside=np.nan,
            noll_normalize=True):
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

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix: int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

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
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[np.where(rho > 1)] = outside
    return zernike_result


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
        Zernike function ordinate, following the convention of
        Noll et al. JOSA 1976

    Additional arguments are defined as in `poppy.zernike.zernike`.

    Returns
    -------
    zern : 2D numpy array
        Z_j evaluated at each (rho, theta)
    """
    n, m = noll_indices(j)
    return zernike(n, m, **kwargs)


@lru_cache()
def cached_zernike1(j, shape, pixelscale, pupil_radius, outside=np.nan, noll_normalize=True):
    """Compute Zernike based on Noll index *j*, using an LRU cache
    for efficiency. Refer to the `zernike1` docstring for details.
    """
    y, x = Wavefront.pupil_coordinates(shape, pixelscale)
    r = np.sqrt(x ** 2 + y ** 2)

    rho = r / pupil_radius
    theta = np.arctan2(y / pupil_radius, x / pupil_radius)

    n, m = noll_indices(j)
    result = zernike(n, m, rho=rho, theta=theta, outside=outside, noll_normalize=noll_normalize)
    result.flags.writeable = False  # don't let caller modify cached copy in-place
    return result


def zernike_basis(nterms=15, npix=512, rho=None, theta=None, **kwargs):
    """
    Return a cube of Zernike terms from 1 to N each as a 2D array
    showing the value at each point. (Regions outside the unit circle on which
    the Zernike is defined are initialized to zero.)

    Parameters
    -----------
    nterms : int, optional
        Number of Zernike terms to return, starting from piston.
        (e.g. ``nterms=1`` would return only the Zernike piston term.)
        Default is 15.
    npix: int
        Desired pixel diameter for circular pupil. Only used if `rho`
        and `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.

    Other parameters are passed through to `poppy.zernike.zernike`
    and are documented there.
    """
    if rho is not None:
        # both are required, but validated in zernike1
        shape = rho.shape
        use_polar = True
    else:
        shape = (npix, npix)
        use_polar = False

    zern_output = np.zeros((nterms,) + shape)

    if use_polar:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, rho=rho, theta=theta, **kwargs)
    else:
        for j in range(nterms):
            zern_output[j] = zernike1(j + 1, npix=npix, **kwargs)
    return zern_output


@lru_cache()
def zernike_basis_faster(nterms=15, npix=512, outside=np.nan):
    """
    Return a cube of Zernike terms from 1 to N each as a 2D array
    showing the value at each point. (Regions outside the unit circle on which
    the Zernike is defined are initialized to np.nan.)

    Same as the original zernike_basis, but optimized to run about 2x faster,
    at a cost of somewhat less flexibility.

    Does not support providing polar coordinates directly - use regular
    zernike_basis for that.

    Parameters
    -----------
    nterms : int, optional
        Number of Zernike terms to return, starting from piston.
        (e.g. ``nterms=1`` would return only the Zernike piston term.)
        Default is 15.
    npix: int
        Desired pixel diameter for circular pupil. Only used if `rho`
        and `theta` are not provided.


    """
    shape = (npix, npix)
    use_polar = False

    zern_output = np.zeros((nterms,) + shape)

    x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
    y = x
    xx, yy = np.meshgrid(x, y)

    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    aperture = np.ones_like(rho)
    aperture[rho > 1] = 0.0  # this is the aperture mask
    noll_normalize=True

    @lru_cache()
    def cached_R(n, m):
        """Compute R[n, m], the Zernike radial polynomial

        Parameters
        ----------
        n, m : int
            Zernike function degree
        rho : array
            Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
            unit circle  (this is found implicitly in the scope of the calling fn)
        """

        m = int(np.abs(m))
        n = int(np.abs(n))
        output = np.zeros(rho.shape)
        if _is_odd(n - m):
            return 0
        else:
            for k in range(int((n - m) / 2) + 1):
                coef = ((-1) ** k * factorial(n - k) /
                        (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
                output += coef * rho ** (n - 2 * k)
            return output

    for j in range(nterms):
        n, m = noll_indices(j+1)


        if m == 0:
            if n == 0:
                zernike_result = aperture
            else:
                norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
                zernike_result = norm_coeff * cached_R(n, m) * aperture
        elif m > 0:
            norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * cached_R(n, np.abs(m)) * np.cos(np.abs(m) * theta) * aperture
        else:
            norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * cached_R(n, np.abs(m)) * np.sin(np.abs(m) * theta) * aperture

        zernike_result[rho > 1] = outside
        zern_output[j] = zernike_result


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
    rho, theta : 2D numpy arrays, optional
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
        # the coordinates here must be consistent with those used elsewhere in poppy
        # see issue #111
        x_ = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / (npix / 2.)
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


def hexike_basis(nterms=15, npix=512, rho=None, theta=None,
                 vertical=False, outside=np.nan):
    """Return a list of hexike polynomials 1-N following the
    method of Mahajan and Dai 2006 for numerical orthonormalization

    This function orders the hexikes in a similar way as the Zernikes.

    See also hexike_basis_wss for an alternative implementation.

    Parameters
    ----------
    nterms : int
        Number of hexike terms to compute, starting from piston.
        (e.g. ``nterms=1`` would return only the hexike analog to the
        Zernike piston term.) Default is 15.
    npix : int
        Size, in pixels, of the aperture array. The hexagon will span
        the whole array from edge to edge in the direction aligned
        with its flat sides.
    rho, theta : 2D numpy arrays, optional
        For some square aperture, rho and theta contain each pixel's
        coordinates in polar form. The hexagon will be defined such
        that it can be circumscribed in a rho = 1 circle.
    vertical : bool
        Make flat sides parallel to the Y axis instead of the default X.
        Default is False.
    outside : float
        Value for pixels outside the hexagonal aperture.
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    """

    if rho is not None:
        shape = rho.shape
        assert len(shape) == 2 and shape[0] == shape[1], \
            "only square rho and theta arrays supported"
    else:
        shape = (npix, npix)

    aperture = hex_aperture(npix=npix, rho=rho, theta=theta, vertical=vertical)
    A = aperture.sum()

    # precompute zernikes
    Z = np.zeros((nterms + 1,) + shape)
    Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta, outside=0.0)


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

        nextH = nextG / np.sqrt((nextG ** 2).sum() / A)

        G.append(nextG)
        H.append(nextH)

        #TODO - contemplate whether the above algorithm is numerically stable
        # cf. modified gram-schmidt algorithm discussion on wikipedia.

    # drop the 0th null element, return the rest
    return H[1:]

def hexike_basis_wss(nterms=9, npix=512, rho=None, theta=None,
                x=None,y=None,
                 vertical=False, outside=np.nan):
    """Return a list of hexike polynomials 1-N based on analytic
    expressions. Note, this is strictly consistent with the
    JWST WSS hexikes in both ordering and normalization.

    ***The ordering of hexike terms is DIFFERENT FROM that returned by
    the zernike_basis or regular hexike_basis functions. Use this one
    in particular if you need something consistent with JWST WSS internals.
    ***

    That ordering is:
        H1 = Piston
        H2 = X tilt
        H3 = Y tilt
        H4 = Astigmatism-45
        H5 = Focus
        H6 = Astigmatism-00
        H7 = Coma X
        H8 = Coma Y
        H9 = Spherical
        H10 = Trefoil-0
        H11 = Trefoil-30

    The last two are included for completeness of that hexike order but
    are not actually used in the WSS.
    This function has an attributed hexike_basis_wss.label_strings for
    convenient use in plot labeling.


    Parameters
    ----------
    nterms : int
        Number of hexike terms to compute, starting from piston.
        (e.g. ``nterms=1`` would return only the hexike analog to the
        Zernike piston term.) Default is 15.
    npix : int
        Size, in pixels, of the aperture array. The hexagon will span
        the whole array from edge to edge in the direction aligned
        with its flat sides.
    rho, theta : 2D numpy arrays, optional
        For some square aperture, rho and theta contain each pixel's
        coordinates in polar form. The hexagon will be defined such
        that it can be circumscribed in a rho = 1 circle.
    x,y : 1D numpy arrays, optional
        Alternative way of specifying the coordinates.
    vertical : bool
        Make flat sides parallel to the Y axis instead of the default X.
        Default is False.
    outside : float
        Value for pixels outside the hexagonal aperture.
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    """

    if rho is not None and theta is not None:
        _log.debug("User supplied radial coords")
        shape = rho.shape
        assert len(shape) == 2 and shape[0] == shape[1], \
            "only square rho and theta arrays supported"
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        r2 = rho**2
    elif x is not None and y is not None:
        _log.debug("User supplied cartesian coords")
        r2 = x**2+y**2
        rho = np.sqrt(r2)
        theta = np.arctan2(y,x)
    else:
        _log.debug("User supplied only the number of pixels")
        #create 2D arrays of coordinates between 0 and 1
        shape = (npix, npix)
        y,x = np.indices(shape, dtype=float)
        y -= npix/2.
        x -= npix/2.
        y /= (npix/2)
        x /= (npix/2)

        r2 = x**2+y**2
        rho = np.sqrt(r2)
        theta = np.arctan2(y,x)


    aperture = hex_aperture(npix=npix, rho=rho, theta=theta, vertical=vertical)
    #print(rho[aperture==1].max())

    A = aperture.sum()

    # first 9 hexikes (those used in WAS for JWST)
    # create array of hexikes, plus pad for 0th term
    H = [np.zeros_like(x),       # placeholder for 0th term to allow 1-indexing
         np.ones_like(x),    # Piston
         y,                 # tilt around x
         x,                 # tilt around y
         2*x*y,              # astig-45
         r2-0.5,             # focus -- yes this is really exactly what the WAS uses
         x**2-y**2,          # astig-00
         ((25./11.)*r2-14./11.) * x,  # Coma x
         ((25./11.)*r2-14./11.) * y,  # Coma y
         ((860./231.)*r2**2 - (5140./1617.)*r2 + (67./147.)),  # Spherical
         (10./7.)*(rho*r2) * np.sin(3.*theta),     # Trefoil-0
         (10./7.)*(rho*r2) * np.cos(3.*theta),           # Trefoil-30
         ]

    if nterms > len(H)-1:
        raise NotImplementedError("hexicke_basis_analytic doesn't support that many terms yet")
    else:
        # apply aperture mask
        for i in range(1,nterms+1):
            H[i] *= aperture
        return H[1:nterms+1]

hexike_basis_wss.label_strings = ['Piston','X tilt', 'Y tilt', 'Astigmatism-45','Focus','Astigmatism-00','Coma X','Coma Y','Spherical','Trefoil-0','Trefoil-30']


def arbitrary_basis(aperture, nterms=15, rho=None, theta=None):
    """ Orthonormal basis on arbitrary aperture, via Gram-Schmidt

    Return a cube of Zernike-like terms from 1 to N, calculated on an
    arbitrary aperture, each as a 2D array showing the value at each
    point. (Regions outside the unit circle on which the Zernike is
    defined are initialized to zero.)

    This implements Gram-Schmidt orthonormalization numerically,
    starting from the regular Zernikes, to generate an orthonormal basis
    on some other aperture

    Parameters
    -----------
    aperture : array_like
        2D binary array representing the arbitrary aperture
    nterms : int, optional
        Number of Zernike terms to return, starting from piston.
        (e.g. ``nterms=1`` would return only the Zernike piston term.)
        Default is 15.
    rho, theta : array_like, optional
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the pupil. `theta` should be
        the angle in radians.
    """
    # code submitted by Arthur Vigan - see https://github.com/mperrin/poppy/issues/166

    shape = aperture.shape
    npix  = shape[0]

    A = aperture.sum()

    # precompute zernikes
    Z = np.zeros((nterms + 1,) + shape)
    Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta, outside=0.0)


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

        nextH = nextG / np.sqrt((nextG ** 2).sum() / A)

        G.append(nextG)
        H.append(nextH)

        #TODO - contemplate whether the above algorithm is numerically stable
        # cf. modified gram-schmidt algorithm discussion on wikipedia.

    # drop the 0th null element, return the rest
    return H[1:]


def opd_expand(opd, aperture=None, nterms=15, basis=zernike_basis,
              **kwargs):
    """Given a wavefront OPD map, return the list of coefficients in a
    given basis set (by default, Zernikes) that best fit the OPD map.

    Note that this implementation of the function treats the Zernikes as
    an orthonormal basis, which is only true on the unobscured unit circle.
    See also `opd_expand_nonorthonormal` for an alternative approach.

    Parameters
    ----------
    opd : 2D numpy.ndarray
        The wavefront OPD map to expand in terms of the requested basis.
        Must be square.
    aperture : 2D numpy.ndarray, optional
        ndarray giving the aperture mask to use
        (1.0 where light passes, 0.0 where it is blocked).
        If not explicitly specified, all finite points in the `opd`
        array (i.e. not NaNs) are assumed to define the pupil aperture.
    nterms : int
        Number of terms to use. (Default: 15)
    basis : callable, optional
        Callable (e.g. a function) that generates a sequence
        of basis arrays given arguments `nterms`, `npix`, and `outside`.
        Default is `poppy.zernike.zernike_basis`.

    Additional keyword arguments to this function are passed
    through to the `basis` callable.

    Note: Recovering coefficients used to generate synthetic/test data
    depends greatly on the sampling (as one might expect). Generating
    test data using zernike_basis with npix=256 and passing the result
    through opd_expand reproduces the input coefficients within <0.1%.

    Returns
    -------
    coeffs : list
        List of coefficients (of length `nterms`) from which the
        input OPD map can be constructed in the given basis.
        (No additional unit conversions are performed. If the input
        wavefront is in waves, coeffs will be in waves.)
        Note that the first coefficient (element 0 in Python indexing)
        corresponds to the Z=1 Zernike piston term, and so on.
    """

    if aperture is None:
        _log.warn("No aperture supplied - "
                  "using the finite (non-NaN) part of the OPD map as a guess.")
        aperture = np.isfinite(opd).astype(np.float)

    basis_set = basis(
        nterms=nterms,
        npix=opd.shape[0],
        outside=np.nan,
        **kwargs
    )

    wgood = np.where((aperture != 0.0) & np.isfinite(basis_set[1]))
    ngood = (wgood[0]).size

    coeffs = [(opd * b)[wgood].sum() / ngood
              for b in basis_set]

    return coeffs


def opd_expand_nonorthonormal(opd, aperture=None, nterms=15, basis=zernike_basis_faster,
                  iterations=5, **kwargs):
    """ Modified version of opd_expand, for cases where the basis function is
    *not* orthonormal, for instance using the regular Zernike functions on
    obscured apertures.

    This version subtracts off each term as it is fit, to avoid over-fitting the
    same WFE multiple times. It also iterates the fitting multiple times by re-fitting
    the residuals, in order to allow for capturing any WFE which is missed by the
    first pass at fitting.

    Based on various empirical experimentation for what is necessary to get
    reasonable behavior in this non-ideal case. Factors to consider:
    1) Masking to use just pixels good in both the zernike unit circle and the
       asymmetric numerical aperture
    2) Subtracting off the fit terms as you go, so as to not fit the same WFE
       multiple times
    3) Iterating multiples by re-fitting the residual, to include as much WFE
       as possible.

    Parameters
    -----------
    opd : the OPD you want to fit
    aperture : an aperture mask of which pixels are valid. ANything non-NaN is
               considered valid
    nterms : number of terms to fit
    basis: which basis function to use. Defaults to Zernike
    iterations : int
        Number of iterations for convergence. Default is 5

    """

    if aperture is None:
        _log.warn("No aperture supplied - "
                  "using the finite (non-NaN) part of the OPD map as a guess.")
        aperture = np.isfinite(opd).astype(np.float)

    basis_set = basis(
        nterms=nterms,
        npix=opd.shape[0],
        outside=np.nan,
        **kwargs
    )

    wgood = np.where((aperture != 0.0) & np.isfinite(aperture) & np.isfinite(basis_set[1]))
    ngood = (wgood[0]).size

    coeffs = np.zeros(nterms)
    opd_copy = np.copy(opd)

    for count in range(iterations):
        for i,b in enumerate(basis_set):
            this_coeff = (opd_copy * b)[wgood].sum() / ngood
            opd_copy  -= this_coeff * b
            coeffs[i] += this_coeff

    return coeffs


def opd_from_zernikes(coeffs, basis=zernike_basis_faster, aperture=None, outside=np.nan,
        **kwargs):
    """ Synthesize an OPD from a set of coefficients

    Parameters
    -----------
    coeffs : list or ndarray
        Coefficients for the Zernike terms
    basis : callable
        Which basis set. Defaults to Zernike
    aperture : ndarray, optional
        Aperture mask, if provided. Defaults to the unit circle filling the array

    Other parameters are supported via **kwargs, in particular setting the
    size of the OPD via npix.

    Example
    --------
    opd = opd_from_zernikes([0,0,-5,1,0,4,0,8], npix=512)

    """
    basis_set = basis(
        nterms=len(coeffs),
        outside=outside,
        **kwargs
    )

    output = np.zeros_like(basis_set[0])

    for i, b in enumerate(basis_set):
        if coeffs[i] !=0:
            output += coeffs[i]*b
    if aperture is not None:
        output[~np.isfinite(aperture)] = np.nan
        output[aperture==0] = 0
    return output
