#  Functions for reasonably exact geometry on discrete arrays
#  These codes allow you to calculate circles and other such
#  shapes discretized onto arrays with proper handling of areas
#  at subpixel precision. (or at least reasonably proper; no
#  guarantees for utter mathematical exactness at machine precision.)

import array_api_extra as xpx
import numpy as np

from . import accel_math
from .accel_math import xp

if accel_math._NUMEXPR_AVAILABLE:
    import numexpr as ne

import logging

_log = logging.getLogger('poppy')


# original code in pixwt.c by Marc Buie
#    See http://www.boulder.swri.edu/~buie/idl/downloads/custom/32bit/pixwt.c
#
# ported to pixwt.pro (IDL) by Doug Loucks, Lowell Observatory, 1992 Sep
#
# subsequently ported to python by Michael Fitzgerald, 2007-10-16
# LLNL / UCLA

def _arc(x, y0, y1, r):
    """
    Compute the area within an arc of a circle.  The arc is defined by
    the two points (x,y0) and (x,y1) in the following manner: The
    circle is of radius r and is positioned at the origin.  The origin
    and each individual point define a line which intersects the
    circle at some point.  The angle between these two points on the
    circle measured from y0 to y1 defines the sides of a wedge of the
    circle.  The area returned is the area of this wedge.  If the area
    is traversed clockwise then the area is negative, otherwise it is
    positive.

    Where y0 == y1 the wedge is empty and the area is exactly zero (this avoids
    0/0 for x == 0).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        if accel_math._USE_NUMEXPR:
            area = ne.evaluate("0.5 * r**2 * (arctan(y1/x) - arctan(y0/x))")
        else:
            area = 0.5 * r**2 * (xp.atan(y1 / x) - xp.atan(y0 / x))
    return xp.where(y0 == y1, xp.zeros_like(area), area)

def _chord(x, y0, y1):
    """
    Compute the area of a triangle defined by the origin and two
    points, (x,y0) and (x,y1).  This is a signed area.  If y1 > y0
    then the area will be positive, otherwise it will be negative.
    """
    return 0.5 * x * (y1 - y0)

def _oneside(x, y0, y1, r):
    """
    Compute the area of intersection between a triangle and a circle.
    The circle is centered at the origin and has a radius of r.  The
    triangle has vertices at the origin and at (x,y0) and (x,y1).
    This is a signed area.  The path is traversed from y0 to y1.  If
    this path takes you clockwise the area will be negative.

    This is evaluated branch-free. With yh the half-chord of the circle at x
    (zero if |x| >= r), and c0, c1 the endpoints clipped to [-yh, yh], the path
    from y0 to y1 splits into an outside arc (y0 to c0), an inside chord (c0 to
    c1) and a second outside arc (c1 to y1). Each piece is zero when empty.
    """
    x, y0, y1 = xp.broadcast_arrays(*(xp.asarray(v, dtype=xp.float64) for v in (x, y0, y1)))

    yh = xp.sqrt(xp.clip(r**2 - x**2, 0, None))
    c0 = xp.clip(y0, -yh, yh)
    c1 = xp.clip(y1, -yh, yh)

    return _arc(x, y0, c0, r) + _chord(x, c0, c1) + _arc(x, c1, y1, r)

def _intarea(xc, yc, r, x0, x1, y0, y1):
    """
    Compute the area of overlap of a circle and a rectangle.
      xc, yc  :  Center of the circle.
      r       :  Radius of the circle.
      x0, y0  :  Corner of the rectangle.
      x1, y1  :  Opposite corner of the rectangle.
    """
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return _oneside(x1, y0, y1, r) + _oneside(y1, -x1, -x0, r) + \
           _oneside(-x0, -y1, -y0, r) + _oneside(-y0, x0, x1, r)

def pixwt(xc, yc, r, x, y):
    """
    Compute the fraction of a unit pixel that is interior to a circle.
    The circle has a radius r and is centered at (xc, yc).  The center
    of the unit pixel (length of sides = 1) is at (x, y).

    Divides the circle and rectangle into a series of sectors and
    triangles.  Determines which of nine possible cases for the
    overlap applies and sums the areas of the corresponding sectors
    and triangles.

    area = pixwt( xc, yc, r, x, y )

    xc, yc : Center of the circle, numpy scalars
    r      : Radius of the circle, numpy scalars
    x, y   : Center of the unit pixel, numpy scalar or vector
    """
    return _intarea(xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5)



def filled_circle_aa(shape, xcenter, ycenter, radius, xarray=None, yarray=None,
        fillvalue=1, clip=True, cliprange=(0,1)):
    """Draw a filled circle with subpixel antialiasing into an array.

    Parameters
    ----------
    shape : 2d ndarray
        shape of array to return
    xcenter, ycenter : floats
        (X, Y) coordinates for the center of the circle (in the coordinate
        system specified by the xarray and yarray parameters, if those are given)
    radius : float
        Radius of the circle
    xarray, yarray : 2d ndarrays
        X and Y coordinates corresponding to the center of each pixel
        in the main array. If not present, integer pixel indices are assumed.
        WARNING - code currently is buggy with pixel scales != 1
    fillvalue : float
        Value to add into the array, for pixels that are entirely within the radius.
        This is *added* to each pixel at the specified coordinates. Default is 1
    clip : bool
        Clip the output array values to between the values given by the cliprange parameter.
    cliprange : array_like
        if clip is True, give values to use in the clip function.
    """

    if xarray is None or yarray is None:
        yarray, xarray = xp.meshgrid(xp.arange(shape[0]), xp.arange(shape[1]), indexing="ij")

    xarray = xp.asarray(xarray, dtype=xp.float64)
    yarray = xp.asarray(yarray, dtype=xp.float64)

    r = xp.sqrt((xarray - xcenter) ** 2 + (yarray - ycenter) ** 2)

    pixscale = xp.abs(xarray[0, 1] - xarray[0, 0])
    area_per_pix = pixscale**2

    if xp.abs(pixscale - 1.0) > 0.01:
        import warnings
        warnings.warn('filled_circle_aa may not yield exact results for grey pixels when pixel scale <1')

    array = xp.astype(r < radius, xp.float64) * fillvalue

    # Only the thin ring of border pixels needs the (expensive) exact area calculation.
    border = xp.abs(r - radius) < pixscale
    weights = pixwt(xcenter, ycenter, radius, xarray[border], yarray[border])
    array = xpx.at(array, border).set(weights * fillvalue / area_per_pix)

    if clip:
        assert len(cliprange) == 2
        return xp.clip(array, *cliprange)
    else:
        return array
