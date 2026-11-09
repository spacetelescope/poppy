# Licensed under a 3-clause BSD style license - see LICENSE.md
"""Array-backend-aware wrappers for the scipy functions that POPPY uses.

Why this module exists
----------------------
POPPY does its array math through the Array API namespace ``poppy.accel_math.xp`` (numpy, cupy, ...,
chosen once at import time). The Array API standard does not include scipy-style routines such as
image resampling, rotation, shifting, FFT convolution or Bessel functions, and scipy itself is
numpy-centric: scipy's own Array API dispatch is opt-in, partial, and version dependent. Rather than
scattering backend checks through the code, every call site that needs one of these functions calls
the wrapper of the same name here and never needs to know which backend is active.

Dispatch order (non-numpy arrays)
---------------------------------
1. **scipy native dispatch**, but ONLY if the user has set the environment variable
   ``SCIPY_ARRAY_API=1`` before importing scipy. The result is accepted only if it comes back in the
   same array namespace as the input; some scipy functions silently return numpy arrays instead.
2. **The backend's own scipy**, where it has an equivalent (``cupyx.scipy`` for cupy, ``jax.scipy``
   for jax). Functions or options it does not support (e.g. spline orders above 1 in
   ``jax.scipy.ndimage.map_coordinates``) fall through to the next step.
3. **Host round trip**: convert to numpy, run scipy, convert the result back with ``xp.asarray``.
   Always correct, but on a GPU it costs two device/host transfers, so a debug message is logged.

For numpy arrays scipy is called directly, with no dispatch overhead.

Notes for maintainers
---------------------
* **POPPY must never set ``SCIPY_ARRAY_API`` itself.** It is a process-wide switch that changes the
  behavior of all scipy code in the user's session and has to be set before scipy is imported, so it
  is left to the user. These wrappers merely detect it (via the environment) and use it if present.
* Without ``SCIPY_ARRAY_API``, scipy does not raise for non-numpy arrays; it silently converts them to
  numpy and returns numpy. That is why step 3 converts explicitly and step 1 checks the result type,
  instead of relying on exceptions.
* Which functions dispatch natively depends on the scipy version, so nothing here is hard-coded from a
  table; each step is attempted and verified at call time. As a snapshot (scipy 1.18, jax 0.10): native
  dispatch works for ``ndimage.zoom/rotate/shift/map_coordinates/gaussian_filter``, ``signal.fftconvolve``
  and ``special.j1``, but ``zoom/rotate/shift/gaussian_filter`` on jax are an internal host round trip
  and cannot be jitted; only ``map_coordinates``, ``fftconvolve`` and ``j1`` are truly native.
  ``special.jn`` and ``linalg.lstsq`` are not dispatched at all. ``jax.scipy`` has only
  ``map_coordinates`` and ``signal.fftconvolve`` of the functions below.
* All array arguments of one call must be in the same namespace. With ``SCIPY_ARRAY_API=1`` scipy raises
  a ``TypeError`` for mixed numpy/jax inputs, so wrappers coerce their array arguments to the
  namespace of the first one.
* Scipy functions that are only applied to small host-side data (``interpolate.interp1d``,
  ``RectBivariateSpline``, ``binary_erosion`` on setup masks, ``gaussian_filter`` on FITS data) are
  deliberately NOT wrapped; they stay on numpy behind ``accel_math.to_numpy``.
* Stage 2 (jit-compatible jax): the host-round-trip functions above will need pure-``xp``
  implementations (for example built from ``map_coordinates`` or matrix-multiply interpolation),
  and data-dependent output shapes must be avoided.
"""

import importlib
import logging
import os

import scipy.ndimage
import scipy.signal
import scipy.special
from array_api_compat import array_namespace, is_cupy_namespace, is_jax_namespace, is_numpy_namespace

from .accel_math import to_numpy

_log = logging.getLogger('poppy')

__all__ = ['zoom', 'rotate', 'shift', 'map_coordinates', 'fftconvolve', 'j1']

# Read once; scipy itself reads this variable when it is first imported.
_NATIVE_DISPATCH = os.environ.get("SCIPY_ARRAY_API", "").lower() not in ("", "0", "false")

_BACKEND_SCIPY = (
    (is_cupy_namespace, "cupyx.scipy"),
    (is_jax_namespace, "jax.scipy"),
)


def _backend_function(xp_, module, name):
    """Return the equivalent function in the backend's own scipy, or None if it has none."""
    for is_backend, package in _BACKEND_SCIPY:
        if is_backend(xp_):
            try:
                return getattr(importlib.import_module(f"{package}.{module}"), name)
            except (ImportError, AttributeError):
                return None
    return None


def _same_namespace(result, xp_):
    try:
        return array_namespace(result) is xp_
    except TypeError:
        return False


def _dispatch(module, name, arrays, call):
    """Call ``scipy.<module>.<name>`` for the namespace of ``arrays``.

    Parameters
    ----------
    module, name : str
        scipy submodule and function name, e.g. ``"ndimage"``, ``"zoom"``.
    arrays : sequence of arrays
        The array arguments; they are coerced to the namespace of the first one.
    call : callable
        ``call(function, *arrays)`` performs the call with the given function and (coerced) arrays.
    """
    xp_ = array_namespace(arrays[0])
    scipy_fn = getattr(getattr(scipy, module), name)
    if is_numpy_namespace(xp_):
        return call(scipy_fn, *arrays)

    arrays = [xp_.asarray(a) for a in arrays]
    qualname = f"scipy.{module}.{name}"

    if _NATIVE_DISPATCH:
        try:
            result = call(scipy_fn, *arrays)
            if _same_namespace(result, xp_):
                return result
            _log.debug(f"{qualname} did not return a {xp_.__name__} array; trying alternatives")
        except (TypeError, NotImplementedError, ValueError) as err:
            _log.debug(f"{qualname} native array dispatch failed ({err}); trying alternatives")

    backend_fn = _backend_function(xp_, module, name)
    if backend_fn is not None:
        try:
            return call(backend_fn, *arrays)
        except (TypeError, NotImplementedError, ValueError) as err:
            _log.debug(f"{backend_fn.__module__}.{name} could not handle this call ({err}); using host fallback")

    _log.debug(f"{qualname}: host round trip for {xp_.__name__} arrays")
    result = call(scipy_fn, *[to_numpy(a) for a in arrays])
    return xp_.asarray(result)


def zoom(input, zoom, **kwargs):
    """Wrapper for :py:func:`scipy.ndimage.zoom`; output has the same dtype as the input."""
    return _dispatch("ndimage", "zoom", (input,), lambda fn, x: fn(x, zoom, **kwargs))


def rotate(input, angle, **kwargs):
    """Wrapper for :py:func:`scipy.ndimage.rotate`."""
    return _dispatch("ndimage", "rotate", (input,), lambda fn, x: fn(x, angle, **kwargs))


def shift(input, shift, **kwargs):
    """Wrapper for :py:func:`scipy.ndimage.shift`."""
    return _dispatch("ndimage", "shift", (input,), lambda fn, x: fn(x, shift, **kwargs))


def map_coordinates(input, coordinates, **kwargs):
    """Wrapper for :py:func:`scipy.ndimage.map_coordinates`."""
    return _dispatch("ndimage", "map_coordinates", (input, coordinates), lambda fn, x, c: fn(x, c, **kwargs))


def fftconvolve(in1, in2, **kwargs):
    """Wrapper for :py:func:`scipy.signal.fftconvolve`."""
    return _dispatch("signal", "fftconvolve", (in1, in2), lambda fn, a, b: fn(a, b, **kwargs))


def j1(x):
    """Wrapper for :py:func:`scipy.special.j1`."""
    return _dispatch("special", "j1", (x,), lambda fn, a: fn(a))
