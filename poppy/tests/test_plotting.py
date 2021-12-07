# see http://matplotlib.org/devel/testing.html

pass
# Never mind the following - I tried it but it seems to fail due to
# a matplotlib issue:
#
# ___________________ ERROR collecting poppy/tests/test_plotting.py ____________________
# poppy/tests/test_plotting.py:5: in <module>
# >   from matplotlib.testing.decorators import image_comparison
# /Users/spacetelescope/software/macports/Library/Frameworks/Python.framework/Versions/2.7/lib/
# python2.7/site-packages/matplotlib/testing/decorators.py:7: in <module>
# >   import matplotlib.tests
# E   ImportError: No module named tests
# ======================== 13 passed, 1 error in 65.11 seconds =========================


# import numpy as np
# import matplotlib
# from matplotlib.testing.decorators import image_comparison
# import matplotlib.pyplot as plt
#
# from .. import poppy_core
# from .. import utils
#
# @image_comparison(baseline_images=['simple_airy_display'])
# def test_simple_airy_display():
#    """ For one specific geometry, test that we get the expected plot based on a prior reference
#    calculation."""
#
#    osys = poppy_core.OpticalSystem("test", oversample=1)
#    osys.add_pupil(function='Circle', radius=1)
#    osys.add_detector(pixelscale=0.1, fov_arcsec=5.0)
#      # use a large FOV so we grab essentially all the light and conserve flux
#
#    psf = osys.calc_psf(wavelength=1.0e-6)
#
#    utils.display_psf(psf)
#
#    # we need to be a little careful here due to floating point math comparision equality issues...
#    # Can't just do a strict equality
#    assert abs(psf[0].data.max() - 0.201) < 0.001
#
#

