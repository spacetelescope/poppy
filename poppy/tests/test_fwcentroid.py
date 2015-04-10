
import numpy as np
from .. import fwcentroid
from ..fwcentroid import test_fwcentroid

from .test_errorhandling import _exception_message_starts_with

# fwcentroid is a standalong package that's just included as a copy in
# poppy. It has its own test function

def test_fwcentroid_square(n=20,):
    test_fwcentroid(n=n,verbose=False)

def test_fwcentroid_rectangle(n=20,):
    test_fwcentroid(n=20,halfwidth=[5,7],verbose=False)


try:
    import pytest
    _HAVE_PYTEST = True
except:
    _HAVE_PYTEST = False

if _HAVE_PYTEST:
    def test_checkbox_disabled():
        """ This is mostly here to get Pytest 100% coverage on fwcentroid - 
        should be replaced when/if we implement the checkbox option.
        """

        with pytest.raises(NotImplementedError) as excinfo:
            fwcentroid.fwcentroid(np.ones((10,10)), checkbox=5)
        _exception_message_starts_with(excinfo, "Checkbox smoothing not done yet")



#
#def gaussian(height, center_x, center_y, width_x, width_y):
#    """Returns a gaussian function with the given parameters"""
#    width_x = float(width_x)
#    width_y = float(width_y)
#    return lambda x,y: height*np.exp(
#                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
#
#
#def makegaussian(size=128, center=(64,64), width=5):
#    x = np.arange(size)[np.newaxis,:]
#    y = np.arange(size)[:,np.newaxis]
#    arr = gaussian(1, center[0], center[1], width, width)(x,y)
#    return arr
#
#
#def test_fwcentroid(n=20, width=5, halfwidth=5, **kwargs):
#    """ Test floating window centroid position is accurate to within
#    5e-3 pixels for a set of randomly positioned Gaussians, based
#    on the rms position offset over that set
#    """
#
#    #print("Performing {0} tests using Gaussian PSF with width={1:.1f}, centroid halfwidth= {2:.1f}".format(n,width, halfwidth))
#    
#    diffx = np.zeros(n)
#    diffy = np.zeros(n)
#    size = 100
#    for i in range(n):
#        coords = np.random.uniform(halfwidth+1,size-halfwidth-1,(2))
#        im = makegaussian(size=size, center=coords, width=width) #, **kwargs)
#        measy, measx = fwcentroid(im, halfwidth=halfwidth, **kwargs)
#        diffx[i] = coords[0] - measx
#        diffy[i] = coords[1] - measy
#
#    assert np.sqrt(np.mean(diffx**2+diffy**2)) < 5e-3
#
#    #print("RMS measured position error, X: {0} pixels".format(diffx.std()) )
#    #print("RMS measured position error, Y: {0} pixels".format(diffy.std()) )
#


