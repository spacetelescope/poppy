import numpy as np

from .. import utils

def test_padToSize():
    square = np.ones((300,300))

    for desiredshape in [ (500, 500), (400,632), (2048, 312)]:
        newshape = utils.padToSize(square, desiredshape).shape 
        for i in [0,1]: assert newshape[i] == desiredshape[i]


