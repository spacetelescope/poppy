import poppy
from poppy.poppy_core import PlaneType, _USE_CUDA, _USE_NUMEXPR
import numpy as np

if _USE_NUMEXPR:
    import numexpr as ne

if _USE_CUDA:
    from numba import cuda

def _exp(x):
    """
    Function to speed up taking exponential of an array if NumExpr is available.
    Otherwise defaults to np.exp()

    """
    if _USE_NUMEXPR:
        return  ne.evaluate("exp(x)", optimization='moderate',)
    else:
        return np.exp(x)

def _fftshift(x):
    
    N=x.shape[0]
    if (_USE_CUDA) & (N==x.shape[1]):
        blockdim = (32, 32) # threads per block
        numBlocks = (int(N/blockdim[0]),int(N/blockdim[1]))
        cufftShift_2D_kernel[numBlocks, blockdim](x.ravel(),N)
        return x
    else:
        return np.fft.fftshift(x)
    
if  _USE_CUDA:  
    @cuda.jit()
    def cufftShift_2D_kernel(data, N):
        '''
        adopted CUDA FFT shift code from:
        https://github.com/marwan-abdellah/cufftShift
        (GNU Lesser Public License)
        '''
        
        #// 2D Slice & 1D Line
        sLine = N
        sSlice = N * N
        #// Transformations Equations
        sEq1 = int((sSlice + sLine) / 2)
        sEq2 = int((sSlice - sLine) / 2)
        x, y = cuda.grid(2)
        #// Thread Index Converted into 1D Index
        index = (y * N) + x
            
        if (x < N / 2):
            if (y < N / 2):
                #// First Quad
                temp = data[index]
                data[index] = data[index + sEq1]
            #// Third Quad
                data[index + sEq1] = temp
        else:
            if (y < N / 2):
                #// Second Quad
                temp=data[index]
                data[index] = data[index + sEq2]
                data[index + sEq2] = temp
                
                
