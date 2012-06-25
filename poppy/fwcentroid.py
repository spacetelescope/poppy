#!/usr/bin/env python

"""

Implements the robust floating-window first moment centroid algorithm
adopted for JWST target acquisitions.


In brief, the benefit of this particular algorithm is that it provides
robust subpixel centroid measurements. The more straightforward
first-moment centroid calculation becomes biased when the peak is
not precisely centered on a pixel, since then there will necessarily be more 
field of view (and thus more light) on one side or the other. (For instance, if you
use a hypothetical 5x5 box and the peak is located at coordinates (2, 2.25) then an
integer-pixel 5x5 centroid box includes more field of view below the peak than above it.
The floating window algorithm implemented here avoids this by calculating the centroid
over a box that moves in fractional pixel coordinates to align itself with the measured 
centroid, and iterating until convergence.

See JWST technical reports JWST-STScI-001117 and JWST-STScI-001134 for details.

M. Perrin, 2011-02-17, based on IDL code by Jeff Valenti et al. 

"""

import numpy as np

def fwcentroid(image, checkbox=1, maxiterations=20, threshold=1e-4, halfwidth=5, verbose=False):
    """ Implement the Floating-window first moment centroid algorithm
        chosen for JWST target acquisition.

        See JWST-STScI-001117 and JWST-STScI-001134 for details.

        This code makes no attempt to vectorize or optimize for speed;
        it's pretty much just a straight verbatim implementation of the
        IDL-like pseudocode provided in JWST-STScI-001117


        Parameters
        ----------
        image : array_like
            image to centroid
        checkbox : int
            size of moving checkbox for initial peak pixel guess. Default 1
        halfwidth : int
            Half width of the centroid box size (less 1). Specify as a scalar, or a tuple Xhalfwidth, Yhalfwidth.
            Empirical tests suggest this parameter should be at *least* the PSF FWHM for convergence,
            preferably some small factor larger
        maxiterations : int
            Max number of loops. Default 5
        threshold : float
            Position threshold for convergence

        Returns
        --------
        (ycen, xcen) : float tuple
            Measured centroid position. Note that this is returned in Pythonic
            Y,X order for use as array indices, etc.




        -Marshall Perrin 2011-02-11


    """

    if hasattr(halfwidth, '__iter__'):
        XHW, YHW = halfwidth[0:2]
    else:
        XHW, YHW = halfwidth, halfwidth
       


    # Determine starting peak location
    if checkbox >1:
        raise NotImplemented("Checkbox smoothing not done yet")
    else:
        # just use brightest pixel
        w = np.where(image == image.max())
        YPEAK, XPEAK = w[0][0], w[1][0]
        if verbose: print "Peak pixels are %d, %d" % (XPEAK, YPEAK)

   
    # Calculate centroid for first iteration

    SUM = 0.0
    XSUM = 0.0
    XSUM2 = 0.0
    XSUM3 = 0.0
    YSUM = 0.0
    YSUM2 = 0.0
    YSUM3 = 0.0
    CONVERGENCEFLAG = False

    for i in np.arange( 2*XHW+1)+ XPEAK-XHW :
        for j in np.arange( 2*YHW+1) +YPEAK-YHW :
            #print "(%d, %d)" % (i,j)
            #XLOC = i - XPEAK + XHW +1
            #YLOC = j - YPEAK + YHW +1
            XLOC = i
            YLOC = j
            #print "  (%d, %d)" % (XLOC, YLOC)
            SUM += image[j,i]
            XSUM += XLOC * image[j,i]
            XSUM2 += XLOC**2 * image[j,i]
            XSUM3 += XLOC**3 * image[j,i]
            YSUM += YLOC * image[j,i]
            YSUM2 += YLOC**2 * image[j,i]
            YSUM3 += YLOC**3 * image[j,i]
    XCEN = XSUM / SUM
    XMOMENT2 = XSUM2 / SUM
    XMOMENT3 = XSUM3 / SUM
    YCEN = YSUM / SUM
    YMOMENT2 = YSUM2 / SUM
    YMOMENT3 = YSUM3 / SUM

    # MDP edit in debugging: convert from local box coords to full image.
    #XCEN += XPEAK -XHW-1
    #YCEN += YPEAK -YHW-1   #this would be equivalent to removing the XLOC lines?
    # end MDP edit
    
    oldXCEN = XCEN
    oldYCEN = YCEN

    

    if verbose: print( "After initial calc, cent pos is  (%f, %f)" % (XCEN, YCEN))

    # Iteratively calculate centroid until solution converges,
    # use more neighboring pixels and apply weighting: 
    for k in range(maxiterations):
        SUM = 0.0
        XSUM = 0.0
        XSUM2 = 0.0
        XSUM3 = 0.0
        YSUM = 0.0
        YSUM2 = 0.0
        YSUM3 = 0.0
        for i in np.arange( 2*(XHW+1)+1)+ int(oldXCEN)-(XHW+1) :
            for j in np.arange( 2*(YHW+1)+1) +int(oldYCEN)-(YHW+1) :
                #stop()
                #-- Calculate weights
                #Initialize weights to zero:
                XWEIGHT = 0
                YWEIGHT = 0
                #Adjust weights given distance from current centroid:
                XOFF = np.abs(i - oldXCEN)
                YOFF = np.abs(j - oldYCEN)
                #If within original centroid box, set the weight to one:
                if (XOFF <= XHW): XWEIGHT = 1
                elif (XOFF > XHW) and (XOFF < XHW+1):
                    #Else if on the border, then weight needs to be scaled:
                    XWEIGHT = XHW + 1 - XOFF
                #If within original centroid box, set the weight to one:
                if (YOFF <= YHW): YWEIGHT = 1
                elif (YOFF > YHW) and (YOFF < YHW+1):
                    #Else if on the border, then weight needs to be scaled:
                    YWEIGHT = YHW + 1 - YOFF
                WEIGHT = XWEIGHT * YWEIGHT

                #Centroid, second moment, and third moment calculations
                #XLOC = i - int(XCEN) + XHW + 2
                #YLOC = j - int(YCEN) + YHW + 2
                XLOC = i
                YLOC = j

                #print "pix (%d, %d) weight %f" % (i, j, WEIGHT)
                SUM = SUM + image[j,i] * WEIGHT
                XSUM = XSUM + XLOC * image[j,i] * WEIGHT
                XSUM2 = XSUM2 + XLOC**2 * image[j,i] * WEIGHT
                XSUM3 = XSUM3 + XLOC**3 * image[j,i] * WEIGHT
                YSUM = YSUM + YLOC * image[j,i] * WEIGHT
                YSUM2 = YSUM2 + YLOC**2 * image[j,i] * WEIGHT
                YSUM3 = YSUM3 + YLOC**3 * image[j,i] * WEIGHT
        XCEN = XSUM / SUM
        XMOMENT2 = XSUM2 / SUM
        XMOMENT3 = XSUM3 / SUM
        YCEN = YSUM / SUM
        YMOMENT2 = YSUM2 / SUM
        YMOMENT3 = YSUM3 / SUM

        # MDP edit in debugging: convert from local box coords to full image.
        #XCEN += oldXCEN -XHW-1
        #YCEN += oldYCEN -YHW-1   #this would be equivalent to removing the XLOC lines?
 
        if verbose: print( "After iter %d , cent pos is  (%f, %f)" % (k, XCEN, YCEN))
        #Check for convergence:
        if (np.abs(XCEN - oldXCEN) <= threshold and
            np.abs(YCEN - oldYCEN) <= threshold):
            CONVERGENCEFLAG = True
            break
        else:
            CONVERGENCEFLAG = False
            oldXCEN = XCEN
            oldYCEN = YCEN
    if not CONVERGENCEFLAG:
        print("Algorithm terminated at max iterations without convergence.")

    return  YCEN, XCEN


############################

def test_fwcentroid(n=100, width=5, halfwidth=5, **kwargs):
    def gaussian(height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(
                    -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


    def makegaussian(size=128, center=(64,64), width=5):
        x = np.arange(size)[np.newaxis,:]
        y = np.arange(size)[:,np.newaxis]
        arr = gaussian(1, center[0], center[1], width, width)(x,y)
        return arr


    print "Performing %d tests using Gaussian PSF with width=%.1f, centroid halfwidth= %.1f" % (n,width, halfwidth)
    
    diffx = np.zeros(n)
    diffy = np.zeros(n)
    size = 100
    for i in range(n):
        coords = np.random.uniform(halfwidth+1,size-halfwidth-1,(2))
        im = makegaussian(size=size, center=coords, width=width) #, **kwargs)
        measy, measx = fwcentroid(im, halfwidth=halfwidth, **kwargs)
        diffx[i] = coords[0] - measx
        diffy[i] = coords[1] - measy

    print "RMS measured position error, X: %f pixels" % diffx.std()
    print "RMS measured position error, Y: %f pixels" % diffy.std()



if __name__ == "__main__":

    test_fwcentroid()



