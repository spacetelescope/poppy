Appendix B: Optimizing FFT Performance with FFTW
================================================

.. warning::

    This page is obsolete, and superceded by subsequent library development. You probably want
    to use Intel MKL rather than FFTW. See :ref:`_performance_and_parallelization`.

Optimizing numerical performance of FFTs is a complicated subject. Just using the FFTW library is no guarantee of optimal performance; you need to know how to configure it.

.. note::
   The following tests were performed using the older PyFFTW3 package, and have not yet been updated for the newer pyFFTW package. However, performance considerations are expected to be fairly similar for both packages since the underlying FFTW library is the same.

   See discussion and test results at https://github.com/spacetelescope/webbpsf/issues/10

This is probably fairly sensitive to hardware details. The following benchmarks were performed on a Mac Pro, dual quad-core 2.66 GHz Xeon, 12 GB RAM.

 * Unlike many of the array operations in :py:mod:`numpy`, the :py:mod:`~numpy.fft` operation is not threaded for execution across multiple processors. It is thus slow and inefficient.
 * Numpy and Scipy no longer include FFTW, but luckily there is an independently maintained pyfftw3 module.  See https://launchpad.net/pyfftw/
 * Using pyfftw3 can result in a 3-4x speedup for moderately large arrays.  However, there are two significant gotchas to be aware of:

  1) the pyfftw3.Plan() function has a default parameter of nthreads=1. You have to
     explicitly tell it to use multiple threads if you wish to reap the
     rewards of multiple processors.  By default with nthreads=1 it is in fact a
     bit slower than numpy.fft!
  2) The FFTW3 documentation asserts that greater speed can be achieved by using
     arrays which are aligned in memory to 16-byte boundaries. There is a
     fftw3.create_aligned_array() function that created numpy arrays which have
     this property. While I expected using this would make the transforms faster,
     in fact I see significantly better performance when using unaligned arrays.
     (The speed difference becomes larger as array size increases, up to 2x!)
     This is unexpected and not understood, so it may vary by machine and I
     suggest one ought to test this on different machines to see if it is reliable.


Planning in FFTW3
------------------



* Performing plans can take a *long* time, especially if you select exhaustive or patient modes: 
* The default option is 'estimate' which turns out to be really a poor choice.  
* It appears that you can get most of the planning benefit from using the 'measure' option. 
* Curiously, the really time-consuming planning only appears to take place if you do use aligned arrays. 
  If you use regular unaligned arrays, then a very abbreviated planning set is performed, and yet you still
  appear to reap most of the benefits of 



A comparison of different FFT methods
-------------------------------------

This test involves, in each iteration, allocating a new numpy array filled
with random values, passing it to a function, FFTing it, and then returning the
result. Thus it is a fairly realistic test but takes longer per iteration than some of the
other tests presented below on this page. This is noted here in way of explanation for why
there are discrepant values for how long an optimized FFT of a given size takes. 


Test results::

    Doing complex FFT with array size = 1024 x 1024
       for         numpy fft, elapsed time is: 0.094331 s
       for             fftw3, elapsed time is: 0.073848 s
       for    fftw3 threaded, elapsed time is: 0.063143 s
       for fftw3 thr noalign, elapsed time is: 0.020411 s
       for fftw3 thr na inplace, elapsed time is: 0.017340 s
    Doing complex FFT with array size = 2048 x 2048
       for         numpy fft, elapsed time is: 0.390593 s
       for             fftw3, elapsed time is: 0.304292 s
       for    fftw3 threaded, elapsed time is: 0.224193 s
       for fftw3 thr noalign, elapsed time is: 0.061629 s
       for fftw3 thr na inplace, elapsed time is: 0.047997 s
    Doing complex FFT with array size = 4096 x 4096
       for         numpy fft, elapsed time is: 2.190670 s
       for             fftw3, elapsed time is: 1.911555 s
       for    fftw3 threaded, elapsed time is: 1.414653 s
       for fftw3 thr noalign, elapsed time is: 0.332999 s
       for fftw3 thr na inplace, elapsed time is: 0.293531 s



Conclusions: It appears that the most efficient algorithm is a non-aligned in-place FFT.  Therefore, this is the algorithm adopted into POPPY.

In this case, it makes sense that avoiding the alignment is beneficial, since it avoids a memory copy of the
entire array (from regular python unaligned into the special aligned array). 
Another set of tests (results not shown here) indicated that there is no gain in performance from FFTing from an unaligned input array to an aligned output array.


A test comparing all four planning methods
------------------------------------------

This test involves creating one single input array (specifically, a large circle in the central half of the array)
and then repeatedly FFTing that same array. Thus it is pretty much the best possible case and the speeds are very fast.  ::

    For arrays of size 512x512
    Building input circular aperture
            that took 0.024070 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.041177        0.005638 s
            Executing took  0.017639        0.017181 s
     Plan method= measure
            Array alignment True            False
            Planning took   0.328468        0.006960 s
            Executing took  0.001991        0.002741 s
     Plan method= patient
            Array alignment True            False
            Planning took   39.816985       0.020944 s
            Executing took  0.002081        0.002475 s
     Plan method= exhaustive
            Array alignment True            False
            Planning took   478.421909      0.090302 s
            Executing took  0.004974        0.002467 s



A comparison of 'estimate' and 'measure' for different sizes
------------------------------------------------------------


This test involves creating one single input array (specifically, a large circle in the central half of the array)
and then repeatedly FFTing that same array. Thus it is pretty much the best possible case and the speeds are very fast.  ::

    For arrays of size 1024x1024
    Building input circular aperture
            that took 0.120378 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.006557        0.014652 s
            Executing took  0.041282        0.041586 s
     Plan method= measure
            Array alignment True            False
            Planning took   1.434870        0.015797 s
            Executing took  0.008814        0.011852 s

    For arrays of size 2048x2048
    Building input circular aperture
            that took 0.469819 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.006753        0.032270 s
            Executing took  0.098976        0.098925 s
     Plan method= measure
            Array alignment True            False
            Planning took   5.347839        0.033213 s
            Executing took  0.028528        0.047729 s

    For arrays of size 4096x4096
    Building input circular aperture
            that took 2.078152 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.007102        0.056571 s
            Executing took  0.395048        0.326832 s
     Plan method= measure
            Array alignment True            False
            Planning took   17.890278       0.057363 s
            Executing took  0.126414        0.133602 s

    For arrays of size 8192x8192
    Building input circular aperture
            that took 93.043509 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.245359        0.425931 s
            Executing took  2.800093        1.426851 s
     Plan method= measure
            Array alignment True            False
            Planning took   41.203768       0.235688 s
            Executing took  0.599916        0.526022 s


Caching of plans means that irunning the same script a second time is much faster
-----------------------------------------------------------------------------------
Immediately after executing the above, I ran the same script again. Now the planning times all become essentially negligible. 

Oddly, the exection time for the largest array gets longer. I suspect this has something to do with memory or system load.  ::

    For arrays of size 1024x1024
    Building input circular aperture
            that took 0.115704 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.005147        0.015813 s
            Executing took  0.006883        0.011428 s
     Plan method= measure
            Array alignment True            False
            Planning took   0.009078        0.012562 s
            Executing took  0.007057        0.010706 s

    For arrays of size 2048x2048
    Building input circular aperture
            that took 0.421966 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.004888        0.032564 s
            Executing took  0.026869        0.043273 s
     Plan method= measure
            Array alignment True            False
            Planning took   0.019813        0.032273 s
            Executing took  0.027532        0.045452 s

    For arrays of size 4096x4096
    Building input circular aperture
            that took 1.938918 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.005327        0.057813 s
            Executing took  0.123481        0.131502 s
     Plan method= measure
            Array alignment True            False
            Planning took   0.030474        0.057851 s
            Executing took  0.119786        0.134453 s

    For arrays of size 8192x8192
    Building input circular aperture
            that took 78.352433 s
     Plan method= estimate
            Array alignment True            False
            Planning took   0.020330        0.325254 s
            Executing took  0.593469        0.530125 s
     Plan method= measure
            Array alignment True            False
            Planning took   0.147264        0.227571 s
            Executing took  4.640368        0.528359 s


The Payoff: Speed improvements in POPPY
----------------------------------------


For a monochromatic propagation through a 1024x1024 pupil, using 4x oversampling, 
using FFTW results in about a 3x increase in performance. ::

        Using FFTW:         FFT time elapsed:      0.838939 s
        Using Numpy.fft:    FFT time elapsed:      3.010586 s


This leads to substantial savings in total computation time::

        Using FFTW:          TIME 1.218268 s for propagating one wavelength
        Using Numpy.fft:     TIME 3.396681 s for propagating one wavelength





Users are encouraged to try different approaches to optimizing performance on their own machines. 
To enable some rudimentary benchmarking for the FFT section of the code, set `poppy.conf.enable_speed_tests=True` and configure
your logging display to show debug messages. (i.e. `webbpsf.configure_logging('debug')`).
Measured times will be printed in the log stream, for instance like so::

    poppy     : INFO     Calculating PSF with 1 wavelengths
    poppy     : INFO      Propagating wavelength = 1e-06 meters  with weight=1.00
    poppy     : DEBUG    Creating input wavefront with wavelength=0.000001, npix=511, pixel scale=0.007828 meters/pixel
    poppy     : DEBUG      Wavefront and optic Optic from fits.HDUList object already at same plane type, no propagation needed.
    poppy     : DEBUG      Multiplied WF by phasor for Pupil plane: Optic from fits.HDUList object
    poppy     : DEBUG    normalizing at first plane (entrance pupil) to 1.0 total intensity
    poppy     : DEBUG      Propagating wavefront to Image plane: -empty- (Analytic).
    poppy     : DEBUG    conf.use_fftw is True
    poppy     : INFO     using numpy FFT of (511, 511) array
    poppy     : DEBUG    using numpy FFT of (511, 511) array, direction=forward
    poppy     : DEBUG       TIME 0.051085 s  for the FFT                                     # This line
    poppy     : DEBUG      Multiplied WF by phasor for Image plane: -empty- (Analytic)
    poppy     : DEBUG       TIME 0.063745 s for propagating one wavelength                   # and this one
    poppy     : INFO       Calculation completed in 0.082 s
    poppy     : INFO     PSF Calculation completed.
