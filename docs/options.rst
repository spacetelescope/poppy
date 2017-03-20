Options
=================


Output PSF Normalization
--------------------------


Output PSFs can be normalized in different ways, based on the ``normalization`` keyword to ``calc_psf``. The options are: 

  * ``normalize="first"``: The wavefront is normalized to total intensity 1 over the entrance pupil. If there are obstructions downstream of the entrance pupil (e.g. coronagraph masks) then the output PSF intensity will be < 1. 
  * ``normalize="last"``: The output PSF's integrated total intensity is normalized to 1.0, over whatever FOV that PSF has. (Note that this 
  * ``normalize="exit_pupil"``: The wavefront is normalized to total intensity 1 at the exit pupil, i.e. the last pupil in the optical system. This means that the output PSF will have total intensity 1.0 if integrated over an arbitrarily large aperture. The total intensity over any finite aperture will be some number less than one. In other words, this option is equivalent to saying "Normalize the PSF to have integrated intensity 1 over an infinite aperture."


Logging
------------------

As noted on the :ref:`examples` page, Poppy uses the Python ``logging`` mechanism for log message display. The default "info" level provides a modest amount of insight into the major steps of a calculation; the "debug" level provides an exhaustive and lengthy description of everything you could possibly want to know. You can switch between these like so::


        import logging
        logging.basicConfig(level=logging.INFO)
        logging.basicConfig(level=logging.DEBUG)

See the `python logging docs <https://docs.python.org/2/library/logging.html>`_ for more information and extensive options for directing log output to screen or file.



Configuration 
-------------------

Poppy makes use of the `Astropy configuration system <http://astropy.readthedocs.org/en/stable/config/index.html>`_ to store settings persistently between sessions. 
These settings are stored in a file in the user's home directory, for instance ``~/.astropy/config/poppy.cfg``. Edit this text file to adjust settings. 


=========================== =============================================================   ===================
Setting                     Description                                                     Default
=========================== =============================================================   ===================
use_multiprocessing         Should PSF calculations run in parallel using multiple          False
                            processors?                             

n_processes                 Maximum number of additional worker processes to spawn.         4
use_fftw                    Should the pyFFTW library be used (if it is present)?           True
autosave_fftw_wisdom        Should POPPY automatically save and reload FFTW 'wisdom'        True
                            (i.e. timing measurements of different FFT variants)
default_image_display_fov   Default display field of view for PSFs, in arcsec               5
default_logging_level       Default verbosity of logging to Python's logging framework      INFO
enable_speed_tests          Enable additional verbose logging of execution timing           False
enable_flux_tests           Enable additional verbose logging of flux conservation tests    False
=========================== =============================================================   ===================

