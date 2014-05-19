import astropy.config

# Package-global configuration items here. 
# segregated into a file "conf" to ease migration to the revamped config system in astropy >= 0.4

use_multiprocessing = astropy.config.ConfigurationItem('use_multiprocessing', False, 'Should PSF calculations run in parallel using multiple processers using the Python multiprocessing framework (if True; faster but does not allow display of each wavelength) or run serially in a single process(if False; slower but shows the calculation in progress. Also a bit more robust.?)')


# Caution: Do not make this next too large on high-CPU-count machines
# because this is a memory-intensive calculation and you willg
# just end up thrashing IO and swapping out a ton, so everything
# becomes super slow.
n_processes = astropy.config.ConfigurationItem('n_processes', 4, 'Maximum number of additional worker processes to spawn. PSF calculations are likely RAM limited more than CPU limited for higher N on modern machines.')

use_fftw = astropy.config.ConfigurationItem('use_fftw', True, 'Use FFTW for FFTs (assuming it is available)?  Set to False to force numpy.fft always, True to try importing and using FFTW via PyFFTW.')
autosave_fftw_wisdom=  astropy.config.ConfigurationItem('autosave_fftw_wisdom', True, 'Should POPPY automatically save and reload FFTW "wisdom" for improved speed?')

default_logging_level = astropy.config.ConfigurationItem('logging_level', 'INFO', 'Logging verbosity: one of {DEBUG, INFO, WARN, ERROR, or CRITICAL}')

enable_speed_tests =  astropy.config.ConfigurationItem('enable_speed_tests', False, 'Enable additional verbose printout of computation times. Useful for benchmarking.')
enable_flux_tests =  astropy.config.ConfigurationItem('enable_flux_tests', False, 'Enable additional verbose printout of fluxes and flux conservation during calculations. Useful for testing.')



# should be science state in astropy >=0.4 schema:
default_image_display_fov =  astropy.config.ConfigurationItem('default_image_display_fov', 5.0, 'Default image display field of view, in arcseconds. Adjust this to display only a subregion of a larger output array.')


