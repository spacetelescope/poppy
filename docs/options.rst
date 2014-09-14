Options
=================


Output PSF Normalization
--------------------------


Output PSFs can be normalized in different ways, based on the ``normalization`` keyword to ``calcPSF``. The options are: 

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
