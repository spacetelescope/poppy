

Installation
==================

POPPY may be installed from PyPI in the usual manner for Python packages::

  % pip install poppy --upgrade


The source code is hosted in `this repository on GitHub <https://github.com/mperrin/poppy>`_.


Requirements
--------------

* The standard Python scientific stack: numpy, scipy, matplotlib
* `astropy <http://astropy.org>`_, 0.2 or more recent, in particular its ``astropy.io.fits`` and ``astropy.io.ascii`` components.

These are optional but recommended:


* `pysynphot <https://trac6.assembla.com/astrolib>`_ enables the simulation of PSFs with proper spectral response to realistic source spectra.  Without this, PSF fidelity is reduced. See below for :ref:`installation instructions for pysynphot <pysynphot_install>`. 
* `pyFFTW <https://pypi.python.org/pypi/pyFFTW>`_. The FFTW library can speed up the FFTs used in coronagraphic simulations and slit spectroscopy. Since direct imaging simulations use a discrete matrix FFT instead, direct imaging simulation speed is unchanged.  pyFFTW is recommended if you expect to perform many coronagraphic calculations, particularly for MIRI.  (Note: POPPY previously made use of the PyFFTW3 package, which is *different* from pyFFTW. The latter is more actively maintained and supported today, hence the switch.) 
.. _pysynphot_install:

Installing or updating pysynphot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pysynphot is an optional dependency, but is highly recommended. 

To install or update ``pysynphot``, do the following. (See also http://stsdas.stsci.edu/pysynphot/ and https://trac6.assembla.com/astrolib). WebbPSF has most recently been tested using pysynphot 0.9.5 but is known to work well with earlier versions as well.


.. warning::
   You may have trouble installing pysynphot, as the zip file of the source on pypi is broken. This has been
   communicated upstream but not yet fixed. You may have more luck installing from an updated zip file 
   on testpypi: https://testpypi.python.org/pypi/pysynphot/0.9.5
   To install this, use this command::

     pip install -i https://testpypi.python.org/pypi pysynphot

.. comment 
        work without this update but computations will be slower than the current version, so we recommend updating it. 
    1. Download the most recent version of pysynphot from https://trac6.assembla.com/astrolib. 
    2. Untar that file into a temporary working directory. 
    3. run ``python setup.py install`` in that directory.  You can delete the setup files there after you do this step. 

If this is your initial installation of ``pysynphot`` you need to install the CDBS files. See the `pysynphot installation guide <https://trac6.assembla.com/astrolib/wiki/PysynphotInstallationGuide>`_. The necessary files are available from https://trac6.assembla.com/astrolib; follow the download links for "throughput files" and "model spectra". If you already have CDBS installed, then you're all set and can skip this step.



