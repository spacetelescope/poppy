

Installation
==================

POPPY may be installed from PyPI in the usual manner for Python packages::

  % pip install poppy --upgrade


The source code is hosted in `this repository on GitHub <https://github.com/mperrin/poppy>`_.

.. note::
                                                                                                                                             Users at STScI may also access POPPY through the standard `SSB software distributions <http://ssb.stsci.edu/ssb_software.shtml>`_.




.. comment:
        .. caution::
        Some users have reported problems installing on recent versions of Mac OS, due to a known issue with
        Mac OS no longer defining certain environment variables by default for text encoding format.
        If you have trouble installing ``poppy``, try setting the following::
                # set if on OSX (tcsh shell syntax)
                setenv LANG "en_US.UTF-8"
                setenv LC_ALL "en_US.UTF-8"
                setenv LC_CTYPE "en_US.UTF-8"
                # set if on OSX (bash shell syntax)
                export LANG="en_US.UTF-8"
                export LC_ALL="en_US.UTF-8"
                export LC_CTYPE="en_US.UTF-8"
        For more information see http://stackoverflow.com/questions/7165108/in-osx-lion-lang-is-not-set-to-utf8-how-fix



Requirements
--------------

* The standard Python scientific stack: ``numpy``, ``scipy``, ``matplotlib``
* `astropy <http://astropy.org>`_, 0.4 or more recent.

The following are *optional*.
The first, ``pysynphot``, is recommended for most users. The other optional installs are only worth adding for speed improvements if you are spending substantial time running calculations.

* `pysynphot <https://pypi.python.org/pypi/pysynphot>`_ enables the simulation of PSFs with proper spectral response to realistic source spectra.  Without this, PSF fidelity is reduced. See below for :ref:`installation instructions for pysynphot <pysynphot_install>`. 
* `psutil <https://pypi.python.org/pypi/psutil>`_ enables slightly better automatic selection of numbers of processes for multiprocess calculations.
* `pyFFTW <https://pypi.python.org/pypi/pyFFTW>`_. The FFTW library can speed up the FFTs used in coronagraphic simulations and slit spectroscopy. Since direct imaging simulations use a discrete matrix FFT instead, direct imaging simulation speed is unchanged.  pyFFTW is recommended if you expect to perform many coronagraphic calculations, particularly for MIRI.  (Note: POPPY previously made use of the PyFFTW3 package, which is *different* from pyFFTW. The latter is more actively maintained and supported today, hence the switch.) 

.. _pysynphot_install:

Installing or updating pysynphot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Pysynphot <https://pypi.python.org/pypi/pysynphot>`_ is an optional dependency, but as noted above is highly recommended. 



.. comment 
        To install or update ``pysynphot``, do the following. (See also http://stsdas.stsci.edu/pysynphot/ and https://trac6.assembla.com/astrolib). WebbPSF has most recently been tested using pysynphot 0.9.5 but is known to work well with earlier versions as well.
        .. warning::
   You may have trouble installing pysynphot, as the zip file of the source on pypi is broken. This has been
   communicated upstream but not yet fixed. You may have more luck installing from an updated zip file 
   on testpypi: https://testpypi.python.org/pypi/pysynphot/0.9.5
        work without this update but computations will be slower than the current version, so we recommend updating it. 
    1. Download the most recent version of pysynphot from https://trac6.assembla.com/astrolib. 
    2. Untar that file into a temporary working directory. 
    3. run ``python setup.py install`` in that directory.  You can delete the setup files there after you do this step. 

Pysynphot is now also available from PyPI::

   %  pip install pysynphot

If this is your initial installation of ``pysynphot`` you need to install the CDBS files. See the `pysynphot installation guide <https://trac6.assembla.com/astrolib/wiki/PysynphotInstallationGuide>`_. The necessary files are available from http://www.stsci.edu/hst/observatory/crds/cdbs_throughput.html; follow the download links for "throughput files" and "Castelli and Kurucz spectral atlast". If you already have CDBS installed, then you're all set and can skip this step.



