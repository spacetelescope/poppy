Installation
==================

POPPY may be installed from PyPI in the usual manner for Python packages::

   % pip install poppy --upgrade

The source code is hosted in `this repository on GitHub <https://github.com/mperrin/poppy>`_. It is possible to directly install the latest development version from git::

   % git clone https://github.com/mperrin/poppy.git
   % cd poppy
   % pip install -e .

.. note::
   Users at STScI may also access POPPY through the standard `SSB software distributions <http://ssb.stsci.edu/ssb_software.shtml>`_.

Requirements
--------------

* Either Python 2.7 or Python 3.4.  (Under :ref:`certain circumstances on Mac OS <accelerated_multiprocessing>`, Python 3.4 is required if you want parallelized calculations, but otherwise POPPY functionality is identical on both versions.)
* The standard Python scientific stack: :py:mod:`numpy`, :py:mod:`scipy`, :py:mod:`matplotlib`
* :py:mod:`astropy`, 0.4 or more recent, from http://astropy.org

The following are *optional*.
The first, :py:mod:`pysynphot`, is recommended for most users. The other optional installs are only worth adding for speed improvements if you are spending substantial time running calculations.

* `pysynphot <https://pypi.python.org/pypi/pysynphot>`_ enables the simulation
  of PSFs with proper spectral response to realistic source spectra.  Without
  this, PSF fidelity is reduced. See below for :ref:`installation instructions
  for pysynphot <pysynphot_install>`. 
* `psutil <https://pypi.python.org/pypi/psutil>`_ enables slightly better
  automatic selection of numbers of processes for multiprocess calculations.
* `pyFFTW <https://pypi.python.org/pypi/pyFFTW>`_. The FFTW library can speed
  up the FFTs used in multi-plane optical simulations such as coronagraphiy or
  slit spectroscopy. Since direct imaging simulations use a discrete matrix FFT
  instead, direct imaging simulation speed is unchanged.  pyFFTW is recommended
  if you expect to perform many coronagraphic calculations, particularly for
  MIRI.  (Note: POPPY previously made use of the PyFFTW3 package, which is
  *different* from pyFFTW.  The latter is more actively maintained and
  supported today, hence the switch.  Note also that some users have reported
  intermittent stability issues with pyFFTW for reasons that are not yet
  clear.) *At this time we recommend most users should skip installing pyFFTW
  while getting started with poppy*.

.. _pysynphot_install:

Installing or updating pysynphot
----------------------------------

Pysynphot is an optional dependency, but is highly recommended.

To install or update to the latest version of :py:mod:`pysynphot`, simply invoke ``pip install -U pysynphot``.

If you already have the CDBS data package installed, or are using a machine at STScI, then you can simply set the ``PYSYN_CDBS`` environment variable to point to the CDBS files.

If this is your initial installation of :py:mod:`pysynphot`, you will need to install the CDBS files. These are available from STScI in DMG form for Mac users, as well as in gzipped tar format.

**Installing CDBS on Mac:** To obtain the DMG, consult the "Installing CDBS locally on a Mac" section of http://ssb.stsci.edu/ssb_software.shtml. Download the DMG and open it to find ``cdbs.pkg``. Running this graphical installer will place the CDBS files in ``/usr/stsci/stdata``. Set the environment variable ``PYSYN_CDBS`` to point to that directory, e.g. ``setenv PYSYN_CDBS /usr/stsci/stdata`` for tcsh/csh or ``export PYSYN_CDBS="/usr/stsci/stdata"`` for bash.

**Installing CDBS from tar archives**: To obtain the tar files, consult http://www.stsci.edu/hst/observatory/crds/cdbs_throughput.html. Download the archives numbered ``synphot[1-6].tar.gz`` and extract them to a directory such as ``$HOME/data/CDBS``.
Set the environment variable ``PYSYN_CDBS`` to point to that directory. e.g. ``setenv PYSYN_CDBS $HOME/data/CDBS`` for tcsh/csh or ``export PYSYN_CDBS="$HOME/data/CDBS"``.


Testing your installation of poppy
----------------------------------

Poppy includes a suite of unit tests that exercise its functionality and verify outputs match expectations. You can optionally 
run this test suite to verify that your installation is working properly::

   >>> import poppy
   >>> poppy.test()
   ============================ test session starts =====================================
   platform darwin -- Python 2.7.8 -- pytest-2.5.1
   Running tests with Astropy version 0.4.1.
   ... [etc] ...
   ================= 66 passed, 1 skipped, 1 xfailed in 124.68 seconds ==================

Some tests may be automatically skipped depending on whether certain optional packaged are
installed, and other tests in development may be marked "expected to fail" (``xfail``), but 
as long as no tests actually fail then your installation is working as expected.
