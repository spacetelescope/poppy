Installation
==================

POPPY may be installed one of three different ways.

1. Using ``conda`` through the `AstroConda channel <https://astroconda.readthedocs.io/en/latest/>`__. This is the recommended channel for most users on MacOS and Linux. But note that AstroConda does not support Windows.

2. Using PyPi in the usual manner for Python packages::

    % pip install poppy --upgrade

3. Cloning the source code hosted in `this repository on GitHub <https://github.com/spacetelescope/poppy>`_. It is possible to directly install the latest development version using your locally installed ``git`` package::

    % git clone https://github.com/spacetelescope/poppy.git
    % cd poppy
    % pip install -e .


Requirements
--------------

* Python 3.7, or more recent.
* The standard Python scientific stack: :py:mod:`numpy`, :py:mod:`scipy`,
  :py:mod:`matplotlib`
* POPPY relies upon the `astropy
  <http://www.astropy.org>`__ community-developed core library for astronomy.


The following are *optional*.  The first, :py:mod:`synphot`, is recommended
for most users. The other optional installs are only worth adding for speed
improvements if you are spending substantial time running calculations. See
:ref:`the appendix on performance optimization <performance_and_parallelization>` for details.

* `synphot <https://synphot.readthedocs.io>`_ enables the simulation
  of PSFs with proper spectral response to realistic source spectra.  Without
  this, PSF fidelity is reduced. See below for :ref:`installation instructions
  for synphot <synphot_install>`.
* `psutil <https://pypi.python.org/pypi/psutil>`__ enables slightly better
  automatic selection of numbers of processes for multiprocess calculations.
* `pyFFTW <https://pypi.python.org/pypi/pyFFTW>`__. The FFTW library can speed
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
* Anaconda `accelerate <https://docs.anaconda.com/accelerate/>`_ and
  `numexpr <http://numexpr.readthedocs.io/en/latest/user_guide.html>`_.
  These optionally can provide improved performance particularly in the
  Fresnel code.

.. _synphot_install:

Installing or updating synphot
--------------------------------

`synphot <https://synphot.readthedocs.io>`_ is an optional dependency, but is highly recommended.
See the `synphot installation docs here <https://synphot.readthedocs.io/en/latest/#installation-and-setup>`_
to install ``synphot`` and (even more optionally) some of its TRDS data files.

*The minimum needed to have stellar spectral models available for use when
creating PSFs is synphot itself plus just one of the TRDS data files: the Castelli & Kurucz stellar atlas, file*
`synphot3_castelli-kurucz-2004.tar <https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_castelli-kurucz-2004-atlas_multi_v1_synphot3.tar>`_ (18
MB). Feel free to ignore the rest of the many GB of synphot TRDS files unless you know you want a larger set of
input spectra or need the reference files for other purposes.


Testing your installation of poppy
----------------------------------

Poppy includes a suite of unit tests that exercise its functionality and verify
outputs match expectations. If you have cloned the repository, you can optionally
run this test suite to verify that your installation is working properly::

   % cd poppy/tests/
   % pytest
   ============================ test session starts =====================================
   Python 3.7.9, pytest-6.2.3, py-1.9.0, pluggy-0.13.1
   ... [etc] ...
   ===================== 166 passed, 9 skipped in 410.12s (0:06:50 ======================

Some tests may be automatically skipped depending on whether certain optional packaged are
installed, and other tests in development may be marked "expected to fail" (``xfail``), but
as long as no tests actually fail then your installation is working as expected.
