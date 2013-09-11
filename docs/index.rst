Documentation for POPPY
=========================

POPPY (Physical Optics Propagation in PYthon) is 
a Python package that simulates physical optical propagation including diffraction. 
It implements a flexible framework for modeling Fraunhofer (far-field) diffraction
and point spread function formation, particularly in the context of astronomical telescopes.
POPPY was developed as part of a simulation package for JWST, but is more broadly applicable to many kinds of 
imaging simulations. 

While this current version only supports far-field calculations, future versions may add
near-field (Fresnel) calculations as well, if interest and usage warrant that. 



POPPY is developed and maintained primarily by Marshall Perrin. Questions, comments, and
code additions always welcome.




**What this software does:**

* Allows users to define an optical system consisting of multiple image and pupil planes
* Provides flexible and extensible optical element classes, including a wide variety of stops, masks, lenses and other optics
* Computes monochromatic and polychromatic point spread functions through those optics
* Provides an extensible framework for defining models of astronomical instruments, including
  selection of broad- and narrow-band filters, selectable optical components such as pupil stops, etc.

**What this software does not do:**

* Fresnel, Talbot, or Huygens propagation.
* Any kind of detector noise or imperfections modeling. 


Requirements
--------------

* The standard Python scientific stack: numpy, scipy, matplotlib
* `astropy <http://astropy.org>`_, 0.2 or more recent, in particular its ``astropy.io.fits`` and ``astropy.io.ascii`` components.

These are optional but recommended:

* `pysynphot <https://trac6.assembla.com/astrolib>`_ enables the simulation of PSFs with proper spectral response to realistic source spectra.  Without this, PSF fidelity is reduced. See below for :ref:`installation instructions for pysynphot <pysynphot_install>`.
* `pyFFTW3 <http://pypi.python.org/pypi/PyFFTW3/0.2.1>`_. The FFTW library will significantly speed up the FFTs used in coronagraphic simulations. Since direct imaging simulations use a discrete matrix FT instead, direct imaging simulation speed is unchanged.  pyFFTW3 is highly recommended if you expect to perform many coronagraphic calculations.



Installation
-------------

POPPY may be installed from PyPI in the usual manner for Python packages::

  % pip install poppy --upgrade


The source code is hosted in `this repository on GitHub <https://github.com/mperrin/poppy>`_.


Contents
-----------

.. toctree::
  :maxdepth: 2

  relnotes.rst
  overview.rst
  examples.rst
  classes.rst

