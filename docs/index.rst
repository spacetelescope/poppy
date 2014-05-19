Documentation for POPPY
=========================

POPPY (Physical Optics Propagation in PYthon) is 
a Python package that simulates physical optical propagation including diffraction. 
It implements a flexible framework for modeling Fraunhofer (far-field) diffraction
and point spread function formation, particularly in the context of astronomical telescopes.
POPPY was developed as part of a simulation package for JWST, but is more broadly applicable to many kinds of 
imaging simulations. 



Summary
------------


**What this software does:**

* Allows users to define an optical system consisting of multiple image and pupil planes
* Provides flexible and extensible optical element classes, including a wide variety of stops, masks, lenses and other optics
* Computes monochromatic and polychromatic point spread functions through those optics
* Provides an extensible framework for defining models of astronomical instruments, including
  selection of broad- and narrow-band filters, selectable optical components such as pupil stops, etc.

**What this software does not do:**

* Fresnel, Talbot, or Huygens propagation.
* Any kind of detector noise or imperfections modeling. 

While this current version only supports far-field calculations, future versions may add
near-field (Fresnel) calculations as well, if interest and usage warrant that. 



Contents
-----------

.. toctree::
  :maxdepth: 2

<<<<<<< HEAD
  installation.rst
  relnotes.rst
  overview.rst
  examples.rst
  classes.rst


Getting Help
-------------------
POPPY is developed and maintained primarily by Marshall Perrin. Questions, comments, and
code additions always welcome.

The source code is available `on Github <https://github.com/mperrin/poppy>`_.


