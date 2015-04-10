.. _poppy_home:

Documentation for POPPY
=========================

POPPY (Physical Optics Propagation in PYthon) 
simulates physical optical propagation including diffraction. 
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
* Provides an extensible framework for defining models of optical instruments, including
  selection of broad- and narrow-band filters, selectable optical components such as pupil stops, etc.

**What this software does not do:**

* Fresnel, Talbot, or Huygens propagation.
* Modelling of any kind of detector noise or imperfections. 

While this current version only supports far-field calculations, future versions may add
near-field (Fresnel) calculations as well, if interest and usage warrant that. 



.. admonition:: Quickstart IPython Notebook

       This documentation is complemented by an `IPython Notebook quickstart tutorial <http://nbviewer.ipython.org/github/mperrin/poppy/blob/master/notebooks/POPPY_tutorial.ipynb>`_.

       Downloading and running that notebook is a great way to get started using POPPY. The documentation following here provides greater details on the algorithms and API.

:ref:`What's New in the latest version? <whatsnew>`
       

Contents
-----------

.. toctree::
  :maxdepth: 1

  installation.rst
  relnotes.rst
  overview.rst
  examples.rst
  options.rst
  classes.rst
  extending.rst
  api.rst
  about.rst
  performance.rst
  fft_optimization.rst


Getting Help
-------------------
POPPY is developed and maintained by :ref:`Marshall Perrin, Joseph Long, and collaborators <about_team>`. Questions, comments, and
pull requests always welcome, either via the `Github repository <https://github.com/mperrin/poppy>`_ or email. 


