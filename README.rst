============================================
POPPY: Physical Optics Propagation in Python
============================================

.. image:: docs/figures/readme_fig.png

.. image:: https://img.shields.io/pypi/v/poppy.svg
   :target: https://pypi.python.org/pypi/poppy
   :alt: Badge showing current released PyPI version

.. image:: https://travis-ci.org/mperrin/poppy.svg?branch=master
   :target: https://travis-ci.org/mperrin/poppy
   :alt: Badge showing continuous integration test status

.. image:: https://coveralls.io/repos/mperrin/poppy/badge.svg
   :target: https://coveralls.io/r/mperrin/poppy
   :alt: Badge showing testing code coverage percentage

POPPY (**P**\ hysical **O**\ ptics **P**\ ropagation in **Py**\ thon) is a Python package that simulates physical optical propagation including diffraction. It implements a flexible framework for modeling Fraunhofer and Fresnel diffraction and point spread function formation, particularly in the context of astronomical telescopes.

POPPY was developed as part of a simulation package for the James Webb Space Telescope, but is more broadly applicable to many kinds of imaging simulations. It is not, however, a substitute for high fidelity optical design software such as Zemax or Code V, but rather is intended as a lightweight alternative for cases for which diffractive rather than geometric optics is the topic of interest, and which require portability between platforms or ease of scripting.

For documentation, see https://pythonhosted.org/poppy/

Code by Marshall Perrin, Joseph Long, Ewan Douglas, with additional contributions from Anand Sivaramakrishnan, Remi Soummer, Christine Slocum, and others on the Astropy team.

Projects using POPPY
--------------------

POPPY provides the optical modeling framework used in:

* WebbPSF, a PSF simulator for NASA's JWST and WFIRST space telescopes. See https://pypi.python.org/pypi/webbpsf
* ``gpipsfs``, a PSF simulator for the Gemini Planet Imager coronagraph. See https://github.com/geminiplanetimager/gpipsfs 

