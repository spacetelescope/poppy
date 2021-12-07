============================================
POPPY: Physical Optics Propagation in Python
============================================

.. image:: docs/figures/readme_fig.png

.. image:: https://img.shields.io/pypi/v/poppy.svg
   :target: https://pypi.python.org/pypi/poppy
   :alt: Badge showing current released PyPI version

.. image:: https://github.com/spacetelescope/poppy/workflows/CI/badge.svg?branch=develop
   :target: https://github.com/spacetelescope/poppy/actions
   :alt: Github Actions CI Status

.. image:: https://codecov.io/gh/spacetelescope/poppy/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/spacetelescope/poppy

.. |Documentation Status| image:: https://img.shields.io/readthedocs/poppy-optics/latest.svg?logo=read%20the%20docs&logoColor=white&label=Docs&version=stable
   :target: https://poppy-optics.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/ascl-1602.018-blue.svg?colorB=262255
   :target: http://ascl.net/1602.018

POPPY (**P**\ hysical **O**\ ptics **P**\ ropagation in **Py**\ thon) is a Python package that simulates physical optical propagation including diffraction. It implements a flexible framework for modeling Fraunhofer and Fresnel diffraction and point spread function formation, particularly in the context of astronomical telescopes.

POPPY was developed as part of a simulation package for the James Webb Space Telescope, but is more broadly applicable to many kinds of imaging simulations. It is not, however, a substitute for high fidelity optical design software such as Zemax or Code V, but rather is intended as a lightweight alternative for cases for which diffractive rather than geometric optics is the topic of interest, and which require portability between platforms or ease of scripting.

For documentation, see http://poppy-optics.readthedocs.io/

Code by Marshall Perrin, Joseph Long, Ewan Douglas, Neil Zimmerman, Anand Sivaramakrishnan, Shannon Osborne, Kyle Douglass, Maciek Grochowicz, Phillip Springer, & Ted Corcovilos, with additional contributions from Remi Soummer, Kyle Van Gorkom, Jonathan Fraine, Christine Slocum, Roman Yurchak, and others on the Astropy team.

Projects using POPPY
--------------------

POPPY provides the optical modeling framework used in:

* WebbPSF, a PSF simulator for NASA's JWST and WFIRST space telescopes. See https://pypi.python.org/pypi/webbpsf
* ``gpipsfs``, a PSF simulator for the Gemini Planet Imager coronagraph. See https://github.com/geminiplanetimager/gpipsfs 

