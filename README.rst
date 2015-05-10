================================================
poppy: Physical Optics Propagation in PYthon
================================================


POPPY (Physical Optics Propagation in PYthon) is a Python package that
simulates physical optical propagation including diffraction. It implements a
flexible framework for modeling Fraunhofer (far-field) diffraction and point
spread function formation, particularly in the context of astronomical
telescopes. 

POPPY was developed as part of a simulation package for the James Webb Space
Telescope, but is more broadly applicable to many kinds of imaging simulations.
It is not, however, a substitute for high fidelity physical optical design
software such as Zemax or Code V, but rather is intended as a lightweight
alternative for cases for which far-field diffraction is sufficient and which
require portability between platforms or ease of scripting. 


For documentation, see https://pythonhosted.org/poppy/


Code by Marshall Perrin, Joseph Long, Ewan Douglas, with additional
contributions from Anand Sivaramakrishnan, Remi Soummer, Christine Slocum, 
and others on the Astropy team. 




Status reports for developers
-----------------------------

.. image:: https://pypip.in/v/poppy/badge.png
    :target: https://pypi.python.org/pypi/poppy

.. image:: https://pypip.in/d/poppy/badge.png
    :target: https://pypi.python.org/pypi/poppy

.. image:: https://travis-ci.org/mperrin/poppy.png?branch=master
    :target: https://travis-ci.org/mperrin/poppy
    :alt: Test Status

.. image:: https://coveralls.io/repos/mperrin/poppy/badge.svg 
    :target: https://coveralls.io/r/mperrin/poppy
    :alt: Test Coverage
