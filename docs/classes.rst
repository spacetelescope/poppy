.. JWST-PSFs documentation master file, created by
   sphinx-quickstart on Mon Nov 29 15:57:01 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: poppy


POPPY Class Documentation
============================

The key classes for POPPY are OpticalSystem and the various OpticalElement classes (of which there are many). There is also a Wavefront class that is used internally, but users will rarely
need to instantiate that directly. Results are returned as FITS HDUList arrays. 

The Instrument class provides a framework for developing high-level models of astronomical instruments; it should not be used directly but rather is subclassed to implement the details of your particular instrument. See its class documentation for more details.


.. inheritance-diagram:: poppy.Detector poppy.Wavefront poppy.OpticalSystem poppy.Rotation poppy.CircularAperture poppy.HexagonAperture poppy.SquareAperture poppy.IdealFieldStop poppy.IdealCircularOcculter poppy.IdealBarOcculter poppy.BandLimitedCoron poppy.IdealFQPM poppy.FQPM_FFT_aligner poppy.CompoundAnalyticOptic poppy.FITSOpticalElement poppy.Instrument poppy.SecondaryObscuration poppy.InverseTransmission poppy.NgonAperture poppy.MultiHexagonAperture


.. _Wavefront:

Wavefront
---------

.. autoclass:: poppy.Wavefront
    :members:

.. OpticalSystem:

Optical System 
--------------

.. autoclass:: poppy.OpticalSystem
    :members:

.. autoclass:: poppy.SemiAnalyticCoronagraph
    :members:




.. OpticalElement:

Optical Elements
----------------

.. autoclass:: poppy.OpticalElement
   :members:


General Optical Elements
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: poppy.FITSOpticalElement
   :members:

.. autoclass:: poppy.AnalyticOpticalElement
   :show-inheritance:
.. autoclass:: poppy.CompoundAnalyticOptic
   :show-inheritance:
.. autoclass:: poppy.InverseTransmission
   :show-inheritance:
.. autoclass:: poppy.Rotation
   :show-inheritance:


Pupil Plane Optical Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: poppy.CircularAperture
   :show-inheritance:
.. autoclass:: poppy.SquareAperture
   :show-inheritance:
.. autoclass:: poppy.RectangleAperture
   :show-inheritance:
.. autoclass:: poppy.HexagonAperture
   :show-inheritance:
.. autoclass:: poppy.MultiHexagonAperture
   :show-inheritance:
.. autoclass:: poppy.NgonAperture
   :show-inheritance:
.. autoclass:: poppy.SecondaryObscuration
   :show-inheritance:

.. autoclass:: poppy.ThinLens
   :show-inheritance:



.. autoclass:: poppy.FQPM_FFT_aligner
   :show-inheritance:




Image Plane Optical Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. autoclass:: poppy.IdealFieldStop
   :show-inheritance:
.. autoclass:: poppy.IdealCircularOcculter
   :show-inheritance:
.. autoclass:: poppy.IdealBarOcculter
   :show-inheritance:
.. autoclass:: poppy.BandLimitedCoron
   :show-inheritance:
.. autoclass:: poppy.IdealFQPM
   :show-inheritance:



The Detector Optical Element
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: poppy.Detector
   :show-inheritance:




.. Instrument:

Instrument
----------------

.. autoclass:: poppy.Instrument
   :members:

------


--------------

Documentation last updated on |today|

