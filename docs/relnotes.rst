
Release Notes
===============


0.3
-----

 * Added documentation and examples for POPPY, separate from the WebbPSF documentation.
 * Added support for rectangular subarray calculations. You can invoke these by setting fov_pixels or fov_arcsec with a 2-element iterable::

    >> nc = webbpsf.NIRCam()
    >> nc.calcPSF('F212N', fov_arcsec=[3,6])
    >> nc.calcPSF('F187N', fov_pixels=(300,100) )

   Those two elements give the desired field size as (Y,X) following the usual Python axis order convention.
 * InverseTransmission of an AnalyticElement is now allowed inside a CompoundAnalyticOptic

