.. _fitsheaders:

POPPY FITS Header Keywords Definitions
======================================


* `WAVELEN`: Wavelength in meters
* `DIFFLMT`:  Diffraction limit lambda/D in *arcsecond*
* `OVERSAMP`: Oversampling factor for FFTs in computation of PSF
* `DET_SAMP`: Oversampling factor for MFT to detector plane
* `PIXELSCL` : Scale in *arcsecond/pixel*  or *meter/pixel* (after oversampling)
* `PIXUNIT` : units of the pixels in the header, typically either *arcsecond* or *meter*
* `FOV`: Field of view in *arcsecond* (full array)
* `FOV_X`: Field of view in *arcsecond* (full array), X direction
* `FOV_Y`: Field of view in *arcsecond* (full array), Y direction
* `FFTTYPE`: Algorithm for FFTs (e.g. numpy or fftw)
* `NORMALIZ` : Which plane normalization was applied in (*first* or *last*)
* `DIFFLMT`: Scale in arcsec/pix (after oversampling)
* `DIAM`: Pupil diameter in meters (not incl padding)
* `NWAVES`: Number of wavelengths used in calculation
* `BUNIT`: units of OPD error. Default is 'meters'. Can be 'meter', 'meters', 'micron(s)', 'nanometer(s)', or their SI abbreviations. Can also be 'radian' or 'radians', which makes POPPY treat the phase pattern as wavelength-independent (i.e. Pancharatnam-Berry phase).
