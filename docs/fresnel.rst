.. _fresnel:

Fresnel Propagation
===========================

POPPY now includes support for Fresnel propagation as well as Fraunhofer.
Particular credit is due to `Ewan Douglas <http://blogs.bu.edu/douglase/>`_ for
initially developing this code.  This substantial upgrade to ``poppy`` enables
calculation of wavefronts propagated arbitrary distances in free space, for applications
such as Gaussian beam propagation and modeling of Talbot effect mixing between phase and
amplitude aberrations. 


.. caution::
        The Fresnel code has
        been cross-checked against the `PROPER library by John Krist
        <http://proper-library.sourceforge.net>` to verify accuracy and correctness of
        output. A test suite is provided along with ``poppy`` in the tests subdirectory
        and users are encouraged to run these tests themselves. However this is still 
        a relatively new addition to ``poppy``, and it is possible there may still be
        some bugs. 



Usage of the Fresnel code
--------------------------------


The API has been kept as similar as possible to the original Fraunhofer mode of
poppy. There are FresnelWavefront and FresnelOpticalSystem classes, which can
be used for the most part similar to the Wavefront and OpticalSystem classes. 

Users are encouraged to consult the Jupyter notebook `Fresnel_Propagation_Demo
<https://github.com/mperrin/poppy/blob/master/notebooks/Fresnel_Propagation_Demo.ipynb>`_
for examples of how to use the Fresnel code. 

Key Differences from Fraunhofer mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Fresnel propagation API necessarily differs in several ways from the original Fraunhofer API in poppy. Let's highlight a few of the key differences.
First, when we define a Fresnel wavefront, the first argument specifies the desired diameter of the wavefront, and must be given as an `astropy.Quantity <http://docs.astropy.org/en/stable/units/>`_ of dimension length::

        import astropy.units as u
        wf_fresnel = poppy.FresnelWavefront(0.5*u.m,wavelength=2200e-9,npix=npix,oversample=4)
        # versus:
        wf_fraunhofer = poppy.Wavefront(diam=0.5, wavelength=2200e-9,npix=npix,oversample=4)

The Fresnel code relies on the Quantity framework to enforce consistent units and dimensionality. You can use any desired unit of length, from nanometers to parsecs and beyond, and the code will convert units appropriately.
This also shows up when requesting an optical propagation. Rather than having implicit transformations between pupil and image planes, for Fresnel propagation a specific distance must be given. This too is a Quantity giving a length. ::

        wf.propagate_fresnel(5*u.km)


The parameters of a Gaussian beam may be modified (making it converging or
diverging) by adding optical power. In poppy this is represented with the
``QuadraticLens`` class. This is so named because it applies a purely quadratic
phase term, i.e. representative of a parabolic mirror or a lens considered in
the paraxial approximation.  Right now, only the Fresnel ``QuadraticLens`` class
will actually cause the Gaussian beam parameters to change. You won't get that
effect by adding wavefront error with some other ``OpticalElement`` class.


Just like the ``OpticalSystem`` serves as a high-level container for
``OpticalElements`` in Fraunhofer propagation, the ``FresnelOpticalSystem``
serves the same purpose in Fresnel propagation.  Note that when adding
``OpticalElements`` to the ``FresnelOpticalSystem``, you use an ``add_optic``
function and must specify a physical distance separating that optic from the
previous optic, again as an `astropy.Quantity
<http://docs.astropy.org/en/stable/units/>`_ of dimension length. This replaces
the ``addImage`` and ``addPupil`` methods used in Fraunhofer propagation.



For more details and examples of code usage, consult the Jupyter notebook
`Fresnel_Propagation_Demo
<https://github.com/mperrin/poppy/blob/master/notebooks/Fresnel_Propagation_Demo.ipynb>`_.

A worked example of a compound microscope in POPPY is available
`here <https://github.com/douglase/poppy_example_notebooks/blob/master/Fresnel/Microscope_Example.ipynb>`_, 
reproducing the microscope example case provided in the PROPER manual.

References
-------------

The following references were helpful in the development of this code. 

    - Goodman, `Fourier Optics <http://www.amazon.com/Introduction-Fourier-Optics-Joseph-Goodman/dp/0974707724>`_

    - Lawrence, G. N. (1992), Optical Modeling, in Applied Optics and Optical Engineering., vol. XI,
        edited by R. R. Shannon and J. C. Wyant., Academic Press, New York.

    - IDEX Optics and Photonics(n.d.), 
      `Gaussian Beam Optics <https://marketplace.idexop.com/store/SupportDocuments/All_About_Gaussian_Beam_OpticsWEB.pdf>`_

    - Krist, J. E. (2007), `PROPER: an optical propagation library for IDL <http://dx.doi.org/10.1117/12.731179>`_
       vol. 6675, p. 66750P-66750P-9.

    - Andersen, T., and A. Enmark (2011), 
      `Integrated Modeling of Telescopes <http://www.amazon.com/Integrated-Modeling-Telescopes-Astrophysics-Science/dp/1461401488>`_,
      Springer Science & Business Media.


