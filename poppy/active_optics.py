import numpy as np
import astropy.units as u
import poppy

# active optics classes

__all__ = ['TipTiltStage']

class TipTiltStage(poppy.poppy_core.OpticalElement):
    """Put an arbitrary optic on a tip-tilt stage, so we can actively adjust its tip and tilt.

    This is implemented internally like a compound optic between the supplied OpticalElement
    and a ZernikeWFE instance for the tip and tilt.

    Use the .set_tip_tilt() method to adjust the position of this stage, moving the resulting PSF.


    Parameters
    ----------
    optic : OpticalElement
        Optic that defines the transmission, and/or some fixed part of the OPD.
    radius : astropy Quantity of dimension length
        radius of the circle over which to apply Zernike tip/tilt.
        This is used in converting angular tip/tilt to OPD in length units.
    include_factor_of_two : bool
        include factor of 2 on WFE due to reflection? In other words, if True, the
        applied wavefront tilt will be twice the commanded tilt of the stage.

    """
    def __init__(self, optic=None, radius=1*u.m, include_factor_of_two=False, **kwargs):
        name = optic.name + " on a tip/tilt stage"
        super().__init__(name=name, **kwargs)
        self.optic = optic
        self.include_factor_of_two = include_factor_of_two
        self.tilt_optic = poppy.ZernikeWFE(coefficients=[0,0,0], radius=radius)
        self.pupil_diam = 2 * radius

    @property
    def radius(self):
        return self.tilt_optic.radius

    @poppy.utils.quantity_input(tip=u.arcsec, tilt=u.arcsec)
    def set_tip_tilt(self, tip, tilt):
        """ Set tip & tilt, in angular units

        These are internally converted into Zernike coefficients with units = meters of optical path difference,
        and summed with the OPD (if any) of the optic on this tip tilt stage.

        If the input beam is a flat wavefront, then setting e.g. tip=1 arcsec will result in the
        PSF being displaced to +1 arcsec in X in the subsequent focal plane, and so on.
        (The displacement distance will be doubled if .include_factor_of_two is set.)

        Parameters
        ----------
        tip, tilt : astropy Quantities of dimension angle
            Tip (around X axis) and Tilt (around Y axis) to apply using this stage.

        """

        # SIGN CONVENTION: positive displacements of the PSF require opposite sign in the
        #  OPD across the pupil. See e.g.
        # https://poppy-optics.readthedocs.io/en/latest/sign_conventions_for_coordinates_and_phase.html#A-wavefront-with-positive-OPD-tilt-in-+x-should-shift-the-focused-PSF-toward--x
        coeff = -2 if self.include_factor_of_two else -1

        self.tilt_optic.coefficients[1] = tip.to_value(u.radian) * self.radius * coeff/2
        self.tilt_optic.coefficients[2] = tilt.to_value(u.radian) * self.radius * coeff/2

    def get_tip_tilt(self):
        """Return tip & tilt, in Zernike coefficients with units = meters of optical path difference """
        return self.tilt_optic.coefficients[1:3]

    def get_transmission(self, wave):
        """Return transmission of this optic. See poppy.OpticalElement parent class docstring"""
        return self.optic.get_transmission(wave)

    def get_opd(self, wave):
        """Return OPD of this optic. See poppy.OpticalElement parent class docstring

        In this case, we sum optical tip and tilt onto whatever the OPD is from the optic on this stage.

        """
        self.opd = self.optic.get_opd(wave) + self.tilt_optic.get_opd(wave)
        return self.opd
