import numpy as np
import astropy.units as u
import poppy

# active optics classes

class TipTiltStage(poppy.poppy_core.OpticalElement):
    """Put an arbitrary optic on a tip-tilt stage, so we can actively adjust its tip and tilt.

    This is implemented internally like a compound optic between the supplied OpticalElement
    and a ZernikeWFE instance for the tip and tilt.

    Parameters
    ----------
    optic : OpticalElement
        Optic that defines the transmission, and/or some fixed part of the OPD.
    radius : astropy Quantity of dimension length
        radius of the circle over which to apply Zernike tip/tilt
    include_factor_of_two : book
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

    @poppy.utils.quantity_input(tip=u.meter, tilt=u.meter)
    def set_tip_tilt(self, tip, tilt):
        """ Set tip & tilt, in Zernike coefficients with units = meters of optical path difference (not waves)

        Parameters
        ----------
        tip, tilt : astropy Quantities of dimension length
            meters of optical path difference for tilt

        """
        self.tilt_optic.coefficients[1] = tip
        self.tilt_optic.coefficients[2] = tilt

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
        opd = np.zeros(wave.shape, dtype=poppy.accel_math._float())

        opd += self.optic.get_opd(wave)

        coeff = 2 if self.include_factor_of_two else 1

        opd += coeff * self.tilt_optic.get_opd(wave)

        self.opd = opd
        return self.opd
