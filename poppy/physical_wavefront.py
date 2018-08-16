# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as const
import astropy.units as u
from copy import deepcopy
from scipy.linalg import lstsq

from poppy.fresnel import FresnelWavefront, QuadraticLens


class PhysicalFresnelWavefront(FresnelWavefront):
    """
    This class extends the capabilities of poppy's FresnelWavefront class.

    The extension includes the redefinition of the wavefront array to
    represent an electric field with the appropriate units V.m^-1.
    Some utility functions have been defined to be able to compute the beam
    radius and beam quality factor. Additionally, a wavelength scaling has
    been introduced to simulate a beam quality factor.

    NOTE: This class expects a vacuum wavelength and a refractive index. It
    internally scales the wavelength appropriately.

    Parameters
    ----------
    M2 : float
        Beam quality factor.
    n0 : float
        Ambient refractive index.
    """

    def __init__(self,
                 beam_radius,
                 units=u.m,
                 rayleigh_factor=2.0,
                 oversample=2,
                 wavelength=1.0e-6,
                 npix=1024,
                 M2=1.0,
                 n0=1.00027398,  # refractive index of air @ 15 deg C, lambda=1064nm
                 **kwargs):
        super(PhysicalFresnelWavefront, self).__init__(
            beam_radius=beam_radius,
            units=units,
            rayleigh_factor=rayleigh_factor,
            oversample=oversample,
            wavelength=wavelength * M2 / n0,  # scale the wavelength and apply refractive index
            npix=npix,
            **kwargs
        )
        self.lam = wavelength / n0  # its unscaled value is needed to compute M2
        self.n0 = n0
        self.npix = npix * oversample

    @property
    def x(self):
        """Spatial grid (m)."""
        y, x = self.coordinates()

        return x[0, :]

    @property
    def dx(self):
        """Spatial grid sampling (m)."""

        return self.pixelscale.to(u.m / u.pixel).value

    @property
    def xy(self):
        """Spatial grid (m) as 2 dimensional array."""
        y, x = self.coordinates()

        return np.array([x.flatten(), y.flatten()]).T

    @property
    def q(self):
        """Momentum grid (m^-1)."""

        return np.fft.fftfreq(self.npix, d=self.dx) * 2.0 * np.pi

    @property
    def dq(self):
        """Momentum grid sampling (m^-1)."""

        return 2.0 * np.pi / self.npix / self.dx

    @property
    def intensity(self):
        """Intensity distribution (W.m^-2)."""

        return const.c * self.n0 * const.epsilon_0 * np.abs(self.amplitude) ** 2 / 2.0

    @property
    def power(self):
        """Power of the wavefront (W)."""

        return (self.dx ** 2) * self.total_intensity

    def scale_power(self, P):
        """
        Scales the wavefront to a desired power.

        Parameters
        ----------
        P : float
            The desired power of the wavefront.
        """

        P0 = self.power
        self.wavefront *= np.sqrt(P / P0)

    def normalize(self):
        # for PhysicalFresnelWavefronts use scale_power instead of
        # normalizing total intensity to 1
        pass

    def propagate_fresnel(self, z, attenuation_coeff=0.0, **kwargs):
        """
        Propagates the wavefront a specified distance while keeping its
        power in agreement with the Beer-Lambert law.

        Parameters
        ----------
        z : float
            Distance to propagate (m).
        attenuation_coeff : float
            Attenuation coefficient (m^-1).
        """

        pow = self.power
        super(PhysicalFresnelWavefront, self).propagate_fresnel(z, **kwargs)
        self.scale_power(pow * np.exp(-attenuation_coeff * z.to(u.m).value))

    def center(self, mask=1.0):
        """
        Calculates the first moments (m), e.g. the center coordinates of the
        wavefront.

        Parameters
        ----------
        mask : float or numpy.ndarray
            Mask to multiply intensity distribution with. This is useful to
            mimic integration borders.
        """

        power = self.power
        x = self.x
        dx = self.dx
        intensity = self.intensity * mask
        center_y = dx ** 2 * np.sum(np.dot(intensity, x)) / power
        center_x = dx ** 2 * np.sum(np.dot(x, intensity)) / power

        return center_x, center_y

    def sigma2(self, mask=1.0):
        """
        Calculates the squared second moments (m^2).

        Parameters
        ----------
        mask : float or numpy.ndarray
            Mask to multiply intensity distribution with. This is useful to
            mimic integration borders.
        """

        center_x, center_y = self.center(mask=mask)
        pow = self.power
        x = self.x
        dx = self.dx
        intensity = self.intensity * mask
        sigma_xx = dx ** 2 * np.sum(np.dot((x - center_x) ** 2, intensity)) / pow
        sigma_yy = dx ** 2 * np.sum(np.dot(intensity, (x - center_y) ** 2)) / pow
        sigma_xy = dx ** 2 * np.dot(np.dot((x - center_y), intensity), (x - center_y)) / pow

        return sigma_xx, sigma_yy, sigma_xy

    @property
    def radius(self):
        """
        Calculates the beam radius (m) and the ellipticity based on the
        2nd moments according to DIN EN ISO 11146-1.
        """

        w_x = self.diam.to(u.m).value / 2  # Initial guess
        w_y = self.diam.to(u.m).value / 2  # Initial guess
        w = self.diam.to(u.m).value / 2  # Initial guess
        eps = min(w_x, w_y) / max(w_x, w_y)
        num = self.npix
        mask = np.ones((num, num), dtype=float)
        x = self.x

        for idx in range(30):
            w_buf = w

            center_x, center_y = self.center(mask)

            mask[:, :] = 0.0
            for idx_x in range(num):
                for idx_y in range(num):
                    if abs(x[idx_x] - center_x) < 3.0 * w_x and abs(x[idx_y] - center_y) < 3.0 * w_y:
                        mask[idx_x, idx_y] = 1.0

            sigma_xx, sigma_yy, sigma_xy = self.sigma2(mask)

            gam = 1.0
            if np.abs((sigma_xx - sigma_yy) / sigma_xx) > 1.0e-3:
                gam = (sigma_xx - sigma_yy) / np.abs(sigma_xx - sigma_yy)

            tmp = (sigma_xx - sigma_yy) ** 2 + 4.0 * sigma_xy ** 2

            hlp = sigma_xx + sigma_yy
            hlp2 = sigma_xx + sigma_yy

            if tmp > 0.0:
                hlp += gam * np.sqrt(tmp)
            w_x = np.sqrt(2.0) * np.sqrt(hlp)

            if tmp > 0.0:
                hlp2 -= gam * np.sqrt(tmp)
            w_y = np.sqrt(2.0) * np.sqrt(hlp2)

            w = np.sqrt((w_x ** 2 + w_y ** 2) / 2.0)

            eps = min(w_x, w_y) / max(w_x, w_y)

            if abs((w_buf - w) / w_buf) < 1.0e-12:
                break

            if idx == 29:
                raise StopIteration('Maximal number of iterations reached while calculating beam radius.')

        return w_x, w_y, w, eps

    def M2(self, direction='xy'):
        """
        Calculates the beam quality factor according to DIN EN ISO 11146-1.

        Parameters
        ----------
        direction : string
            The direction in which to compute the beam quality factor. Must
            be one of 'x', 'y', or 'xy'.

        Returns
        -------
        M2 : float
            Beam quality factor.
        z : numpy.ndarray or shape (11,)
            Position (m) at which the caustic is evaluated to be used in the
            fitting procedure. 5 points lie within the Rayleigh length, 6
            lie outside of the twofold Rayleigh length.
        caustic : numpy.ndarray or shape (11,)
            Caustic (m) evaluated at z.
        z_fine : numpy.ndarray or shape (10000,)
            Better resolved position (m) array.
        w_fit : numpy.ndarray or shape (100,)
            Fit of the caustic (m) evaluated at z_fine.
        rayleigh_length : float
            Raleigh length (m).
        """

        wf_ini = deepcopy(self)

        # Apply a very short focal length
        f = 1.0
        L = 2.0 * f
        wf_ini *= QuadraticLens(f_lens=f * u.m)

        num_z = 11
        dz = np.ones(num_z, dtype=float)
        dz *= L / num_z
        z = np.zeros(num_z, dtype=float)
        caustic = np.zeros(num_z, dtype=float)
        A = np.zeros((num_z, 3), dtype=float)
        k = 2.0 * np.pi / wf_ini.lam
        M2 = 0.0
        M2_old = 1.0

        for idx in range(30):  # try to do it in 30 steps
            wf_work = deepcopy(wf_ini)

            for idx_z in range(num_z):
                wf_work.propagate_fresnel(dz[idx_z] * u.m)
                if direction == 'x':
                    w, _, _, _ = wf_work.radius
                elif direction == 'y':
                    _, w, _, _ = wf_work.radius
                elif direction == 'xy':
                    _, _, w, _ = wf_work.radius
                else:
                    raise AttributeError('Direction not correctly defined in M2.')

                caustic[idx_z] = w
                z[idx_z] = np.sum(dz[0:idx_z + 1])

            A[:, :] = 0.0
            A[:, 0] = 1.0
            for idx_1 in range(num_z):
                for idx_2 in range(1, 3):
                    A[idx_1, idx_2] = z[idx_1] ** idx_2

            b = 4.0 * caustic ** 2

            x, _, _, _ = lstsq(A, b, lapack_driver='gelsd')

            z_fine = np.linspace(z.min(), z.max(), 100000)
            w_fit = np.sqrt(x[0] + x[1] * z_fine + x[2] * z_fine ** 2) / 2

            rayleigh_length = np.sqrt(4.0 * x[0] * x[2] - x[1] ** 2) / 2.0 / x[2]
            M2 = k * np.sqrt(4.0 * x[0] * x[2] - x[1] ** 2) / 16.0

            if abs((M2_old - M2) / M2_old) < 1.0e-3 and idx > 3:
                break
            else:
                dz[0] = z_fine[np.argmin(w_fit)] - 2.9 * rayleigh_length
                dz[1] = 0.4 * rayleigh_length
                dz[2] = 0.4 * rayleigh_length
                dz[3] = 1.2 * rayleigh_length
                dz[4] = 0.45 * rayleigh_length
                dz[5] = 0.45 * rayleigh_length
                dz[6] = 0.45 * rayleigh_length
                dz[7] = 0.45 * rayleigh_length
                dz[8] = 1.2 * rayleigh_length
                dz[9] = 0.4 * rayleigh_length
                dz[10] = 0.4 * rayleigh_length

            M2_old = M2

            if idx == 29:
                raise RuntimeError('Maximal number of iterations reached \
                                    while calculating beam quality factor.')

        return M2, z, caustic, z_fine, w_fit, rayleigh_length
