#Cosmology Module

import numpy as np

#: Number of points used for the 1/E quadrature grid.
N_GRID = 2001
SERIES_THRESHOLD = 1e-6

#Scalar or Array
def _match_shape(out, z):
    return out[()] if z.ndim == 0 else out


def _cumulative_integral(x, integrand):
    x_max = float(np.max(x))
    if x_max <= 0.0:
        return np.zeros_like(x)

    grid = np.linspace(0.0, x_max, N_GRID)
    h = x_max / (N_GRID - 1)
    f = integrand(grid)

    total = np.empty(N_GRID)
    total[0] = 0.0
    np.cumsum(0.5 * h * (f[:-1] + f[1:]), out=total[1:])
    i = np.clip((x / h).astype(np.int64), 0, N_GRID - 2)
    t = (x - grid[i]) / h
    t2 = t * t
    t3 = t2 * t
    with np.errstate(invalid="ignore"):
        out = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * total[i]
            + (t3 - 2.0 * t2 + t) * h * f[i]
            + (-2.0 * t3 + 3.0 * t2) * total[i + 1]
            + (t3 - t2) * h * f[i + 1]
        )
    return np.where(np.isnan(out), np.inf, out)

#Background Cosmology Class. I put some tests to validate the physical limits of the parameters. 
class FLRW:
    """FLRW background with matter and a cosmological constant, no radiation.

    E^2(z) = omega_m (1+z)^3 + omega_k (1+z)^2 + omega_lambda
    """

    def __init__(self, omega_m, omega_lambda):
        self.omega_m = float(omega_m)
        self.omega_lambda = float(omega_lambda)

    @property
    def omega_k(self):
        """Curvature density."""
        return 1.0 - self.omega_m - self.omega_lambda

    def E_squared(self, z):
        """Squared expansion rate."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._e_squared(z), z)

    def E(self, z):
        """Returns 0, not NaN, wherever E^2 <= 0."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._e(z), z)

    def comoving_distance(self, z):
        """
        Line-of-sight comoving distance I(z) = int_0^z dz'/E(z').
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._comoving(z), z)

    def transverse_comoving_distance(self, z):
        """Transverse comoving distance D_M = S_k(I).

        The three curvature branches are one analytic function. With
        x = omega_k * I^2,

            D_M = I * sinh(sqrt(x)) / sqrt(x)
                = I * (1 + x/6 + x^2/120 + x^3/5040 + ...)
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._transverse(self._comoving(z)), z)

    def angular_diameter_distance(self, z):
        """D_A = D_M / (1+z)."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._transverse(self._comoving(z)) / (1.0 + z), z)

    def luminosity_distance(self, z):
        """Dimensionless d_L."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._luminosity(z), z)

    def distance_modulus(self, z, m_offset=0.0):
        """mu_th = mu_shape + m_offset, with mu_shape = 5 log10(d_L).

        The shape is dimensionless: there is no +25 and no c/H0 in it. Both,
        together with the absolute magnitude, live in

            m_offset = M_B + 25 + 5 log10(c/(H0 Mpc)).
        """
        z = np.asarray(z, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 5.0 * np.log10(self._luminosity(z)) + m_offset
        return _match_shape(np.where(np.isnan(mu), np.inf, mu), z)

    def lookback_time(self, z):
        """Time between emission at z and today, in units of the Hubble time.

        t_L(z) = int_0^z dz' / ((1+z') E(z')), so H0 t_L is returned.
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._lookback(z), z)

    def age(self, z=0.0):
        """Cosmic time elapsed since the Big Bang at z, in Hubble times.

        Substituting a = 1/(1+z) and then a = x^2 turns the improper integral
        over z in [z, inf) into a proper one over x in [0, sqrt(a)]:

            H0 t(a) = int_0^a da' / sqrt(Om/a' + Ok + OL a'^2)
                    = int_0^sqrt(a) 2 x^2 dx / sqrt(Om + Ok x^2 + OL x^6).
        """
        z = np.asarray(z, dtype=float)
        x = np.sqrt(1.0 / (1.0 + z))
        return _match_shape(_cumulative_integral(x, self._age_integrand), z)

    def comoving_volume_element(self, z):
        """
        dV_C / (dz dOmega) = D_M^2 / E.
        """
        z = np.asarray(z, dtype=float)
        d_m = self._transverse(self._comoving(z))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = d_m * d_m / self._e(z)
        return _match_shape(np.where(np.isnan(out), np.inf, out), z)

    def deceleration_parameter(self, z):
        """
        q(z) = [Om (1+z)^3 / 2 - OL] / E^2(z).
        """
        z = np.asarray(z, dtype=float)
        zp1 = 1.0 + z
        numerator = 0.5 * self.omega_m * zp1**3 - self.omega_lambda
        with np.errstate(divide="ignore", invalid="ignore"):
            q = numerator / np.maximum(self._e_squared(z), 0.0)
        return _match_shape(np.where(np.isnan(q), np.inf, q), z)

    @property
    def q0(self):
        """Deceleration parameter today, Om/2 - OL."""
        return 0.5 * self.omega_m - self.omega_lambda

    def is_valid(self, z_max):
        """
        True if the model is usable on [0, z_max].
        """
        z_max = float(z_max)
        grid = np.linspace(0.0, z_max, N_GRID)
        if np.any(self._e_squared(grid) <= 0.0):
            return False
        if z_max <= 0.0:
            return True
        d_m = self._transverse(self._comoving(grid))
        return bool(np.all(d_m[1:] > 0.0))

    def __repr__(self):
        return (
            f"FLRW(omega_m={self.omega_m!r}, omega_lambda={self.omega_lambda!r})"
            f"  # omega_k={self.omega_k:+.4g}, q0={self.q0:+.4g}"
        )

    def _e_squared(self, z):
        zp1 = 1.0 + z
        return self.omega_m * zp1**3 + self.omega_k * zp1**2 + self.omega_lambda

    def _e(self, z):
        return np.sqrt(np.maximum(self._e_squared(z), 0.0))

    def _comoving(self, z):
        with np.errstate(divide="ignore"):
            return _cumulative_integral(z, lambda g: 1.0 / self._e(g))

    def _lookback(self, z):
        with np.errstate(divide="ignore"):
            return _cumulative_integral(z, lambda g: 1.0 / ((1.0 + g) * self._e(g)))

    def _age_integrand(self, x):
        """2 x^2 / sqrt(Om + Ok x^2 + OL x^6), the age integrand in x = sqrt(a)."""
        radicand = self.omega_m + self.omega_k * x**2 + self.omega_lambda * x**6
        with np.errstate(divide="ignore", invalid="ignore"):
            value = 2.0 * x * x / np.sqrt(np.maximum(radicand, 0.0))
        at_zero = np.inf if self.omega_m == 0.0 and self.omega_k == 0.0 else 0.0
        return np.where(x > 0.0, value, at_zero)

    def _transverse(self, i_z):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = self.omega_k * i_z * i_z
            series = 1.0 + x / 6.0 + x * x / 120.0
            s = np.sqrt(np.abs(x))
            if self.omega_k >= 0.0:
                factor = np.where(x > SERIES_THRESHOLD, np.sinh(s) / s, series)
            else:
                factor = np.where(x < -SERIES_THRESHOLD, np.sin(s) / s, series)
            out = i_z * factor
        return np.where(np.isnan(out), np.inf, out)

    def _luminosity(self, z):
        return (1.0 + z) * self._transverse(self._comoving(z))
