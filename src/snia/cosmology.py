"""Dimensionless FLRW background distances.

Every distance here is DIMENSIONLESS. The factor c/H0 is absorbed, together
with the absolute magnitude M_B and the +25, into the single nuisance
parameter ``m_offset`` of the distance modulus. Never multiply these by c/H0.

The sampled space is (omega_m, omega_lambda); omega_k = 1 - omega_m -
omega_lambda is derived, so flatness is a line in the sampled plane rather
than an imposed constraint.
"""

import numpy as np

#: Number of points used for the 1/E quadrature grid. The composite trapezoid
#: error is a boundary term (Euler-Maclaurin), so the RELATIVE accuracy is
#: uniform in z at ~0.02*h**2; here that is ~1e-8, i.e. ~2e-8 mag.
N_GRID = 2001

#: Switch between the series and the closed form of S_k at |omega_k| * I**2.
#: Both representations are accurate to ~1e-16 relative here, so the switch is
#: smaller than one ulp. See transverse_comoving_distance.
SERIES_THRESHOLD = 1e-6


def _match_shape(out, z):
    """Return a numpy scalar if the input was scalar, else the array."""
    return out[()] if z.ndim == 0 else out


class FLRW:
    """FLRW background with matter and a cosmological constant, no radiation.

    E^2(z) = omega_m (1+z)^3 + omega_k (1+z)^2 + omega_lambda

    Parameter points may be unphysical (E^2 <= 0, or distances past the
    hyperspherical antipode). Nothing raises and nothing returns NaN: invalid
    points yield +inf, and ``is_valid`` is the gate the likelihood should use.
    """

    def __init__(self, omega_m, omega_lambda):
        self.omega_m = float(omega_m)
        self.omega_lambda = float(omega_lambda)

    @property
    def omega_k(self):
        """Curvature density, derived and never sampled."""
        return 1.0 - self.omega_m - self.omega_lambda

    def E_squared(self, z):
        """Squared expansion rate. May be <= 0 off the physical region."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._e_squared(z), z)

    def E(self, z):
        """Expansion rate. Returns 0, not NaN, wherever E^2 <= 0."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._e(z), z)

    def comoving_distance(self, z):
        """Line-of-sight comoving distance I(z) = int_0^z dz'/E(z').

        Integrated ONCE with a cumulative trapezoid on a uniform grid of
        N_GRID points over [0, max(z)], then interpolated onto z with a cubic
        Hermite polynomial. The Hermite nodal derivatives are free: I'(z) is
        the integrand 1/E, already evaluated on the grid. The grid is rebuilt
        per call, so a low-z-only call automatically gets a fine grid.
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._comoving(z), z)

    def transverse_comoving_distance(self, z):
        """Transverse comoving distance D_M = S_k(I).

        The three curvature branches are one analytic function. With
        x = omega_k * I^2,

            D_M = I * sinh(sqrt(x)) / sqrt(x)
                = I * (1 + x/6 + x^2/120 + x^3/5040 + ...)

        The series contains only even powers of sqrt(x), so it is entire in x
        and covers sinh (x>0), the flat limit (x=0) and sin (x<0) alike; for
        x<0 it is sin(y)/y with y = sqrt(|x|). The sinh/sin split below is
        therefore a numerical device, not a mathematical branch: the closed
        form is 0/0 at exactly omega_k = 0.

        This uses 1 + x/6 + x^2/120 for |x| < SERIES_THRESHOLD = 1e-6, where
        the first dropped term is x^3/5040 ~ 2e-22, far below machine epsilon.
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._transverse(self._comoving(z)), z)

    def angular_diameter_distance(self, z):
        """D_A = D_M / (1+z)."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._transverse(self._comoving(z)) / (1.0 + z), z)

    def luminosity_distance(self, z):
        """Dimensionless d_L = (1+z) D_M. No c/H0 factor."""
        z = np.asarray(z, dtype=float)
        return _match_shape(self._luminosity(z), z)

    def distance_modulus(self, z, m_offset=0.0):
        """mu = 5 log10(d_L) + m_offset, with d_L dimensionless.

        m_offset = M_B + 25 + 5 log10(c/(H0 Mpc)) carries the whole zero point,
        so it enters as a pure additive constant, independent of z. That is
        what allows it to be marginalized analytically.
        """
        z = np.asarray(z, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 5.0 * np.log10(self._luminosity(z)) + m_offset
        return _match_shape(np.where(np.isnan(mu), np.inf, mu), z)

    def is_valid(self, z_max):
        """True if the model is usable on [0, z_max].

        False in the no-Big-Bang wedge (E^2 <= 0 somewhere, so the integrand
        diverges) and for closed models whose light cone passes the antipode
        (D_M <= 0). D_L = (1+z) D_M shares the sign of D_M, so testing D_M
        is equivalent. The caller turns False into -inf.
        """
        grid = np.linspace(0.0, float(z_max), N_GRID)
        if np.any(self._e_squared(grid) <= 0.0):
            return False
        d_m = self._transverse(self._comoving(grid))
        return bool(np.all(d_m[1:] > 0.0))

    def _e_squared(self, z):
        zp1 = 1.0 + z
        return self.omega_m * zp1**3 + self.omega_k * zp1**2 + self.omega_lambda

    def _e(self, z):
        # Clipping keeps the sqrt real: E = 0 gives 1/E = +inf downstream,
        # never NaN. is_valid is what actually rejects these points.
        return np.sqrt(np.maximum(self._e_squared(z), 0.0))

    def _comoving(self, z):
        z_max = float(np.max(z))
        if z_max <= 0.0:
            return np.zeros_like(z)

        grid = np.linspace(0.0, z_max, N_GRID)
        h = z_max / (N_GRID - 1)
        with np.errstate(divide="ignore"):
            f = 1.0 / self._e(grid)

        integral = np.empty(N_GRID)
        integral[0] = 0.0
        np.cumsum(0.5 * h * (f[:-1] + f[1:]), out=integral[1:])

        # Uniform spacing means the enclosing cell is a division, not a search.
        i = np.clip((z / h).astype(np.int64), 0, N_GRID - 2)
        t = (z - grid[i]) / h
        t2 = t * t
        t3 = t2 * t
        with np.errstate(invalid="ignore"):
            out = (
                (2.0 * t3 - 3.0 * t2 + 1.0) * integral[i]
                + (t3 - 2.0 * t2 + t) * h * f[i]
                + (-2.0 * t3 + 3.0 * t2) * integral[i + 1]
                + (t3 - t2) * h * f[i + 1]
            )
        return np.where(np.isnan(out), np.inf, out)

    def _transverse(self, i_z):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = self.omega_k * i_z * i_z
            # At an invalid point I is +inf, so for omega_k < 0 this is
            # -inf + inf = NaN. It is discarded by the np.where below and
            # sanitized on the way out, but it must not warn.
            series = 1.0 + x / 6.0 + x * x / 120.0
            s = np.sqrt(np.abs(x))
            # omega_k is a scalar and I^2 >= 0, so the SIGN of x is uniform
            # across the array and can be branched on once. The MAGNITUDE is
            # not: at small omega_k the same array straddles the threshold, so
            # that test stays elementwise. omega_k == 0 gives x == 0 and falls
            # through to the series, which is then exactly 1.
            if self.omega_k >= 0.0:
                factor = np.where(x > SERIES_THRESHOLD, np.sinh(s) / s, series)
            else:
                factor = np.where(x < -SERIES_THRESHOLD, np.sin(s) / s, series)
            out = i_z * factor
        return np.where(np.isnan(out), np.inf, out)

    def _luminosity(self, z):
        return (1.0 + z) * self._transverse(self._comoving(z))
