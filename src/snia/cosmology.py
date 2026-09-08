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


def _cumulative_integral(x, integrand):
    """int_0^x f dx' for every entry of x, from ONE grid over [0, max(x)].

    A cumulative trapezoid gives the integral at the N_GRID nodes; a cubic
    Hermite polynomial then interpolates to the requested points. The Hermite
    nodal derivatives are free, because the derivative of the integral is the
    integrand, which has already been evaluated on the grid. That makes the
    interpolation error O(h^4) and keeps the trapezoid's O(h^2) dominant.

    The grid is rebuilt per call and spans only as far as it must, so a
    low-x-only call automatically gets a proportionally finer grid.
    """
    x_max = float(np.max(x))
    if x_max <= 0.0:
        return np.zeros_like(x)

    grid = np.linspace(0.0, x_max, N_GRID)
    h = x_max / (N_GRID - 1)
    f = integrand(grid)

    total = np.empty(N_GRID)
    total[0] = 0.0
    np.cumsum(0.5 * h * (f[:-1] + f[1:]), out=total[1:])

    # Uniform spacing means the enclosing cell is a division, not a search.
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
    # An invalid point makes f infinite, and inf * 0 in the Hermite weights is
    # NaN. Those points are rejected by is_valid anyway; +inf is the sentinel.
    return np.where(np.isnan(out), np.inf, out)


class FLRW:
    """FLRW background with matter and a cosmological constant, no radiation.

    E^2(z) = omega_m (1+z)^3 + omega_k (1+z)^2 + omega_lambda

    Parameter points may be unphysical (E^2 <= 0, or distances past the
    hyperspherical antipode). Nothing raises and nothing returns NaN: invalid
    points yield +inf, and ``is_valid`` is the gate the likelihood should use.

    All methods take z >= 0 and accept a scalar or an array of any shape,
    returning the matching shape. Distances are dimensionless (units of c/H0)
    and times are in Hubble times (1/H0); multiply times by 9.778/h for Gyr.
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

        Conditioning: for closed models the map I -> D_M is ill-conditioned
        near the antipode. With u = sqrt(|Ok|) I, a relative error dI/I is
        amplified by u cot(u), which diverges as u -> pi. At (Om, OL) =
        (1.6, 3.0) the light cone reaches u = 3.132, so cot(u) = -104 turns a
        4e-8 error in I into 1e-5 in D_M. That is geometry, not quadrature --
        no integration scheme avoids it -- and such models are rejected by
        is_valid a little further on anyway.
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
        """mu_th = mu_shape + m_offset, with mu_shape = 5 log10(d_L).

        The shape is dimensionless: there is no +25 and no c/H0 in it. Both,
        together with the absolute magnitude, live in

            m_offset = M_B + 25 + 5 log10(c/(H0 Mpc)),

        which is a pure additive constant, independent of z -- and that is what
        allows it to be marginalized analytically.

        m_offset is added to the SHAPE, never to a complete distance modulus:
        5 log10(D_L/Mpc) + 25 already contains the +25 and the c/H0, so adding
        m_offset on top would count them twice, a 43 mag error.

        For the released Union2.1 mu, which already had a fiducial M_B removed,
        the M_B above is the offset relative to that fiducial, and m_offset
        comes out near 43.16 for h = 0.7. Fitting apparent magnitudes instead
        would put the full M_B ~ -19.3 in it, giving 23.86.
        """
        z = np.asarray(z, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 5.0 * np.log10(self._luminosity(z)) + m_offset
        return _match_shape(np.where(np.isnan(mu), np.inf, mu), z)

    def lookback_time(self, z):
        """Time between emission at z and today, in units of the Hubble time.

        t_L(z) = int_0^z dz' / ((1+z') E(z')), so H0 t_L is returned. Multiply
        by 9.778/h Gyr for years.
        """
        z = np.asarray(z, dtype=float)
        return _match_shape(self._lookback(z), z)

    def age(self, z=0.0):
        """Cosmic time elapsed since the Big Bang at z, in Hubble times.

        Substituting a = 1/(1+z) and then a = x^2 turns the improper integral
        over z in [z, inf) into a proper one over x in [0, sqrt(a)]:

            H0 t(a) = int_0^a da' / sqrt(Om/a' + Ok + OL a'^2)
                    = int_0^sqrt(a) 2 x^2 dx / sqrt(Om + Ok x^2 + OL x^6).

        The second form is the one used. It is worth the substitution twice
        over: the first kills the infinite upper limit, and the second removes
        the inverse-square-root singularity that the integrand would otherwise
        have at the Big Bang, restoring the trapezoid's O(h^2) convergence.

        Returns +inf for models with no Big Bang, where the integral diverges.
        """
        z = np.asarray(z, dtype=float)
        x = np.sqrt(1.0 / (1.0 + z))
        return _match_shape(_cumulative_integral(x, self._age_integrand), z)

    def comoving_volume_element(self, z):
        """dV_C / (dz dOmega) = D_M^2 / E, in units of (c/H0)^3 per steradian.

        This is the survey-volume weighting: how much comoving volume sits in
        a shell dz at redshift z, per unit solid angle.
        """
        z = np.asarray(z, dtype=float)
        d_m = self._transverse(self._comoving(z))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = d_m * d_m / self._e(z)
        return _match_shape(np.where(np.isnan(out), np.inf, out), z)

    def deceleration_parameter(self, z):
        """q(z) = [Om (1+z)^3 / 2 - OL] / E^2(z).

        From q = (1/2) sum_i Omega_i(z) (1 + 3 w_i): matter (w=0) contributes
        Om(z)/2, Lambda (w=-1) contributes -OL(z), and curvature (w=-1/3)
        contributes nothing. q < 0 means accelerating.
        """
        z = np.asarray(z, dtype=float)
        zp1 = 1.0 + z
        numerator = 0.5 * self.omega_m * zp1**3 - self.omega_lambda
        with np.errstate(divide="ignore", invalid="ignore"):
            q = numerator / np.maximum(self._e_squared(z), 0.0)
        return _match_shape(np.where(np.isnan(q), np.inf, q), z)

    @property
    def q0(self):
        """Deceleration parameter today, Om/2 - OL. Negative if accelerating."""
        return 0.5 * self.omega_m - self.omega_lambda

    def is_valid(self, z_max):
        """True if the model is usable on [0, z_max].

        False in the no-Big-Bang wedge (E^2 <= 0 somewhere, so the integrand
        diverges) and for closed models whose light cone passes the antipode
        (D_M <= 0). D_L = (1+z) D_M shares the sign of D_M, so testing D_M
        is equivalent. The caller turns False into -inf.
        """
        z_max = float(z_max)
        grid = np.linspace(0.0, z_max, N_GRID)
        if np.any(self._e_squared(grid) <= 0.0):
            return False
        if z_max <= 0.0:
            # Degenerate range: E^2(0) > 0 is all there is to check, and the
            # D_M > 0 test below would reject every model because D_M(0) = 0.
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
        # Clipping keeps the sqrt real: E = 0 gives 1/E = +inf downstream,
        # never NaN. is_valid is what actually rejects these points.
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
        # At x = 0 this is 0/0 unless Om > 0. The limit is 0 whenever Om or Ok
        # is positive; for de Sitter (Om = Ok = 0) it is +inf, which is right,
        # because that model has no Big Bang and its age genuinely diverges.
        at_zero = np.inf if self.omega_m == 0.0 and self.omega_k == 0.0 else 0.0
        return np.where(x > 0.0, value, at_zero)

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
