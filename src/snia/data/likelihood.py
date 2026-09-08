"""Union2.1 likelihood with the supernova zero point marginalized out.

The observed distance modulus fixes the SHAPE of the distance-redshift
relation but not its normalization. The model is

    mu_th(z) = mu_shape(z; Om, OL) + M,   mu_shape = 5 log10(d_L),

where the dimensionless shape carries all the cosmology and

    M = M_B + 25 + 5 log10(c/(H0 Mpc))

carries everything else. M is added to the shape, never to a complete
distance modulus. Writing the residual as Delta - M 1, the chi-square is a
quadratic in M,

    chi2(M) = A - 2 M B + M^2 E,

    A = Delta^T C^-1 Delta,   B = 1^T C^-1 Delta,   E = 1^T C^-1 1,

and integrating M out under a flat improper prior is a Gaussian integral:

    chi2 = A - B^2 / E.

E depends only on the covariance, never on (Omega_m, Omega_Lambda), so the
ln E term from that integral is an additive constant and is dropped. That
same fact makes this numerically identical to profiling M out.
"""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from snia.cosmology import FLRW


class Union21Likelihood:
    """Marginalized Union2.1 likelihood over the 2-D space (Om, OL).

    Precomputed once, in the constructor:

        L       Cholesky factor of C, lower triangular      O(n^3/3)
        v       C^-1 1, by one cho_solve (two solves)       O(n^2)
        E       1^T C^-1 1, which is just v.sum()           O(n)

    Per likelihood call:

        is_valid + d_L      two FLRW quadrature grids       O(N_grid)
        Delta               mu - 5 log10(d_L)               O(n)
        y = L^-1 Delta      ONE triangular solve            O(n^2)
        A = y . y           dot product                     O(n)
        B = v . Delta       dot product, v is precomputed   O(n)

    Only the forward substitution is needed: A = ||L^-1 Delta||^2 because
    Delta^T C^-1 Delta = Delta^T L^-T L^-1 Delta. Calling cho_solve here
    instead would do a back substitution as well and compute C^-1 Delta,
    which is never needed. C^-1 is never formed.

    Precision: write Delta = M_hat 1 + r with M_hat = B/E. That split is
    orthogonal in the C^-1 inner product, since 1^T C^-1 r = B - (B/E) E = 0,
    so exactly

        A = M_hat^2 E + chi2,      B^2/E = M_hat^2 E.

    This form therefore recovers chi2 by cancelling two numbers whose ratio
    to it is 1 + M_hat^2 E / chi2 = 7.3e3 on Union2.1: about four digits
    lost, leaving ~5e-12 relative, far below anything that matters here.

    The large M_hat ~ 43 is a direct consequence of the dimensionless
    convention, because 5 log10(d_L) is O(1) while mu_obs ~ 43. Using
    distances in Mpc would put M_hat near zero and remove the cancellation
    entirely -- but it would also drag H0 back into the model, which is a far
    worse trade. Four digits is the cheap price of keeping H0 out.

    If it were ever needed, the cancellation-free form is
    chi2 = ||y - M_hat w||^2 with w = L^-1 1 precomputed. It is the same one
    triangular solve per call and measures ~5e-15 relative instead of 5e-12,
    because it subtracts the vectors before squaring rather than after.
    """

    def __init__(self, z, mu, cov):
        z = np.asarray(z, dtype=float)
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        n = z.size
        if z.ndim != 1 or mu.shape != (n,) or cov.shape != (n, n):
            raise ValueError(
                f"inconsistent shapes: z {z.shape}, mu {mu.shape}, cov {cov.shape}"
            )

        self.z = z
        self.mu = mu
        self.n_sn = n
        self.z_max = float(z.max())

        # The one O(n^3) operation in the whole analysis. scipy's cholesky is
        # used rather than cho_factor because it returns a clean lower factor;
        # cho_factor leaves the opposite triangle filled with unrelated values,
        # which solve_triangular ignores but which makes L harder to inspect.
        self.L = cholesky(cov, lower=True)

        # v = C^-1 1 and E = 1^T C^-1 1 are parameter-independent, so they are
        # paid for once here and never again.
        self.v = cho_solve((self.L, True), np.ones(n))
        self.E = float(self.v.sum())

    def chi2(self, theta):
        """Marginalized chi2 = A - B^2/E. Returns +inf at invalid points."""
        terms = self._terms(theta)
        if terms is None:
            return np.inf
        a, b = terms
        return a - b * b / self.E

    def log_likelihood(self, theta):
        """-chi2/2, up to the constant dropped with ln E and the normalization.

        Returns -inf, never NaN and never an exception, in the no-Big-Bang
        wedge and past the antipode of a closed model.
        """
        return -0.5 * self.chi2(theta)

    __call__ = log_likelihood

    def chi2_at_offset(self, theta, m_offset):
        """Un-marginalized chi2(M) = A - 2 M B + M^2 E, for cross-checking."""
        terms = self._terms(theta)
        if terms is None:
            return np.inf
        a, b = terms
        return a - 2.0 * m_offset * b + m_offset * m_offset * self.E

    def m_offset(self, theta):
        """The M that minimizes chi2(M), B/E.

        With a flat prior this is also the posterior mean and the peak of the
        Gaussian in M, so it is what to use when overlaying a model curve on
        the Hubble diagram. NaN at invalid points, where it is undefined.
        """
        terms = self._terms(theta)
        if terms is None:
            return np.nan
        return terms[1] / self.E

    def _terms(self, theta):
        """Return (A, B), or None if the parameter point must be rejected."""
        omega_m, omega_lambda = theta
        model = FLRW(omega_m, omega_lambda)
        if not model.is_valid(self.z_max):
            return None

        delta = self.mu - 5.0 * np.log10(model.luminosity_distance(self.z))

        # The single per-call triangular solve. check_finite is off because
        # is_valid has already guaranteed d_L > 0 and finite on [0, z_max],
        # and mu comes from the validated loader.
        y = solve_triangular(self.L, delta, lower=True, check_finite=False)
        return float(y @ y), float(self.v @ delta)
