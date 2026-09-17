"""Union2.1 likelihood with the supernova zero point marginalized out.

    mu_th(z) = mu_shape(z; Om, OL) + M,   mu_shape = 5 log10(d_L),
    where the dimensionless shape carries all the cosmology and
    M = M_B + 25 + 5 log10(c/(H0 Mpc))."""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from snia.cosmology import FLRW

class Union21Likelihood:
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
        # np.tril: SciPy 1.18.1 can leave garbage above the diagonal for n >= 64.
        self.L = np.tril(cholesky(cov, lower=True))
        self.v = cho_solve((self.L, True), np.ones(n))
        self.E = float(self.v.sum())

    def chi2(self, theta):
        """Marginalized chi2 = A - B^2/E."""
        terms = self._terms(theta)
        if terms is None:
            return np.inf
        a, b = terms
        return a - b * b / self.E

    def log_likelihood(self, theta):
        """-chi2/2, up to the constant dropped with ln E and the normalization."""
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
        """The M that minimizes chi2(M), B/E."""
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
        y = solve_triangular(self.L, delta, lower=True, check_finite=False)
        return float(y @ y), float(self.v @ delta)
