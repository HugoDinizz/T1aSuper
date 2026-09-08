"""Proposal distributions.

A proposal is an algorithmic choice, not part of the statistical model. It is
represented by BOTH ``propose`` and ``logpdf`` so that one sampler handles
symmetric and asymmetric moves alike; ``is_symmetric`` lets the sampler skip
the Hastings ratio when it provably cancels.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import cholesky, solve_triangular


class Proposal(ABC):
    """Interface every proposal must satisfy."""

    #: True when q(a|b) == q(b|a), so the Hastings ratio cancels.
    is_symmetric = False

    @abstractmethod
    def propose(self, state, rng):
        """Draw a candidate from q(. | state), using the supplied generator."""

    @abstractmethod
    def logpdf(self, target, given):
        """log q(target | given)."""


class GaussianRandomWalk(Proposal):
    """theta' = theta + scale * L z, with z ~ N(0, I) and L L^T = cov.

    Centred on the current state, so q(a|b) depends only on a - b and the
    proposal is symmetric. The covariance is what steers exploration: an
    axis-aligned proposal wastes attempts across a narrow degeneracy, which is
    exactly the geometry of the supernova posterior, so ``cov`` should
    approximate the posterior covariance.

    ``scale`` multiplies the whole step. Keeping it separate from ``cov`` lets
    a pilot run fix the SHAPE once and then tune only the SIZE, which is the
    one-dimensional part of the problem.
    """

    is_symmetric = True

    def __init__(self, cov, scale=1.0):
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be square, got {cov.shape}")
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=0.0):
            raise ValueError("cov must be symmetric")
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.cov = cov
        self.scale = float(scale)
        self.n_params = cov.shape[0]
        # Factor once. A proposal covariance that is not positive definite is a
        # programming error, not a parameter point, so this is allowed to raise.
        self.cholesky = cholesky(cov, lower=True)

        # log q needs -0.5 log det(2 pi scale^2 cov), constant across calls.
        log_det_cov = 2.0 * np.sum(np.log(np.diag(self.cholesky)))
        self._log_norm = -0.5 * (
            self.n_params * np.log(2.0 * np.pi)
            + log_det_cov
            + 2.0 * self.n_params * np.log(self.scale)
        )

    def propose(self, state, rng):
        z = rng.standard_normal(self.n_params)
        return np.asarray(state, dtype=float) + self.scale * (self.cholesky @ z)

    def logpdf(self, target, given):
        """Multivariate normal log density, via the same triangular factor.

        Never forms cov^-1: with d = target - given and cov = L L^T,
        d^T cov^-1 d = ||L^-1 d||^2.
        """
        d = np.asarray(target, dtype=float) - np.asarray(given, dtype=float)
        y = solve_triangular(self.cholesky, d, lower=True, check_finite=False)
        return self._log_norm - 0.5 * (y @ y) / (self.scale * self.scale)

    def rescaled(self, scale):
        """A copy with a different step size, sharing the same shape."""
        return GaussianRandomWalk(self.cov, scale)

    def __repr__(self):
        return f"GaussianRandomWalk(cov={self.cov.tolist()}, scale={self.scale!r})"
