"""Combining a prior and a likelihood into one log density.

Nothing here knows about cosmology or supernovae. The sampler must be
exercisable against a target whose answer is known analytically, so every
object in this package takes plain callables and plain arrays.
"""

import numpy as np


class UniformPrior:
    """Flat prior on a box, 0 inside and -inf outside.

    The value inside is 0, not the normalized -sum(log(upper-lower)): only
    ratios of the target enter Metropolis-Hastings, so an overall constant is
    irrelevant. It would matter for evidence, which this exercise does not
    compute.
    """

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower {self.lower.shape} and upper {self.upper.shape} differ"
            )
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must exceed its lower bound")

    @property
    def n_params(self):
        return self.lower.size

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        inside = np.all(theta >= self.lower) and np.all(theta <= self.upper)
        return 0.0 if inside else -np.inf

    def sample(self, rng, size=None):
        """Draw from the prior, for dispersed chain starts."""
        shape = self.lower.shape if size is None else (size, *self.lower.shape)
        return rng.uniform(self.lower, self.upper, size=shape)

    def __repr__(self):
        return f"UniformPrior(lower={self.lower.tolist()}, upper={self.upper.tolist()})"


class Posterior:
    """log posterior = log prior + log likelihood, up to a constant.

    The prior is evaluated FIRST and the likelihood is skipped when the prior
    is -inf. That is not a micro-optimization: the likelihood here integrates a
    cosmology on a 2001-point grid, and a random-walk proposal spends a good
    fraction of its attempts outside the box.

    A non-finite likelihood is passed through as -inf rather than raised on, so
    that the sampler sees a rejection instead of an exception.
    """

    def __init__(self, log_likelihood, log_prior):
        if not callable(log_likelihood) or not callable(log_prior):
            raise TypeError("log_likelihood and log_prior must be callables")
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.n_calls = 0

    def __call__(self, theta):
        self.n_calls += 1
        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf
        log_like = self.log_likelihood(theta)
        if np.isnan(log_like):
            return -np.inf
        return log_prior + log_like

    def __repr__(self):
        return f"Posterior({self.log_likelihood!r}, {self.log_prior!r})"
