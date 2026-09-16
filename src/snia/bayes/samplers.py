"""Metropolis-Hastings"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MCMCResult:
    samples: np.ndarray
    log_prob: np.ndarray
    accepted: np.ndarray

    @property
    def acceptance_rate(self):
        """Fraction of proposals accepted; per chain when there are several."""
        return self.accepted.mean(axis=-1)

    @property
    def n_params(self):
        return self.samples.shape[-1]

    @property
    def n_steps(self):
        return self.samples.shape[-2]

    @property
    def n_chains(self):
        return 1 if self.samples.ndim == 2 else self.samples.shape[0]

    def burned(self, n_burn):
        """Drop the first n_burn steps, keeping every other axis."""
        return MCMCResult(
            self.samples[..., n_burn:, :],
            self.log_prob[..., n_burn:],
            self.accepted[..., n_burn:],
        )

    def flat(self):
        """All chains stacked into one (n_chains*n_steps, n_params) array."""
        return self.samples.reshape(-1, self.n_params)
class MetropolisHastings:
    """Random-walk Metropolis-Hastings with an explicit accept/reject step.

    Per iteration:

        theta' ~ q(. | theta)
        log alpha = log pi(theta') - log pi(theta)
                    + log q(theta | theta') - log q(theta' | theta)
        accept if log u < log alpha,  u ~ U(0,1)
    """

    def __init__(self, log_target, proposal):
        if not callable(log_target):
            raise TypeError("log_target must be callable")
        for name in ("propose", "logpdf"):
            if not hasattr(proposal, name):
                raise TypeError(f"proposal must provide {name}()")
        self.log_target = log_target
        self.proposal = proposal

    def sample(self, initial_state, n_steps, rng):
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "pass an explicit numpy.random.Generator, e.g. "
                "np.random.default_rng(seed); the global np.random is never used"
            )
        if n_steps < 1:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        state = np.atleast_1d(np.asarray(initial_state, dtype=float)).copy()
        log_p = float(self.log_target(state))
        if not np.isfinite(log_p):
            raise ValueError(
                f"initial state {state.tolist()} has non-finite log density "
                f"({log_p}); start the chain somewhere the target is defined"
            )

        n_params = state.size
        samples = np.empty((n_steps, n_params))
        log_prob = np.empty(n_steps)
        accepted = np.zeros(n_steps, dtype=bool)

        symmetric = getattr(self.proposal, "is_symmetric", False)

        for step in range(n_steps):
            candidate = self.proposal.propose(state, rng)
            log_p_new = float(self.log_target(candidate))

            log_alpha = log_p_new - log_p
            if not symmetric:
                log_alpha += self.proposal.logpdf(state, candidate)
                log_alpha -= self.proposal.logpdf(candidate, state)
            with np.errstate(divide="ignore"):
                log_u = np.log(rng.random())

            if log_u < log_alpha:
                state, log_p = candidate, log_p_new
                accepted[step] = True

            samples[step] = state
            log_prob[step] = log_p

        return MCMCResult(samples, log_prob, accepted)


def run_chains(log_target, proposal, initial_states, n_steps, rng):
    """Run one chain per initial state and stack them."""
    initial_states = np.atleast_2d(np.asarray(initial_states, dtype=float))
    children = rng.spawn(initial_states.shape[0])

    results = [
        MetropolisHastings(log_target, proposal).sample(start, n_steps, child)
        for start, child in zip(initial_states, children)
    ]
    return MCMCResult(
        np.stack([r.samples for r in results]),
        np.stack([r.log_prob for r in results]),
        np.stack([r.accepted for r in results]),
    )
