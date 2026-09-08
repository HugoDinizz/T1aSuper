"""Metropolis-Hastings.

The sampler takes any callable returning a log density, so it can be checked
against a target whose moments are known in closed form. It never touches the
global numpy random state: a generator is required, and the same generator
seed reproduces a chain bit for bit.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MCMCResult:
    """Output of a run.

    ``samples`` is (n_steps, n_params) for a single chain and
    (n_chains, n_steps, n_params) for a set of them; ``log_prob`` and
    ``accepted`` carry the matching leading axes. The (n_chains, n_steps,
    n_params) layout is what the diagnostics package expects.
    """

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

    Three details that are easy to get wrong and that the tests pin down:

    * **A rejection is still a sample.** The current state is appended again.
      Keeping only accepted proposals changes the target frequencies and biases
      every estimate.
    * **The current log density is cached.** After a rejection it is reused, so
      each iteration costs exactly one target evaluation.
    * **A non-finite target is a rejection, not an error.** -inf and NaN both
      make the comparison false, so unphysical proposals are simply refused.

    ``min(0, log alpha)`` is not applied because log u < 0 always, which makes
    the clamp redundant.
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
        """Run one chain of n_steps iterations from initial_state."""
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

            # log(0) is -inf, which correctly always accepts; NaN compares
            # false, which correctly always rejects.
            with np.errstate(divide="ignore"):
                log_u = np.log(rng.random())

            if log_u < log_alpha:
                state, log_p = candidate, log_p_new
                accepted[step] = True

            samples[step] = state
            log_prob[step] = log_p

        return MCMCResult(samples, log_prob, accepted)


def run_chains(log_target, proposal, initial_states, n_steps, rng):
    """Run one chain per initial state and stack them.

    Each chain gets its own child generator via ``rng.spawn``, so the streams
    are independent and the whole set is still reproducible from one seed.
    Dispersed starting points are what make split-Rhat meaningful: agreement
    between chains that began far apart is much stronger evidence than one
    stable-looking trace.
    """
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
