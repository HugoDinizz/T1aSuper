"""Acceptance tests for snia.bayes (milestone 2).

CLAUDE.md asks for three things: recover a known 2-D Gaussian to within MCSE,
reproduce a run from a seed, and verify that rejected proposals appear as
repeated states. All three are here, plus the checks that catch the mistakes
those three would miss -- most importantly that the Hastings ratio is actually
applied, which a symmetric proposal can never reveal.

Nothing in this file imports snia.cosmology or snia.data: the target is a
Gaussian whose mean and covariance are known in closed form.
"""

import numpy as np
import pytest

from snia.bayes.posterior import Posterior, UniformPrior
from snia.bayes.proposals import GaussianRandomWalk
from snia.bayes.samplers import MCMCResult, MetropolisHastings, run_chains

MU = np.array([1.0, -2.0])
COV = np.array([[2.0, 0.8], [0.8, 0.5]])
COV_INV = np.linalg.inv(COV)

#: 2.38/sqrt(d) times the target covariance is the classic optimal scaling.
OPTIMAL_SCALE = 2.38 / np.sqrt(2)


def gaussian_log_target(theta):
    d = np.asarray(theta, dtype=float) - MU
    return -0.5 * d @ COV_INV @ d


@pytest.fixture(scope="module")
def long_run():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    return run_chains(
        gaussian_log_target,
        proposal,
        [[0.0, 0.0], [5.0, -6.0], [-4.0, 2.0], [3.0, 3.0]],
        20000,
        np.random.default_rng(20260907),
    ).burned(2000)


# --------------------------------------------------------------------------
# 1. Does it sample the right distribution?
# --------------------------------------------------------------------------


#: Measured integrated autocorrelation time for this target and proposal.
#: 4 chains x 18000 retained steps therefore carry ESS ~ 9.5e3, NOT 7.2e4:
#: chain length is not sample size.
TAU_INT = 7.6


def test_recovers_a_known_gaussian_mean_within_mcse(long_run):
    """The headline acceptance test: the sampler must reproduce known moments.

    MCSE = s/sqrt(N_eff) with N_eff = N/tau_int, which is ~0.015 and ~0.007 on
    the two means. The threshold is 5 MCSE; across twelve seeds the observed
    error never exceeded 2.9 MCSE, so this is a real check with margin rather
    than a tolerance fitted to one lucky run.
    """
    flat = long_run.flat()
    ess = flat.shape[0] / TAU_INT
    mcse = flat.std(axis=0) / np.sqrt(ess)
    error = np.abs(flat.mean(axis=0) - MU)
    assert np.all(error < 5.0 * mcse), f"error {error} vs MCSE {mcse}"


def test_recovers_a_known_gaussian_covariance(long_run):
    assert np.allclose(np.cov(long_run.flat().T), COV, rtol=0.05, atol=0.03)


def test_dispersed_chains_agree_with_each_other(long_run):
    """Four chains started far apart must forget where they began."""
    per_chain = long_run.samples.mean(axis=1)
    assert np.all(np.abs(per_chain - MU).max(axis=0) < 0.1)
    spread = per_chain.std(axis=0)
    assert np.all(spread < 0.05), f"between-chain spread {spread}"


def test_acceptance_rate_is_sensible_for_the_optimal_scaling(long_run):
    """d=2 random walk at 2.38/sqrt(d) sits near 0.35."""
    assert np.all((long_run.acceptance_rate > 0.25) & (long_run.acceptance_rate < 0.45))


# --------------------------------------------------------------------------
# 2. The Hastings ratio
# --------------------------------------------------------------------------


class AutoregressiveProposal:
    """q(theta' | theta) = N(theta'; drift * theta, sigma^2 I), drift != 1.

    Deliberately ASYMMETRIC: proposing from a to b is not as likely as from b
    to a, so the Hastings ratio does not cancel. A sampler that drops it will
    converge to the wrong stationary distribution, which is what makes this
    the sharpest available test of the acceptance step.
    """

    is_symmetric = False

    def __init__(self, drift, sigma, n_params):
        self.drift, self.sigma, self.n_params = drift, sigma, n_params
        self._log_norm = -n_params * np.log(sigma * np.sqrt(2.0 * np.pi))

    def propose(self, state, rng):
        return self.drift * np.asarray(state) + self.sigma * rng.standard_normal(
            self.n_params
        )

    def logpdf(self, target, given):
        d = np.asarray(target) - self.drift * np.asarray(given)
        return self._log_norm - 0.5 * (d @ d) / self.sigma**2


def test_asymmetric_proposal_still_targets_the_right_distribution():
    """With the Hastings correction, an AR(1) proposal recovers N(MU, COV)."""
    proposal = AutoregressiveProposal(drift=0.9, sigma=1.2, n_params=2)
    result = MetropolisHastings(gaussian_log_target, proposal).sample(
        MU, 120000, np.random.default_rng(11)
    ).burned(5000)
    flat = result.flat()
    assert np.allclose(flat.mean(axis=0), MU, atol=0.06)
    assert np.allclose(np.cov(flat.T), COV, rtol=0.08, atol=0.05)


def test_symmetric_shortcut_agrees_with_the_general_formula():
    """GaussianRandomWalk sets is_symmetric, so the sampler skips the two
    logpdf calls. Forcing them back on must not change the chain."""

    class NotDeclaredSymmetric(GaussianRandomWalk):
        is_symmetric = False

    cov, scale = COV, 0.8
    a = MetropolisHastings(
        gaussian_log_target, GaussianRandomWalk(cov, scale)
    ).sample([0.0, 0.0], 3000, np.random.default_rng(12))
    b = MetropolisHastings(
        gaussian_log_target, NotDeclaredSymmetric(cov, scale)
    ).sample([0.0, 0.0], 3000, np.random.default_rng(12))
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.accepted, b.accepted)


# --------------------------------------------------------------------------
# 3. Reproducibility
# --------------------------------------------------------------------------


def test_identical_seed_gives_byte_identical_chains():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    sampler = MetropolisHastings(gaussian_log_target, proposal)
    a = sampler.sample([0.0, 0.0], 2000, np.random.default_rng(99))
    b = sampler.sample([0.0, 0.0], 2000, np.random.default_rng(99))
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.log_prob, b.log_prob)
    assert np.array_equal(a.accepted, b.accepted)


def test_different_seeds_give_different_chains():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    sampler = MetropolisHastings(gaussian_log_target, proposal)
    a = sampler.sample([0.0, 0.0], 2000, np.random.default_rng(1))
    b = sampler.sample([0.0, 0.0], 2000, np.random.default_rng(2))
    assert not np.array_equal(a.samples, b.samples)


def test_run_chains_is_reproducible_and_chains_are_independent():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    starts = [[0.0, 0.0], [2.0, -1.0], [-2.0, 1.0]]
    a = run_chains(gaussian_log_target, proposal, starts, 500, np.random.default_rng(5))
    b = run_chains(gaussian_log_target, proposal, starts, 500, np.random.default_rng(5))
    assert np.array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples[0], a.samples[1])


def test_each_chain_is_driven_by_its_own_spawned_stream():
    """Chain i must be reproducible in isolation from the i-th child generator.

    Sharing one generator across chains would still give a valid sampler --
    the chains would consume disjoint consecutive segments of one stream --
    which is why nothing else here catches it. It matters anyway: with a
    shared generator a chain cannot be re-run or parallelized on its own
    without changing its result, and a parallel map would race.
    """
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    starts = [[0.0, 0.0], [2.0, -1.0], [-2.0, 1.0]]

    together = run_chains(
        gaussian_log_target, proposal, starts, 400, np.random.default_rng(31)
    )
    children = np.random.default_rng(31).spawn(len(starts))
    for i, (start, child) in enumerate(zip(starts, children)):
        alone = MetropolisHastings(gaussian_log_target, proposal).sample(
            start, 400, child
        )
        assert np.array_equal(together.samples[i], alone.samples)


def test_the_global_numpy_random_state_is_never_touched():
    """A run must not depend on, or perturb, np.random."""
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    sampler = MetropolisHastings(gaussian_log_target, proposal)

    np.random.seed(1234)
    before = np.random.get_state()[1].copy()
    sampler.sample([0.0, 0.0], 500, np.random.default_rng(7))
    assert np.array_equal(np.random.get_state()[1], before)


def test_an_integer_seed_is_refused():
    """Silently accepting a seed would make the generator implicit."""
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    sampler = MetropolisHastings(gaussian_log_target, proposal)
    with pytest.raises(TypeError, match="Generator"):
        sampler.sample([0.0, 0.0], 10, 42)


# --------------------------------------------------------------------------
# 4. A rejection is still a sample
# --------------------------------------------------------------------------


def test_rejected_proposals_repeat_the_previous_state():
    """Deleting repeated states changes the target frequencies and biases
    every estimate, so a rejection must append the current state again."""
    proposal = GaussianRandomWalk(COV, scale=6.0)      # large steps: many rejections
    result = MetropolisHastings(gaussian_log_target, proposal).sample(
        MU, 3000, np.random.default_rng(13)
    )
    rejected = np.flatnonzero(~result.accepted)
    assert rejected.size > 500, "expected plenty of rejections at this scale"

    for i in rejected:
        if i == 0:
            continue      # the state before step 0 is the initial state, not stored
        assert np.array_equal(result.samples[i], result.samples[i - 1])
        assert result.log_prob[i] == result.log_prob[i - 1]


def test_accepted_proposals_move_the_state():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    result = MetropolisHastings(gaussian_log_target, proposal).sample(
        MU, 3000, np.random.default_rng(14)
    )
    for i in np.flatnonzero(result.accepted):
        if i == 0:
            continue
        assert not np.array_equal(result.samples[i], result.samples[i - 1])


def test_acceptance_rate_matches_the_indicator_array():
    proposal = GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    result = MetropolisHastings(gaussian_log_target, proposal).sample(
        MU, 1000, np.random.default_rng(15)
    )
    assert result.acceptance_rate == pytest.approx(result.accepted.mean())


def test_uphill_moves_are_always_accepted():
    """log alpha >= 0 must accept regardless of u, since log u < 0 always."""

    class UphillProposal:
        is_symmetric = True

        def propose(self, state, rng):
            return np.asarray(state) * 0.0 + MU     # jump straight to the mode

        def logpdf(self, target, given):
            return 0.0

    result = MetropolisHastings(gaussian_log_target, UphillProposal()).sample(
        [4.0, 4.0], 50, np.random.default_rng(16)
    )
    assert result.accepted.all()
    assert np.allclose(result.samples, MU)


# --------------------------------------------------------------------------
# 5. Cost and robustness
# --------------------------------------------------------------------------


def test_exactly_one_target_evaluation_per_step():
    """The current log density is cached, so a rejection costs nothing extra.
    Recomputing it would roughly double the cost of the whole analysis."""
    calls = {"n": 0}

    def counted(theta):
        calls["n"] += 1
        return gaussian_log_target(theta)

    proposal = GaussianRandomWalk(COV, scale=6.0)     # mostly rejections
    n_steps = 500
    MetropolisHastings(counted, proposal).sample(MU, n_steps, np.random.default_rng(17))
    assert calls["n"] == n_steps + 1                  # +1 to validate the start


def test_non_finite_target_is_a_rejection_not_an_error():
    """Outside the support the target is -inf; the chain must simply refuse."""
    prior = UniformPrior([-1.0, -1.0], [1.0, 1.0])
    target = Posterior(gaussian_log_target, prior)
    proposal = GaussianRandomWalk(np.eye(2), scale=2.0)
    result = MetropolisHastings(target, proposal).sample(
        [0.0, 0.0], 4000, np.random.default_rng(18)
    )
    assert np.all(np.isfinite(result.log_prob))
    assert np.all(result.samples >= -1.0) and np.all(result.samples <= 1.0)
    assert 0.0 < result.acceptance_rate < 1.0


def test_nan_from_the_target_is_a_rejection():
    def nan_beyond_one(theta):
        return np.nan if np.any(np.abs(theta) > 1.0) else gaussian_log_target(theta)

    proposal = GaussianRandomWalk(np.eye(2), scale=1.5)
    result = MetropolisHastings(nan_beyond_one, proposal).sample(
        [0.0, 0.0], 2000, np.random.default_rng(19)
    )
    assert not np.any(np.isnan(result.log_prob))
    assert np.all(np.abs(result.samples) <= 1.0)


def test_invalid_initial_state_is_refused_with_a_clear_message():
    prior = UniformPrior([0.0, 0.0], [1.0, 1.0])
    target = Posterior(gaussian_log_target, prior)
    proposal = GaussianRandomWalk(np.eye(2))
    with pytest.raises(ValueError, match="non-finite log density"):
        MetropolisHastings(target, proposal).sample(
            [5.0, 5.0], 10, np.random.default_rng(20)
        )


@pytest.mark.parametrize("n_steps", [0, -1])
def test_non_positive_step_count_is_refused(n_steps):
    proposal = GaussianRandomWalk(COV)
    with pytest.raises(ValueError, match="n_steps"):
        MetropolisHastings(gaussian_log_target, proposal).sample(
            MU, n_steps, np.random.default_rng(21)
        )


def test_sampler_construction_validates_its_arguments():
    with pytest.raises(TypeError, match="callable"):
        MetropolisHastings("not callable", GaussianRandomWalk(COV))

    with pytest.raises(TypeError, match="propose"):
        MetropolisHastings(gaussian_log_target, object())

    class ProposeOnly:
        def propose(self, state, rng):
            return state

    # logpdf is checked separately: a proposal without it cannot be used for
    # an asymmetric move, so the omission must be caught up front.
    with pytest.raises(TypeError, match="logpdf"):
        MetropolisHastings(gaussian_log_target, ProposeOnly())


# --------------------------------------------------------------------------
# 6. The result container
# --------------------------------------------------------------------------


def test_result_shapes_single_and_multiple_chains(long_run):
    assert long_run.samples.shape == (4, 18000, 2)
    assert long_run.log_prob.shape == (4, 18000)
    assert long_run.accepted.shape == (4, 18000)
    assert long_run.n_chains == 4 and long_run.n_params == 2 and long_run.n_steps == 18000
    assert long_run.acceptance_rate.shape == (4,)
    assert long_run.flat().shape == (4 * 18000, 2)

    single = MetropolisHastings(
        gaussian_log_target, GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    ).sample(MU, 100, np.random.default_rng(22))
    assert single.samples.shape == (100, 2)
    assert single.n_chains == 1
    assert np.isscalar(single.acceptance_rate) or single.acceptance_rate.ndim == 0


def test_burned_drops_a_prefix_from_every_array():
    result = MetropolisHastings(
        gaussian_log_target, GaussianRandomWalk(COV, scale=OPTIMAL_SCALE)
    ).sample(MU, 100, np.random.default_rng(23))
    burned = result.burned(30)
    assert burned.samples.shape == (70, 2)
    assert np.array_equal(burned.samples, result.samples[30:])
    assert np.array_equal(burned.log_prob, result.log_prob[30:])


def test_log_prob_matches_the_target_at_the_stored_samples(long_run):
    """The cached log density must correspond to the state actually stored."""
    rng = np.random.default_rng(24)
    chain, idx = 0, rng.integers(0, long_run.n_steps, 200)
    for i in idx:
        assert long_run.log_prob[chain, i] == pytest.approx(
            gaussian_log_target(long_run.samples[chain, i]), rel=1e-12
        )


# --------------------------------------------------------------------------
# 7. Posterior and prior
# --------------------------------------------------------------------------


def test_posterior_is_the_sum_of_prior_and_likelihood():
    prior = UniformPrior([-5.0, -5.0], [5.0, 5.0])
    post = Posterior(gaussian_log_target, prior)
    theta = np.array([0.5, -1.0])
    assert post(theta) == pytest.approx(gaussian_log_target(theta))


def test_posterior_short_circuits_outside_the_prior():
    """The likelihood must not even be called when the prior rejects: on the
    real problem that call integrates a cosmology on a 2001-point grid."""
    called = {"n": 0}

    def expensive(theta):
        called["n"] += 1
        return 0.0

    post = Posterior(expensive, UniformPrior([0.0, 0.0], [1.0, 1.0]))
    assert post([5.0, 5.0]) == -np.inf
    assert called["n"] == 0
    assert post([0.5, 0.5]) == 0.0
    assert called["n"] == 1


def test_posterior_turns_a_nan_likelihood_into_minus_inf():
    post = Posterior(lambda t: np.nan, UniformPrior([-1.0, -1.0], [1.0, 1.0]))
    assert post([0.0, 0.0]) == -np.inf


def test_uniform_prior_boundaries_are_inclusive():
    prior = UniformPrior([0.0, -1.0], [2.0, 3.0])
    assert prior([0.0, -1.0]) == 0.0
    assert prior([2.0, 3.0]) == 0.0
    assert prior([2.0 + 1e-12, 0.0]) == -np.inf
    assert prior([-1e-12, 0.0]) == -np.inf


def test_uniform_prior_matches_the_exercise_box():
    """0 <= Om <= 2, -1 <= OL <= 3 from the lecture."""
    prior = UniformPrior([0.0, -1.0], [2.0, 3.0])
    assert prior([0.3, 0.7]) == 0.0
    assert prior([2.5, 0.7]) == -np.inf
    assert prior([0.3, 3.5]) == -np.inf


def test_uniform_prior_sampling_stays_in_the_box():
    prior = UniformPrior([0.0, -1.0], [2.0, 3.0])
    draws = prior.sample(np.random.default_rng(25), size=5000)
    assert draws.shape == (5000, 2)
    assert np.all(draws >= prior.lower) and np.all(draws <= prior.upper)


def test_uniform_prior_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="upper bound"):
        UniformPrior([1.0], [0.0])
    with pytest.raises(ValueError, match="differ"):
        UniformPrior([0.0, 0.0], [1.0])
