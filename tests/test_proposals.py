"""Acceptance tests for snia.bayes.proposals.

A proposal has to do two things correctly: draw from q(.|state), and report
log q. The second is what lets one sampler handle asymmetric moves, so it is
checked against scipy's multivariate normal rather than against itself.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from snia.bayes.proposals import GaussianRandomWalk, Proposal

COV = np.array([[2.0, 0.8], [0.8, 0.5]])


def test_proposal_interface_cannot_be_instantiated():
    """Proposal is an interface: propose and logpdf are both required."""
    with pytest.raises(TypeError):
        Proposal()


def test_gaussian_random_walk_is_symmetric():
    assert GaussianRandomWalk(COV).is_symmetric is True


def test_proposals_are_centred_on_the_current_state():
    """A random walk adds zero-mean noise, so the mean displacement vanishes."""
    rng = np.random.default_rng(1)
    proposal = GaussianRandomWalk(COV)
    state = np.array([3.0, -1.0])
    draws = np.array([proposal.propose(state, rng) for _ in range(20000)])
    # standard error on each mean is sqrt(diag(cov)/n) ~ 0.010, 0.005
    assert np.allclose(draws.mean(axis=0), state, atol=0.05)


def test_proposal_covariance_is_reproduced():
    rng = np.random.default_rng(2)
    proposal = GaussianRandomWalk(COV)
    draws = np.array([proposal.propose(np.zeros(2), rng) for _ in range(40000)])
    assert np.allclose(np.cov(draws.T), COV, rtol=0.05, atol=0.02)


def test_scale_multiplies_the_step_covariance_quadratically():
    """cov is the shape, scale is the size: the step covariance is scale^2 cov."""
    rng = np.random.default_rng(3)
    proposal = GaussianRandomWalk(COV, scale=0.5)
    draws = np.array([proposal.propose(np.zeros(2), rng) for _ in range(40000)])
    assert np.allclose(np.cov(draws.T), 0.25 * COV, rtol=0.06, atol=0.02)


@pytest.mark.parametrize("scale", [0.3, 1.0, 2.5])
def test_logpdf_matches_scipy(scale):
    """The normalization matters for asymmetric proposals, so check it."""
    proposal = GaussianRandomWalk(COV, scale=scale)
    reference = multivariate_normal(mean=np.zeros(2), cov=scale**2 * COV)
    rng = np.random.default_rng(4)
    given = rng.normal(size=(20, 2))
    target = given + rng.normal(size=(20, 2))
    for a, b in zip(target, given):
        assert proposal.logpdf(a, b) == pytest.approx(reference.logpdf(a - b), rel=1e-12)


def test_logpdf_is_symmetric_in_its_arguments():
    """q(a|b) == q(b|a) for a centred walk: this is why the Hastings ratio
    cancels and the sampler may skip it."""
    proposal = GaussianRandomWalk(COV)
    rng = np.random.default_rng(5)
    for _ in range(50):
        a, b = rng.normal(size=2), rng.normal(size=2)
        assert proposal.logpdf(a, b) == pytest.approx(proposal.logpdf(b, a), rel=1e-14)


def test_logpdf_peaks_at_no_movement():
    proposal = GaussianRandomWalk(COV)
    origin = np.zeros(2)
    best = proposal.logpdf(origin, origin)
    rng = np.random.default_rng(6)
    for _ in range(50):
        assert proposal.logpdf(rng.normal(size=2), origin) <= best


def test_same_generator_seed_reproduces_the_same_proposals():
    proposal = GaussianRandomWalk(COV)
    state = np.array([1.0, 1.0])
    a = [proposal.propose(state, np.random.default_rng(7)) for _ in range(1)]
    b = [proposal.propose(state, np.random.default_rng(7)) for _ in range(1)]
    assert np.array_equal(a, b)


def test_propose_does_not_mutate_the_state():
    proposal = GaussianRandomWalk(COV)
    state = np.array([1.0, 2.0])
    proposal.propose(state, np.random.default_rng(8))
    assert np.array_equal(state, [1.0, 2.0])


def test_one_dimensional_proposal_works():
    proposal = GaussianRandomWalk([[0.25]])
    assert proposal.n_params == 1
    rng = np.random.default_rng(9)
    draws = np.array([proposal.propose([0.0], rng) for _ in range(20000)])
    assert draws.std() == pytest.approx(0.5, rel=0.05)


def test_rescaled_keeps_the_shape_and_changes_the_size():
    proposal = GaussianRandomWalk(COV, scale=1.0)
    smaller = proposal.rescaled(0.4)
    assert np.array_equal(smaller.cov, proposal.cov)
    assert smaller.scale == 0.4


@pytest.mark.parametrize(
    "cov, scale, match",
    [
        (np.ones((2, 3)), 1.0, "square"),
        (np.array([[1.0, 0.5], [0.2, 1.0]]), 1.0, "symmetric"),
        (COV, 0.0, "positive"),
        (COV, -1.0, "positive"),
    ],
)
def test_invalid_construction_is_rejected(cov, scale, match):
    with pytest.raises(ValueError, match=match):
        GaussianRandomWalk(cov, scale)


def test_non_positive_definite_covariance_is_rejected():
    """A proposal covariance that cannot be factored is a programming error,
    so it is allowed to raise rather than being silently repaired."""
    with pytest.raises(np.linalg.LinAlgError):
        GaussianRandomWalk(np.array([[1.0, 2.0], [2.0, 1.0]]))
