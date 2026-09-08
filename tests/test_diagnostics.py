"""Acceptance tests for snia.diagnostics (milestone 4).

CLAUDE.md asks for an AR(1) process with a known tau_int. That process is the
right target for all three modules, because everything about it is analytic:

    x_{t+1} = phi x_t + eps,   eps ~ N(0, sigma^2)

    rho_l    = phi^l
    var      = sigma^2 / (1 - phi^2)
    tau_int  = (1 + phi) / (1 - phi)
    S(k)     = sigma^2 / (1 - 2 phi cos k + phi^2)

Expanding S for small k gives the Dunkley form exactly, with

    P0 = sigma^2/(1-phi)^2,   k_star = (1-phi)/sqrt(phi),   alpha = 2,

and since P0 = var * tau_int, the spectral ratio r = P0/(N s^2) equals
tau_int/N = 1/N_eff. So the autocorrelation route and the spectral route are
two independent estimators of the same number, and the tests compare them.

Nothing here imports snia.cosmology or snia.data.
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from snia.diagnostics.autocorr import (
    autocorrelation,
    effective_sample_size,
    integrated_time,
    monte_carlo_error,
)
from snia.diagnostics.gelman_rubin import rank_normalized_rhat, split_rhat
from snia.diagnostics.spectral import ALPHA_BOUNDS, fit_dunkley, power_spectrum


def ar1(phi, n_steps, rng, sigma=1.0, burn_in=2000):
    """Stationary AR(1) draw. lfilter runs the recursion in C."""
    noise = rng.normal(0.0, sigma, n_steps + burn_in)
    return lfilter([1.0], [1.0, -phi], noise)[burn_in:]


def analytic_tau(phi):
    return (1.0 + phi) / (1.0 - phi)


@pytest.fixture(scope="module")
def ar1_long():
    rng = np.random.default_rng(20260908)
    return {phi: ar1(phi, 200000, rng) for phi in (0.0, 0.5, 0.8, 0.9, 0.95)}


# --------------------------------------------------------------------------
# 1. Autocorrelation and integrated time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.8, 0.9])
def test_integrated_time_matches_the_analytic_ar1_value(ar1_long, phi):
    """The headline acceptance test: tau_int = (1+phi)/(1-phi)."""
    tau = integrated_time(ar1_long[phi])[0]
    assert tau == pytest.approx(analytic_tau(phi), rel=0.05)


def test_slow_chain_is_estimated_without_a_systematic_bias(ar1_long):
    """At phi = 0.95 the true tau is 39, and the estimate straddles it.

    Geyer's initial positive sequence keeps every pair while the pairs are
    positive, so it does not systematically truncate the tail the way a window
    rule does. What remains is scatter: across seeds this estimate moves by
    about +-7% at N = 200000, so the tolerance is set from that and not from a
    single lucky run.
    """
    tau = integrated_time(ar1_long[0.95])[0]
    assert tau == pytest.approx(analytic_tau(0.95), rel=0.15)


@pytest.mark.parametrize("phi", [-0.2, -0.5, -0.8, -0.9])
def test_anti_correlated_chains_give_a_positive_correlation_time(phi):
    """Regression: a negative tau_int is arithmetically impossible here.

    A window rule that watches the running sum 1 + 2 sum rho_l can stop at the
    very first lag of an anti-correlated chain, while that total is still
    negative, and hand back a negative tau_int -- and therefore a NEGATIVE
    effective sample size and a NaN Monte Carlo error. Summing adjacent lags
    in pairs cancels the alternation inside each pair instead.

    The Metropolis-Hastings sampler in snia.bayes never produces such chains,
    so this was latent rather than active, but the numbers it returned were
    meaningless rather than merely imprecise.
    """
    chain = ar1(phi, 50000, np.random.default_rng(0))
    tau = integrated_time(chain)[0]
    assert tau > 0.0
    assert effective_sample_size(chain)[0] > 0.0
    assert np.isfinite(monte_carlo_error(chain)[0])


def test_effective_sample_size_is_capped_at_n_log10_n():
    """The floor on tau_int caps ESS, which is what keeps an antithetic chain
    from claiming an unbounded number of independent draws."""
    alternating = np.array([(-1.0) ** i for i in range(10000)])
    alternating = alternating + 1e-9 * np.random.default_rng(0).normal(size=10000)
    tau = integrated_time(alternating)[0]
    assert tau == pytest.approx(1.0 / np.log10(10000), rel=1e-9)
    assert effective_sample_size(alternating)[0] == pytest.approx(
        10000 * np.log10(10000), rel=1e-9
    )


@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9])
def test_autocorrelation_matches_phi_to_the_lag(ar1_long, phi):
    rho = autocorrelation(ar1_long[phi], max_lag=5)[:, 0]
    assert rho[0] == pytest.approx(1.0)
    for lag in range(1, 6):
        assert rho[lag] == pytest.approx(phi**lag, abs=0.01)


def test_white_noise_has_unit_integrated_time(ar1_long):
    assert integrated_time(ar1_long[0.0])[0] == pytest.approx(1.0, abs=0.05)


def test_effective_sample_size_is_length_over_tau(ar1_long):
    chain = ar1_long[0.9]
    ess = effective_sample_size(chain)[0]
    assert ess == pytest.approx(chain.size / integrated_time(chain)[0], rel=1e-12)
    assert ess < chain.size / 15.0        # correlation really does cost


def test_monte_carlo_error_is_std_over_sqrt_ess(ar1_long):
    chain = ar1_long[0.9]
    expected = chain.std(ddof=1) / np.sqrt(effective_sample_size(chain)[0])
    assert monte_carlo_error(chain)[0] == pytest.approx(expected, rel=1e-12)


def test_the_sample_mean_lands_within_a_few_mcse_of_zero(ar1_long):
    """The point of MCSE: it must actually bracket the error."""
    for phi in (0.5, 0.8, 0.9):
        chain = ar1_long[phi]
        assert abs(chain.mean()) < 4.0 * monte_carlo_error(chain)[0]


def test_mcse_shrinks_as_the_square_root_of_the_chain_length():
    rng = np.random.default_rng(1)
    long_chain = ar1(0.8, 160000, rng)
    short = monte_carlo_error(long_chain[:10000])[0]
    full = monte_carlo_error(long_chain)[0]
    assert full / short == pytest.approx(np.sqrt(10000 / 160000), rel=0.25)


def test_several_short_chains_give_the_same_tau_as_one_long_one():
    rng = np.random.default_rng(2)
    chains = np.stack([ar1(0.8, 40000, rng) for _ in range(4)])[:, :, np.newaxis]
    pooled_tau = integrated_time(chains)[0]
    assert pooled_tau == pytest.approx(analytic_tau(0.8), rel=0.05)
    assert effective_sample_size(chains)[0] == pytest.approx(4 * 40000 / pooled_tau)


def test_a_stuck_chain_reports_infinite_correlation_time():
    """A chain that never moved carries no information, and must say so
    rather than dividing by zero and claiming perfect mixing."""
    stuck = np.full(1000, 2.5)
    assert integrated_time(stuck)[0] == np.inf
    assert effective_sample_size(stuck)[0] == 0.0
    assert np.isnan(monte_carlo_error(stuck)[0])


# --------------------------------------------------------------------------
# 2. Gelman-Rubin
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def converged_chains():
    rng = np.random.default_rng(3)
    return np.stack([ar1(0.8, 8000, rng) for _ in range(4)])[:, :, np.newaxis]


def test_converged_chains_give_rhat_near_one(converged_chains):
    assert split_rhat(converged_chains)[0] == pytest.approx(1.0, abs=0.01)
    assert rank_normalized_rhat(converged_chains)[0] == pytest.approx(1.0, abs=0.01)


def test_an_offset_chain_is_caught(converged_chains):
    offset = converged_chains.copy()
    offset[0] += 3.0
    assert split_rhat(offset)[0] > 1.05
    assert rank_normalized_rhat(offset)[0] > 1.05


def test_splitting_catches_a_drift_shared_by_every_chain():
    """Four chains all drifting the same way have no BETWEEN-chain spread, so
    an unsplit Rhat would be ~1. Cutting each chain in half exposes it."""
    rng = np.random.default_rng(4)
    trend = np.linspace(0.0, 4.0, 6000)
    drifting = np.stack([ar1(0.8, 6000, rng) + trend for _ in range(4)])[:, :, np.newaxis]
    assert split_rhat(drifting)[0] > 1.05


def test_rank_normalization_catches_a_scale_difference_that_rhat_misses(converged_chains):
    """One chain three times wider, with the same mean.

    The plain statistic compares locations, so it is nearly blind here. The
    folded version, built on |x - median|, turns a difference in scale into a
    difference in location and sees it. This is the reason both are provided.
    """
    wider = converged_chains.copy()
    wider[0] *= 3.0
    assert split_rhat(wider)[0] < 1.02             # missed
    assert rank_normalized_rhat(wider)[0] > 1.05   # caught


def test_rank_normalized_rhat_is_finite_for_a_distribution_without_variance():
    """Cauchy has no finite variance, so a variance-ratio statistic rests on
    quantities that do not exist. Ranking first removes that dependence."""
    rng = np.random.default_rng(5)
    cauchy = np.stack([rng.standard_cauchy(4000) for _ in range(4)])[:, :, np.newaxis]
    value = rank_normalized_rhat(cauchy)[0]
    assert np.isfinite(value)
    assert value == pytest.approx(1.0, abs=0.05)


def test_rhat_handles_several_parameters_at_once():
    rng = np.random.default_rng(6)
    chains = rng.normal(size=(4, 3000, 3))
    assert split_rhat(chains).shape == (3,)
    assert np.allclose(split_rhat(chains), 1.0, atol=0.05)


def test_too_short_a_chain_is_refused():
    with pytest.raises(ValueError, match="at least 4 steps"):
        split_rhat(np.zeros((2, 3, 1)))


# --------------------------------------------------------------------------
# 3. The Dunkley power spectrum
# --------------------------------------------------------------------------


def test_power_spectrum_normalization_is_the_spectral_density():
    """For white noise the spectrum is flat at the variance, so the mean of
    the periodogram must reproduce it. This pins the 1/sqrt(N) convention."""
    white = np.random.default_rng(7).normal(size=40000)
    j, power = power_spectrum(white)
    assert power.mean() == pytest.approx(white.var(), rel=0.02)


def test_power_spectrum_drops_the_zero_mode():
    j, power = power_spectrum(np.random.default_rng(8).normal(size=1000))
    assert j[0] == 1
    assert power.size == j.size == 500


@pytest.mark.parametrize("phi", [0.9, 0.95])
def test_dunkley_fit_recovers_the_analytic_ar1_spectrum(phi):
    """P0 = sigma^2/(1-phi)^2, k_star = (1-phi)/sqrt(phi), alpha = 2.

    Only phi close to 1 is tested, because the Dunkley form is the SMALL-k
    expansion of the AR(1) spectrum. At phi = 0.5 the turnover sits at
    k_star = 0.71, far from small, and the fit compensates with a shallower
    alpha -- a model mismatch, not an implementation error.
    """
    chain = ar1(phi, 100000, np.random.default_rng(9))
    fit = fit_dunkley(chain)
    assert fit.p0 == pytest.approx(1.0 / (1.0 - phi) ** 2, rel=0.08)
    assert fit.k_star == pytest.approx((1.0 - phi) / np.sqrt(phi), rel=0.08)
    assert fit.alpha == pytest.approx(2.0, abs=0.2)
    assert fit.j_star == pytest.approx(fit.k_star * fit.n_steps / (2 * np.pi), rel=1e-12)


@pytest.mark.parametrize("phi", [0.9, 0.95])
def test_spectral_r_equals_one_over_the_effective_sample_size(phi):
    """The cross-check between the two halves of this package.

    r = P0/(N s^2) and P0 = s^2 tau_int, so r = tau_int/N = 1/N_eff. The
    spectral fit and the autocorrelation window share no code and no
    assumptions beyond stationarity, so agreement is meaningful.

    The tolerance is set by how noisy a tau estimate is, not by the code:
    the standard error of tau_int is roughly tau sqrt(2(2M+1)/N) with a window
    M ~ 5 tau, which is 6% at phi = 0.9 and 9% at phi = 0.95. Measured over
    eight seeds the ratio spans [0.94, 1.14], so 20% is the honest bound.
    """
    chain = ar1(phi, 100000, np.random.default_rng(10))
    assert fit_dunkley(chain).r == pytest.approx(
        1.0 / effective_sample_size(chain)[0], rel=0.20
    )


def test_short_chains_fail_and_long_chains_pass():
    """The progression in the lecture: the thresholds must actually bite."""
    rng = np.random.default_rng(11)
    short = fit_dunkley(ar1(0.9, 500, rng))
    assert not short.white_regime_resolved      # j_star < 20
    assert not short.mean_is_converged          # r > 0.01
    assert not short.passes

    long = fit_dunkley(ar1(0.9, 100000, rng))
    assert long.white_regime_resolved
    assert long.mean_is_converged
    assert long.passes


@pytest.mark.parametrize("seed", [0, 1, 2, 12])
def test_white_noise_has_no_turnover_in_range(seed):
    """With no correlation there is no turnover, so k_star runs off past the
    highest mode and the plateau extends over the whole spectrum.

    alpha is genuinely UNIDENTIFIED here: once k_star exceeds every k in the
    data, any slope describes the same flat model, and across seeds the fit
    settles anywhere between the two bounds. Only k_star and P0 -- and hence
    r -- carry information, so only those are asserted.
    """
    white = np.random.default_rng(seed).normal(size=50000)
    fit = fit_dunkley(white)
    assert fit.j_star > 1e4
    assert ALPHA_BOUNDS[0] <= fit.alpha <= ALPHA_BOUNDS[1]
    assert fit.r == pytest.approx(1.0 / 50000, rel=0.2)
    assert fit.passes


def test_alpha_stays_inside_its_bounds():
    """At alpha = 0 the model collapses to the constant P0/2, so a flat
    spectrum would be fitted with P0 a factor two too large."""
    for chain in (np.random.default_rng(13).normal(size=20000),
                  ar1(0.9, 20000, np.random.default_rng(14))):
        fit = fit_dunkley(chain)
        assert ALPHA_BOUNDS[0] <= fit.alpha <= ALPHA_BOUNDS[1]


def test_repr_reports_the_verdict():
    passing = fit_dunkley(ar1(0.9, 100000, np.random.default_rng(15)))
    failing = fit_dunkley(ar1(0.9, 500, np.random.default_rng(16)))
    assert "PASS" in repr(passing)
    assert "FAIL" in repr(failing)


# --------------------------------------------------------------------------
# 4. Input handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "function", [integrated_time, effective_sample_size, monte_carlo_error, split_rhat]
)
def test_one_two_and_three_dimensional_inputs(function):
    """1-D is one chain of one parameter; 2-D is one chain of several, which
    is what MetropolisHastings.sample returns; 3-D is (chains, steps, params)."""
    rng = np.random.default_rng(17)
    assert function(rng.normal(size=5000)).shape == (1,)
    assert function(rng.normal(size=(5000, 3))).shape == (3,)
    assert function(rng.normal(size=(4, 5000, 3))).shape == (3,)


def test_four_dimensional_input_is_refused():
    with pytest.raises(ValueError, match="1, 2 or 3 dimensions"):
        integrated_time(np.zeros((2, 2, 2, 2)))


def test_fft_autocorrelation_matches_a_direct_sum():
    """The FFT shortcut must reproduce the definition exactly.

    An unpadded transform computes the CIRCULAR correlation, which wraps the
    end of the chain onto its start. The error scales as 1/N, so it hides
    completely in a long chain: at N = 200000 it is 6e-6, but at N = 300 it is
    4e-2. Testing on a short, strongly correlated chain is what makes the
    zero-padding visible.
    """
    chain = ar1(0.9, 300, np.random.default_rng(18))
    centred = chain - chain.mean()
    direct = np.array(
        [np.sum(centred[: chain.size - lag] * centred[lag:]) for lag in range(21)]
    ) / np.sum(centred**2)

    assert np.allclose(autocorrelation(chain, max_lag=20)[:, 0], direct, atol=1e-12)


def test_the_fit_does_not_collapse_on_repeated_chains():
    """Regression: j_star must not occasionally fall to zero.

    A shallow alpha opens a ridge where k_star -> 0 and P0 grows to match,
    turning the model into a slowly falling power law that fits the data
    almost as well. The fit then reports j_star = 0 and a spurious FAIL, on a
    chain no different from its neighbours. Warm-starting each iteration from
    the previous fit made it permanent.

    With alpha >= 1 and a fresh start per iteration, 25 statistically
    identical chains give 25 consistent answers. Before the fix, 2 of these
    25 collapsed.
    """
    j_stars = np.array(
        [
            fit_dunkley(ar1(0.8, 18000, np.random.default_rng(seed))).j_star
            for seed in range(25)
        ]
    )
    assert np.all(j_stars > 200.0), f"collapsed fits: {j_stars[j_stars <= 200]}"
    assert j_stars.max() / j_stars.min() < 3.0
