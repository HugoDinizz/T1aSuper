#!/usr/bin/env python
"""End-to-end Union2.1 analysis: data in, constraints and diagnostics out.

    python scripts/run_union21.py                 # default production run
    python scripts/run_union21.py --steps 2000    # quick smoke test
    python scripts/run_union21.py --auto-steps    # size the run from tau_int
    python scripts/run_union21.py --cross-check   # validate the marginalization

Everything is driven by one seed, so a rerun reproduces the chains bit for
bit. The script exits non-zero if the convergence diagnostics fail, so it can
be used as a check and not only as a report.

Products, all written to results/:

    summary.txt             the full report, exactly as printed
    chains.npz              retained samples, log posterior, acceptance
    hubble_diagram.png      the fit against the data, with residuals
    contours.png            the (Omega_m, Omega_Lambda) marginal posterior
    parameter_space.png     the same, on a scale that shows what is excluded
    convergence.png         traces, running means, autocorrelation
    dunkley_spectrum.png    the chain in Fourier space
"""

import argparse
import sys
from pathlib import Path

import matplotlib

import corner
import matplotlib.pyplot as plt
import numpy as np

from snia.bayes.posterior import Posterior, UniformPrior
from snia.bayes.proposals import GaussianRandomWalk
from snia.bayes.samplers import MetropolisHastings, run_chains
from snia.cosmology import FLRW
from snia.data.likelihood import Union21Likelihood
from snia.data.union21 import load_union21
from snia.diagnostics.autocorr import (
    autocorrelation,
    effective_sample_size,
    integrated_time,
    monte_carlo_error,
)
from snia.diagnostics.gelman_rubin import rank_normalized_rhat, split_rhat
from snia.diagnostics.spectral import MAX_R, MIN_J_STAR, fit_dunkley, power_spectrum

REPOSITORY = Path(__file__).resolve().parents[1]

NAMES = ["Omega_m", "Omega_Lambda"]
LABELS = [r"$\Omega_m$", r"$\Omega_\Lambda$"]
PRIOR_LOWER = [0.0, -1.0]
PRIOR_UPPER = [2.0, 3.0]

#: 2.38/sqrt(d) times the posterior covariance is the classic optimal scaling.
OPTIMAL_SCALE = 2.38 / np.sqrt(2)

#: Rhat above this is treated as a failure. A guideline, not a proof.
MAX_RHAT = 1.01

EINSTEIN_DE_SITTER = (1.0, 0.0)


class Tee:
    """Write the report to stdout and to a file at the same time."""

    def __init__(self, path):
        self.handle = open(path, "w")

    def __call__(self, text=""):
        print(text)
        self.handle.write(text + "\n")

    def close(self):
        self.handle.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=20000, help="steps per chain")
    parser.add_argument("--chains", type=int, default=4, help="number of chains")
    parser.add_argument("--warmup", type=int, default=8000, help="pilot run length")
    parser.add_argument("--burn", type=int, default=None,
                        help="steps discarded per chain (default: steps // 10)")
    parser.add_argument("--auto-steps", action="store_true",
                        help="choose --steps from a calibration run instead")
    parser.add_argument("--target-r", type=float, default=MAX_R,
                        help="target Dunkley r = tau_int/N used to size the run")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--no-systematics", action="store_true",
                        help="use the statistics-only covariance")
    parser.add_argument("--cross-check", action="store_true",
                        help="also run a 3-parameter chain with M sampled")
    parser.add_argument("--outdir", type=Path, default=REPOSITORY / "results")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def tune_proposal(posterior, n_warmup, rng):
    """Learn the shape of the posterior, then freeze it.

    A pilot chain with a crude diagonal proposal is enough to estimate the
    covariance. Adapting during the production run would break the Markov
    property, so the tuned covariance is fixed from here on.
    """
    pilot = MetropolisHastings(
        posterior, GaussianRandomWalk(np.diag([0.05**2, 0.08**2]))
    ).sample([0.3, 0.7], n_warmup, rng)
    kept = pilot.samples[n_warmup // 4 :]
    return np.cov(kept.T), kept.mean(axis=0), pilot.acceptance_rate


def calibrate_length(posterior, proposal, start, rng, target_r, probe_steps=6000):
    """How many steps per chain are needed, measured rather than guessed.

    The Dunkley ratio is r = tau_int/N exactly, and r is also 1/N_eff, so a
    target r fixes the length directly:

        N = tau_int / r_target.

    r = 0.01 means the standard error of the mean is a tenth of the posterior
    width. The other Dunkley threshold, j_star > 20, needs only N > 20 pi tau
    (about 63 tau) for a random walk, so the r condition is the binding one.

    tau_int is measured with the FINAL proposal: the pilot used a deliberately
    crude one and would give an unrepresentative answer.
    """
    probe = MetropolisHastings(posterior, proposal).sample(start, probe_steps, rng)
    tau = float(np.max(integrated_time(probe.samples[probe_steps // 4 :])))
    return int(np.ceil(tau / target_r)), tau


def dispersed_starts(posterior, centre, covariance, n_chains, rng, spread=3.0):
    """Valid starting points scattered well beyond the bulk of the posterior.

    Chains that begin far apart and end up agreeing are much stronger evidence
    than one stable-looking trace, which is the whole point of split-Rhat.
    """
    factor = np.linalg.cholesky(covariance)
    for _ in range(1000 * n_chains):
        starts = [
            centre + spread * (factor @ rng.standard_normal(centre.size))
            for _ in range(n_chains)
        ]
        if all(np.isfinite(posterior(start)) for start in starts):
            return np.array(starts)
    raise RuntimeError("could not find enough valid starting points")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def summarize(samples, likelihood, say):
    flat = samples.reshape(-1, samples.shape[-1])
    omega_m, omega_lambda = flat[:, 0], flat[:, 1]
    rows = [
        ("Omega_m", omega_m),
        ("Omega_Lambda", omega_lambda),
        ("Omega_k", 1.0 - omega_m - omega_lambda),
        ("q0", omega_m / 2.0 - omega_lambda),
    ]

    say("\n" + "-" * 74)
    say("CONSTRAINTS")
    say("-" * 74)
    say(f"{'parameter':14s} {'mean':>9} {'sd':>8} {'16%':>9} {'50%':>9} {'84%':>9}")
    for name, column in rows:
        low, mid, high = np.percentile(column, [16, 50, 84])
        say(f"{name:14s} {column.mean():9.4f} {column.std():8.4f} "
            f"{low:9.4f} {mid:9.4f} {high:9.4f}")

    q0 = rows[3][1]
    curvature = rows[2][1]
    best = flat.mean(axis=0)
    say("")
    say(f"accelerating (q0 < 0)          : {np.mean(q0 < 0.0):6.2%} of the posterior")
    say(f"consistent with flat (|Ok|<.05): {np.mean(np.abs(curvature) < 0.05):6.2%}")
    say(f"chi2 at the mean               : {likelihood.chi2(best):.2f} "
        f"for {likelihood.n_sn} SNe, chi2/dof = "
        f"{likelihood.chi2(best) / (likelihood.n_sn - 3):.4f}")
    say(f"delta chi2 vs Einstein-de Sitter: "
        f"{likelihood.chi2(EINSTEIN_DE_SITTER) - likelihood.chi2(best):.1f}")
    say(f"fitted zero point M            : {likelihood.m_offset(best):.4f} mag "
        f"-> H0 = {299792.458 / 10 ** ((likelihood.m_offset(best) - 25) / 5):.2f}")


def report_convergence(samples, acceptance, target_r, say):
    """Print every diagnostic and return whether they all passed."""
    tau = integrated_time(samples)
    ess = effective_sample_size(samples)
    mcse = monte_carlo_error(samples)
    rhat = split_rhat(samples)
    rhat_rank = rank_normalized_rhat(samples)
    n_chains, n_steps, _ = samples.shape
    spread = samples.reshape(-1, samples.shape[-1]).std(axis=0)

    say("\n" + "-" * 74)
    say("CONVERGENCE")
    say("-" * 74)
    say(f"acceptance per chain: {np.round(acceptance, 3)}   "
        f"(0.2-0.5 is healthy for a 2-D random walk)")
    say("")
    say(f"{'parameter':14s} {'tau_int':>8} {'ESS':>8} {'MCSE':>9} {'MCSE/sd':>9} "
        f"{'Rhat':>8} {'rank-Rhat':>10}")
    for index, name in enumerate(NAMES):
        say(f"{name:14s} {tau[index]:8.2f} {ess[index]:8.0f} {mcse[index]:9.5f} "
            f"{mcse[index] / spread[index]:9.4f} {rhat[index]:8.4f} "
            f"{rhat_rank[index]:10.4f}")
    say("")
    say("  tau_int   steps before a sample is effectively independent")
    say("  ESS       independent-equivalent draws, pooled over all chains")
    say("  MCSE/sd   error on the mean as a fraction of the posterior width")
    say(f"  rank-Rhat between-chain agreement; must be < {MAX_RHAT}")

    say("\nDunkley spectrum, per chain and parameter (unthinned):")
    say(f"{'parameter':14s} {'chain':>6} {'j*':>9} {'r':>9} {'alpha':>7}   verdict")
    spectra_pass = True
    for index, name in enumerate(NAMES):
        for chain in range(n_chains):
            fit = fit_dunkley(samples[chain, :, index])
            spectra_pass &= fit.passes
            say(f"{name:14s} {chain:6d} {fit.j_star:9.1f} {fit.r:9.5f} "
                f"{fit.alpha:7.2f}   {'PASS' if fit.passes else 'FAIL'}")
    say(f"\n  thresholds: j* > {MIN_J_STAR:.0f} (white plateau resolved), "
        f"r < {MAX_R} (mean precise)")

    say("\nWas the run long enough?")
    needed = int(np.ceil(tau.max() / target_r))
    say(f"  measured tau_int (worst parameter) : {tau.max():.2f}")
    say(f"  N needed per chain for r < {target_r:g}   : {needed:,}")
    say(f"  N actually used per chain          : {n_steps:,}  "
        f"({n_steps / needed:.1f}x)")

    rhat_pass = bool(np.all(rhat_rank < MAX_RHAT))
    say("")
    say(f"rank-normalized Rhat < {MAX_RHAT}  : {rhat_pass}")
    say(f"Dunkley thresholds passed  : {spectra_pass}")
    return rhat_pass and spectra_pass


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _style():
    """One consistent look for every figure."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "font.size": 10,
    })


DATA_COLOR = "0.55"
FIT_COLOR = "C3"
EDS_COLOR = "C2"


def figure_hubble(samples, likelihood, redshift, mu, sigma):
    """One message: the decelerating model fails, and the residuals show how."""
    flat = samples.reshape(-1, samples.shape[-1])
    best = flat.mean(axis=0)
    model, eds = FLRW(*best), FLRW(*EINSTEIN_DE_SITTER)
    offset = likelihood.m_offset(best)
    eds_offset = likelihood.m_offset(EINSTEIN_DE_SITTER)

    smooth = np.logspace(np.log10(0.014), np.log10(1.5), 300)
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 6.5), sharex=True, height_ratios=[2, 1]
    )

    top.errorbar(redshift, mu, yerr=sigma, fmt=".", ms=3, elinewidth=0.5,
                 alpha=0.35, color=DATA_COLOR, zorder=1)
    top.plot(smooth, model.distance_modulus(smooth, offset), color=FIT_COLOR,
             lw=2, zorder=3,
             label=rf"$\Omega_m$={best[0]:.2f}, $\Omega_\Lambda$={best[1]:.2f}")
    top.plot(smooth, eds.distance_modulus(smooth, eds_offset), color=EDS_COLOR,
             lw=2, ls="--", zorder=2, label=r"Einstein-de Sitter")
    top.set_ylabel(r"$\mu$  [mag]")
    top.set_title(f"Union2.1: {redshift.size} supernovae")
    top.legend(loc="lower right")

    edges = np.logspace(np.log10(0.014), np.log10(1.5), 12)
    centres = np.sqrt(edges[1:] * edges[:-1])
    which = np.digitize(redshift, edges) - 1

    def binned(residual):
        return np.array([residual[which == b].mean() if np.any(which == b) else np.nan
                         for b in range(len(centres))])

    bottom.errorbar(redshift, mu - model.distance_modulus(redshift, offset),
                    yerr=sigma, fmt=".", ms=3, elinewidth=0.5, alpha=0.25,
                    color=DATA_COLOR, zorder=0)
    bottom.axhline(0, color="k", lw=1, zorder=1)
    bottom.plot(centres, binned(mu - model.distance_modulus(redshift, offset)),
                "o-", color=FIT_COLOR, ms=5, lw=1.8, zorder=3)
    bottom.plot(centres, binned(mu - eds.distance_modulus(redshift, eds_offset)),
                "s--", color=EDS_COLOR, ms=5, lw=1.8, zorder=2)
    bottom.set_xscale("log")
    bottom.set_xlabel("redshift  $z$")
    bottom.set_ylabel(r"$\mu_{\rm obs}-\mu_{\rm th}$")
    bottom.set_ylim(-0.35, 0.55)
    bottom.set_title("binned residuals", fontsize=10)

    figure.tight_layout()
    return figure


def _credible_levels(histogram, probabilities=(0.683, 0.954)):
    """Density levels enclosing the given posterior mass, lowest first."""
    ordered = np.sort(histogram.ravel())[::-1]
    enclosed = np.cumsum(ordered) / ordered.sum()
    return [ordered[np.searchsorted(enclosed, p)] for p in probabilities][::-1]


def figure_contours(samples):
    """One message: the marginal constraints and their degeneracy."""
    flat = samples.reshape(-1, samples.shape[-1])
    figure = corner.corner(
        flat, labels=LABELS, levels=(0.683, 0.954),
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".3f",
        title_kwargs={"fontsize": 11, "pad": 8},
        label_kwargs={"fontsize": 14},
        color="#2171b5", fill_contours=True, plot_datapoints=False,
        contour_kwargs={"linewidths": 0.9},
        contourf_kwargs={"colors": [(1, 1, 1, 0), "#c6dbef", "#4292c6"]},
        hist_kwargs={"color": "#2171b5", "lw": 1.6},
        smooth=0.8, smooth1d=0.8,
    )
    for axis in figure.axes:
        axis.grid(True, alpha=0.2, lw=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    # the flat line, faint, on the joint panel only
    joint = np.array(figure.axes).reshape((2, 2))[1, 0]
    span = np.array(joint.get_xlim())
    joint.plot(span, 1 - span, "k--", lw=1.2, alpha=0.7, zorder=0)
    joint.set_xlim(*span)

    figure.set_size_inches(6.4, 6.4)
    return figure


def figure_parameter_space(samples, likelihood):
    """One message: what the data exclude, on a scale where it is visible.

    The corner plot auto-scales to the posterior, which puts Einstein-de Sitter
    at (1, 0) off the edge. Here the range is fixed so the comparison shows.
    """
    flat = samples.reshape(-1, samples.shape[-1])
    histogram, x_edges, y_edges = np.histogram2d(
        flat[:, 0], flat[:, 1], bins=80,
        range=[[0.0, 1.15], [-0.3, 1.5]], density=True,
    )
    histogram = histogram.T
    x = 0.5 * (x_edges[1:] + x_edges[:-1])
    y = 0.5 * (y_edges[1:] + y_edges[:-1])
    levels = _credible_levels(histogram)

    figure, axis = plt.subplots(figsize=(6.8, 5.8))
    axis.contourf(x, y, histogram, levels=levels + [histogram.max()],
                  colors=["#c6dbef", "#4292c6"])
    axis.contour(x, y, histogram, levels=levels, colors="0.3", linewidths=0.8)

    line = np.linspace(0.0, 1.15, 20)
    axis.plot(line, 1 - line, "k--", lw=1.3, label=r"$\Omega_k = 0$")
    axis.plot(line, line / 2, "k:", lw=1.3, label=r"$q_0 = 0$")

    best = flat.mean(axis=0)
    delta = likelihood.chi2(EINSTEIN_DE_SITTER) - likelihood.chi2(best)
    axis.plot(*EINSTEIN_DE_SITTER, "s", color=EDS_COLOR, ms=9,
              label=rf"EdS  ($\Delta\chi^2$={delta:.0f})")

    axis.set_xlim(0.0, 1.15)
    axis.set_ylim(-0.3, 1.5)
    axis.set_xlabel(r"$\Omega_m$", fontsize=13)
    axis.set_ylabel(r"$\Omega_\Lambda$", fontsize=13)
    axis.set_title("Union2.1, 68.3% and 95.4% credible regions")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def figure_convergence(samples):
    """Three questions, one row each, in words rather than symbols."""
    n_chains, n_steps, _ = samples.shape
    rho = autocorrelation(samples, max_lag=60)
    tau = integrated_time(samples)
    mcse = monte_carlo_error(samples)
    window = min(3000, n_steps)

    figure, axes = plt.subplots(3, 2, figsize=(10, 8.5))
    for index, label in enumerate(LABELS):
        trace, running, correlation = axes[:, index]

        for chain in range(n_chains):
            trace.plot(samples[chain, :window, index], lw=0.4, alpha=0.8)
        trace.set_ylabel(label, fontsize=13)
        trace.set_xlabel(f"step (first {window:,} of {n_steps:,})")
        trace.set_title(f"{n_chains} chains, started far apart, now overlapping")

        final = samples[:, :, index].mean()
        for chain in range(n_chains):
            cumulative = np.cumsum(samples[chain, :, index]) / np.arange(1, n_steps + 1)
            running.plot(cumulative, lw=0.9)
        # mcse is for the POOLED mean; a single chain carries n_chains times
        # fewer samples, so its own error bar is sqrt(n_chains) wider.
        single = mcse[index] * np.sqrt(n_chains)
        running.axhspan(final - single, final + single, color="0.6", alpha=0.4)
        running.set_xlim(0, n_steps)
        running.set_ylim(final - 8 * single, final + 8 * single)
        running.set_ylabel(f"mean of {label} so far", fontsize=11)
        running.set_xlabel("step")
        running.set_title("grey band: expected spread of one chain's mean")

        correlation.plot(rho[:, index], color="C0")
        correlation.axhline(0, color="k", lw=0.8)
        correlation.axvline(tau[index], color=FIT_COLOR, ls="--", lw=1.2)
        correlation.set_ylabel("correlation", fontsize=11)
        correlation.set_xlabel("separation between samples (steps)")
        correlation.set_title(f"independent after about {tau[index]:.0f} steps")

    figure.tight_layout()
    return figure


def figure_dunkley(samples):
    """One message: the power is flat at small j, then falls."""
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for index, label in enumerate(LABELS):
        chain = samples[0, :, index]
        modes, power = power_spectrum(chain)
        fit = fit_dunkley(chain)
        wavenumber = 2.0 * np.pi * modes / chain.size

        # Coarse log bins: the raw periodogram scatters exponentially, and
        # a bin holding one mode is just noise.
        edges = np.unique(np.geomspace(1, modes[-1], 22).astype(int))
        centres, means = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (modes >= low) & (modes < high)
            if np.count_nonzero(inside) >= 2:
                centres.append(np.sqrt(low * high))
                means.append(power[inside].mean())

        axis = axes[index]
        axis.loglog(centres, means, "o", color="C0", ms=4)
        axis.loglog(modes, fit.p0 / (1 + (wavenumber / fit.k_star) ** fit.alpha),
                    color=FIT_COLOR, lw=2)
        axis.set_xlabel("Fourier mode  $j$")
        axis.set_ylabel("$P_j$")
        axis.set_title(rf"{label}    $j_\star$ = {fit.j_star:.0f},   "
                       rf"$r$ = {fit.r:.4f}")

    figure.tight_layout()
    return figure


def cross_check(likelihood, tuned_covariance, n_steps, rng, marginalized_flat, say):
    """Sample M as a third parameter and compare with the analytic result."""
    prior = UniformPrior([0.0, -1.0, 41.0], [2.0, 3.0, 45.0])
    posterior = Posterior(
        lambda theta: -0.5 * likelihood.chi2_at_offset(theta[:2], theta[2]), prior
    )

    start_covariance = np.zeros((3, 3))
    start_covariance[:2, :2] = tuned_covariance
    start_covariance[2, 2] = 0.004**2
    pilot = MetropolisHastings(
        posterior, GaussianRandomWalk(start_covariance)
    ).sample([0.3, 0.7, 43.17], max(2000, n_steps // 4), rng)

    result = run_chains(
        posterior,
        GaussianRandomWalk(np.cov(pilot.samples[500:].T), scale=2.38 / np.sqrt(3)),
        pilot.samples[-4:],
        n_steps,
        rng,
    )
    flat = result.burned(n_steps // 10).flat()

    say("\n" + "-" * 74)
    say("CROSS-CHECK: M sampled instead of marginalized analytically")
    say("-" * 74)
    say(f"{'parameter':14s} {'analytic':>21s} {'M sampled':>21s} {'agreement':>13s}")
    for index, name in enumerate(NAMES):
        analytic, sampled = marginalized_flat[:, index], flat[:, index]
        error = np.hypot(monte_carlo_error(analytic)[0],
                         monte_carlo_error(sampled)[0])
        say(f"{name:14s} {analytic.mean():9.4f} +- {analytic.std():.4f} "
            f"{sampled.mean():9.4f} +- {sampled.std():.4f} "
            f"{abs(analytic.mean() - sampled.mean()) / error:8.2f} MCSE")
    say(f"{'M':14s} {'(marginalized out)':>21s} "
        f"{flat[:, 2].mean():9.4f} +- {flat[:, 2].std():.4f}")
    say("\nA sign error in B, or a swapped residual convention, would separate")
    say("these. Nothing else in the pipeline would notice.")


# --------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    _style()
    say = Tee(args.outdir / "summary.txt")
    rng = np.random.default_rng(args.seed)

    redshift, mu, covariance = load_union21(systematics=not args.no_systematics)
    likelihood = Union21Likelihood(redshift, mu, covariance)
    posterior = Posterior(
        likelihood.log_likelihood, UniformPrior(PRIOR_LOWER, PRIOR_UPPER)
    )

    say("=" * 74)
    say("UNION2.1  ->  (Omega_m, Omega_Lambda)")
    say("=" * 74)
    say(f"{redshift.size} supernovae, "
        f"{'with' if not args.no_systematics else 'without'} systematics")
    say(f"seed {args.seed}, {args.chains} chains, warm-up {args.warmup}")

    tuned, centre, pilot_acceptance = tune_proposal(posterior, args.warmup, rng)
    say(f"\npilot acceptance {pilot_acceptance:.3f}; tuned proposal covariance, "
        f"now frozen:")
    for row in tuned:
        say("  " + "  ".join(f"{value: .6f}" for value in row))

    proposal = GaussianRandomWalk(tuned, scale=OPTIMAL_SCALE)
    n_steps = args.steps
    recommended, probe_tau = calibrate_length(
        posterior, proposal, centre, rng, args.target_r
    )
    say(f"\ncalibration run: tau_int = {probe_tau:.2f}")
    say(f"  N = tau_int / r_target = {probe_tau:.2f} / {args.target_r:g} "
        f"= {recommended:,} steps per chain")
    if args.auto_steps:
        n_steps = max(recommended, 1000)
        say(f"  --auto-steps: using {n_steps:,}")
    else:
        say(f"  using --steps = {n_steps:,} "
            f"({n_steps / recommended:.1f}x the requirement)")

    burn = args.burn if args.burn is not None else n_steps // 10
    starts = dispersed_starts(posterior, centre, tuned, args.chains, rng)
    say(f"\ndispersed starting points (3 sigma from the pilot):")
    for start in starts:
        say("  " + "  ".join(f"{value: .4f}" for value in start))

    production = run_chains(posterior, proposal, starts, n_steps, rng)
    samples = production.burned(burn).samples
    say(f"\n{args.chains} x {n_steps:,} steps, discarding {burn:,} as burn-in")
    say(f"{posterior.n_calls:,} posterior evaluations")

    summarize(samples, likelihood, say)
    converged = report_convergence(samples, production.acceptance_rate,
                                   args.target_r, say)

    figures = {
        "hubble_diagram": figure_hubble(samples, likelihood, redshift, mu,
                                        np.sqrt(np.diag(covariance))),
        "contours": figure_contours(samples),
        "parameter_space": figure_parameter_space(samples, likelihood),
        "convergence": figure_convergence(samples),
        "dunkley_spectrum": figure_dunkley(samples),
    }
    for name, figure in figures.items():
        figure.savefig(args.outdir / f"{name}.png", bbox_inches="tight")
        plt.close(figure)

    np.savez_compressed(
        args.outdir / "chains.npz",
        samples=samples,
        log_prob=production.burned(burn).log_prob,
        accepted=production.burned(burn).accepted,
        seed=args.seed,
        names=np.array(NAMES),
    )

    if args.cross_check:
        cross_check(likelihood, tuned, n_steps, rng,
                    samples.reshape(-1, samples.shape[-1]), say)

    say("\n" + "=" * 74)
    say("CONVERGED" if converged else "NOT CONVERGED: run longer or retune")
    say("=" * 74)
    say(f"\nwritten to {args.outdir}:")
    for name in sorted(p.name for p in args.outdir.glob("*")):
        say(f"  {name}")
    say.close()
    return 0 if converged else 1


if __name__ == "__main__":
    # Headless only when run as a script. Setting this at import time would
    # also silence the inline backend, so a notebook importing this module
    # would get no figures at all.
    matplotlib.use("Agg")
    sys.exit(main())
