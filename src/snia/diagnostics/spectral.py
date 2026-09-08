"""The Dunkley power-spectrum convergence test.

A chain viewed in Fourier space separates two questions that a trace plot
mixes together: whether the long-timescale modes have flattened into white
noise (has the chain reached stationarity?) and how precisely the mean is
known (is the run long enough?).

Dunkley et al. (2005) fit the chain power with

    P(k) = P0 / (1 + (k/k_star)^alpha),      j_star = k_star N / (2 pi),

and require j_star > 20 and r = P0/(N s^2) < 0.01. Those thresholds are
paper-specific; a chain trapped in one mode can still look stationary and
white, so this must be combined with dispersed chains.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

#: Dunkley's thresholds: enough long modes to resolve the plateau, and a mean
#: known to 10% of the posterior width (r = 0.01 means MCSE/s = 0.1).
MIN_J_STAR = 20.0
MAX_R = 0.01

#: Range allowed for the high-frequency slope. A random walk gives alpha ~ 2,
#: and alpha < 1 does not describe a turnover at all, so the lower bound is
#: both physical and necessary: see the note in fit_dunkley.
ALPHA_BOUNDS = (1.0, 8.0)


@dataclass
class DunkleySpectrum:
    """Result of fitting P(k) = P0/(1 + (k/k_star)^alpha) to one chain."""

    p0: float
    k_star: float
    alpha: float
    j_star: float
    r: float
    n_steps: int
    variance: float
    n_modes_fitted: int

    @property
    def white_regime_resolved(self):
        """j_star > 20: enough modes below the turnover to see the plateau."""
        return self.j_star > MIN_J_STAR

    @property
    def mean_is_converged(self):
        """r < 0.01, i.e. the standard error of the mean is under 10% of s."""
        return self.r < MAX_R

    @property
    def passes(self):
        return self.white_regime_resolved and self.mean_is_converged

    def __repr__(self):
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"DunkleySpectrum(P0={self.p0:.4g}, j_star={self.j_star:.1f}, "
            f"alpha={self.alpha:.2f}, r={self.r:.2e}) [{verdict}]"
        )


def power_spectrum(chain):
    """Fourier modes of a centred chain.

        a_j = (1/sqrt(N)) sum_n (x_n - x_bar) exp(2 pi i j n / N),
        P_j = |a_j|^2,   k_j = 2 pi j / N.

    Returns ``(j, power)`` for j = 1 .. N//2, where small j probes the longest
    timescales in the chain.

    Subtracting the mean is belt and braces: sum_n exp(-2 pi i j n / N) = 0 for
    every j != 0, so a constant offset only ever lands in the j = 0 mode, which
    is dropped anyway. It is kept because it states the intent, and because
    j = 0 would otherwise dwarf everything if this function ever returned it.

    With this normalization the expected power is the spectral density, so
    P(k -> 0) = s^2 tau_int. That is what ties this test to the effective
    sample size: r = P0/(N s^2) = tau_int/N = 1/N_eff.
    """
    x = np.asarray(chain, dtype=float).ravel()
    amplitude = np.fft.rfft(x - x.mean()) / np.sqrt(x.size)
    power = np.abs(amplitude) ** 2
    return np.arange(power.size)[1:], power[1:]


def fit_dunkley(chain, n_iterations=3):
    """Fit the Dunkley model to one unthinned chain of one parameter.

    The periodogram ordinates are asymptotically independent and exponentially
    distributed about the true spectrum, so the fit maximizes

        log L = -sum_j [ log P(k_j) + P_hat_j / P(k_j) ]

    rather than least-squares on the raw power. Least squares would be wrong
    twice over: the noise is multiplicative, not additive, and log P_hat is
    biased low by the Euler-Mascheroni constant.

    Only the low-k modes are fitted, since the model describes the turnover
    and not whatever the spectrum does at high frequency. Following Dunkley,
    the range is refined iteratively to about 10 j_star.
    """
    x = np.asarray(chain, dtype=float).ravel()
    n_steps = x.size
    variance = x.var(ddof=1)
    j, power = power_spectrum(x)
    k = 2.0 * np.pi * j / n_steps

    n_modes = min(power.size, max(100, n_steps // 50))

    # Two guards, both needed, and each of which was found by a fit that went
    # wrong rather than anticipated:
    #
    #  * alpha is bounded below at 1. A shallow alpha opens a degenerate ridge
    #    where k_star collapses towards zero and P0 grows to compensate, since
    #    P0/(1 + (k/k_star)^alpha) -> P0 (k_star/k)^alpha is then a slowly
    #    falling power law that mimics the data. It reports j_star = 0 and a
    #    spurious failure. alpha < 1 does not describe a turnover anyway.
    #
    #  * each iteration restarts from a guess derived on its own mode window,
    #    rather than warm-starting from the previous fit. Warm starting
    #    inherits a degenerate state and cannot escape it.
    bounds = [
        (None, None),
        (np.log(k[0]) - 5.0, np.log(k[-1]) + 10.0),
        (np.log(ALPHA_BOUNDS[0]), np.log(ALPHA_BOUNDS[1])),
    ]

    for _ in range(n_iterations):
        fit = minimize(
            _negative_log_likelihood,
            _initial_guess(k[:n_modes], power[:n_modes]),
            args=(k[:n_modes], power[:n_modes]),
            method="Nelder-Mead",
            bounds=bounds,
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000},
        )
        p0, k_star, alpha = np.exp(fit.x)
        j_star = k_star * n_steps / (2.0 * np.pi)
        n_modes = int(np.clip(10 * j_star, 50, power.size))

    with np.errstate(divide="ignore", invalid="ignore"):
        r = p0 / (n_steps * variance)

    return DunkleySpectrum(
        p0=float(p0),
        k_star=float(k_star),
        alpha=float(alpha),
        j_star=float(j_star),
        r=float(r),
        n_steps=n_steps,
        variance=float(variance),
        n_modes_fitted=int(n_modes),
    )


def _model(k, log_params):
    p0, k_star, alpha = np.exp(log_params)
    return p0 / (1.0 + (k / k_star) ** alpha)


def _negative_log_likelihood(log_params, k, power):
    """-log L for exponentially distributed periodogram ordinates."""
    if not np.all(np.isfinite(log_params)):
        return np.inf
    model = _model(k, log_params)
    if not np.all(model > 0.0) or not np.all(np.isfinite(model)):
        return np.inf
    return float(np.sum(np.log(model) + power / model))


def _initial_guess(k, power):
    """Rough (log P0, log k_star, log alpha) to start the optimizer.

    The plateau is estimated from the lowest modes and the turnover from where
    the running mean first falls to half of it. alpha starts at 2, the slope a
    random walk produces.
    """
    n_low = max(5, power.size // 200)
    p0 = max(power[:n_low].mean(), np.finfo(float).tiny)

    running = np.cumsum(power) / np.arange(1, power.size + 1)
    below = np.flatnonzero(running < 0.5 * p0)
    k_star = k[below[0]] if below.size else k[power.size // 2]
    k_star = max(k_star, k[0])

    return np.log([p0, k_star, 2.0])
