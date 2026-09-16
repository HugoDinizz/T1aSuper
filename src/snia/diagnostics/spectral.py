"""The Dunkley power-spectrum convergence test.

Dunkley fit the chain power with

    P(k) = P0 / (1 + (k/k_star)^alpha),      j_star = k_star N / (2 pi),

and require j_star > 20 and r = P0/(N s^2) < 0.01.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

#: Dunkley's thresholds: enough long modes to resolve the plateau, and a mean
#: known to 10% of the posterior width (r = 0.01 means MCSE/s = 0.1).
MIN_J_STAR = 20.0
MAX_R = 0.01
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
    timescales in the chain."""
    x = np.asarray(chain, dtype=float).ravel()
    amplitude = np.fft.rfft(x - x.mean()) / np.sqrt(x.size)
    power = np.abs(amplitude) ** 2
    return np.arange(power.size)[1:], power[1:]


def fit_dunkley(chain, n_iterations=3):
    x = np.asarray(chain, dtype=float).ravel()
    n_steps = x.size
    variance = x.var(ddof=1)
    j, power = power_spectrum(x)
    k = 2.0 * np.pi * j / n_steps

    n_modes = min(power.size, max(100, n_steps // 50))
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
    """Rough (log P0, log k_star, log alpha) to start the optimizer."""
    n_low = max(5, power.size // 200)
    p0 = max(power[:n_low].mean(), np.finfo(float).tiny)

    running = np.cumsum(power) / np.arange(1, power.size + 1)
    below = np.flatnonzero(running < 0.5 * p0)
    k_star = k[below[0]] if below.size else k[power.size // 2]
    k_star = max(k_star, k[0])

    return np.log([p0, k_star, 2.0])
