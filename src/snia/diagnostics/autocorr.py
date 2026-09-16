"""Autocorrelation, integrated time, effective sample size and MCSE."""

import numpy as np

from ._chains import as_chains


def autocorrelation(samples, max_lag=None):
    """rho_l = Cov(x_t, x_{t+l}) / Var(x_t), averaged over chains."""
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape
    if max_lag is None:
        max_lag = n_steps - 1
    max_lag = min(max_lag, n_steps - 1)

    result = np.empty((max_lag + 1, n_params))
    for parameter in range(n_params):
        per_chain = [_normalized_acf(chains[c, :, parameter]) for c in range(n_chains)]
        result[:, parameter] = np.mean(per_chain, axis=0)[: max_lag + 1]
    return result


def integrated_time(samples):
    """tau_int by Geyer's initial positive sequence.
        Gamma_m = rho_{2m} + rho_{2m+1},
        tau_int = -1 + 2 sum_{m=0}^{M} Gamma_m .
    """
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape
    floor = 1.0 / np.log10(n_chains * n_steps)

    times = np.empty(n_params)
    for parameter in range(n_params):
        values = chains[:, :, parameter]
        if np.all(values == values.flat[0]):
            times[parameter] = np.inf
            continue
        rho = np.mean(
            [_normalized_acf(values[c]) for c in range(n_chains)], axis=0
        )
        times[parameter] = max(_initial_positive_sequence(rho), floor)
    return times


def effective_sample_size(samples):
    """N_eff = n_chains * n_steps / tau_int, the number of independent draws."""
    chains = as_chains(samples)
    n_chains, n_steps, _ = chains.shape
    total = n_chains * n_steps
    with np.errstate(divide="ignore"):
        return total / integrated_time(chains)


def monte_carlo_error(samples):
    """MCSE of the posterior mean, s / sqrt(N_eff)."""
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape
    pooled = chains.reshape(n_chains * n_steps, n_params)
    ess = effective_sample_size(chains)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pooled.std(axis=0, ddof=1) / np.sqrt(ess)


def _normalized_acf(x):
    """Normalized autocorrelation of a 1-D array, by FFT."""
    centred = x - x.mean()
    size = 1 << (2 * x.size - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[: x.size]
    if acf[0] <= 0.0:
        return np.zeros(x.size)
    return acf / acf[0]


def _initial_positive_sequence(rho):
    """Sum adjacent pairs of rho while they stay positive; return tau_int."""
    n_pairs = rho.size // 2
    pairs = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)

    non_positive = np.flatnonzero(pairs <= 0.0)
    cut = int(non_positive[0]) if non_positive.size else pairs.size
    return -1.0 + 2.0 * float(pairs[:cut].sum())
