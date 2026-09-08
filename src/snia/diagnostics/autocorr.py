"""Autocorrelation, integrated time, effective sample size and MCSE.

Chain length is not sample size. A Markov chain of N correlated draws carries
the information of roughly N/tau_int independent ones, and it is that number
which sets the Monte Carlo error on any reported quantity.
"""

import numpy as np

from ._chains import as_chains


def autocorrelation(samples, max_lag=None):
    """rho_l = Cov(x_t, x_{t+l}) / Var(x_t), averaged over chains.

    Returns an array of shape (max_lag + 1, n_params), starting at rho_0 = 1.
    Computed by FFT, which is O(N log N) instead of the O(N^2) of a direct
    sum -- the difference between seconds and hours on a long chain.
    """
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

    tau_int = 1 + 2 sum_{l>=1} rho_l is a sum that has to be truncated: at
    large lag rho_l is estimated from ever fewer pairs and is pure noise.
    Geyer's rule pairs adjacent lags,

        Gamma_m = rho_{2m} + rho_{2m+1},

    and keeps terms only while Gamma_m stays positive. For a reversible Markov
    chain Gamma_m is provably positive and decreasing, so the first
    non-positive pair is noise and everything from there on is dropped. Since
    rho_0 = 1, summing in pairs gives

        tau_int = -1 + 2 sum_{m=0}^{M} Gamma_m .

    Pairing is what makes this safe on ANTI-correlated chains, and that is why
    it replaced a window rule here. A rule that watches the running sum -- as
    Sokal's does -- can stop at the very first lag while the total is still
    negative, and return a negative tau_int, hence a negative effective sample
    size and a NaN Monte Carlo error. Adjacent lags of an alternating chain
    cancel inside a pair instead.

    The result is floored at 1/log10(N), which caps the effective sample size
    at N log10(N). This is Stan's safeguard: an estimate above that is not
    trustworthy, and without it a nearly antithetic chain can still drive
    tau_int to zero or below. It errs towards reporting too few independent
    samples, never too many.

    Returns +inf for a chain that never moved.
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
    """MCSE of the posterior mean, s / sqrt(N_eff).

    This is the number to quote a result against: a constraint is only
    meaningful to the precision with which its own mean is known.
    """
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape
    pooled = chains.reshape(n_chains * n_steps, n_params)
    ess = effective_sample_size(chains)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pooled.std(axis=0, ddof=1) / np.sqrt(ess)


def _normalized_acf(x):
    """Normalized autocorrelation of a 1-D array, by FFT.

    The transform is zero-padded to at least 2N so that the circular
    correlation the FFT computes equals the linear one we want.
    """
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
