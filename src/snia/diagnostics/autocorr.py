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


def integrated_time(samples, window_constant=5.0):
    """tau_int = 1 + 2 sum_{l>=1} rho_l, with Sokal's automatic truncation.

    The tail of the sum is dominated by noise -- rho_l at large l is estimated
    from ever fewer effective pairs -- so it must be truncated. Sokal's rule
    takes the smallest window M satisfying M >= c * tau(M), with c = 5 the
    usual choice. Returns +inf for a chain that never moved.
    """
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape

    times = np.empty(n_params)
    for parameter in range(n_params):
        values = chains[:, :, parameter]
        if np.all(values == values.flat[0]):
            times[parameter] = np.inf
            continue
        rho = np.mean(
            [_normalized_acf(values[c]) for c in range(n_chains)], axis=0
        )
        # cumulative[m] = 2 * sum_{l=0..m} rho_l - 1 = 1 + 2 sum_{l=1..m} rho_l
        cumulative = 2.0 * np.cumsum(rho) - 1.0
        times[parameter] = cumulative[_sokal_window(cumulative, window_constant)]
    return times


def effective_sample_size(samples, window_constant=5.0):
    """N_eff = n_chains * n_steps / tau_int, the number of independent draws."""
    chains = as_chains(samples)
    n_chains, n_steps, _ = chains.shape
    total = n_chains * n_steps
    with np.errstate(divide="ignore"):
        return total / integrated_time(chains, window_constant)


def monte_carlo_error(samples, window_constant=5.0):
    """MCSE of the posterior mean, s / sqrt(N_eff).

    This is the number to quote a result against: a constraint is only
    meaningful to the precision with which its own mean is known.
    """
    chains = as_chains(samples)
    n_chains, n_steps, n_params = chains.shape
    pooled = chains.reshape(n_chains * n_steps, n_params)
    ess = effective_sample_size(chains, window_constant)
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


def _sokal_window(cumulative, window_constant):
    """Smallest M with M >= c * tau(M); the last index if none qualifies."""
    inside = np.arange(cumulative.size) < window_constant * cumulative
    if np.all(inside):
        return cumulative.size - 1
    return int(np.argmin(inside))
