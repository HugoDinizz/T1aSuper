"""Gelman-Rubin convergence statistics."""

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from ._chains import as_chains


def split_rhat(samples):
    """Split-Rhat = sqrt(V_hat / W) for each parameter.
    W is the mean within-half variance, B the between-half variance, and
    V_hat = (n-1)/n W + B/n the pooled estimate."""
    halves = _split_in_half(as_chains(samples))
    n_halves, n_steps, _ = halves.shape

    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n_steps * halves.mean(axis=1).var(axis=0, ddof=1)
    pooled = (n_steps - 1) / n_steps * within + between / n_steps

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(pooled / within)


def rank_normalized_rhat(samples):
    chains = as_chains(samples)
    median = np.median(chains, axis=(0, 1), keepdims=True)
    bulk = split_rhat(_rank_normalize(chains))
    folded = split_rhat(_rank_normalize(np.abs(chains - median)))
    return np.maximum(bulk, folded)


def _split_in_half(chains):
    """Turn n chains of length N into 2n chains of length N//2."""
    n_steps = chains.shape[1]
    if n_steps < 4:
        raise ValueError(f"need at least 4 steps per chain to split, got {n_steps}")
    half = n_steps // 2
    return np.concatenate([chains[:, :half, :], chains[:, half : 2 * half, :]], axis=0)


def _rank_normalize(chains):
    n_chains, n_steps, n_params = chains.shape
    total = n_chains * n_steps
    out = np.empty_like(chains)
    for parameter in range(n_params):
        ranks = rankdata(chains[:, :, parameter].ravel())
        out[:, :, parameter] = ndtri((ranks - 0.375) / (total - 0.25)).reshape(
            n_chains, n_steps
        )
    return out
