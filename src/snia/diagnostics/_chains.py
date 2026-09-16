import numpy as np

def as_chains(samples):
    """Promote samples to the canonical (n_chains, n_steps, n_params) shape."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return samples[np.newaxis, :, np.newaxis]
    if samples.ndim == 2:
        return samples[np.newaxis, :, :]
    if samples.ndim == 3:
        return samples
    raise ValueError(
        f"expected samples with 1, 2 or 3 dimensions, got {samples.ndim}"
    )
