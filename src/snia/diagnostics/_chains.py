"""Shared input handling for the diagnostics.

Every public function in this package takes plain arrays, never sampler
objects, so the diagnostics can be tested against processes whose correlation
structure is known analytically.
"""

import numpy as np


def as_chains(samples):
    """Promote samples to the canonical (n_chains, n_steps, n_params) shape.

    A 1-D array is read as one chain of one parameter, and a 2-D array as one
    chain of several parameters -- which is exactly what
    ``MetropolisHastings.sample`` returns. ``run_chains`` already returns 3-D.
    """
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
