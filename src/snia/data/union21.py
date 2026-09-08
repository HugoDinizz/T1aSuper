"""Loader for the Union2.1 Type Ia supernova compilation.

The file format was verified once and is recorded in CLAUDE.md and in
data/download_union21.sh. This module does not re-derive it; it enforces it,
so that a corrupted or mismatched download fails loudly at load time rather
than silently biasing a chain.

The raw files are gitignored. Fetch them with::

    bash data/download_union21.sh
"""

from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError, cholesky

#: Supernovae passing the Union2.1 release cuts.
N_SN = 580

TABLE_FILE = "SCPUnion2.1_mu_vs_z.txt"
COVMAT_SYS_FILE = "SCPUnion2.1_covmat_sys.txt"
COVMAT_NOSYS_FILE = "SCPUnion2.1_covmat_nosys.txt"

#: <repo>/data/raw, reached from <repo>/src/snia/data/union21.py.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

DOWNLOAD_COMMAND = "bash data/download_union21.sh"

#: covmat_sys round-trips through text and is symmetric only to ~1e-14
#: absolute, so symmetry is checked with a tolerance rather than exactly.
SYMMETRY_ATOL = 1e-10

#: Slack for the diagonal-versus-sigma_mu check. The nosys diagonal equals
#: sigma_mu**2 to ~2e-12, which is text round-tripping, not disagreement.
DIAGONAL_ATOL = 1e-9


def load_union21(path=None, systematics=True):
    """Load the Union2.1 compilation.

    Parameters
    ----------
    path : path-like or None
        Directory holding the raw files. Defaults to ``<repo>/data/raw``.
    systematics : bool
        Use the covariance including systematics (the default, and what the
        final results use) or the statistics-only one.

    Returns
    -------
    z : ndarray, shape (580,)
        CMB-frame redshift, spanning 0.015 to 1.414.
    mu : ndarray, shape (580,)
        Observed distance modulus in magnitudes, already standardized for
        stretch, colour and host mass with an arbitrary fiducial M_B. That
        arbitrariness is why the fit must keep a free additive offset.
    cov : ndarray, shape (580, 580)
        Covariance in mag**2, symmetric and positive definite.

    Notes
    -----
    Rows come in supernova-name order and are NOT sorted by redshift. Nothing
    downstream needs them sorted; if you sort anyway, permute all three
    together, ``idx = np.argsort(z)`` and ``cov = cov[np.ix_(idx, idx)]``.
    Sorting z and mu without the covariance corrupts the fit with no error.

    The diagonal of ``cov`` already contains sigma_mu**2 from column 3 of the
    table. Never add it again.
    """
    directory = DEFAULT_DATA_DIR if path is None else Path(path)
    covmat_name = COVMAT_SYS_FILE if systematics else COVMAT_NOSYS_FILE
    table_path = _require_file(directory / TABLE_FILE)
    covmat_path = _require_file(directory / covmat_name)

    # Column 0 is the SN name, a string, so it must be skipped. Column 4,
    # P(low-mass host), is already folded into mu through the delta term
    # recorded in the file header, so it is not needed here.
    z, mu, sigma_mu = np.loadtxt(table_path, usecols=(1, 2, 3), unpack=True)

    # No delimiter argument: the covariance rows carry a trailing tab, which
    # delimiter='\t' would turn into a 581st empty column.
    cov = np.loadtxt(covmat_path)

    if z.shape != (N_SN,):
        raise ValueError(
            f"expected {N_SN} supernovae in {table_path}, got {z.shape[0]}"
        )
    if cov.shape != (N_SN, N_SN):
        raise ValueError(
            f"expected a ({N_SN}, {N_SN}) covariance in {covmat_path}, "
            f"got {cov.shape}"
        )

    asymmetry = np.abs(cov - cov.T).max()
    if asymmetry > SYMMETRY_ATOL:
        raise ValueError(
            f"covariance in {covmat_path} is not symmetric: "
            f"max|C - C.T| = {asymmetry:.3e}"
        )
    # Now that the asymmetry is known to be round-off, remove it, so the
    # returned matrix is exactly symmetric and the Cholesky factor downstream
    # cannot depend on which triangle scipy happens to read.
    cov = 0.5 * (cov + cov.T)

    try:
        cholesky(cov, lower=True)
    except LinAlgError as exc:
        raise ValueError(
            f"covariance in {covmat_path} is not positive definite, so it "
            f"cannot be Cholesky-factored: {exc}"
        ) from exc

    # The only check that ties the two files together: systematics can add
    # variance to the diagonal but never remove it, so a mismatched pair of
    # files, or a shifted column, shows up here.
    deficit = (sigma_mu**2 - np.diag(cov)).max()
    if deficit > DIAGONAL_ATOL:
        raise ValueError(
            f"covariance diagonal in {covmat_path} falls below sigma_mu**2 "
            f"from {table_path} by {deficit:.3e} mag**2; the two files do "
            f"not describe the same supernovae"
        )

    return z, mu, cov


def _require_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Union2.1 data file not found: {path}\n"
            f"The raw data is gitignored. Fetch it with:\n"
            f"    {DOWNLOAD_COMMAND}"
        )
    return path
