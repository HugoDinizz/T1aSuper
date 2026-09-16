"""Loader for the Union2.1 Type Ia supernova compilation."""

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
SYMMETRY_ATOL = 1e-10
DIAGONAL_ATOL = 1e-9


def load_union21(path=None, systematics=True):
    """Load the Union2.1 compilation."""
    directory = DEFAULT_DATA_DIR if path is None else Path(path)
    covmat_name = COVMAT_SYS_FILE if systematics else COVMAT_NOSYS_FILE
    table_path = _require_file(directory / TABLE_FILE)
    covmat_path = _require_file(directory / covmat_name)
    z, mu, sigma_mu = np.loadtxt(table_path, usecols=(1, 2, 3), unpack=True)
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
    cov = 0.5 * (cov + cov.T)

    try:
        cholesky(cov, lower=True)
    except LinAlgError as exc:
        raise ValueError(
            f"covariance in {covmat_path} is not positive definite, so it "
            f"cannot be Cholesky-factored: {exc}"
        ) from exc
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
