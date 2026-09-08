"""Acceptance tests for the Union2.1 loader and likelihood (milestone 3).

Tests that need the raw files skip cleanly when they are absent, since
data/raw is gitignored. Tests that fabricate their own data always run, so
the validation and the marginalization algebra stay covered on a fresh clone.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from snia.cosmology import FLRW
from snia.data.likelihood import Union21Likelihood
from snia.data.union21 import (
    COVMAT_NOSYS_FILE,
    COVMAT_SYS_FILE,
    DEFAULT_DATA_DIR,
    N_SN,
    TABLE_FILE,
    load_union21,
)

requires_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / TABLE_FILE).is_file(),
    reason="Union2.1 raw data absent; fetch with: bash data/download_union21.sh",
)


@pytest.fixture(scope="module")
def sys_data():
    return load_union21()


@pytest.fixture(scope="module")
def nosys_data():
    return load_union21(systematics=False)


# --------------------------------------------------------------------------
# Fabricated files, mimicking the real format exactly
# --------------------------------------------------------------------------


def _write_table(directory, n=N_SN, sigma=0.1):
    """Write a mu_vs_z file: 5 '#' header lines, then tab-separated rows
    whose first column is a string."""
    directory.mkdir(parents=True, exist_ok=True)
    # The real file carries z to 6 decimals; round so the values written below
    # round-trip exactly and the caller can compare against them.
    z = np.round(np.linspace(0.015, 1.414, n), 6)
    mu = 5.0 * np.log10(z) + 43.0
    lines = [
        "# alpha 0.121851859725",
        "# beta 2.46569277393",
        "# delta -0.0363405630486",
        "# M(h=0.7, statistical only) -19.3182761161",
        "# M(h=0.7, with systematics) -19.3081547178",
    ]
    lines += [
        f"sn{i:04d}\t{z[i]:.6f}\t{mu[i]:.10f}\t{sigma:.10f}\t0.5" for i in range(n)
    ]
    (directory / TABLE_FILE).write_text("\n".join(lines) + "\n")
    return z, mu


def _write_covmat(directory, cov, name=COVMAT_SYS_FILE):
    """Write a covariance the way the SCP does: tab separated, ONE TRAILING
    TAB per line, plus a trailing blank line."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = ["\t".join(f"{v:.10g}" for v in row) + "\t" for row in np.asarray(cov)]
    (directory / name).write_text("\n".join(rows) + "\n\n")


# --------------------------------------------------------------------------
# 1. Shapes and basic sanity  (CLAUDE.md acceptance test 3)
# --------------------------------------------------------------------------


@requires_data
def test_shapes(sys_data):
    z, mu, cov = sys_data
    assert z.shape == (N_SN,)
    assert mu.shape == (N_SN,)
    assert cov.shape == (N_SN, N_SN)


@requires_data
def test_dtypes_are_float64(sys_data):
    for array in sys_data:
        assert array.dtype == np.float64


@requires_data
def test_nothing_is_nan_or_infinite(sys_data):
    for array in sys_data:
        assert np.all(np.isfinite(array))


@requires_data
def test_redshift_range_matches_the_release(sys_data):
    z = sys_data[0]
    assert np.all(z > 0.0)
    assert z.min() == pytest.approx(0.015, abs=1e-6)
    assert z.max() == pytest.approx(1.414, abs=1e-6)


@requires_data
def test_distance_moduli_are_plausible_magnitudes(sys_data):
    """mu spans roughly 33.8 to 45.4 mag. This catches a column shift: the
    sigma_mu column would come back as O(0.1)."""
    mu = sys_data[1]
    assert 33.0 < mu.min() < 35.0
    assert 44.0 < mu.max() < 46.0


@requires_data
def test_rows_are_not_sorted_by_redshift(sys_data):
    """Documented footgun: rows follow SN name order. Anyone who sorts must
    permute the covariance with np.ix_ at the same time."""
    z = sys_data[0]
    assert not np.all(np.diff(z) >= 0.0)


# --------------------------------------------------------------------------
# 2. Covariance properties
# --------------------------------------------------------------------------


@requires_data
def test_covariance_is_exactly_symmetric(sys_data):
    """The file is symmetric only to ~1e-14; the loader removes that
    round-off, so what comes back is symmetric bit for bit."""
    cov = sys_data[2]
    assert np.array_equal(cov, cov.T)


@requires_data
def test_covariance_is_positive_definite(sys_data):
    from scipy.linalg import cho_factor

    cov = sys_data[2]
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    cho_factor(cov)  # must not raise


@requires_data
def test_covariance_spectrum_matches_the_recorded_values(sys_data):
    """CLAUDE.md records eigenvalues in [7.5e-3, 1.02] and condition ~137.
    If a future download differs, this is where it surfaces."""
    eigenvalues = np.linalg.eigvalsh(sys_data[2])
    assert eigenvalues.min() == pytest.approx(7.47e-3, rel=1e-2)
    assert eigenvalues.max() == pytest.approx(1.024, rel=1e-2)
    assert eigenvalues.max() / eigenvalues.min() == pytest.approx(137.0, rel=1e-2)


@requires_data
def test_no_regularization_is_needed(sys_data):
    """The smallest eigenvalue is ~7e-3, nine orders above the level where
    the 137-condition matrix would need a jitter term."""
    assert np.linalg.eigvalsh(sys_data[2]).min() > 1e-3


# --------------------------------------------------------------------------
# 3. The two covariances, and the sigma_mu cross-check
# --------------------------------------------------------------------------


@requires_data
def test_nosys_diagonal_equals_sigma_mu_squared(nosys_data):
    """The statistics-only diagonal IS sigma_mu**2 from column 3, to 1.3e-11.

    This is the sharpest available proof that usecols=(1, 2, 3) picks the
    right columns, and it is the reason sigma_mu**2 must never be added to
    the covariance a second time.
    """
    sigma_mu = np.loadtxt(DEFAULT_DATA_DIR / TABLE_FILE, usecols=(3,))
    diagonal = np.diag(nosys_data[2])
    assert np.allclose(diagonal, sigma_mu**2, rtol=1e-9, atol=0.0)


@requires_data
def test_systematics_only_add_variance(sys_data, nosys_data):
    """Same supernovae, strictly larger diagonal with systematics."""
    assert np.array_equal(sys_data[0], nosys_data[0])
    assert np.array_equal(sys_data[1], nosys_data[1])
    assert np.all(np.diag(sys_data[2]) > np.diag(nosys_data[2]))


@requires_data
def test_systematics_flag_selects_a_different_file(sys_data, nosys_data):
    assert not np.array_equal(sys_data[2], nosys_data[2])


@requires_data
def test_explicit_path_matches_the_default(sys_data):
    z, mu, cov = load_union21(path=DEFAULT_DATA_DIR)
    assert np.array_equal(z, sys_data[0])
    assert np.array_equal(mu, sys_data[1])
    assert np.array_equal(cov, sys_data[2])


# --------------------------------------------------------------------------
# 4. Missing files
# --------------------------------------------------------------------------


def test_missing_table_names_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_union21(path=tmp_path)
    message = str(excinfo.value)
    assert "download_union21.sh" in message
    assert TABLE_FILE in message


def test_missing_covariance_names_the_download_command(tmp_path):
    _write_table(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_union21(path=tmp_path)
    assert "download_union21.sh" in str(excinfo.value)
    assert COVMAT_SYS_FILE in str(excinfo.value)


def test_missing_nosys_covariance_is_reported_separately(tmp_path):
    """systematics=False must look for the other file, not silently reuse."""
    _write_table(tmp_path)
    _write_covmat(tmp_path, np.eye(N_SN) * 0.05, name=COVMAT_SYS_FILE)
    with pytest.raises(FileNotFoundError) as excinfo:
        load_union21(path=tmp_path, systematics=False)
    assert COVMAT_NOSYS_FILE in str(excinfo.value)


# --------------------------------------------------------------------------
# 5. Validation of corrupted inputs
# --------------------------------------------------------------------------


def test_trailing_tabs_and_blank_line_are_tolerated(tmp_path):
    """The documented trap: these rows must NOT parse as 581 columns.

    _write_covmat reproduces the SCP layout exactly, so this test fails if
    anyone adds delimiter='\\t' to the np.loadtxt call.
    """
    z_in, mu_in = _write_table(tmp_path)
    cov_in = np.eye(N_SN) * 0.05
    _write_covmat(tmp_path, cov_in)
    z, mu, cov = load_union21(path=tmp_path)
    assert cov.shape == (N_SN, N_SN)
    assert np.allclose(cov, cov_in)
    assert np.allclose(z, z_in)
    assert np.allclose(mu, mu_in)


def test_asymmetric_covariance_is_rejected(tmp_path):
    _write_table(tmp_path)
    cov = np.eye(N_SN) * 0.05
    cov[0, 1] = 1.0
    _write_covmat(tmp_path, cov)
    with pytest.raises(ValueError, match="not symmetric"):
        load_union21(path=tmp_path)


def test_non_positive_definite_covariance_is_rejected(tmp_path):
    """Symmetric, positive diagonal, but 579 negative eigenvalues."""
    _write_table(tmp_path)
    cov = np.full((N_SN, N_SN), 0.2)
    np.fill_diagonal(cov, 0.05)
    _write_covmat(tmp_path, cov)
    with pytest.raises(ValueError, match="not positive definite"):
        load_union21(path=tmp_path)


def test_wrong_covariance_shape_is_rejected(tmp_path):
    _write_table(tmp_path)
    _write_covmat(tmp_path, np.eye(10) * 0.05)
    with pytest.raises(ValueError, match=r"\(580, 580\) covariance"):
        load_union21(path=tmp_path)


def test_wrong_supernova_count_is_rejected(tmp_path):
    _write_table(tmp_path, n=10)
    _write_covmat(tmp_path, np.eye(N_SN) * 0.05)
    with pytest.raises(ValueError, match="expected 580 supernovae"):
        load_union21(path=tmp_path)


def test_mismatched_table_and_covariance_are_rejected(tmp_path):
    """A covariance whose diagonal sits below sigma_mu**2 cannot belong to
    this table: systematics add variance, they never remove it."""
    _write_table(tmp_path, sigma=0.1)
    _write_covmat(tmp_path, np.eye(N_SN) * 0.001)
    with pytest.raises(ValueError, match="falls below sigma_mu"):
        load_union21(path=tmp_path)


# ==========================================================================
# Union21Likelihood
# ==========================================================================


TOY_OFFSET = 43.2


@pytest.fixture(scope="module")
def toy():
    """A small, exactly-solvable dataset: 12 SNe drawn from a known model.

    mu is the model plus a constant offset and NO noise, so the best fit is
    known analytically: chi2 must vanish at the truth and M must come back as
    TOY_OFFSET. The covariance is dense, so a diagonal-only bug would show.
    """
    rng = np.random.default_rng(20260907)
    n = 12
    z = np.linspace(0.02, 1.2, n)
    mu = 5.0 * np.log10(FLRW(0.3, 0.7).luminosity_distance(z)) + TOY_OFFSET
    scatter = rng.normal(size=(n, n))
    cov = 0.01 * np.eye(n) + 0.002 * (scatter @ scatter.T) / n
    return z, mu, cov


@pytest.fixture(scope="module")
def toy_likelihood(toy):
    return Union21Likelihood(*toy)


@pytest.fixture(scope="module")
def real_likelihood(sys_data):
    return Union21Likelihood(*sys_data)


# --------------------------------------------------------------------------
# 6. The marginalization algebra
# --------------------------------------------------------------------------


def test_matches_brute_force_with_an_explicit_inverse(toy, toy_likelihood):
    """Reference implementation, written the forbidden way on purpose.

    Forming C^-1 is fine in a 12x12 test; the point is that the module's
    Cholesky route reproduces A - B^2/E.

    The tolerance is absolute, not relative, because A - B^2/E cancels two
    numbers of size A ~ 1.8e6 down to a chi2 of ~0.5. The floating-point
    floor is therefore eps*A ~ 4e-10, independent of how small chi2 is, and
    no implementation can do better in this form.
    """
    z, mu, cov = toy
    cov_inv = np.linalg.inv(cov)
    ones = np.ones(z.size)
    theta = (0.4, 0.8)
    delta = mu - 5.0 * np.log10(FLRW(*theta).luminosity_distance(z))
    a = delta @ cov_inv @ delta
    b = ones @ cov_inv @ delta
    e = ones @ cov_inv @ ones
    assert toy_likelihood.E == pytest.approx(e, rel=1e-10)
    assert toy_likelihood.chi2(theta) == pytest.approx(a - b * b / e, abs=1e-8)


def test_E_is_ones_Cinv_ones(toy, toy_likelihood):
    cov_inv = np.linalg.inv(toy[2])
    assert toy_likelihood.E == pytest.approx(np.ones(12) @ cov_inv @ np.ones(12), rel=1e-10)


def test_v_solves_C_v_equals_ones(toy, toy_likelihood):
    """v = C^-1 1 is characterised by C v = 1, checkable without inverting."""
    assert np.allclose(toy[2] @ toy_likelihood.v, 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("theta", [(0.3, 0.7), (0.5, 0.2), (0.1, 0.9), (1.0, 0.0)])
def test_marginalized_chi2_equals_the_minimum_over_the_offset(toy_likelihood, theta):
    """The whole point: A - B^2/E is the minimum of the quadratic in M.

    Because E does not depend on (Om, OL), marginalizing and profiling give
    the same answer up to a constant, so an independent numerical
    minimization over M must land on the analytic value.
    """
    result = minimize_scalar(
        lambda m: toy_likelihood.chi2_at_offset(theta, m),
        bounds=(TOY_OFFSET - 20.0, TOY_OFFSET + 20.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert toy_likelihood.chi2(theta) == pytest.approx(result.fun, abs=1e-9)
    assert toy_likelihood.m_offset(theta) == pytest.approx(result.x, abs=1e-6)


@pytest.mark.parametrize("theta", [(0.3, 0.7), (0.5, 0.2), (1.0, 0.0)])
def test_chi2_at_offset_is_a_parabola_with_curvature_2E(toy_likelihood, theta):
    """chi2(M) = A - 2MB + M^2 E exactly, so its second difference is 2E."""
    f = lambda m: toy_likelihood.chi2_at_offset(theta, m)
    h = 1.0
    second = (f(TOY_OFFSET + h) - 2.0 * f(TOY_OFFSET) + f(TOY_OFFSET - h)) / h**2
    assert second == pytest.approx(2.0 * toy_likelihood.E, rel=1e-8)


def test_chi2_is_invariant_under_a_constant_shift_of_mu(toy):
    """THE defining property of marginalizing M: adding a constant to every
    observed magnitude cannot change the inference on (Om, OL).

    Algebraically A -> A + 2cB + c^2 E and B -> B + cE, and the two shifts
    cancel exactly in A - B^2/E.

    Exactly, that is, in real arithmetic. In float64 the shift inflates A to
    ~2.4e7 for c = 100, so the cancellation floor eps*A rises with it; the
    tolerance below is that floor, not a fudge.
    """
    z, mu, cov = toy
    reference = Union21Likelihood(z, mu, cov).chi2((0.4, 0.8))
    for shift in (2.5, -7.0, 100.0):
        shifted = Union21Likelihood(z, mu + shift, cov).chi2((0.4, 0.8))
        assert shifted == pytest.approx(reference, abs=1e-7)


def test_m_offset_absorbs_the_shift(toy):
    z, mu, cov = toy
    base = Union21Likelihood(z, mu, cov).m_offset((0.4, 0.8))
    shifted = Union21Likelihood(z, mu + 3.0, cov).m_offset((0.4, 0.8))
    assert shifted == pytest.approx(base + 3.0, rel=1e-12)


def test_noiseless_data_gives_zero_chi2_at_the_truth(toy_likelihood):
    """The toy mu is the (0.3, 0.7) model plus a constant, so the residual is
    exactly proportional to 1 and the marginalization removes all of it."""
    assert toy_likelihood.chi2((0.3, 0.7)) == pytest.approx(0.0, abs=1e-6)
    assert toy_likelihood.m_offset((0.3, 0.7)) == pytest.approx(TOY_OFFSET, abs=1e-8)


def test_chi2_is_non_negative_across_the_prior_box(toy_likelihood):
    for om in np.linspace(0.0, 2.0, 15):
        for ol in np.linspace(-1.0, 3.0, 15):
            value = toy_likelihood.chi2((om, ol))
            assert value >= 0.0


# --------------------------------------------------------------------------
# 7. Cost: factor once, one triangular solve per call, never invert
# --------------------------------------------------------------------------


def _count_linalg(monkeypatch):
    from snia.data import likelihood as module

    calls = {"solve_triangular": 0, "cho_solve": 0, "cholesky": 0}
    for name in calls:
        original = getattr(module, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, counted)
    return calls


def test_constructor_factors_exactly_once(monkeypatch, toy):
    calls = _count_linalg(monkeypatch)
    Union21Likelihood(*toy)
    assert calls == {"solve_triangular": 0, "cho_solve": 1, "cholesky": 1}


def test_exactly_one_triangular_solve_per_likelihood_call(monkeypatch, toy):
    """The requirement from CLAUDE.md, asserted rather than assumed.

    A = ||L^-1 Delta||^2 needs only the forward substitution. cho_solve would
    do the back substitution too and hand back C^-1 Delta, which is never
    used; on the real 580x580 problem that costs 274 us instead of 29 us.
    """
    calls = _count_linalg(monkeypatch)
    likelihood = Union21Likelihood(*toy)
    calls.update(dict.fromkeys(calls, 0))

    for _ in range(7):
        likelihood.log_likelihood((0.3, 0.7))

    assert calls == {"solve_triangular": 7, "cho_solve": 0, "cholesky": 0}


def test_covariance_is_never_inverted(monkeypatch, toy):
    """Make every inversion routine explode, then evaluate anyway."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the covariance must never be inverted")

    import scipy.linalg

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    monkeypatch.setattr(scipy.linalg, "inv", forbidden)

    likelihood = Union21Likelihood(*toy)
    assert np.isfinite(likelihood.log_likelihood((0.3, 0.7)))


def test_repeated_calls_are_bitwise_identical(toy_likelihood):
    """No hidden state, no accumulation: required for reproducible chains."""
    first = toy_likelihood.log_likelihood((0.37, 0.61))
    for _ in range(5):
        assert toy_likelihood.log_likelihood((0.37, 0.61)) == first


# --------------------------------------------------------------------------
# 8. Interface and rejection behaviour
# --------------------------------------------------------------------------


def test_log_likelihood_is_minus_half_chi2(toy_likelihood):
    theta = (0.42, 0.66)
    assert toy_likelihood.log_likelihood(theta) == -0.5 * toy_likelihood.chi2(theta)


def test_instance_is_callable_for_the_sampler(toy_likelihood):
    """bayes/ takes any callable returning a log density."""
    theta = (0.42, 0.66)
    assert callable(toy_likelihood)
    assert toy_likelihood(theta) == toy_likelihood.log_likelihood(theta)


def test_no_big_bang_wedge_is_rejected(toy_likelihood):
    assert toy_likelihood.chi2((0.1, 2.5)) == np.inf
    assert toy_likelihood.log_likelihood((0.1, 2.5)) == -np.inf


def test_past_the_antipode_is_rejected(toy_likelihood):
    """The second rejection mode, and it depends on the data's z_max.

    (0.28, 1.68) is past the antipode by z = 1.414 but NOT by the toy's
    z_max = 1.2, so a point that is genuinely rejected here has to be chosen
    for this z_max. E^2 stays positive throughout, which is what makes this
    a test of the D_L <= 0 branch rather than of the no-Big-Bang branch.
    """
    theta = (0.36, 1.8)
    grid = np.linspace(0.0, toy_likelihood.z_max, 2001)
    assert np.all(FLRW(*theta).E_squared(grid) > 0.0)
    assert not FLRW(*theta).is_valid(toy_likelihood.z_max)

    assert toy_likelihood.chi2(theta) == np.inf
    assert toy_likelihood.log_likelihood(theta) == -np.inf


def test_m_offset_is_nan_where_the_model_is_invalid(toy_likelihood):
    assert np.isnan(toy_likelihood.m_offset((0.1, 2.5)))


def test_never_nan_and_never_raises_across_the_prior_box(toy_likelihood):
    """Every point is either a finite log density or exactly -inf."""
    for om in np.linspace(0.0, 2.0, 21):
        for ol in np.linspace(-1.0, 3.0, 21):
            value = toy_likelihood.log_likelihood((om, ol))
            assert not np.isnan(value)
            assert np.isfinite(value) or value == -np.inf


def test_inconsistent_shapes_are_rejected(toy):
    z, mu, cov = toy
    with pytest.raises(ValueError, match="inconsistent shapes"):
        Union21Likelihood(z, mu[:-1], cov)
    with pytest.raises(ValueError, match="inconsistent shapes"):
        Union21Likelihood(z, mu, cov[:-1, :-1])


# --------------------------------------------------------------------------
# 9. The real compilation
# --------------------------------------------------------------------------


@requires_data
def test_real_best_fit_is_near_flat_lcdm(real_likelihood):
    """A coarse grid must land on the published Union2.1 result."""
    grid_m = np.linspace(0.1, 0.6, 26)
    grid_l = np.linspace(0.4, 1.0, 31)
    best = min(
        ((real_likelihood.chi2((om, ol)), om, ol) for om in grid_m for ol in grid_l)
    )
    chi2, om, ol = best
    assert 0.2 < om < 0.4
    assert 0.55 < ol < 0.85
    assert abs(om + ol - 1.0) < 0.1  # consistent with flatness
    assert chi2 / (real_likelihood.n_sn - 3) == pytest.approx(0.945, abs=0.05)


@requires_data
def test_real_zero_point_recovers_h_equal_0p7(real_likelihood):
    """M = 25 + 5 log10(c/(H0 Mpc)) for the released mu, whose fiducial M_B
    was set at h = 0.7. Inverting the fitted M must give H0 back.

    This closes the loop on the dimensionless-distance convention: it would
    fail if c/H0 or the +25 had leaked into the cosmology module.
    """
    m_hat = real_likelihood.m_offset((0.295, 0.705))
    hubble_distance = 10.0 ** ((m_hat - 25.0) / 5.0)  # Mpc
    h0 = 299792.458 / hubble_distance
    assert h0 == pytest.approx(70.0, rel=0.01)


@requires_data
def test_accelerating_expansion_is_strongly_preferred(real_likelihood):
    """Einstein-de Sitter is disfavoured by delta chi2 > 100."""
    best = real_likelihood.chi2((0.29, 0.69))
    assert real_likelihood.chi2((1.0, 0.0)) - best > 100.0


@requires_data
def test_real_marginalization_matches_numerical_minimization(real_likelihood):
    theta = (0.3, 0.7)
    result = minimize_scalar(
        lambda m: real_likelihood.chi2_at_offset(theta, m),
        bounds=(20.0, 60.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert real_likelihood.chi2(theta) == pytest.approx(result.fun, abs=1e-6)
    assert real_likelihood.m_offset(theta) == pytest.approx(result.x, abs=1e-6)


def test_A_splits_orthogonally_into_offset_and_residual(toy_likelihood):
    """A = M_hat^2 E + chi2 exactly, which is why the spec form cancels.

    chi2_at_offset(theta, 0) is A by definition. The identity holds because
    Delta = M_hat 1 + r is a C^-1-orthogonal decomposition: the cross term
    2 M_hat (1^T C^-1 r) vanishes since 1^T C^-1 r = B - (B/E) E = 0. So A is
    a hypotenuse, M_hat^2 E is the long leg, and chi2 is the short one.
    """
    theta = (0.4, 0.8)
    a = toy_likelihood.chi2_at_offset(theta, 0.0)
    m_hat = toy_likelihood.m_offset(theta)
    assert a == pytest.approx(m_hat**2 * toy_likelihood.E + toy_likelihood.chi2(theta),
                              rel=1e-12)


@requires_data
def test_cancellation_is_understood_and_bounded(real_likelihood):
    """Quantify the digit loss instead of hoping it is small.

    chi2 is invariant under a constant shift of mu, so re-centring the data
    to put M_hat at zero gives a reference computed with no cancellation at
    all. The specified form must agree with it to the predicted floor.
    """
    theta = (0.3, 0.7)
    z, mu = real_likelihood.z, real_likelihood.mu
    m_hat = real_likelihood.m_offset(theta)

    amplification = 1.0 + m_hat**2 * real_likelihood.E / real_likelihood.chi2(theta)
    assert amplification == pytest.approx(7.3e3, rel=0.1)

    recentred = Union21Likelihood(z, mu - m_hat, _covariance_of(real_likelihood))
    reference = recentred.chi2(theta)
    assert recentred.m_offset(theta) == pytest.approx(0.0, abs=1e-9)

    relative_error = abs(real_likelihood.chi2(theta) / reference - 1.0)
    assert relative_error < 1e-10
    assert relative_error > 1e-14  # it really is lossier than a plain norm


def _covariance_of(likelihood):
    """Rebuild C from the stored Cholesky factor: C = L L^T."""
    return likelihood.L @ likelihood.L.T
