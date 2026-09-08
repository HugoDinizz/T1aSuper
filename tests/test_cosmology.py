"""Acceptance tests for the cosmology core (milestone 1).

Tolerances here are measured, not guessed. The quadrature is a cumulative
trapezoid on N_GRID points; Euler-Maclaurin makes its error a boundary term,
so the RELATIVE error is uniform in z and equal to ~(h^2/12)|f''(0)| with
f = 1/E. That predicts 1.2e-8 for LCDM and 1.6e-7 for Einstein-de Sitter
(f''(0) = 3.75 there), both of which are what the tests below actually see.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from snia.cosmology import FLRW, N_GRID, SERIES_THRESHOLD

EPS = np.finfo(float).eps

#: Redshifts spanning the Union2.1 range, including its low-z anchor.
Z = np.array([0.015, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.414])

#: Well-behaved models used for accuracy checks.
MODELS = [(0.3, 0.7), (1.0, 0.0), (0.3, 0.0), (0.5, 1.2), (0.0, 1.0), (0.2, 0.9)]

#: The prior box of the exercise: 0 <= omega_m <= 2, -1 <= omega_lambda <= 3.
PRIOR_BOX = [
    (om, ol)
    for om in np.linspace(0.0, 2.0, 21)
    for ol in np.linspace(-1.0, 3.0, 21)
]


# --------------------------------------------------------------------------
# 1. Expansion rate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("om, ol", MODELS + [(0.03, 3.0), (2.0, 3.0), (0.0, 0.0)])
def test_E_at_zero_is_unity(om, ol):
    """E^2(0) = Om + Ok + OL = 1 by construction, up to float64 rounding.

    The sum is exact in exact arithmetic, so the only error is the rounding of
    a three-term float64 sum. Scanning the prior box, the worst case is
    4.4e-16 = 2 eps; it is not zero in general, which is why this is a
    tolerance test and not an equality test.
    """
    assert abs(FLRW(om, ol).E(0.0) - 1.0) <= 4 * EPS


def test_E_at_zero_is_bitwise_exact_for_lcdm():
    """For (0.3, 0.7) the rounding happens to cancel: 0.3 + 0.0 + 0.7 == 1.0."""
    assert FLRW(0.3, 0.7).E(0.0) == 1.0


def test_E_squared_hand_computed():
    """E^2(1) = 0.3*2^3 + (-0.05)*2^2 + 0.75 = 2.4 - 0.2 + 0.75 = 2.95."""
    c = FLRW(0.3, 0.75)
    assert c.omega_k == pytest.approx(-0.05, abs=4 * EPS)
    assert c.E_squared(1.0) == pytest.approx(2.95, rel=1e-15)


def test_E_is_square_root_of_E_squared():
    c = FLRW(0.3, 0.7)
    assert np.allclose(c.E(Z) ** 2, c.E_squared(Z), rtol=1e-15, atol=0.0)


def test_E_is_zero_not_nan_where_E_squared_is_negative():
    """The no-Big-Bang wedge must degrade to 0 (hence 1/E = +inf), never NaN."""
    c = FLRW(0.1, 2.5)
    e2 = c.E_squared(Z)
    assert np.any(e2 < 0.0)
    e = c.E(Z)
    assert not np.any(np.isnan(e))
    assert np.all(e[e2 < 0.0] == 0.0)


# --------------------------------------------------------------------------
# 2. Comoving distance: quadrature accuracy
# --------------------------------------------------------------------------


def _quad_comoving(c, z):
    value, _ = quad(
        lambda zz: 1.0 / np.sqrt(c.E_squared(zz)), 0.0, z, epsabs=1e-13, epsrel=1e-13
    )
    return value


@pytest.mark.parametrize("om, ol", MODELS)
def test_comoving_distance_matches_adaptive_quadrature(om, ol):
    """Independent check against scipy's adaptive quadrature."""
    c = FLRW(om, ol)
    got = c.comoving_distance(Z)
    want = np.array([_quad_comoving(c, z) for z in Z])
    assert np.allclose(got, want, rtol=5e-7, atol=0.0)


def test_quadrature_error_is_negligible_in_magnitudes():
    """The only figure of merit that matters: error on mu, against 0.15 mag data."""
    worst = 0.0
    for om, ol in MODELS:
        c = FLRW(om, ol)
        want = np.array([_quad_comoving(c, z) for z in Z])
        # mu = 5 log10(d_L), and d_L is proportional to I at fixed z, so a
        # relative error eps on I is (5/ln10)*eps magnitudes.
        rel = np.abs(c.comoving_distance(Z) / want - 1.0).max()
        worst = max(worst, 5.0 / np.log(10.0) * rel)
    assert worst < 1e-5


def test_relative_accuracy_is_uniform_down_to_the_lowest_redshift():
    """Euler-Maclaurin claim: relative error does not blow up as z -> 0.

    This is the reason a uniform grid in z is enough and a log grid buys
    nothing, even though the lowest Union2.1 supernova sits ~21 cells from
    the origin.
    """
    c = FLRW(0.3, 0.7)
    got = c.comoving_distance(Z)
    want = np.array([_quad_comoving(c, z) for z in Z])
    rel = np.abs(got / want - 1.0)
    assert rel[0] < 5.0 * rel[-1]


def test_comoving_distance_vanishes_at_zero():
    assert FLRW(0.3, 0.7).comoving_distance(0.0) == 0.0
    assert np.all(FLRW(0.3, 0.7).comoving_distance(np.zeros(4)) == 0.0)


def test_comoving_distance_is_monotonic():
    z = np.linspace(0.0, 1.414, 200)
    assert np.all(np.diff(FLRW(0.3, 0.7).comoving_distance(z)) > 0.0)


# --------------------------------------------------------------------------
# 3. Analytic benchmarks: exact solutions, one per curvature branch
# --------------------------------------------------------------------------


def test_de_sitter_is_exact():
    """Om=0, OL=1: E == 1, so I = z and d_L = z(1+z).

    The trapezoid rule is exact for a constant integrand, so this isolates the
    interpolation and the flat (x = 0) branch of S_k at machine precision.
    """
    got = FLRW(0.0, 1.0).luminosity_distance(Z)
    assert np.allclose(got, Z * (1.0 + Z), rtol=1e-13, atol=0.0)


def test_milne_matches_closed_form():
    """Om=OL=0, so Ok=1: E = 1+z, I = ln(1+z), and the sinh branch gives

        d_L = (1+z) sinh(ln(1+z)) = ((1+z)^2 - 1)/2 = z + z^2/2 exactly.

    This is a genuine test of the Ok > 0 branch against a closed form.
    """
    got = FLRW(0.0, 0.0).luminosity_distance(Z)
    assert np.allclose(got, Z + Z**2 / 2.0, rtol=5e-7, atol=0.0)


def test_einstein_de_sitter_matches_closed_form():
    """Om=1, OL=0: E = (1+z)^{3/2} and d_L = 2(1+z)(1 - 1/sqrt(1+z))."""
    got = FLRW(1.0, 0.0).luminosity_distance(Z)
    want = 2.0 * (1.0 + Z) * (1.0 - 1.0 / np.sqrt(1.0 + Z))
    assert np.allclose(got, want, rtol=5e-7, atol=0.0)


def test_closed_model_matches_sin_branch():
    """Direct check of the Ok < 0 branch against sin(sqrt|Ok| I)/sqrt|Ok|."""
    c = FLRW(0.3, 1.2)
    assert c.omega_k < 0.0
    i_z = c.comoving_distance(Z)
    want = np.sin(np.sqrt(-c.omega_k) * i_z) / np.sqrt(-c.omega_k)
    assert np.allclose(c.transverse_comoving_distance(Z), want, rtol=1e-14, atol=0.0)


# --------------------------------------------------------------------------
# 4. Distance relations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("om, ol", MODELS + [(0.3, 1.2), (0.9, 0.1)])
def test_etherington_distance_duality(om, ol):
    """D_L = (1+z)^2 D_A holds for every parameter value, curved or not."""
    c = FLRW(om, ol)
    assert np.allclose(
        c.luminosity_distance(Z),
        (1.0 + Z) ** 2 * c.angular_diameter_distance(Z),
        rtol=1e-14,
        atol=0.0,
    )


def test_transverse_equals_comoving_when_flat_bitwise():
    """Ok is exactly 0.0 for (0.3, 0.7), so x = 0, the series is exactly 1,
    and D_M must equal I bit for bit -- not merely to a tolerance."""
    c = FLRW(0.3, 0.7)
    assert c.omega_k == 0.0
    assert np.array_equal(c.transverse_comoving_distance(Z), c.comoving_distance(Z))


def test_distance_modulus_definition():
    c = FLRW(0.3, 0.7)
    assert np.allclose(
        c.distance_modulus(Z), 5.0 * np.log10(c.luminosity_distance(Z)), atol=1e-14
    )


def test_distance_modulus_offset_is_a_pure_constant():
    """m_offset shifts mu uniformly in z. This z-independence is exactly what
    makes the analytic marginalization over M valid."""
    c = FLRW(0.3, 0.7)
    shift = c.distance_modulus(Z, 43.1) - c.distance_modulus(Z, 0.0)
    assert np.allclose(shift, 43.1, rtol=0.0, atol=1e-12)


def test_distance_modulus_carries_no_hidden_zero_point():
    """Guard against someone re-introducing c/H0 or the +25: with d_L
    dimensionless, mu at m_offset=0 is O(1), not O(40)."""
    assert abs(FLRW(0.3, 0.7).distance_modulus(1.0)) < 5.0


# --------------------------------------------------------------------------
# 5. The S_k branch switch
# --------------------------------------------------------------------------


def _sinhc_series(x):
    """Reference: sinh(u)/u with u^2 = x, four terms. Valid for either sign."""
    return 1.0 + x / 6.0 + x**2 / 120.0 + x**3 / 5040.0


@pytest.mark.parametrize(
    "ok", [1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 1e-4, -1e-8, -1e-7, -1e-6, -3e-6, -1e-5, -1e-4]
)
def test_sk_factor_matches_series_across_the_threshold(ok):
    """D_M/I must equal 1 + x/6 + x^2/120 + ... on BOTH sides of the switch.

    The omega_k values here put |x| = |Ok| I^2 on both sides of
    SERIES_THRESHOLD, so this exercises the truncated series and the closed
    form and demands they agree with the same reference.
    """
    c = FLRW(0.3, 1.0 - 0.3 - ok)
    z = np.array([0.5, 1.0, 1.414])
    i_z = c.comoving_distance(z)
    factor = c.transverse_comoving_distance(z) / i_z
    x = c.omega_k * i_z**2
    assert np.allclose(factor, _sinhc_series(x), rtol=5e-15, atol=0.0)


@pytest.mark.parametrize("z", [0.5, 1.0, 1.414])
def test_sk_is_smooth_through_omega_k_zero(z):
    """Sweep omega_k through 0 and require smoothness to machine precision.

    The sweep spans +-3e-6, which straddles both switch points (at
    |Ok| = 1e-6/I^2) and omega_k = 0 itself. Testing the geometric factor
    D_M/I rather than D_M removes the quadrature's last-ulp jitter, leaving
    only the branch switch. Measured second differences are ~5e-16 (3 ulp);
    a one-term series would jump by x/6 ~ 1.7e-7, eight orders larger.
    """
    omega_k = np.linspace(-3e-6, 3e-6, 201)
    factor = []
    for ok in omega_k:
        c = FLRW(0.3, 1.0 - 0.3 - ok)
        factor.append(c.transverse_comoving_distance(z) / c.comoving_distance(z))
    factor = np.array(factor)
    assert not np.any(np.isnan(factor))
    assert np.abs(np.diff(factor, n=2)).max() < 1e-14


def test_sk_at_exactly_zero_curvature_does_not_divide_by_zero():
    """The closed form is 0/0 at Ok = 0; the series branch must catch it."""
    c = FLRW(0.3, 0.7)
    assert c.omega_k == 0.0
    out = c.transverse_comoving_distance(Z)
    assert np.all(np.isfinite(out))


def test_series_threshold_is_where_truncation_is_below_machine_epsilon():
    """The dropped x^3/5040 term at the switch must be far below eps, or the
    two branches would not agree to machine precision."""
    assert SERIES_THRESHOLD**3 / 5040.0 < 1e-20


# --------------------------------------------------------------------------
# 6. Low-redshift limits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("om, ol", [(0.3, 0.7), (1.0, 0.0), (0.28, 1.68), (0.0, 0.0)])
def test_low_z_hubble_law(om, ol):
    """d_L -> z as z -> 0, for every model."""
    z = np.array([1e-6, 1e-5, 1e-4])
    d_l = FLRW(om, ol).luminosity_distance(z)
    assert np.all(np.abs(d_l / z - 1.0) < 2.0 * z)


@pytest.mark.parametrize("om, ol", [(0.3, 0.7), (1.0, 0.0), (0.28, 1.68), (0.5, 0.1)])
def test_low_z_second_order_expansion(om, ol):
    """d_L = z + (1-q0)/2 z^2 + O(z^3) with q0 = Om/2 - OL.

    Derivation: E^2 = 1 + 2(1+q0) z + O(z^2), so 1/E = 1 - (1+q0) z + O(z^2)
    and I = z - (1+q0) z^2/2. Curvature enters D_M only at O(I^3), so
    d_L = (1+z) I = z + [1 - (1+q0)/2] z^2 = z + (1-q0)/2 z^2.

    Checking the coefficient at one z would not distinguish a correct scheme
    from one with the wrong convergence order, so this also asserts that the
    residual falls by a factor 10 per decade in z, i.e. that the neglected
    term really is O(z^3).
    """
    c = FLRW(om, ol)
    target = (1.0 - (om / 2.0 - ol)) / 2.0
    err = [abs((c.luminosity_distance(z) - z) / z**2 - target) for z in (1e-2, 1e-3, 1e-4)]
    assert err[-1] < 1e-4
    assert 9.0 < err[0] / err[1] < 11.0
    assert 9.0 < err[1] / err[2] < 11.0


def test_milne_has_no_cubic_term():
    """Sanity check on the test above: Milne is d_L = z + z^2/2 EXACTLY, so it
    is the one model whose residual cannot be used to measure the order."""
    c = FLRW(0.0, 0.0)
    for z in (1e-2, 1e-3):
        assert abs((c.luminosity_distance(z) - z) / z**2 - 0.5) < 1e-8


# --------------------------------------------------------------------------
# 7. Validity gate
# --------------------------------------------------------------------------


def test_is_valid_true_for_lcdm():
    assert FLRW(0.3, 0.7).is_valid(1.414) is True


def test_is_valid_false_in_the_no_big_bang_wedge():
    """(0.1, 2.5) has Ok = -1.6 and E^2 < 0 before z = 0.4: the model has a
    turning point, so there is no continuous past light cone to integrate."""
    c = FLRW(0.1, 2.5)
    assert np.any(c.E_squared(np.linspace(0.0, 1.414, N_GRID)) <= 0.0)
    assert c.is_valid(1.414) is False


def test_is_valid_false_past_the_antipode():
    """(0.28, 1.68) is the other rejection mode, and it is NOT dead code.

    E^2 stays positive (min ~0.008), but the model loiters near that turning
    point, so I grows to 4.75 and sqrt|Ok| I = 4.65 > pi: the light cone has
    passed the antipode of the 3-sphere and D_M has come back through zero
    (at z ~ 1.25) to negative values.
    """
    c = FLRW(0.28, 1.68)
    grid = np.linspace(0.0, 1.414, N_GRID)
    assert np.all(c.E_squared(grid) > 0.0)
    assert np.any(c.transverse_comoving_distance(grid) <= 0.0)
    assert c.is_valid(1.414) is False


def test_everything_is_finite_and_positive_wherever_is_valid():
    """The contract the likelihood relies on: is_valid True implies usable."""
    for om, ol in PRIOR_BOX:
        c = FLRW(om, ol)
        if c.is_valid(Z.max()):
            assert np.all(np.isfinite(c.distance_modulus(Z, 43.0)))
            assert np.all(c.luminosity_distance(Z) > 0.0)


# --------------------------------------------------------------------------
# 8. Interface contract
# --------------------------------------------------------------------------


METHODS = [
    "E",
    "E_squared",
    "comoving_distance",
    "transverse_comoving_distance",
    "angular_diameter_distance",
    "luminosity_distance",
    "distance_modulus",
    "lookback_time",
    "age",
    "comoving_volume_element",
    "deceleration_parameter",
]


@pytest.mark.parametrize("name", METHODS)
def test_shape_is_preserved(name):
    method = getattr(FLRW(0.3, 0.7), name)
    assert np.shape(method(0.5)) == ()
    assert np.shape(method(np.array([0.1, 0.5, 1.0]))) == (3,)
    assert np.shape(method(np.array([[0.1, 0.5], [0.7, 0.9]]))) == (2, 2)


@pytest.mark.parametrize("name", ["E", "E_squared"])
def test_pointwise_methods_are_bitwise_identical_scalar_or_array(name):
    """E and E^2 involve no quadrature, so there is nothing to differ."""
    method = getattr(FLRW(0.3, 1.2), name)
    assert np.array_equal([method(z) for z in Z], method(Z))


@pytest.mark.parametrize("name", METHODS[2:])
def test_distances_agree_scalar_or_array_to_quadrature_accuracy(name):
    """Distances agree to ~1e-8, NOT bit for bit, and that is by design.

    The grid spans [0, max(z)] of the call, so asking for one redshift alone
    integrates on a finer grid than asking for it inside an array reaching to
    z = 1.414. Both answers are correct to the quoted accuracy. What matters
    for a reproducible chain is determinism -- the likelihood passes the same
    580 redshifts every call, so it always gets the same grid -- not
    invariance under regrouping the inputs.
    """
    method = getattr(FLRW(0.3, 1.2), name)
    assert np.allclose([method(z) for z in Z], method(Z), rtol=1e-6, atol=1e-12)


def test_no_nan_and_no_warnings_anywhere_in_the_prior_box():
    """Nothing raises, nothing warns, nothing returns NaN on 0 <= Om <= 2,
    -1 <= OL <= 3 -- including the unphysical parts. Invalid points return
    +inf, which the likelihood turns into -inf via is_valid.

    Turning warnings into errors is deliberate: it verifies that every
    divide-by-zero and 0/0 in the module is inside an explicit np.errstate,
    rather than merely being silent by default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for om, ol in PRIOR_BOX:
            c = FLRW(om, ol)
            outputs = [
                c.E(Z),
                c.E_squared(Z),
                c.comoving_distance(Z),
                c.transverse_comoving_distance(Z),
                c.angular_diameter_distance(Z),
                c.luminosity_distance(Z),
                c.distance_modulus(Z, 43.0),
                c.lookback_time(Z),
                c.age(Z),
                c.comoving_volume_element(Z),
                c.deceleration_parameter(Z),
            ]
            c.is_valid(Z.max())
            for out in outputs:
                assert not np.any(np.isnan(out)), f"NaN at ({om}, {ol})"


def test_omega_k_is_derived_not_stored():
    c = FLRW(0.3, 0.6)
    assert c.omega_k == pytest.approx(0.1, abs=4 * EPS)
    c.omega_lambda = 0.7
    assert c.omega_k == pytest.approx(0.0, abs=4 * EPS)


# --------------------------------------------------------------------------
# 9. Times, volume, and the deceleration parameter
# --------------------------------------------------------------------------


def _quad_age(om, ol, z=0.0):
    """H0 t(z) = int_0^a da'/sqrt(Om/a' + Ok + OL a'^2), integrated adaptively."""
    ok = 1.0 - om - ol
    value, _ = quad(
        lambda a: 1.0 / np.sqrt(om / a + ok + ol * a * a),
        0.0,
        1.0 / (1.0 + z),
        epsabs=1e-13,
        epsrel=1e-13,
    )
    return value


def test_einstein_de_sitter_age_is_two_thirds():
    """The textbook result H0 t0 = 2/3 for a flat matter-only universe."""
    assert FLRW(1.0, 0.0).age(0.0) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_milne_age_is_one_hubble_time():
    """Empty and coasting: a = t, so H0 t0 = 1 exactly."""
    assert FLRW(0.0, 0.0).age(0.0) == pytest.approx(1.0, rel=1e-9)


def test_de_sitter_age_diverges():
    """No Big Bang: the integrand goes as 1/x and the age is infinite."""
    assert FLRW(0.0, 1.0).age(0.0) == np.inf


def test_lcdm_age_is_13_5_gyr():
    """H0 t0 = 0.9641 for (0.3, 0.7), i.e. 13.47 Gyr at h = 0.7.

    The conversion is 1/H0 = 9.778/h Gyr, and this is the standard
    cross-check that the whole E(z) normalization is right.
    """
    hubble_time_gyr = 9.778 / 0.7
    assert FLRW(0.3, 0.7).age(0.0) * hubble_time_gyr == pytest.approx(13.47, abs=0.02)


@pytest.mark.parametrize("om, ol", [(0.3, 0.7), (1.0, 0.0), (0.5, 0.1), (0.2, 0.8)])
@pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
def test_age_matches_adaptive_quadrature(om, ol, z):
    assert FLRW(om, ol).age(z) == pytest.approx(_quad_age(om, ol, z), rel=5e-7)


def test_einstein_de_sitter_age_and_lookback_closed_forms(): 
    """EdS: H0 t(z) = (2/3)(1+z)^-3/2 and H0 t_L = (2/3)[1 - (1+z)^-3/2]."""
    c = FLRW(1.0, 0.0)
    z = np.array([0.5, 1.0, 2.0])
    assert np.allclose(c.age(z), 2.0 / 3.0 * (1.0 + z) ** -1.5, rtol=5e-7)
    assert np.allclose(
        c.lookback_time(z), 2.0 / 3.0 * (1.0 - (1.0 + z) ** -1.5), rtol=5e-7
    )


@pytest.mark.parametrize("om, ol", [(0.3, 0.7), (1.0, 0.0), (0.2, 0.8)])
def test_lookback_time_equals_age_today_minus_age_then(om, ol):
    """t_L(z) = t(0) - t(z), and the two sides are computed by completely
    different integrals: one over z, one over x = sqrt(a). Agreement is a
    strong check that both substitutions are right."""
    c = FLRW(om, ol)
    for z in (0.1, 1.0, 1.414):
        assert c.lookback_time(z) == pytest.approx(c.age(0.0) - c.age(z), abs=2e-7)


def test_lookback_time_vanishes_at_zero_and_increases():
    c = FLRW(0.3, 0.7)
    assert c.lookback_time(0.0) == 0.0
    z = np.linspace(0.0, 1.414, 50)
    assert np.all(np.diff(c.lookback_time(z)) > 0.0)


def test_comoving_volume_element_is_DM_squared_over_E():
    c = FLRW(0.3, 0.7)
    expected = c.transverse_comoving_distance(Z) ** 2 / c.E(Z)
    assert np.allclose(c.comoving_volume_element(Z), expected, rtol=1e-14)
    assert c.comoving_volume_element(0.0) == 0.0


@pytest.mark.parametrize("om, ol", [(0.3, 0.7), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
def test_q0_property_matches_q_at_zero(om, ol):
    c = FLRW(om, ol)
    assert c.q0 == pytest.approx(om / 2.0 - ol, abs=4 * EPS)
    assert c.deceleration_parameter(0.0) == pytest.approx(c.q0, rel=1e-14)


def test_deceleration_changes_sign_at_the_onset_of_acceleration():
    """q = 0 where Om(1+z)^3/2 = OL, i.e. z = (2 OL/Om)^(1/3) - 1 = 0.671
    for (0.3, 0.7). This is the deceleration-acceleration transition."""
    c = FLRW(0.3, 0.7)
    z_transition = (2.0 * 0.7 / 0.3) ** (1.0 / 3.0) - 1.0
    assert z_transition == pytest.approx(0.6711, abs=1e-3)
    assert c.deceleration_parameter(z_transition) == pytest.approx(0.0, abs=1e-12)
    assert c.deceleration_parameter(z_transition - 0.1) < 0.0
    assert c.deceleration_parameter(z_transition + 0.1) > 0.0


def test_einstein_de_sitter_q_is_one_half_at_every_redshift():
    """Matter-only: q = 1/2 independent of z."""
    assert np.allclose(FLRW(1.0, 0.0).deceleration_parameter(Z), 0.5, rtol=1e-14)


def test_age_is_shorter_at_higher_redshift():
    c = FLRW(0.3, 0.7)
    ages = c.age(np.array([0.0, 0.5, 1.0, 2.0, 5.0]))
    assert np.all(np.diff(ages) < 0.0)


def test_is_valid_accepts_a_degenerate_zero_range():
    """is_valid(0) must not reject a good model: D_M(0) = 0 is not a failure."""
    assert FLRW(0.3, 0.7).is_valid(0.0) is True


def test_repr_shows_the_derived_quantities():
    text = repr(FLRW(0.3, 0.7))
    assert "omega_m=0.3" in text and "omega_lambda=0.7" in text
    assert "omega_k" in text and "q0" in text
