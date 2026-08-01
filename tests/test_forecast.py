"""Forecasting: the MA(infinity) weights and the three variances.

Two layers are pinned here. The CORE — psi weights, the error-variance
recursion, the integration that undoes the differencing — is checked against
textbook cases whose answer is known in advance. The LEVEL layer is then checked
end to end against the C's own forecast table, for both series of the canonical
case.

The C's `f1` was read out of the binary directly (a temporary `fprintf` in
`transfer_forecast`) to establish that the forecast of `w` is identical to the
port's, which is what localised the port's last defect to the level layer rather
than to the model.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit  # noqa: E402
from drtran.forecast import (error_variance, forecast,  # noqa: E402
                             forecast_mean, integrated_weights, psi_weights)

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CASES, "ES_CPI_m10.pre")),
    reason="the canonical .pre files from the C repo are missing")


# ── the weights ──────────────────────────────────────────────────────────────
def test_psi_of_a_pure_ar1_is_the_power_of_phi():
    """For a scalar AR(1), psi_l = phi^l. The textbook case, which is the point:
    if this is wrong nothing downstream can be right."""
    phi = np.array([[[0.6]]])
    psi = psi_weights(phi, np.zeros((0, 1, 1)), 5)
    assert psi[0][0, 0] == 1.0
    for l in range(1, 6):
        assert psi[l][0, 0] == pytest.approx(0.6 ** l)


def test_psi_of_a_pure_ma_stops():
    """An MA(q) has psi_l = -Theta_l up to q and exactly zero afterwards."""
    theta = np.array([[[0.4]], [[-0.2]]])
    psi = psi_weights(np.zeros((0, 1, 1)), theta, 5)
    assert psi[1][0, 0] == pytest.approx(-0.4)
    assert psi[2][0, 0] == pytest.approx(0.2)
    assert np.allclose(psi[3:], 0.0)


def test_the_variance_accumulates_one_term_per_horizon():
    """V_l = sum_(j<l) psi_j Sigma psi_j': it grows with the horizon, and for
    white noise it is exactly l*Sigma."""
    psi = np.array([np.eye(2)] * 4)
    var = error_variance(psi, np.eye(2), 3)
    for l in range(1, 4):
        assert np.allclose(var[l], l * np.eye(2))


def test_integrating_undoes_the_differencing():
    """With d=1 the level weights are the cumulative sum of psi: that is what
    dividing by (1-B) means, and why the level's variance has no bound."""
    # WHITE NOISE in differences is psi = [1, 0, 0, ...], not all ones: the
    # random walk. Dividing by (1-B) gives 1, 1, 1, ... and the level variance
    # grows linearly, which is the textbook signature.
    psi = np.zeros((5, 1, 1)); psi[0, 0, 0] = 1.0
    psis = integrated_weights(psi, d=1, D=0, s=1)
    assert [psis[l][0, 0] for l in range(5)] == pytest.approx([1.0] * 5)
    var = error_variance(psis, np.eye(1), 4)
    for l in range(1, 5):
        assert var[l][0, 0] == pytest.approx(float(l))

    # and if psi is all ones, psi/(1-B) is the running sum: 1, 2, 3, ...
    psis2 = integrated_weights(np.array([np.eye(1)] * 5), d=1, D=0, s=1)
    assert [psis2[l][0, 0] for l in range(5)] == pytest.approx([1., 2., 3., 4., 5.])


def test_the_seasonal_operator_only_subtracts_from_lag_s():
    psi = np.zeros((6, 1, 1)); psi[0, 0, 0] = 1.0
    psis = integrated_weights(psi, d=0, D=1, s=4)
    # (1-B^4)^-1 of [1,0,0,...] gives 1,0,0,0,1,0: a one every s
    assert [psis[l][0, 0] for l in range(6)] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ── the point forecast ───────────────────────────────────────────────────────
def test_the_forecast_of_w_decays_to_mu():
    """`w` carries no deterministic part — the cast removed it — so the forecast
    of the stochastic component can only decay to its mean. The C's LEVEL
    forecast does not decay because it adds the harmonics back; that layer is
    not ported yet, and this test is what makes the difference explicit."""
    phi = np.array([[[0.5]]])
    mu = np.array([2.0])
    w = np.array([[3.0], [2.5], [2.2]])
    a = np.zeros((3, 1))
    f = forecast_mean(phi, np.zeros((0, 1, 1)), mu, w, a, 8)
    assert f[0, 0] == pytest.approx(2.0 + 0.5 * (2.2 - 2.0))
    assert abs(f[-1, 0] - 2.0) < 1e-3
    assert np.all(np.abs(np.diff(f[:, 0])) <= 1e-9 + np.abs(np.diff(f[:, 0]))[0])


def test_the_ma_part_dies_after_q():
    """Beyond q the residuals no longer enter: the forecast is pure AR."""
    theta = np.array([[[0.5]]])
    a = np.array([[1.0], [1.0], [2.0]])
    f = forecast_mean(np.zeros((0, 1, 1)), theta, np.zeros(1),
                      np.zeros((3, 1)), a, 3)
    assert f[0, 0] == pytest.approx(-0.5 * 2.0)     # l=1 usa a_n
    assert f[1, 0] == pytest.approx(0.0)
    assert f[2, 0] == pytest.approx(0.0)


# ── against the binary: the structure ────────────────────────────────────────
@pytest.fixture(scope="module")
def fitted():
    cs = build_cast_spec([drtran.load_pre(os.path.join(CASES, "ES_CPI_m10.pre")),
                          drtran.load_pre(os.path.join(CASES, "WTI_ar1.pre"))],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return fit(cs, embed=True)


def test_the_ratios_of_the_standard_errors_match_the_C(fitted):
    """The C reports, for ES_CPI from 12/2019, level s.e. of
    0.24 0.44 0.60 0.74 0.85 0.95 and variation s.e. of 0.24 0.28 0.28 ...

    Those are in index units and this port still works in the modelled scale, so
    the levels cannot be compared directly. The RATIOS can, and they depend on
    exactly what lives here: psi, the integration and Sigma.
    """
    fc = forecast(fitted, L=6)
    level = fc.se("level", 0)
    variacion = fc.se("diff", 0)

    c_nivel = np.array([0.24, 0.44, 0.60, 0.74, 0.85, 0.95])
    c_var = np.array([0.24, 0.28, 0.28, 0.28, 0.28, 0.28])

    # the C publishes two decimals, so the honest comparison is RELATIVE:
    # 0.24 already carries 2% of rounding uncertainty
    assert level / level[0] == pytest.approx(c_nivel / c_nivel[0], rel=0.02)
    assert variacion / variacion[0] == pytest.approx(c_var / c_var[0], rel=0.03)


def test_the_level_variance_grows_and_the_variation_settles(fitted):
    """The signature of a differenced model: integrating gives an unbounded
    level variance, while the variation's converges."""
    fc = forecast(fitted, L=12)
    level, variacion = fc.se("level", 0), fc.se("diff", 0)
    assert np.all(np.diff(level) > 0)
    assert abs(variacion[-1] - variacion[-2]) < 1e-3


def test_the_scale_is_applied_not_left_in_Q(fitted):
    """The cast returns Q, not Sigma — the likelihood is concentrated and sigma2
    comes out separately. Forgetting it leaves the right shape with the wrong
    magnitude, which is the classic misreading of Q as Sigma."""
    fc = forecast(fitted, L=3)
    assert 0.05 < fc.se("level", 0)[0] < 1.0, "sigma's order of magnitude, not Q's"


# ── back to the level ────────────────────────────────────────────────────────
def test_a_series_with_no_incoming_transfer_matches_the_C(fitted):
    """WTI receives nothing, so its level forecast must be the univariate one —
    and the C reports 60.76, 61.02, 61.10 from 12/2019.

    This exercises the integration against delta(B) and the inverse Box-Cox, but
    NOT the deterministic effect: WTI has no interventions, so `xi` is zero here
    however it is built. That is precisely why this test passed while the port
    was still building `xi` from the seeds — it takes `test_the_output_level_
    matches_the_C` to see that.
    """
    from drtran.forecast import to_level

    fc = forecast(fitted, L=6)
    level = to_level(fc, fitted.cast_spec, series=1)
    assert level[:3] == pytest.approx([60.76, 61.02, 61.10], abs=0.01)
    assert level[-1] == pytest.approx(61.14, abs=0.01)


def test_the_output_level_matches_the_C(fitted):
    """ES_CPI receives the transfer, and this is the end-to-end check: the C
    reports 82.01 82.02 82.38 83.17 83.33 83.44 from 12/2019.

    This is the test that caught the port's last real defect, and the reason it
    is worth keeping verbatim. `xi` must be built from the **jointly estimated**
    deterministic coefficients, not from the `.pre` seeds. Both give a plausible
    forecast; the seeds give the *univariate* one, which even matches the C's own
    `-0` run — so the error is invisible unless it is compared against a fit that
    actually has a transfer in it.

    On this case the two `omega_d1` move from their seeds to -0.040867 and
    -0.094588 when the transfer is fitted alongside them; that is exactly the
    gap this test closes.
    """
    from drtran.forecast import to_level

    fc = forecast(fitted, L=6)
    level = to_level(fc, fitted.cast_spec, series=0)
    assert level == pytest.approx([82.01, 82.02, 82.38, 83.17, 83.33, 83.44],
                                  abs=0.01)


def test_the_deterministics_come_from_the_fit_not_from_the_pre(fitted):
    """The guard for the above, stated directly: if `_fitted_deterministics`
    ever returns the seeds again, this fails loudly instead of quietly moving
    the forecast onto the univariate path."""
    from drtran.forecast import _fitted_deterministics

    om, _de = _fitted_deterministics(forecast(fitted, L=1),
                                     fitted.cast_spec, series=0)
    semillas = [list(i.omega) for i in
                fitted.cast_spec.series[0].spec.model.interventions]
    assert om[0][0] == pytest.approx(-0.040867, abs=1e-5)
    assert om[1][0] == pytest.approx(-0.094588, abs=1e-5)
    assert om != semillas


# ── the forecast error variance decomposition ────────────────────────────────
def test_the_decomposition_matches_the_C(fitted):
    """The C reports, for ES_CPI: 68.2/31.8, 54.4/45.6, 50.0/50.0, 48.1/51.9,
    47.0/53.0, 46.3/53.7. It is the number that says whether the multivariate
    model earns its keep — by h=6 more than half of the error of forecasting the
    Spanish CPI comes from the oil innovation."""
    from drtran.forecast import variance_decomposition

    fc = forecast(fitted, L=6)
    shares, why = variance_decomposition(fc, series=0)
    assert why is None, why

    c = np.array([[68.2, 31.8], [54.4, 45.6], [50.0, 50.0],
                  [48.1, 51.9], [47.0, 53.0], [46.3, 53.7]]) / 100.0
    assert shares == pytest.approx(c, abs=0.0005)
    assert shares.sum(axis=1) == pytest.approx(np.ones(6))


def test_the_decomposition_is_STRUCTURAL_not_reduced_form(fitted):
    """The distinction is what makes the table possible at all.

    With b=0 the cast puts omega_0 at lag zero and `normalize_phi0` leaves a
    reduced-form Sigma that is correlated BY CONSTRUCTION — decomposing there
    would be impossible on principle. Undoing the normalisation gives a diagonal
    Q, and the total variance does not move, because
    `psi Sigma psi' = (psi Phi0^-1) Q (psi Phi0^-1)'` identically.

    So: the reduced-form Sigma must be non-diagonal here (otherwise there is
    nothing to undo and this test is vacuous), the structural Q must be diagonal,
    and the level variances must be unchanged by the whole manoeuvre.
    """
    from drtran.forecast import error_variance, variance_decomposition

    fc = forecast(fitted, L=6)
    sigma = np.asarray(fc.sigma, float)
    phi0 = np.asarray(fc.phi0, float)

    off = sigma - np.diag(np.diag(sigma))
    assert np.max(np.abs(off)) > 1e-6, "the reduced form must be correlated here"

    Q = phi0 @ sigma @ phi0.T
    assert np.max(np.abs(Q - np.diag(np.diag(Q)))) < 1e-10, "Q must be diagonal"

    psis = np.array([p @ np.linalg.inv(phi0)
                     for p in np.asarray(fc.psi_level, float)])
    var_struct = error_variance(psis, Q, 6)
    for l in range(1, 7):
        assert var_struct[l][0, 0] == pytest.approx(fc.var_level[l][0, 0],
                                                    rel=1e-10)

    shares, why = variance_decomposition(fc, series=0)
    assert why is None and shares is not None


def test_a_correlated_Q_is_declared_not_decomposed(fitted, tmp_path):
    """Freeing a covariance costs you this table, and that is the honest answer:
    with correlated innovations the decomposition depends on an ORDERING, which
    is the VAR's problem. It is declared, not papered over with a Cholesky."""
    from drtran.forecast import variance_decomposition

    fc = forecast(fitted, L=3)
    fc.phi0 = np.eye(2)                       # pretend no normalisation happened
    shares, why = variance_decomposition(fc, series=0)
    assert shares is None
    assert "NOT UNIQUE" in why and "ORDERING" in why
