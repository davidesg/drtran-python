"""Forecasting: the MA(infinity) weights and the three variances.

What is pinned here is the CORE — psi weights, the error-variance recursion, and
the integration that undoes the differencing. The reporting layer that returns
to the LEVEL (adding the future deterministic component and undoing Box-Cox) is
not ported yet, so the absolute figures of the C's forecast table cannot be
compared; what is compared is the **structure**, which is the part that lives
here.

Two facts make the comparison with the C meaningful anyway:

* the ratios of the standard errors across horizons must match the C exactly —
  they depend only on psi, on the integration and on Sigma;
* the point forecast of `w` must decay to `mu`, because `w` is the series with
  the deterministic part already removed. The C's level forecast does NOT decay,
  precisely because it adds those harmonics back.
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

    # y si psi es todo unos, psi/(1-B) es la suma acumulada: 1, 2, 3, ...
    psis2 = integrated_weights(np.array([np.eye(1)] * 5), d=1, D=0, s=1)
    assert [psis2[l][0, 0] for l in range(5)] == pytest.approx([1., 2., 3., 4., 5.])


def test_the_seasonal_operator_only_subtracts_from_lag_s():
    psi = np.zeros((6, 1, 1)); psi[0, 0, 0] = 1.0
    psis = integrated_weights(psi, d=0, D=1, s=4)
    # (1-B^4)^-1 de [1,0,0,...] da 1,0,0,0,1,0: un uno cada s
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
def ajuste():
    cs = build_cast_spec([drtran.load_pre(os.path.join(CASES, "ES_CPI_m10.pre")),
                          drtran.load_pre(os.path.join(CASES, "WTI_ar1.pre"))],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return fit(cs, embed=True)


def test_the_ratios_of_the_standard_errors_match_the_C(ajuste):
    """The C reports, for ES_CPI from 12/2019, level s.e. of
    0.24 0.44 0.60 0.74 0.85 0.95 and variation s.e. of 0.24 0.28 0.28 ...

    Those are in index units and this port still works in the modelled scale, so
    the levels cannot be compared directly. The RATIOS can, and they depend on
    exactly what lives here: psi, the integration and Sigma.
    """
    fc = forecast(ajuste, L=6)
    nivel = fc.se("level", 0)
    variacion = fc.se("diff", 0)

    c_nivel = np.array([0.24, 0.44, 0.60, 0.74, 0.85, 0.95])
    c_var = np.array([0.24, 0.28, 0.28, 0.28, 0.28, 0.28])

    # el C publica con dos decimales, asi que la comparacion honesta es
    # RELATIVA: 0.24 lleva ya un 2 % de incertidumbre de redondeo
    assert nivel / nivel[0] == pytest.approx(c_nivel / c_nivel[0], rel=0.02)
    assert variacion / variacion[0] == pytest.approx(c_var / c_var[0], rel=0.03)


def test_the_level_variance_grows_and_the_variation_settles(ajuste):
    """The signature of a differenced model: integrating gives an unbounded
    level variance, while the variation's converges."""
    fc = forecast(ajuste, L=12)
    nivel, variacion = fc.se("level", 0), fc.se("diff", 0)
    assert np.all(np.diff(nivel) > 0)
    assert abs(variacion[-1] - variacion[-2]) < 1e-3


def test_the_scale_is_applied_not_left_in_Q(ajuste):
    """The cast returns Q, not Sigma — the likelihood is concentrated and sigma2
    comes out separately. Forgetting it leaves the right shape with the wrong
    magnitude, which is the classic misreading of Q as Sigma."""
    fc = forecast(ajuste, L=3)
    assert 0.05 < fc.se("level", 0)[0] < 1.0, "orden de magnitud de sigma, no de Q"


# ── back to the level ────────────────────────────────────────────────────────
def test_a_series_with_no_incoming_transfer_matches_the_C(ajuste):
    """WTI receives nothing, so its level forecast must be the univariate one —
    and the C reports 60.76, 61.02, 61.10 from 12/2019.

    This is what exonerates the level layer: the future deterministic effect,
    the integration against delta(B) and the inverse Box-Cox are all exercised
    here, and they land on the C's own numbers.
    """
    from drtran.forecast import to_level

    fc = forecast(ajuste, L=6)
    nivel = to_level(fc, ajuste.cast_spec, serie=1)
    assert nivel[:3] == pytest.approx([60.76, 61.02, 61.10], abs=0.01)
    assert nivel[-1] == pytest.approx(61.14, abs=0.01)


@pytest.mark.xfail(reason="la transferencia entra en la previsión, pero su "
                          "efecto converge a una constante mientras que en el C "
                          "crece; ver TODO",
                   strict=True)
def test_the_output_level_matches_the_C(ajuste):
    """ES_CPI receives the transfer, and there the port does not reach the C.

    Lo medido, que acota el problema a una sola cosa:

    * las omega COINCIDEN con las del C (0.016400, -0.010747);
    * sin transferencia el puerto reproduce EXACTAMENTE el `-0` del C
      (81.98 82.01 82.45 83.33 83.54 83.67), luego xi, la integración y la
      Box-Cox son correctas;
    * la transferencia SÍ entra: poniendo omega a cero el nivel se mueve
      [+0.029 +0.038 +0.040 +0.042 +0.042 +0.042];
    * y en el C mueve [+0.03 +0.01 -0.07 -0.16 -0.21 -0.23].

    O sea: en h=1 coincidimos (+0.03), y a partir de ahí el efecto del puerto
    CONVERGE a una constante mientras que el del C crece y cambia de signo. La
    entrada prevista decae rápido (WTI es AR(1) sobre Δlog: 1.46, 0.44, 0.13,
    0.04...), así que su aportación acumulada al nivel tiende a una constante.
    Que en el C no lo haga dice que alimenta a nu(B) con otra senda.

    Comprobado además que no es la vía: reconstruir la previsión a mano
    --prever la entrada, pasarla por nu(B) y sumarla al ruido-- da exactamente
    lo mismo que la recursión conjunta del VARMA (w_Y = 0.1134 frente a 0.1133),
    como debe ser. La hipótesis de que el C hiciera eso y el puerto no, no
    explica la diferencia.
    """
    from drtran.forecast import to_level

    fc = forecast(ajuste, L=6)
    nivel = to_level(fc, ajuste.cast_spec, serie=0)
    assert nivel == pytest.approx([82.01, 82.02, 82.38, 83.17, 83.33, 83.44],
                                  abs=0.01)
