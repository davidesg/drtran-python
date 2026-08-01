"""Transfers: the nu weights, the identity with omega=0, and the joint fit."""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (Link, build_cast_spec, compute_irf,  # noqa: E402
                         loglik_diagonal, x0_from_pre)
from drtran.estimate import fit, unpack  # noqa: E402

C = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")
TARGET = -767.424341

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI),
    reason="the canonical .pre files from the C repo are missing")


# ── the rational filter's weights ────────────────────────────────────────────
def test_irf_box_jenkins_convention():
    """omega(B) = omega_0 - omega_1 B - omega_2 B^2...: the leading term ADDS and
    the rest SUBTRACT.

    Porting this sign the other way round is a one-character error that already
    cost a bug in the C (the Nyquist sign in CalcNonsOp), so it is pinned
    explicitly.
    """
    assert compute_irf([2.0, 0.5], [], b=0, length=5) == pytest.approx(
        [2.0, -0.5, 0.0, 0.0, 0.0])


def test_irf_the_delay_b_shifts_it():
    assert compute_irf([2.0, 0.5], [], b=2, length=5) == pytest.approx(
        [0.0, 0.0, 2.0, -0.5, 0.0])


def test_irf_a_denominator_gives_geometric_decay():
    """1/(1-delta B) with delta=0.5 => nu = 1, .5, .25, .125, ..."""
    assert compute_irf([1.0], [0.5], b=0, length=5) == pytest.approx(
        [1.0, 0.5, 0.25, 0.125, 0.0625])


def test_irf_is_zero_if_omega_is_zero():
    assert compute_irf([0.0, 0.0], [0.7], b=1, length=6) == pytest.approx(
        np.zeros(6))


# ── the identity that proves the bridge is right ─────────────────────────────
@pytest.fixture(scope="module")
def two():
    return drtran.load_pre(ES_CPI), drtran.load_pre(WTI)


def test_omega_zero_is_exactly_the_diagonal(two):
    """BRIDGE_DESIGN: "With omega = 0 the model splits into two independent
    univariate ones — and there is the proof that the bridge is right." """
    Y, X = two
    cs0 = build_cast_spec([Y, X])
    cs1 = build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=0)])
    ll0, _ = loglik_diagonal(x0_from_pre(cs0), cs0)
    ll1, _ = loglik_diagonal(x0_from_pre(cs1), cs1)   # x0 sets omega = 0
    assert ll1 == pytest.approx(ll0, abs=1e-10)
    assert ll0 == pytest.approx(TARGET, abs=1e-4)


def test_a_link_adds_exactly_its_own_parameters(two):
    Y, X = two
    base = build_cast_spec([Y, X]).npar
    for (b, r, s) in ((0, 0, 0), (1, 0, 2), (2, 1, 1)):
        cs = build_cast_spec([Y, X], links=[Link(0, 1, b, r, s)])
        assert cs.npar == base + (s + 1) + r


def test_impossible_links_are_rejected(two):
    Y, X = two
    with pytest.raises(ValueError):
        build_cast_spec([Y, X], links=[Link(out=0, inp=0)])      # to itself
    with pytest.raises(ValueError):
        build_cast_spec([Y, X], links=[Link(out=0, inp=5)])      # out of range


# ── joint estimation ─────────────────────────────────────────────────────────
def test_the_estimated_diagonal_stays_on_target(two):
    Y, X = two
    f = fit(build_cast_spec([Y, X]))
    assert f.ifault == 0
    assert f.loglik == pytest.approx(TARGET, abs=1e-4)


def test_the_transfer_improves_significantly(two):
    """Y<-X with a single omega: the joint fit gains ~61 LR points over the
    diagonal."""
    Y, X = two
    f0 = fit(build_cast_spec([Y, X]))
    f1 = fit(build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=0)]))

    assert f1.ifault == 0
    assert f1.loglik > f0.loglik
    lr = 2.0 * (f1.loglik - f0.loglik)
    assert lr > 10.0, "the WTI->CPI pass-through must be clearly significant"

    omega = unpack(f1)["links"][0][0]
    assert omega[0] > 0.0, "a rise in crude raises the CPI"
    assert omega[0] == pytest.approx(0.016, abs=5e-3)


# ── the full cycle, and robustness to the starting point ─────────────────────
def test_full_cycle_identify_estimate_test(two):
    """Box-Jenkins end to end, without being told the orders."""
    from drtran.identify import identify

    Y, X = two
    idn = identify(build_cast_spec([Y, X]), Link(out=0, inp=1))
    assert (idn.b, idn.r, idn.s) == (0, 0, 1)
    assert idn.exogenous

    cs = build_cast_spec([Y, X], links=[Link(0, 1, idn.b, idn.r, idn.s)])
    f = fit(cs)
    f0 = fit(build_cast_spec([Y, X]))
    assert f.ifault == 0
    assert 2.0 * (f.loglik - f0.loglik) > 50.0        # measured LR: 98.3 on 2 df


def test_the_optimum_is_an_attractor_not_an_echo_of_the_start(two):
    """drtran's M1-milestone robustness test.

    `termcode=3` is "stopped without improvement", and it is only a failure if
    the point is not the optimum. The way to settle it is to perturb the start:
    if every path lands in the same place, the stop is legitimate.
    """
    Y, X = two
    cs = build_cast_spec([Y, X], links=[Link(0, 1, 0, 0, 1)])
    base = fit(cs)
    om_base = unpack(base)["links"][0][0]

    rng = np.random.default_rng(0)
    for _ in range(3):
        x0 = x0_from_pre(cs).copy()
        x0[:2] += rng.normal(0, 0.01, 2)
        x0[2:6] *= rng.uniform(0.5, 1.5, 4)
        f = fit(cs, x0=x0)
        assert f.loglik == pytest.approx(base.loglik, abs=1e-5)
        assert unpack(f)["links"][0][0] == pytest.approx(om_base, abs=1e-5)


# ── the specification of the mean ────────────────────────────────────────────
def test_the_mean_is_the_mean_not_an_intercept():
    """Box-Jenkins writes the model in DEVIATIONS from the mean:

        (w_Y - mu_Y) = nu(B)*(w_X - mu_X) + N_t   =>   E[w_Y] = mu_Y

    so the output's mean inherits NOTHING from the input's. Multiplying by
    delta(B), row 1 is Phi(B)(w - mu) = Theta(B)a with mu = (mu_Y, mu_X): no
    extra term. The alternative (w_Y = c + nu(B)w_X + N) treats mu as an
    INTERCEPT.

    The two are the same family AS LONG AS mu_Y is free; they diverge when it is
    FIXED. Coherence with fue decides: the `.pre`'s mu is the mean fue estimated,
    not an intercept.
    """
    from drtran.embed import cast_embedded

    C = "/home/david/Dropbox/SRC/drtran/tests/cases"
    # output with mu FIXED at 0, input with mu free: the discriminating case
    Y = drtran.load_pre(os.path.join(C, "WTI_ar1.pre"))
    X = drtran.load_pre(os.path.join(C, "ES_CPI_m10.pre"))
    assert Y.model.estimate_mu is False and X.model.estimate_mu is True

    cs = build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=1)])
    x = x0_from_pre(cs)
    x[0], x[1] = 17.0, 5.97           # non-trivial omegas, so the term weighs
    _phi, _th, mu, _w, _s, ifault = cast_embedded(x, cs)

    assert ifault == 0
    assert mu[0] == pytest.approx(0.0), "the output's mu is 0: it inherits nothing"
    assert mu[1] == pytest.approx(0.154472), "the input's mu is the .pre's"


def test_with_the_output_mean_free_the_parametrisation_does_not_matter(two):
    """Proof that adjusting the mean or not is a REPARAMETRISATION: with mu_Y
    free, either one reaches the same optimum. It only matters when mu_Y is
    fixed, and there the deviations convention rules."""
    C = "/home/david/Dropbox/SRC/drtran/tests/cases"
    Y = drtran.load_pre(os.path.join(C, "ES_CPI_m10.pre"))   # mu free
    X = drtran.load_pre(os.path.join(C, "WTI_ar1.pre"))
    cs = build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=1)])
    f = fit(cs, embed=True)
    assert f.ifault == 0
    # homologates with the binary, which in this case (the INPUT's mu = 0) agrees
    assert f.loglik == pytest.approx(-718.287406, abs=1e-4)
