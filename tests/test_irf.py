"""The impulse response of a transfer, with its standard errors.

`nu_k` is what the model is for: how much of a shock in the input reaches the
output, and by when. The cumulative column converges to the gain — on the
canonical case, how much of an oil shock ends up in the Spanish CPI.

The standard errors come from the delta method on the free-parameter covariance.
The C differentiates the nu recursion analytically and maps each slot's
derivative back through the constraints; this port differentiates with respect to
the FREE vector by central differences, so shared slots, products and linear
combinations propagate on their own. The tests below check that the two agree,
including on a case with a denominator, where the recursion is not trivial.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit  # noqa: E402
from drtran.irf import impulse_response, report_irf  # noqa: E402
from drtran.slots import build_slots  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


def _fit(b, r, s):
    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)],
                         links=[Link(0, 1, b=b, r=r, s=s)])
    table = build_slots(cs)
    return fit(cs, embed=True, slots=table)


@pytest.fixture(scope="module")
def canonical():
    return _fit(0, 0, 1)


# ── against the binary ───────────────────────────────────────────────────────
def test_the_weights_and_their_errors_match_the_C(canonical):
    """The C's table for (b=0, r=0, s=1):

        0   0.016400   0.001703    9.63  |   0.016400   0.001703
        1   0.010747   0.001693    6.35  |   0.027146   0.002452
        2+  0.000000   0.000000    0.00  |   0.027146   0.002452
    """
    ir = impulse_response(canonical)
    assert ir.nu[0] == pytest.approx(0.016400, abs=1e-6)
    assert ir.nu[1] == pytest.approx(0.010747, abs=1e-6)
    assert np.allclose(ir.nu[2:], 0.0, atol=1e-12)

    assert ir.se[0] == pytest.approx(0.001703, abs=2e-6)
    assert ir.se[1] == pytest.approx(0.001693, abs=2e-6)
    assert ir.t[0] == pytest.approx(9.63, abs=0.01)
    assert ir.t[1] == pytest.approx(6.35, abs=0.01)

    assert ir.cum[1] == pytest.approx(0.027146, abs=1e-6)
    assert ir.se_cum[1] == pytest.approx(0.002452, abs=2e-6)


def test_the_sign_convention_survives_the_report(canonical):
    """omega1[1] is NEGATIVE (-0.010747) and nu_1 is POSITIVE (+0.010747),
    because BJR writes omega(B) = omega_0 - omega_1 B - ... The C's report once
    got this backwards and published a gain almost 5 times too small; it is the
    first defect the port found in the original."""
    from drtran.estimate import unpack

    omega = unpack(canonical)["links"][0][0]
    assert omega[1] < 0
    ir = impulse_response(canonical, cov=None)
    assert ir.nu[1] > 0
    assert ir.nu[1] == pytest.approx(-omega[1], abs=1e-9)


def test_the_gain_matches_the_C(canonical):
    """The C reports gain 0.027146, s.e. 0.002452, t = 11.07 — a 1 % oil shock
    passing through to about 0.027 % of the CPI, permanently."""
    ir = impulse_response(canonical)
    assert ir.gain == pytest.approx(0.027146, abs=1e-6)
    assert ir.se_gain == pytest.approx(0.002452, abs=2e-6)
    assert ir.gain / ir.se_gain == pytest.approx(11.07, abs=0.02)


def test_with_a_denominator_the_recursion_still_matches_the_C():
    """(b=1, r=1, s=1). With r=0 the weights are just the omegas and the delta
    method is trivial; a denominator makes nu_k depend on nu_{k-1}, so both the
    recursion and its gradient are exercised. The C gives:

        1   0.011203   0.002360    4.75  |   0.011203   0.002360
        2   0.001617   0.002039    0.79  |   0.012820   0.003175
        gain 0.012800   0.003259   3.93
    """
    ir = impulse_response(_fit(1, 1, 1))
    assert ir.nu[0] == pytest.approx(0.0, abs=1e-12), "b=1 delays by one period"
    assert ir.nu[1] == pytest.approx(0.011203, abs=1e-6)
    assert ir.nu[2] == pytest.approx(0.001617, abs=1e-6)
    assert ir.se[1] == pytest.approx(0.002360, abs=2e-6)
    assert ir.se[2] == pytest.approx(0.002039, abs=2e-6)
    assert ir.cum[2] == pytest.approx(0.012820, abs=1e-6)
    assert ir.se_cum[2] == pytest.approx(0.003175, abs=2e-6)
    assert ir.gain == pytest.approx(0.012800, abs=1e-6)
    assert ir.se_gain == pytest.approx(0.003259, abs=2e-6)


# ── the properties that must hold whatever the model ─────────────────────────
def test_the_cumulative_converges_to_the_gain(canonical):
    """nu(1) = omega(1)/delta(1) IS the sum of the weights. Two routes to the
    same number, and they must not drift apart — `check_nu_consistency` exists
    for the same reason."""
    ir = impulse_response(canonical, cov=None)
    assert ir.cum[-1] == pytest.approx(ir.gain, abs=1e-9)

    ir2 = impulse_response(_fit(1, 1, 1), cov=None)
    assert ir2.cum[-1] == pytest.approx(ir2.gain, abs=1e-6)


def test_without_a_covariance_the_errors_are_NaN_not_zero(canonical):
    """A zero standard error reads as infinite precision, which is the opposite
    of "not computed". `-Q` takes this path."""
    ir = impulse_response(canonical, cov=None)
    assert np.all(np.isnan(ir.se))
    assert np.all(np.isnan(ir.se_cum))
    assert not np.any(np.isnan(ir.nu)), "the weights themselves are still there"
    assert np.all(ir.t == 0.0)

    txt = report_irf(ir)
    assert "0.016400" in txt and "IMPULSE RESPONSE" in txt


def test_a_model_with_no_transfer_has_no_impulse_response():
    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)])
    f = fit(cs, embed=True)
    with pytest.raises(ValueError, match="no transfer"):
        impulse_response(f)


def test_the_report_says_where_the_identification_comes_from(canonical):
    """Not decoration. A reader who takes this table for a VAR impulse response
    will look for the ordering; here the restrictions are the model, and they
    are declared and testable."""
    txt = report_irf(impulse_response(canonical, cov=None))
    assert "not identified without an ordering" in txt
    assert "DECLARED" in txt and "TESTED" in txt
