"""Delta(B) = op_out / op_in: the operator the transfer term silently applies.

BUG-8 as arithmetic. The measurements these pin are in
`docs/LEVEL_TRANSFER_PLAN.md` §2d, and the synthetic bank that produced them is
`tests/gen_mixed_operators.py`.
"""

import copy
import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (delta_operator, differencing_poly,  # noqa: E402
                         operators_agree)

M6 = "/home/david/Dropbox/SRC/drtran/tests/data/m6"

pytestmark = pytest.mark.skipif(
    not os.path.exists(M6), reason="the m6 .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def ea():
    """EA: seasonal, and encoded the school's way -- d=2, D=0, ifadf=[0,1,1]."""
    return drtran.load_pre(os.path.join(M6, "M6_EA.pre")).model


@pytest.fixture(scope="module")
def ec():
    """EC: same regular differencing, no seasonal factors."""
    return drtran.load_pre(os.path.join(M6, "M6_EC.pre")).model


def variante(m, d=None, D=None, ifadf=None):
    v = copy.deepcopy(m)
    if d is not None:
        v.d = d
    if D is not None:
        v.D = D
    if ifadf is not None:
        v.ifadf = ifadf
    return v


# ── the reason this compares polynomials and not (d, D) ──────────────────────
def test_the_school_encoding_of_nabla_nabla_4_is_recognised(ea):
    """`d=2, D=0, ifadf=[0,1,1]` and `d=1, D=1` are the SAME operator.

    ∇∇₄ = (1-B)²(1+B)(1+B²): the (1-B) inside ∇₄ adds to the regular
    difference, which is why m6's EA carries d=2 and NOT d=1 (`build_m6.py`
    says so in as many words). Comparing (d, D) tuples would read these two
    encodings as a mismatch and dispatch a correction that is not needed.
    Comparing the polynomial gets it right, and this is the test that keeps it
    that way.
    """
    alt = variante(ea, d=1, D=1, ifadf=[0, 0, 0])
    assert differencing_poly(ea) == pytest.approx(differencing_poly(alt))
    assert operators_agree(ea, alt)
    assert delta_operator(ea, alt)[0] == pytest.approx([1.0])


def test_nabla_nabla_4_is_what_it_should_be(ea):
    """(1-B)(1-B⁴) = 1 - B - B⁴ + B⁵."""
    assert differencing_poly(ea) == pytest.approx([1, -1, 0, 0, -1, 1])


# ── the three cases that matter ──────────────────────────────────────────────
def test_identical_operators_give_delta_one(ec):
    delta, nested, resto = delta_operator(ec, ec)
    assert nested and resto == 0.0
    assert delta == pytest.approx([1.0])
    assert operators_agree(ec, ec)


def test_seasonal_only_mismatch_multiplies_the_gain_by_the_period(ea, ec):
    """EA <- EC would be arm S: same regular differencing, Δ(1) = s.

    Relloso's network never links EA, so this never fired in the legacy; had it
    fired, the reported gain would have been four times the truth.
    """
    delta, nested, _ = delta_operator(ea, ec)
    assert nested
    assert delta == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert float(delta.sum()) == pytest.approx(4.0)
    assert not operators_agree(ea, ec)


def test_an_excess_root_at_frequency_zero_annihilates_the_gain(ec):
    """The observed case: output ∇∇₁₂ against an input at ∇. Δ(1) = 0.

    `∇∇₁₂ = (1-B)²S(B)` is order TWO at frequency zero and the input is order
    one, so Δ = (1-B¹²) and Δ(1) = 0. Measured, arm Z of the synthetic bank
    reports a gain of 0.07 where the truth is 1.0.
    """
    out = variante(ec, d=1, D=1)
    inp = variante(ec, d=1, D=0)
    out.series.freq = inp.series.freq = 12
    delta, nested, _ = delta_operator(out, inp)
    assert nested
    assert float(delta.sum()) == pytest.approx(0.0, abs=1e-12)
    assert len(delta) == 13                      # 1 - B^12
    assert not operators_agree(out, inp)


# ── the case that must be refused rather than dispatched ─────────────────────
def test_an_input_differenced_harder_is_not_nested(ec):
    """Neither operator implies the other: no single vector serves both roles.

    This is not "mismatched and correctable" -- the quotient is not a
    polynomial at all. `nested` is False and the caller must refuse.
    """
    out = variante(ec, d=1, D=0)
    inp = variante(ec, d=1, D=1)
    out.series.freq = inp.series.freq = 12
    _, nested, resto = delta_operator(out, inp)
    assert not nested
    assert resto == float("inf")


def test_partial_overlap_is_not_nested_either(ec):
    """Same degree, different roots: the division leaves a real remainder.

    ∇² against ∇ₛ -- neither contains the other, and the remainder is what says
    so. The `inf` shortcut above only catches the input being LONGER; this is
    the case that needs the division actually performed.
    """
    out = variante(ec, d=0, D=1)
    inp = variante(ec, d=12, D=0)
    out.series.freq = inp.series.freq = 12
    _, nested, resto = delta_operator(out, inp)
    assert not nested
    assert 0.0 < resto < float("inf")


# ── the guard, and it is verified by making it fire ──────────────────────────
WORK = "/home/david/Dropbox/SRC/drtran/examples/work"

guarda = pytest.mark.skipif(
    not os.path.exists(os.path.join(WORK, "WTI_ar1.pre")),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture
def par():
    import drtran as _d
    return (_d.load_pre(os.path.join(WORK, "ES_CPI_m10.pre")),
            _d.load_pre(os.path.join(WORK, "WTI_ar1.pre")))


@guarda
def test_matched_operators_warn_about_nothing(par):
    """The whole legacy is matched, so the guard must cost it nothing.

    If this ever starts warning, the guard has become noise and will be turned
    off -- which is worse than not having it.
    """
    import warnings
    from drtran.cast import Link, build_cast_spec
    y, x = par
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        cs = build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 1)])
    assert cs.delta_warnings == []
    assert [w for w in ws if issubclass(w.category, RuntimeWarning)] == []


@guarda
def test_a_mismatched_link_warns_and_says_the_gain_is_annihilated(par):
    import copy
    import warnings
    from drtran.cast import Link, build_cast_spec
    y, x = par
    y = copy.deepcopy(y)
    y.model.D = 1                      # output at nabla*nabla_12, input at nabla
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        cs = build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 1)])
    assert len(cs.delta_warnings) == 1
    assert len([w for w in ws if issubclass(w.category, RuntimeWarning)]) == 1
    msg = cs.delta_warnings[0]
    # substance, not wording: it must name both series, say the gain is gone,
    # and warn off the correction that looks obvious and is wrong.
    assert "ES_CPI" in msg and "WTI" in msg
    assert "ANNIHILATED" in msg
    assert "Do NOT divide" in msg


@guarda
def test_a_non_nested_pair_is_refused_not_warned(par):
    import copy
    from drtran.cast import Link, build_cast_spec
    y, x = par
    x = copy.deepcopy(x)
    x.model.D = 1                      # the INPUT differenced harder
    with pytest.raises(ValueError, match="neither series"):
        build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 1)])


@guarda
def test_the_diagonal_never_warns(par):
    """No links, no transfer, nothing to mismatch -- the gate stays clean."""
    import warnings
    from drtran.cast import build_cast_spec
    import copy
    y, x = par
    y = copy.deepcopy(y)
    y.model.D = 1
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        cs = build_cast_spec([y, x])
    assert cs.delta_warnings == []
    assert [w for w in ws if issubclass(w.category, RuntimeWarning)] == []
