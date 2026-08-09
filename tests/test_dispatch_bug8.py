"""The dispatch: embedded when the operators agree, subtracting when they do not.

BUG-8's fix. What these pin is the property that made the dispatch worth
choosing over the three alternatives — **the matched cases are not touched** —
and the one that makes it a fix rather than a rearrangement: the mixed cases
land on the oracle.
"""

import os

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec, effective_embed  # noqa: E402
from drtran.estimate import fit, unpack  # noqa: E402

PT8 = "/home/david/Dropbox/SRC/atws/Taste/oracle/data/passthrough8"
WORK = "/home/david/Dropbox/SRC/drtran/examples/work"

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

oraculo = pytest.mark.skipif(
    not os.path.exists(PT8), reason="the TASTE oracle's passthrough data is missing")


def _pair(out_name, s):
    y = drtran.load_pre(os.path.join(PT8, "PT8_%s.pre" % out_name))
    x = drtran.load_pre(os.path.join(PT8, "PT8_WTI.pre"))
    return build_cast_spec([y, x], links=[Link(0, 1, 0, 0, s)])


# ── the property that made this route worth choosing ─────────────────────────
@oraculo
def test_a_matched_pair_keeps_the_embedded_cast():
    """UK is d=1 D=0 on both sides: nothing to dispatch, nothing to change.

    This is the whole argument for the dispatch over a rewrite. Every legacy
    case, m6, the network and the canonical cases are matched, so they keep the
    embedded cast with its exact likelihood and no pre-sample truncation.
    """
    cs = _pair("UK", 0)
    assert not cs.needs_subtracting
    assert cs.alt_est == {}
    assert effective_embed(cs, True) is True
    assert fit(cs).embed is True


@oraculo
def test_a_matched_pairs_estimate_does_not_move():
    """Bit for bit: 0.005223, the value the oracle bank has carried all along."""
    om = unpack(fit(_pair("UK", 0)))["links"][0][0]
    assert float(om[0]) == pytest.approx(0.005223, abs=5e-6)


# ── the case the fix exists for ──────────────────────────────────────────────
@oraculo
def test_a_mismatched_pair_is_dispatched_to_the_subtracting_cast():
    cs = _pair("FR", 1)
    assert cs.needs_subtracting
    assert set(cs.alt_est) == {0}                  # link 0 carries the correction
    assert effective_embed(cs, True) is False
    assert fit(cs).embed is False


@oraculo
@pytest.mark.parametrize("pais, s, oracle", [("FR", 1, 0.009070),
                                             ("DE", 0, 0.011660),
                                             ("EMU", 0, 0.011880)])
def test_the_mixed_cases_now_land_on_the_oracle(pais, s, oracle):
    """From 41-53 % disagreement to under 1.1 %.

    TASTE shares no code with this family and estimates by unconditional
    sum-of-squares with backforecasting rather than exact ML, so agreement to
    the low single digits is the most that can be asked -- the four MATCHED
    cases, which were never in doubt, sit at 0.00-0.93 %. These now sit in the
    same band. Before the dispatch they were at 41-53 %.
    """
    om = unpack(fit(_pair(pais, s)))["links"][0][0]
    rel = abs(float(om[0]) - oracle) / abs(oracle)
    assert rel < 0.011, f"{pais}: {om[0]:.6f} vs oracle {oracle:.6f} = {rel:.1%}"


@oraculo
def test_the_dispatch_can_be_overridden_downwards_but_not_upwards():
    """Asking for the subtracting cast is always honoured; asking for the
    embedded one is not, when the operators differ.

    The asymmetry is the point. `embed=False` is a legitimate request and stays
    legitimate. `embed=True` on a mismatched pair is a request the cast cannot
    honour correctly, and quietly doing it wrong is what BUG-8 was.
    """
    mixed, matched = _pair("FR", 1), _pair("UK", 0)
    assert effective_embed(mixed, False) is False
    assert effective_embed(mixed, True) is False   # NOT honoured
    assert effective_embed(matched, False) is False
    assert effective_embed(matched, True) is True
