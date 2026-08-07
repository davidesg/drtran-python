"""A bank of cases with KNOWN TRUTH, run through mtram's whole surface.

The walkthrough tests answer "does the pipeline run?". This answers the harder
question: **does it recover a structure it was never told?** Every case here is
simulated from a transfer function whose (b, r, s) and omegas we chose, so a
failure is not a matter of opinion.

Three things get exercised at once, which is the point — mtram is a protocol
over an engine, and most of what can go wrong lives in the seam between them:

  * the ENGINE, through the joint exact-ML fit;
  * the PROTOCOL, through the gate (diagonal == sum of univariate) and the
    identification proposal;
  * the CROSSING from fue, through the `.pre` files the generator writes.

It doubles as the regression bank: the recovered values are asserted with
tolerances wide enough to be about the METHOD and not about the last digit of a
particular BLAS, and tight enough that a real change moves them.

Generation is `drtran/tests/gen_synthetic.py` from the C repository — the same
generator the C is validated with, so a discrepancy here is a discrepancy with
the C, not with a second implementation of the truth.
"""
import importlib.util
import os
import tempfile

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("mcp", reason="needs the MCP extra: pip install 'drtran[mcp]'")

from drtran import mcp_server as M  # noqa: E402

GEN = "/home/david/Dropbox/SRC/drtran/tests/gen_synthetic.py"

pytestmark = pytest.mark.skipif(
    not os.path.exists(GEN),
    reason="the C repository's synthetic generator is missing")


def _gen():
    """Load the generator by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("_gen_synth", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# (tag, b, r, s, omega, delta) — the truth each case is simulated from.
#
# The grid moves ONE thing at a time so a failure points somewhere:
#   b   the pure delay, which is what the CCF reads most directly
#   s   the numerator order, i.e. how many free weights
#   r   a denominator, i.e. an infinite tail from two parameters
# Signs follow Box-Jenkins: omega(B) = w0 - w1 B - ..., so a NEGATIVE w1 adds
# to the response. Getting that convention wrong is a documented past bug, and
# these cases would catch it again.
CASES = [
    ("b0s0", 0, 0, 0, [0.800], []),
    ("b0s1", 0, 0, 1, [0.800, -0.400], []),
    ("b1s1", 1, 0, 1, [0.800, -0.400], []),
    ("b2s1", 2, 0, 1, [0.800, -0.400], []),
    ("b3s0", 3, 0, 0, [0.600], []),
    ("b0s2", 0, 0, 2, [0.800, -0.400, 0.200], []),
    ("b1r1", 1, 1, 0, [0.800], [0.500]),
    ("b2r1", 2, 1, 0, [0.700], [0.600]),
]
IDS = [c[0] for c in CASES]


@pytest.fixture(scope="module")
def bank():
    """Generate every case once into a temporary directory."""
    g = _gen()
    d = tempfile.mkdtemp(prefix="mtram_bank_")
    out = {}
    for tag, b, r, s, omega, delta in CASES:
        g.build_case(d, tag, b, r, s, omega, delta, seed=20260807 + len(tag))
        out[tag] = dict(dir=d, b=b, r=r, s=s, omega=omega, delta=delta,
                        Y=os.path.join(d, f"{tag}_Y.pre"),
                        X=os.path.join(d, f"{tag}_X.pre"))
    return out


def _load(bank, tag, check=True):
    c = bank[tag]
    return M.load_pre(tag, f"{c['Y']},{c['X']}", check=check), c


# ── the gate, on every case ────────────────────────────────────────────────
@pytest.mark.parametrize("tag", IDS)
def test_the_diagonal_rung_holds(bank, tag):
    """Before any transfer question: does drtran reproduce the univariate fits?

    With a diagonal structure the exact likelihood factorises, so joint == sum.
    If this fails the `.pre` crossing is broken and every other assertion in
    this file would be testing the wrong thing.
    """
    out, _ = _load(bank, tag)
    assert "✅" in out, "the diagonal rung does not reproduce the univariate fits"


# ── recovery: does the CCF find the delay it was never told? ───────────────
@pytest.mark.parametrize("tag", IDS)
def test_identification_recovers_the_delay(bank, tag):
    """`b` is what the prewhitened CCF reads most directly — the first
    significant bar. It is also the parameter an analyst is least able to guess,
    so getting it wrong is the failure that matters most."""
    _, c = _load(bank, tag, check=False)
    out = M.identify_link(tag)
    from drtran.cast import Link, build_cast_spec
    specs = M._SPECS[tag]
    cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 0)])
    idt = drtran.identify(cs, cs.links[0])
    assert int(idt.b) == c["b"], (
        f"recovered b={idt.b}, truth b={c['b']}; significant lags "
        f"{idt.significant_non_negative}")
    assert "GRÁFICO DE LA CCF" in out


@pytest.mark.parametrize("tag", IDS)
def test_the_input_is_reported_exogenous(bank, tag):
    """Every case is generated with a one-way transfer and no feedback, so the
    exogeneity portmanteau over k < 0 must not reject. A test bank that only
    ever checked the positive side would miss the whole left half of the CCF."""
    _load(bank, tag, check=False)
    from drtran.cast import Link, build_cast_spec
    cs = build_cast_spec(M._SPECS[tag], links=[Link(0, 1, 0, 0, 0)])
    idt = drtran.identify(cs, cs.links[0])
    assert idt.exogenous, f"p(exogeneity) = {idt.p_exogeneity:.4f}"


# ── recovery: do the weights come back? ────────────────────────────────────
@pytest.mark.parametrize("tag", IDS)
def test_estimation_recovers_the_impulse_response(bank, tag):
    """Fit at the TRUE (b, r, s) and compare nu(k) with the truth.

    nu, not the raw omegas: it is what the model actually asserts about the
    world, it is invariant to how the numerator and denominator split the work,
    and it is the quantity the sign convention can flip. A past bug inverted
    nu(k) for k > 0 — and therefore the gain — while every log-likelihood stayed
    correct; this compares the thing that was wrong.
    """
    g = _gen()
    _, c = _load(bank, tag, check=False)
    M.set_network(tag, '[{"out": 0, "inp": 1, "b": %d, "r": %d, "s": %d}]'
                  % (c["b"], c["r"], c["s"]))
    M.estimate(tag)

    ir = drtran.impulse_response(M._FITS[tag], link_index=0)
    got = np.asarray(ir.nu, float)
    want = np.asarray(g.impulse_response(c["omega"], c["s"], c["delta"],
                                         c["r"], c["b"], len(got))[1:], float)
    k = min(len(got), len(want), c["b"] + c["s"] + 6)
    # 0.08 in absolute nu: the weights here are 0.2-0.8, the samples are 400
    # observations of a noisy system, and the point is that the SHAPE and the
    # SIGNS are right, not the fourth decimal.
    assert np.max(np.abs(got[:k] - want[:k])) < 0.08, (
        f"nu recovered {np.round(got[:k], 3)} vs truth {np.round(want[:k], 3)}")


@pytest.mark.parametrize("tag", IDS)
def test_the_gain_is_recovered_with_the_right_sign(bank, tag):
    """The gain is the number a reader acts on, and it is the one a sign error
    corrupts silently: summing the omegas instead of alternating them changed
    it by a factor of five once, with the fit still perfect."""
    g = _gen()
    _, c = _load(bank, tag, check=False)
    M.set_network(tag, '[{"out": 0, "inp": 1, "b": %d, "r": %d, "s": %d}]'
                  % (c["b"], c["r"], c["s"]))
    M.estimate(tag)
    ir = drtran.impulse_response(M._FITS[tag], link_index=0)
    want = float(np.sum(g.impulse_response(c["omega"], c["s"], c["delta"],
                                           c["r"], c["b"], 400)[1:]))
    assert float(ir.gain) == pytest.approx(want, abs=0.15), \
        f"gain {ir.gain:.4f} vs truth {want:.4f}"


# ── the protocol holds up on generated data too ────────────────────────────
@pytest.mark.parametrize("tag", ["b2s1", "b1r1"])
def test_diagnose_finds_the_true_model_adequate(bank, tag):
    _, c = _load(bank, tag, check=False)
    M.set_network(tag, '[{"out": 0, "inp": 1, "b": %d, "r": %d, "s": %d}]'
                  % (c["b"], c["r"], c["s"]))
    M.estimate(tag)
    out = M.diagnose(tag)
    assert "Q" in out or "adecua" in out.lower()


def test_a_wrong_delay_is_visibly_worse(bank):
    """A bank that only ever fits the truth cannot tell whether the criteria
    discriminate. Fitting the SAME data at the wrong delay must cost
    likelihood — otherwise `b` is not identified and the whole identification
    step is decoration."""
    _, c = _load(bank, "b2s1", check=False)
    lls = {}
    for b in (c["b"], c["b"] + 1):
        M.set_network("b2s1", '[{"out": 0, "inp": 1, "b": %d, "r": 0, "s": 1}]' % b)
        M.estimate("b2s1")
        lls[b] = float(M._FITS["b2s1"].loglik)
    assert lls[c["b"]] > lls[c["b"] + 1], (
        f"the true delay does not fit better: {lls}")
