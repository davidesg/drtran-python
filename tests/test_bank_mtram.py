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

def _txt(r):
    """El texto de una respuesta MCP, venga sola o con su figura.

    `identify_link` devuelve `[TextContent, ImageContent]`, como art: el nodo
    N1 se decide MIRANDO la ccf, así que separar el informe de su gráfico
    sería partir el instrumento en dos.
    """
    if isinstance(r, list):
        return "\n".join(c.text for c in r if getattr(c, "type", "") == "text")
    return r



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
    out = _txt(M.identify_link(tag))
    from drtran.cast import Link, build_cast_spec
    specs = M._SPECS[tag]
    cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 0)])
    idt = drtran.identify(cs, cs.links[0])
    assert int(idt.b) == c["b"], (
        f"recovered b={idt.b}, truth b={c['b']}; significant lags "
        f"{idt.significant_non_negative}")
    assert "La CCF va ABAJO, con estos números" in out


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


# ── the stopping rule: some situations make choosing an order premature ────
@pytest.mark.parametrize("seed", [12, 99, 7])
def test_noise_is_reported_as_noise_not_as_feedback(seed):
    """With no relationship at all, mtram must say "I cannot see anything" —
    not propose a delay, and not announce feedback.

    Both wrong answers are available and both are worse than silence.
    `has_relationship` stays True on pure noise, because with ~25 lags at 5 %
    one or two bars are significant by chance, and the `b` read off them is
    random: measured 10, 20, 8, 10, 13 across seeds. And the exogeneity
    portmanteau over k < 0 also rejects by chance sometimes (p = 0.036 in one
    of these), which would send the analyst to `sima` to estimate a
    simultaneous system that does not exist.

    The discriminator that does separate them is how far the PEAK stands above
    the band: 1.0-1.5 on noise, 7.6-7.8 on a real transfer. No grey zone.
    """
    g = _gen()
    d = tempfile.mkdtemp(prefix="mtram_noise_")
    g.build_case(d, "NZ", 2, 0, 0, [0.0005], [], seed=seed)
    M.load_pre(f"NZ{seed}", f"{d}/NZ_Y.pre,{d}/NZ_X.pre", check=False)
    out = _txt(M.identify_link(f"NZ{seed}"))
    assert "NO SE DISTINGUE DEL RUIDO" in out
    assert "No propongo orden" in out
    assert "set_network" not in out, "it proposed an order on noise"


@pytest.mark.parametrize("tag", ["b2s1", "b0s1"])
def test_a_real_transfer_is_not_stopped(bank, tag):
    """The other half of the rule: a stopping rule that also fires on signal
    would just be an off switch."""
    _load(bank, tag, check=False)
    out = _txt(M.identify_link(tag))
    assert "PARA:" not in out
    assert "set_network" in out


def test_the_single_observation_warning_does_not_fire_on_clean_data(bank):
    """A warning that fires on clean simulated data is worth nothing. Measured
    there: the heaviest observation carries 2-5 % of the dominant lag's
    correlation, against a 15 % threshold."""
    _load(bank, "b2s1", check=False)
    out = _txt(M.identify_link("b2s1"))
    assert "UNA OBSERVACIÓN PESA DEMASIADO" not in out


# ── nivel 2: el procedimiento, no sólo el número ───────────────────────────
@pytest.mark.parametrize("tag,truth_r", [("b1r1", 1), ("b2r1", 1),
                                         ("b2s1", 0), ("b0s2", 0)])
def test_refine_reads_the_denominator_off_the_free_weights(bank, tag, truth_r):
    """`r` is the one order the CCF cannot show you: in a sample, an infinite
    geometric tail and a long finite one look alike. The school's answer is to
    estimate a generous free MA and read the SHAPE of the weights — decaying
    gradually means a denominator, cutting off means none.

    This is the test that says whether that reading works, and it is only
    answerable against generated data: the cases carry a denominator or they do
    not, and we chose which.
    """
    _load(bank, tag, check=False)
    out = M.refine_link(tag)
    if truth_r:
        assert "firma de un DENOMINADOR" in out, out[-800:]
        assert '"r": 1' in out
    else:
        assert "se cortan" in out, out[-800:]
        assert '"r": 0' in out


def test_refine_recovers_the_whole_structure(bank):
    """Not just r — the command it proposes must be the truth, all three
    orders. A reading that gets the shape right and the delay wrong sends the
    analyst to a model that fits nothing."""
    _load(bank, "b1r1", check=False)
    out = M.refine_link("b1r1")
    assert '"b": 1, "r": 1, "s": 0' in out, out[-600:]


def test_overfit_confirms_a_model_fitted_at_the_truth(bank):
    """Brajín §2.3.1's doctrine, on data where the truth is known: enlarging a
    correctly specified model must leave the extra parameters non-significant.
    If this failed, `overfit` would be telling analysts to grow every model
    they build."""
    _, c = _load(bank, "b2s1", check=False)
    M.set_network("b2s1", '[{"out": 0, "inp": 1, "b": 2, "r": 0, "s": 1}]')
    M.estimate("b2s1")
    out = M.overfit("b2s1")
    assert "CONFIRMADO" in out, out


def test_overfit_refuses_to_confirm_a_model_that_is_too_small(bank):
    """The other half. Fitted one weight short of the truth, the enlargement
    must come back significant — otherwise `overfit` confirms everything and
    confirms nothing."""
    _load(bank, "b0s2", check=False)
    M.set_network("b0s2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 0}]')
    M.estimate("b0s2")
    out = M.overfit("b0s2")
    assert "NO está cerrado" in out, out


def test_overfit_leaves_the_session_model_alone(bank):
    """An overfitting experiment that leaves the case estimated at the enlarged
    model is a trap: the analyst rejects it, carries on, and everything after
    comes out of the model they just rejected."""
    _load(bank, "b2s1", check=False)
    M.set_network("b2s1", '[{"out": 0, "inp": 1, "b": 2, "r": 0, "s": 1}]')
    M.estimate("b2s1")
    before = list(M._FITS["b2s1"].cast_spec.links), float(M._FITS["b2s1"].loglik)
    M.overfit("b2s1")
    after = list(M._FITS["b2s1"].cast_spec.links), float(M._FITS["b2s1"].loglik)
    assert before == after


def test_the_noise_correction_uses_the_series_own_parameters(bank):
    """The Ljung-Box on ONE series' ACF is a statement about THAT series'
    model. Correcting it by the joint parameter vector is what a naive reading
    of "number of estimated parameters" gives, and it inflates significance on
    every multivariate fit — measured on the canonical case, 21 degrees of
    freedom against 7, p = 0.34 against p = 0.0017."""
    from drtran.school import npar_for_series
    _load(bank, "b2s1", check=False)
    M.set_network("b2s1", '[{"out": 0, "inp": 1, "b": 2, "r": 0, "s": 1}]')
    M.estimate("b2s1")
    f = M._FITS["b2s1"]
    n_series = npar_for_series(f, 0)
    n_joint = M._TABLES["b2s1"].n_free
    assert n_series < n_joint
    # los 2 pesos del enlace entran, porque se estiman del mismo residuo
    assert n_series >= 2


def test_a_redundant_denominator_is_named_as_redundant(bank):
    """Adding r+1 to a model that needs no denominator makes the last numerator
    weight and the new delta describe the same tail, and they correlate at .92.
    That is the EXPECTED outcome, not a data accident, and the report has to say
    so — otherwise "experimento fallido" reads as "bad data" and the analyst
    goes looking for a problem that is not there."""
    _load(bank, "b2s1", check=False)
    M.set_network("b2s1", '[{"out": 0, "inp": 1, "b": 2, "r": 0, "s": 1}]')
    M.estimate("b2s1")
    out = M.overfit("b2s1")
    assert "EXPERIMENTO FALLIDO" in out
    assert "la ampliación es redundante" in out
    # y aun así el modelo queda confirmado por la ampliación que SÍ salió
    assert "CONFIRMADO por 1 de 2" in out
