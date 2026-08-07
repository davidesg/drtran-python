"""mtram end to end, through the MCP surface.

Uses the canonical case from the C repository (ES_CPI <- WTI), which is the one
homologated against the binary and against the TASTE oracle.

WHY ONE TEST PER STEP. The sequence is stateful — load, identify, fix the
network, estimate, then everything that needs a fit — so it runs ONCE in a
module-scoped fixture and the results are asserted step by step. A failure then
names the step that broke instead of the first one, and the later steps still
report.
"""
import os

import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("mcp", reason="needs the MCP extra: pip install 'drtran[mcp]'")

from drtran import mcp_server as M  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


STEPS = [
    ("load_pre", lambda: M.load_pre("K", f"{ES},{WTI}"),
     lambda o: None if ("ES_CPI" in o or "2" in o) else "does not confirm the load"),

    ("identify_link proposes (b, r, s)",
     lambda: M.identify_link("K", input_index=1),
     lambda o: None if ("b=" in o or "(b" in o.lower()) else "proposes no (b,r,s)"),

    ("set_network fixes the identified link",
     lambda: M.set_network("K", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]'),
     lambda o: None if "1" in o else "does not confirm the link"),

    ("estimate", lambda: M.estimate("K"),
     lambda o: None if "log" in o.lower() else "no likelihood"),

    ("the estimate says why it stopped", lambda: M.estimate("K"),
     lambda o: None if ("termcode" in o.lower() or "converg" in o.lower())
     else "does not report the convergence"),

    ("diagnose", lambda: M.diagnose("K"),
     lambda o: None if ("Q" in o or "adecua" in o.lower()) else "no portmanteau"),

    ("impulse_response", lambda: M.impulse_response("K"),
     lambda o: None if any(k in o.lower() for k in ("nu", "ganancia", "gain"))
     else "no nu(k) / gain"),

    ("variance_decomposition", lambda: M.variance_decomposition("K"),
     lambda o: None if "%" in o else "no percentages"),

    ("forecast", lambda: M.forecast("K", horizon=6),
     lambda o: None if ("95" in o or "banda" in o.lower()) else "no band"),

    ("evaluate (rolling)", lambda: M.evaluate("K", window=120, horizon=3),
     lambda o: None if ("RMSE" in o.upper() or "MAE" in o.upper())
     else "no error metrics"),

    ("build_model (full report)", lambda: M.build_model("K", horizon=6),
     lambda o: None if len(o) > 200 else "the report is too short"),
]

LABELS = [lbl for lbl, _, _ in STEPS]


@pytest.fixture(scope="module")
def walk():
    out = {}
    for label, call, check in STEPS:
        try:
            res = call()
        except Exception as exc:                           # noqa: BLE001
            out[label] = "raised %s: %s" % (type(exc).__name__, str(exc)[:160])
            continue
        try:
            out[label] = check(res)
        except Exception as exc:                           # noqa: BLE001
            out[label] = "the check itself raised: %s" % str(exc)[:120]
    return out


@pytest.mark.parametrize("label", LABELS)
def test_step(walk, label):
    reason = walk[label]
    assert reason is None, reason


# ── the refusals, which are part of the contract ───────────────────────────
@pytest.mark.parametrize("tool", ["diagnose", "calibrate", "impulse_response",
                                  "plot_calibration", "plot_impulse_response"])
def test_link_tools_refuse_clearly_without_a_link(tool):
    """A tool that needs a link must SAY so, not raise IndexError.

    These four used to die with `list index out of range` on a model estimated
    diagonally — an exception that reads like a bug in the tool when what is
    missing is a step in the analysis, and the missing step is always the same
    one: `set_network`.
    """
    M.load_pre("DIAG", f"{ES},{WTI}")
    M.estimate("DIAG")
    with pytest.raises(ValueError, match="no transfer|no link"):
        getattr(M, tool)("DIAG")


# ── the gate: load_pre must prove the bridge before anything else ───────────
def test_load_pre_confirms_the_roles():
    """Which series is the output is decision node N0 — the data cannot answer
    it — so the roles must be stated back for confirmation, not assumed."""
    out = M.load_pre("GATE", f"{ES},{WTI}")
    assert "SALIDA" in out and "ES_CPI" in out
    assert "entrada" in out and "WTI" in out
    assert "orden" in out.lower(), "does not warn that the order decides the roles"


def test_load_pre_proves_the_diagonal_rung():
    """THE gate. With a diagonal structure the exact likelihood factorises, so
    the joint fit must equal the sum of the univariate ones. That identity is
    what says the transform, the differencing, the deterministics and the seeds
    all survived the crossing from fue — and until it holds, no transfer result
    from this case means anything."""
    out = M.load_pre("GATE2", f"{ES},{WTI}")
    assert "-767.4243" in out, "the canonical joint/sum value is not reported"
    assert "✅" in out and "consolidado" in out


def test_the_expected_early_stop_is_not_reported_as_a_warning():
    """On the diagonal rung the `.pre` seeds ALREADY are the optimum, so the
    optimiser stops at once. The generic termcode-2 note says "suspect an
    ill-conditioned likelihood, distrust the standard errors, reduce the order"
    — every clause of which is false here, and this is the most prominent place
    it would appear."""
    out = M.load_pre("GATE3", f"{ES},{WTI}")
    assert "ESPERADO" in out
    assert "ill-conditioned" not in out, "the false alarm is back"


def test_check_false_says_it_skipped_the_gate():
    out = M.load_pre("GATE4", f"{ES},{WTI}", check=False)
    assert "OMITIDO" in out


# ── guided mode: identify_link must PROPOSE, not conclude ───────────────────
def test_identify_link_emits_the_plot_with_the_numbers():
    """In guided mode the analyst decides by looking at the CCF. A table of
    r(k) does not carry the SHAPE, which is what separates a decaying tail from
    an isolated spike, so the plot has to arrive in the same call."""
    M.load_pre("ID", f"{ES},{WTI}", check=False)
    out = M.identify_link("ID")
    assert "GRÁFICO DE LA CCF" in out and ".png" in out


def test_identify_link_says_what_it_discarded():
    """The canonical case has a significant weight at k=24 that the contiguous
    block drops. Dropping it may well be right — but it is a JUDGEMENT, and
    until now the analyst was never told one had been made."""
    M.load_pre("ID2", f"{ES},{WTI}", check=False)
    out = M.identify_link("ID2")
    assert "DEJA FUERA" in out
    assert "k = 24" in out
    assert "MÚLTIPLO DE LA FRECUENCIA" in out, "does not flag it as seasonal"


def test_identify_link_offers_the_exact_set_network_call():
    """Confirming must be one copy-paste, and dissenting just as easy."""
    M.load_pre("ID3", f"{ES},{WTI}", check=False)
    out = M.identify_link("ID3")
    assert "set_network" in out and '"b": 0' in out and '"s": 1' in out


def test_identify_link_frames_it_as_a_proposal():
    M.load_pre("ID4", f"{ES},{WTI}", check=False)
    out = M.identify_link("ID4")
    assert "NO UN VEREDICTO" in out
    assert "ESPERA" in out


def test_it_distinguishes_no_tail_from_a_tail_that_does_not_decay():
    """`identify` only assesses the tail when the block has >= 3 lags. With a
    shorter block the rule never ran, and saying "the tail does not decay"
    would claim it was looked at and rejected."""
    M.load_pre("ID5", f"{ES},{WTI}", check=False)
    out = M.identify_link("ID5")          # canonical block is {0,1} = 2 lags
    assert "no hay cola que evaluar" in out
