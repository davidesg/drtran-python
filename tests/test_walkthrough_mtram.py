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
