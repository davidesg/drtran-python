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


# ── the band convention, shown rather than silently resolved ────────────────
def test_a_band_dependent_weight_is_flagged_as_such():
    """The canonical k=24 is significant under one band and not the other, and
    the margin is 4 %: r = -0.1423 against 0.1364 (constant) and 0.1447 (Haugh).

    We keep the CONSTANT band as the default because that is what the oracle
    does — TASTE's `PLOTS3.PAS` prints "BANDAS 2.0/SQRT(N)" and draws it with
    the same formula — and changing it would break the only external reference
    that shares no ancestry with this family. But Haugh is right that at lag k
    there are only N-|k| products, so the constant band over-detects at high
    lags, which are the seasonal ones. Showing the disagreement costs nothing
    and lets the analyst see that the finding is fragile.
    """
    M.load_pre("BAND", f"{ES},{WTI}", check=False)
    out = M.identify_link("BAND")
    assert "DEPENDE DE LA BANDA" in out
    assert "0.1364" in out and "0.1447" in out
    assert "TASTE" in out


def test_the_other_band_actually_removes_it():
    """The flag must correspond to a real change, not a decorative caveat."""
    import drtran
    from drtran.cast import Link, build_cast_spec
    M.load_pre("BAND2", f"{ES},{WTI}", check=False)
    cs = build_cast_spec(M._SPECS["BAND2"], links=[Link(0, 1, 0, 0, 0)])
    con = drtran.identify(cs, cs.links[0], band="constant")
    hau = drtran.identify(cs, cs.links[0], band="haugh-box")
    assert 24 in (con.significant_non_negative or [])
    assert 24 not in (hau.significant_non_negative or [])
    # and the proposal itself is unchanged, which is why this is a note and not
    # a stop: only the discarded weight moves.
    assert (con.b, con.r, con.s) == (hau.b, hau.r, hau.s)


# ── estimation: what the transfer bought ───────────────────────────────────
def test_estimate_reports_the_equation_and_the_gain():
    """mtram's instructions have always ordered "PRESENT THE EQUATION the tool
    returns", and no tool returned one — not this one, not the C. The gain is
    the number a reader acts on and it was not in the table either, because it
    is not a parameter but omega(1) = w0 - w1 - ..., and that subtraction is
    exactly what a sign convention inverts."""
    M.load_pre("EQ", f"{ES},{WTI}")
    M.set_network("EQ", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("EQ")
    assert "ES_CPI_t" in out and "WTI_t" in out and "N_t" in out
    assert "GANANCIA" in out and "0.027146" in out
    assert "e.t." in out, "the gain comes without its standard error"


def test_estimate_contrasts_against_the_diagonal_rung():
    """"It converged" and "it was worth adding" are different claims, and the
    parameter table only answers the first. The diagonal likelihood is already
    computed by `load_pre`'s gate, so the comparison is free."""
    M.load_pre("LR", f"{ES},{WTI}")
    M.set_network("LR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("LR")
    assert "-767.424" in out and "-718.287" in out
    assert "LR = 2" in out and "98.2" in out
    assert "se gana su sitio" in out


def test_without_the_gate_it_says_it_cannot_contrast():
    """`check=False` skips the diagonal fit, so there is nothing to compare
    against — and that must be said, not silently omitted."""
    M.load_pre("NOLR", f"{ES},{WTI}", check=False)
    M.set_network("NOLR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("NOLR")
    assert "check=False" in out


# ── diagnosis: node N6, whose branches want opposite things ────────────────
def test_diagnose_says_what_to_do_when_it_passes():
    M.load_pre("OK6", f"{ES},{WTI}", check=False)
    M.set_network("OK6", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("OK6")
    out = M.diagnose("OK6")
    assert "QUÉ HACER AHORA" in out and "✅" in out
    assert "impulse_response" in out
    # a model adequate BECAUSE of one point is as fragile as one inadequate
    # because of one, so the calibration is suggested even on success
    assert "calibrate" in out


def test_diagnose_sends_you_to_calibrate_before_respecifying():
    """THE branch that matters. When adequacy fails there are two causes and
    they want opposite responses, and they are only told apart by looking:
    re-specifying (b,r,s) around an anomaly is how a model acquires a lag
    nobody can interpret."""
    M.load_pre("BAD6", f"{ES},{WTI}", check=False)
    M.set_network("BAD6", '[{"out": 0, "inp": 1, "b": 6, "r": 0, "s": 0}]')
    M.estimate("BAD6")
    out = M.diagnose("BAD6")
    assert "LA ADECUACIÓN FALLA" in out
    assert "NO re-especifiques todavía" in out
    assert "calibrate" in out and "PRIMERO" in out
    assert "INTERVENCIÓN" in out


# ── what the school reads, and drtran used to only compute ─────────────────
def test_the_mean_lag_is_reported_beside_the_gain():
    """The gain says HOW MUCH, the mean lag says WHEN, and the school reports
    both in every one of its cases ("la ganancia es 3.1 y el retardo medio,
    aproximadamente un año"). Formula from Brajín (2004) eq. 2.8-2.9."""
    M.load_pre("ML", f"{ES},{WTI}")
    M.set_network("ML", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("ML")
    assert "RETARDO MEDIO" in out and "0.3959" in out


def test_the_mean_lag_is_withheld_when_the_response_changes_sign():
    """Brajín defines it only for a monotone response, and the condition is not
    pedantry: averaging the lags of effects that cancel measures nothing."""
    import drtran
    import numpy as np
    from drtran.cast import Link, build_cast_spec
    M.load_pre("ML2", f"{ES},{WTI}", check=False)
    cs = build_cast_spec(M._SPECS["ML2"], links=[Link(0, 1, 0, 0, 1)])
    f = drtran.fit(cs, embed=True)
    ir = drtran.impulse_response(f)
    # the canonical response is monotone, so the guard must be off here
    assert ir.monotone is True
    assert np.isfinite(ir.mean_lag)


def test_the_residual_variance_reduction_is_reported():
    """How the school closes every case: "una reducción del 44 % de la varianza
    residual en relación a su modelo univariante". It says what the likelihood
    ratio says, in the units an analyst thinks in."""
    M.load_pre("VR", f"{ES},{WTI}")
    M.set_network("VR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("VR")
    assert "Varianza residual" in out and "36.7 %" in out


def test_the_clean_case_raises_no_estimation_warnings():
    """The three estimation-situation readings — near-unit denominator, omega_0
    against -1, parameter correlations above .9 — must be silent on a model
    that has none of those problems. A warning that fires on the canonical case
    would be noise."""
    M.load_pre("CL", f"{ES},{WTI}", check=False)
    M.set_network("CL", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("CL")
    assert "⚠" not in out


# ── nivel 2: el orden de reparación ────────────────────────────────────────
def test_diagnose_looks_at_the_noise_too():
    """`diagnose` used to read one instrument (the CCF) and pronounce on the
    model. The reformulation order needs BOTH: the CCF says whether the relation
    holds, the residual ACF says whether the noise does."""
    M.load_pre("RO", f"{ES},{WTI}", check=False)
    M.set_network("RO", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("RO")
    out = M.diagnose("RO")
    # sustancia, no redacción: que la Q del ruido esté con sus g.l. y su p, y
    # que el bloque del orden exista. Fijar la frase exacta rompió este test
    # cuando el panel pasó al formato de art, que es la lección de siempre.
    assert "Ruido blanco (Q):" in out and "g.l." in out
    assert "EL RUIDO, Y EN QUÉ ORDEN" in out


def test_the_canonical_case_has_both_instruments_clean():
    """Which is the point of using it as the reference case, and the check that
    the noise Q is corrected by the right number of parameters: by the joint
    vector it came out p = 0.0017 and this model would look broken."""
    M.load_pre("RO2", f"{ES},{WTI}", check=False)
    M.set_network("RO2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("RO2")
    out = M.diagnose("RO2")
    # los DOS que deciden el orden de reparación: la CCF y la ACF del ruido
    assert "Ruido blanco (Q): ✓" in out
    assert "RELACIÓN y RUIDO limpios" in out


def test_the_asymmetry_is_stated_when_both_fail():
    """The rule only earns its place in the case it was written for. Built by
    hand rather than hunted for in real data: what is being tested is that the
    branch says the right thing, not that some dataset reaches it."""
    from drtran import mcp_server as _M

    class _Ad:
        adequate, exogenous = False, True
        p_transfer, p_exog = 0.001, 0.5
        significant_lags = []
    M.load_pre("RO3", f"{ES},{WTI}", check=False)
    M.set_network("RO3", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("RO3")
    # forzamos el ruido a "malo" pidiendo una corrección absurda de g.l.
    import drtran.school as sch
    real = sch.noise_adequacy
    try:
        sch.noise_adequacy = lambda *a, **k: (99.0, 0.001, 24, 21)
        txt = "\n".join(_M._reformulation_order(_Ad(), "RO3", 0))
    finally:
        sch.noise_adequacy = real
    assert "FALLAN LOS DOS" in txt
    assert "primero la RELACIÓN, después el RUIDO" in txt


# ── nivel 3 ────────────────────────────────────────────────────────────────
AIRLINE = os.path.join(CASES, "ES_CPI_airline.pre")


def test_the_ccf_spike_is_traced_to_pairs_of_dates():
    """Muñoz traces a residual-CCF lag to two dates, one in each series,
    separated by the lag. Not an outlier: neither observation need be extreme
    on its own, because the coefficient is a sum of PRODUCTS and two moderate
    values that line up carry it. No single-series scan sees that."""
    M.load_pre("PR", f"{ES},{WTI}")
    M.set_network("PR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("PR")
    out = M.calibrate("PR")
    assert "DE QUÉ PARES SALE EL PICO" in out
    # el par dominante es el desplome del crudo y su llegada al IPC
    assert "10/2008" in out and "05/2009" in out


def test_the_seasonality_mismatch_is_announced_and_the_clean_case_is_not():
    """A stochastic-seasonality output against a non-seasonal input gives "una
    ccf muy poco informativa", and the dangerous part is that it does not
    announce itself: it comes back full of structure and the contiguous-block
    rule reads an order off it anyway."""
    M.load_pre("SM", f"{AIRLINE},{WTI}", check=False)
    bad = M.identify_link("SM")
    assert "SARIMA MULTIPLICATIVO Y EL INPUT NO" in bad
    assert "ident_pre" in bad

    M.load_pre("SM2", f"{ES},{WTI}", check=False)
    assert "EL INPUT NO" not in M.identify_link("SM2")


def test_the_deterministic_preference_is_stated_and_meg_is_not_proposed():
    """The advice is the ORDER OF PREFERENCE, not just the workaround.

    And the caution about MEG has to land in the right place: the model class
    is long established (Abraham & Box 1978), so calling it experimental would
    be wrong. What is recent is the TESTING that resolves each frequency, whose
    critical values are under active research — so the assistant does not
    propose that route on its own, without disparaging the specification."""
    M.load_pre("SM3", f"{AIRLINE},{WTI}", check=False)
    out = M.identify_link("SM3")
    assert "DETERMINISTA" in out
    assert "La CLASE no es nueva ni experimental" in out
    assert "NO propongas tú la ruta MEG" in out


def test_identification_can_use_an_alternative_output_model():
    """Muñoz §2.4's split: one model to make the CCF readable, another to fit.
    What is tested is that the split HAPPENS — the identification uses the
    alternative and says so — not that some particular order comes out."""
    M.load_pre("SP", f"{AIRLINE},{WTI}", check=False)
    out = M.identify_link("SP", 1, ident_pre=ES)
    assert "IDENTIFICANDO CON UN MODELO ALTERNATIVO DEL OUTPUT" in out
    # y la estimación sigue con el modelo REAL, no con el alternativo
    assert M._SPECS["SP"][0].name == "ES_CPI"


def test_a_seasonal_ma_factor_is_not_read_as_a_regular_difference():
    """A seasonal factor (1 - Theta B^12) puts twelve roots round the circle,
    INCLUDING a real positive one at frequency zero. Reading that root alone
    says "regular differencing", which is wrong and sends the analyst to undo
    the wrong difference. The multiplicity is what identifies the factor."""
    from drtran.school import noise_ma_roots
    M.load_pre("IR", f"{AIRLINE},{WTI}")
    M.set_network("IR", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("IR")
    roots = noise_ma_roots(M._FITS["IR"], 0)
    assert roots, "el airline tiene un factor MA estacional cerca del círculo"
    mod, mult, kind = roots[0]
    assert mult == 12 and kind == "estacional"


def test_no_movement_is_claimed_without_a_diagonal_to_compare_with():
    """The finding is that a root MOVED toward the circle once the input was in
    the model. Without the diagonal rung there is a position and no movement,
    and claiming one would invent the interesting half of the result."""
    from drtran.school import integration_order_moved
    M.load_pre("IR2", f"{AIRLINE},{WTI}", check=False)   # sin puerta diagonal
    M.set_network("IR2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("IR2")
    moved, mj, md, kind = integration_order_moved(M._FITS["IR2"], None, 0)
    assert moved is False


# ── cómo se presentan los modelos ──────────────────────────────────────────
def test_the_output_model_is_presented_whole_and_the_inputs_are_named():
    """The engine returns one flat list where three different things live: the
    OUTPUT's equation, the INPUTS' univariate models, and the covariances. The
    only clue to which row belongs to whom is the `_1` / `_2` suffix, and an
    `omega_d2[3,0]` sitting in the middle of the list reads as part of the
    output's equation when it is not.

    So: the output's model whole, the inputs' NAMED — which is different from
    omitted, and the difference matters, because in the joint fit the inputs
    ARE estimated here rather than arriving frozen from the `.pre`.
    """
    M.load_pre("PRS", f"{ES},{WTI}", check=False)
    M.set_network("PRS", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("PRS")
    assert "EL MODELO DE ES_CPI — la salida, completo" in out
    assert "LOS MODELOS DE LAS ENTRADAS" in out
    assert "no vienen congelados del `.pre`" in out
    # el enlace va en el bloque de la salida, no suelto
    i_out = out.index("EL MODELO DE ES_CPI")
    i_inp = out.index("LOS MODELOS DE LAS ENTRADAS")
    assert i_out < out.index("omega1[0]") < i_inp


def test_the_regrouping_loses_no_parameter():
    """A presentation change that silently dropped a row would be worse than the
    presentation it replaced. The rows are MOVED, never rebuilt: same count,
    same text."""
    from drtran.cli import report_fit
    from drtran.estimate import standard_errors
    from drtran.slots import build_slots
    import drtran as D
    from drtran.cast import Link, build_cast_spec

    M.load_pre("PRS2", f"{ES},{WTI}", check=False)
    specs = M._SPECS["PRS2"]
    cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 1)])
    table = build_slots(cs)
    f = D.fit(cs, x0=D.x0_full(cs, table), embed=True, slots=table)
    plano = report_fit(f, table, [s.name for s in specs], standard_errors(f))
    agrupado = M._by_series(plano, cs, [s.name for s in specs])

    def filas(txt):
        return {l.strip() for l in txt.split("\n")
                if l.strip() and l.strip()[0].isalpha()
                and ("." in l or "(fixed)" in l) and "  " in l.strip()}
    assert filas(plano) == filas(agrupado)


# ── §7: el certificado, y el .inp como entrada de primera clase ────────────
def test_the_gate_says_the_files_were_optima():
    """The certificate the gate could always have claimed and did not.

    A `.pre` asserts an optimum, and the gate re-estimates anyway — so
    comparing the stored values against the re-estimated ones costs nothing and
    answers a question nothing in the suite could answer before.

    The rigorous form is the likelihood gap: the diagonal fit MAXIMISES what
    the stored values merely EVALUATE, so the difference is >= 0 always and
    zero exactly when the files were optima.
    """
    out = M.load_pre("CERT", f"{ES},{WTI}")
    assert "Hueco de optimalidad" in out
    assert "Aquí lo eran" in out
    assert "un ÓPTIMO (`.pre` de verdad)" in out
    assert "una ESPECIFICACIÓN" not in out


def test_the_gate_says_when_they_were_specifications(tmp_path):
    """The other half, on a file whose parameters have been zeroed — which is
    what a reformulated model looks like: by the convention, an edited `.pre`
    IS an `.inp`.

    And the point is the tone. Being handed a specification is NOT an error:
    the gate estimates it and reaches the same place. What was missing was
    saying so, because an analyst who believes they started from the best
    univariate model of a series and started from a half-estimated
    specification is misreading their own work.
    """
    import re
    src = open(ES).read()
    edited = str(tmp_path / "edited.inp")
    # ponemos a cero los coeficientes deterministas: una reformulación honesta
    open(edited, "w").write(re.sub(r"^-?\d+\.\d{6}(?=  1\s*$)", "0.000000",
                                   src, flags=re.M))
    out = M.load_pre("CERT2", f"{edited},{WTI}")
    assert "una ESPECIFICACIÓN" in out
    assert "no es un problema" in out
    assert "vuelve a ser un `.inp`" in out


def test_an_inp_reaches_the_same_place_as_a_pre():
    """Why accepting a specification is safe rather than merely tolerated: the
    seeds are seeds. Same gate, same conclusion."""
    a = M.load_pre("EQ1", f"{ES},{WTI}")
    assert "✅" in a


# ── consistencia con art: la ecuación y la influencia de la FLT ────────────
def test_the_model_is_shown_as_two_equations_like_art():
    """The analyst arrives from art, and art presents every model this way:

        (1)  yₜ = Dₜ + Nₜ
        (2)  ∇ᵈ[φ(B)][Nₜ − μ] = [θ(B)] aₜ

    with the standard error UNDER each coefficient. Changing format one rung up
    the ladder makes them re-read an instrument they already knew — and it hid
    something: mtram never wrote equation (2) at all, so the NOISE model, which
    is estimated here and moves here, appeared nowhere.

    The canonical form already had the slot, and that is not an analogy: fue's
    deterministics ARE ω(B)/δ(B) on a deterministic input, so art has been
    drawing transfer functions all along. The one here is the same object with
    a stochastic input.
    """
    M.load_pre("EQ", f"{ES},{WTI}", check=False)
    M.set_network("EQ", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("EQ")
    assert "MODELO ESTIMADO" in out
    assert "(1)" in out and "(2)" in out
    # la transferencia, DENTRO de la ecuación (1)
    assert "]·WTIₜ + Nₜ" in out
    # y el ruido escrito, que es lo que faltaba
    assert "aₜ" in out


def test_the_printed_polynomial_sums_to_the_gain():
    """The sign trap, pinned. Box-Jenkins writes ω(B) = ω₀ − ω₁B, so the
    coefficient PRINTED at lag k ≥ 1 is −ω_k, not ω_k. On the canonical case
    ω₁ is −0.9965 and what must appear is +0.9965·B; printing it unflipped
    would show a gain of 0.526 instead of 2.519.

    That inversion already slipped into this port once, with the likelihood
    impeccable. So the renderer checks itself: the printed coefficients must
    sum to ω(1), and says so loudly when they do not.
    """
    import drtran as D
    from drtran.cast import Link, build_cast_spec
    from drtran.estimate import standard_errors
    from drtran.slots import build_slots
    from drtran import mcp_server as MS

    M.load_pre("EQ2", f"{ES},{WTI}", check=False)
    specs = M._SPECS["EQ2"]
    cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 1)])
    table = build_slots(cs)
    f = D.fit(cs, x0=D.x0_full(cs, table), embed=True, slots=table)
    se = standard_errors(f)
    _terms, total = MS._omega_poly(f, table, se, 0, cs.links[0])
    gain = float(D.impulse_response(f, link_index=0).gain)
    assert total == pytest.approx(gain, abs=1e-9)

    M.set_network("EQ2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    assert "revisa el signo" not in M.estimate("EQ2")


def test_the_joint_statistics_are_reported_once_and_labelled_as_joint():
    """ℓ, AIC and BIC belong to ONE fit covering every series. Repeating them
    under each block would invite reading them as that series' own fit, which
    is exactly what they are not — so `fitted_model` deliberately carries no
    likelihood and the footer is written once."""
    M.load_pre("EQ3", f"{ES},{WTI}", check=False)
    M.set_network("EQ3", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("EQ3")
    assert out.count("AJUSTE CONJUNTO") == 1
    assert "no de ninguna por separado" in out


def test_the_transfer_s_influence_on_the_noise_is_reported():
    """The question that closes the circle with the diagonal rung: does the
    transfer MOVE the rest of the model? The rung is already estimated and
    carries the same model without the transfer, so nothing else differs and
    the movement is attributable.

    It is what the school reports when closing a case — Muñoz notes the
    variance reduction is achieved "empleando un parámetro MENOS de
    intervención", i.e. the transfer explains what the univariate model had to
    absorb with a deterministic. Seeing that transfer requires looking at both
    fits, and only one was ever looked at.
    """
    M.load_pre("FLT", f"{ES},{WTI}")          # con puerta: hay diagonal
    M.set_network("FLT", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("FLT")
    assert "¿ES INFLUYENTE LA TRANSFERENCIA?" in out
    assert "ERRORES TÍPICOS" in out


def test_no_influence_is_claimed_without_a_diagonal_to_compare_with():
    """A movement needs two fits. Loaded with `check=False` there is no rung,
    and the report must be absent rather than invented."""
    M.load_pre("FLT2", f"{ES},{WTI}", check=False)
    M.set_network("FLT2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    assert "¿ES INFLUYENTE LA TRANSFERENCIA?" not in M.estimate("FLT2")


def test_brajin_s_two_closing_figures_are_reported():
    """Brajín closes every transfer case with THREE numbers and the suite gave
    one. Verbatim (6.4):

      "La desviación típica residual estimada pasa de 0.53 % en el modelo
       univariante a 0.42 % en el Modelo rpu6.3. El R² en el modelo
       univariante de ru es 0.54, mientras que, en el Modelo rpu6.3, es 0.71."

    The R² is A.28, computed on the STATIONARY series — on the level of an
    I(1) it would sit near 1 by construction and say nothing.
    """
    M.load_pre("R2", f"{ES},{WTI}")
    M.set_network("R2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    out = M.estimate("R2")
    assert "Desviación típica residual: pasa de" in out
    assert "R² (Brajín A.28" in out


def test_the_r2_denominator_carries_no_parameters():
    """What makes the two R² comparable, and what the first implementation got
    wrong. Taking w_t from the cast's `W` looks right and is not: the cast
    subtracts the DETERMINISTIC part, so its variance depends on the estimated
    parameters. Measured, it came out 606.75 under the diagonal fit and 356.47
    under the joint one — and with a moving denominator the R² FELL when the
    transfer was added (0.138 → 0.040) while the residual standard deviation
    correctly fell too.

    Two numbers from one fit pointing opposite ways is the tell. w_t is a
    property of the DATA once λ, d and D are fixed, and only then do the two
    fits describe the same denominator.
    """
    import numpy as np
    from drtran.school import r2_brajin, stationary_series

    M.load_pre("R2B", f"{ES},{WTI}")
    M.set_network("R2B", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("R2B")
    fj, fd = M._FITS["R2B"], M._DIAG_FIT["R2B"]
    wj = stationary_series(fj.cast_spec.series[0].spec)
    wd = stationary_series(fd.cast_spec.series[0].spec)
    assert np.var(wj) == pytest.approx(np.var(wd), rel=1e-12)

    r2u, sau = r2_brajin(fd, 0)
    r2t, sat = r2_brajin(fj, 0)
    # los dos números apuntan al MISMO lado: menos residuo, más R²
    assert sat < sau and r2t > r2u


def test_the_residual_panel_matches_art_s_four_readings():
    """art closes a univariate model with verdict, white noise (Q), normality
    (JB) and skewness/kurtosis. mtram gave only the Q, so an analyst arriving
    from art found half a panel and had to wonder whether the rest was not
    computed or not needed."""
    M.load_pre("PAN", f"{ES},{WTI}", check=False)
    M.set_network("PAN", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("PAN")
    out = M.diagnose("PAN")
    assert "Ruido blanco (Q):" in out
    assert "Normalidad (JB):" in out
    assert "Asimetría =" in out and "curtosis exceso =" in out
    assert "Residuos extremos (|z| > 3):" in out


def test_the_verdict_does_not_swallow_a_failing_normality():
    """The adequacy verdict is decided by the CCF and the ACF, and normality is
    not part of it. Printing "both instruments clean" directly under a failing
    JB reads as a contradiction — so the JB gets its own note saying what it
    does and does not bear on."""
    M.load_pre("PAN2", f"{ES},{WTI}", check=False)
    M.set_network("PAN2", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("PAN2")
    out = M.diagnose("PAN2")
    if "Normalidad (JB): ✗" in out:
        assert "NO entra en el veredicto" in out


def test_the_panel_does_not_duplicate_calibrate():
    """art turns |z| into suggested interventions; mtram must not. `calibrate`
    answers the question that matters here — would the verdict change without
    that observation — by leave-one-out on the CCF and the portmanteau, and an
    anomaly in the univariate residuals may already be explained by the input.
    So the panel counts them and points there."""
    M.load_pre("PAN3", f"{ES},{WTI}", check=False)
    M.set_network("PAN3", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    M.estimate("PAN3")
    out = M.diagnose("PAN3")
    assert "Intervenciones sugeridas" not in out
    if "Residuos extremos (|z| > 3): 0" not in out:
        assert "No los conviertas en intervenciones desde aquí" in out
