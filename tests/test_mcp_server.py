"""mtram — the transfer assistant.

Most of what is tested here is not arithmetic: the arithmetic has its own tests.
It is the **architecture's rules**, the ones that make three assistants worth
having instead of one. If they erode, the analyst stops knowing which bench they
are at, and that is the failure this design exists to prevent.
"""

import json
import os

import pytest

pytest.importorskip("mcp")
drtran = pytest.importorskip("drtran")
import drtran.mcp_server as mtram  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
DATA = "/home/david/Dropbox/SRC/drtran/tests/data"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture
def caso():
    name = "t"
    mtram.load_pre(name, f"{ES},{WTI}")
    mtram.set_network(name, json.dumps([{"out": 0, "inp": 1, "b": 0, "r": 0,
                                         "s": 1}]))
    return name


# ── the entry contract: .pre, not raw series ─────────────────────────────────
def test_it_starts_from_pre_files_and_the_first_is_the_output():
    """The roles must be stated back, and the transform each `.pre` carries.

    Asserted by SUBSTANCE, not by layout: this test used to pin the exact
    spacing (`"SALIDA : ES_CPI"`), so re-wording the report broke it while
    nothing about the behaviour had changed. A test that fails on formatting
    trains you to stop reading its failures.
    """
    out = mtram.load_pre("x", f"{ES},{WTI}", check=False)
    salida = next(l for l in out.splitlines() if "SALIDA" in l)
    assert "ES_CPI" in salida, "the output series is not on the SALIDA line"
    entrada = next(l for l in out.splitlines() if "entrada" in l)
    assert "WTI" in entrada, "the input series is not on the entrada line"
    # it reports what each .pre carries, so the analyst can see the rung below
    assert "lambda=0 d=1" in out and "deterministas=11" in out


def test_one_series_is_not_a_transfer_model():
    with pytest.raises(ValueError, match="al menos dos"):
        mtram.load_pre("x", ES)


def test_a_missing_file_says_which():
    with pytest.raises(ValueError, match="no_existe"):
        mtram.load_pre("x", f"{ES},/tmp/no_existe.pre")


# ── THE handoff rule ─────────────────────────────────────────────────────────
def test_a_cyclic_network_is_refused_and_routed_to_sima():
    """The rule that makes three assistants a design and not an accident: a DAG
    with a cycle has no topological order, cannot be cast as a triangular VARMA,
    and therefore describes a SIMULTANEOUS system — which is `sima`'s subject,
    not this one's.

    The assistant must not prune the cycle itself to make the model fit.
    """
    mtram.load_pre("c", f"{ES},{WTI}")
    cyclic = json.dumps([{"out": 0, "inp": 1, "b": 1},
                         {"out": 1, "inp": 0, "b": 1}])
    with pytest.raises(ValueError, match="CYCLE|ciclo"):
        mtram.set_network("c", cyclic)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "m6")),
                    reason="the m6 data is missing")
def test_the_cycle_message_names_sima_and_the_cycle():
    """On m6 the raw proposal IS cyclic — the case the rule was written for. The
    message has to say which cycle and where to go, or the analyst is stuck."""
    m6 = os.path.join(DATA, "m6")
    names = ["EP", "EI", "EU", "EC", "EA", "P"]
    mtram.load_pre("m6", ",".join(os.path.join(m6, f"M6_{n}.pre") for n in names))
    txt = mtram.identify_network("m6")
    assert "CICLO" in txt
    assert "sima" in txt
    assert "->" in txt, "it must name the cycle, not just its existence"
    assert "poda" in txt.lower() or "PODE" in txt


# ── the refusals, which are answers ──────────────────────────────────────────
def test_the_variance_decomposition_refuses_rather_than_ordering(caso):
    """With correlated innovations the decomposition needs an ordering, which is
    the VAR's problem. Refusing IS the answer; the tool must say so in words the
    assistant can pass on."""
    from drtran.forecast import forecast as _f

    f = mtram._require_fit(caso) if caso in mtram._FITS else None
    if f is None:
        mtram.estimate(caso)
    # force the non-diagonal case by pretending there was no normalisation
    fc = _f(mtram._FITS[caso], L=3, embed=True)
    fc.phi0 = None
    from drtran.forecast import variance_decomposition as _v
    shares, why = _v(fc, series=0)
    assert shares is None
    assert "NOT UNIQUE" in why and "ORDERING" in why


def test_tools_refuse_before_the_model_exists():
    mtram.load_pre("u", f"{ES},{WTI}")
    with pytest.raises(ValueError, match="no está estimado"):
        mtram.diagnose("u")
    with pytest.raises(ValueError, match="no hay ningún caso"):
        mtram.diagnose("no_cargado")


# ── the numbers, spot-checked against the C ──────────────────────────────────
def test_the_canonical_case_end_to_end(caso):
    est = mtram.estimate(caso)
    assert "-718.287406" in est
    assert "0.016400" in est and "0.001703" in est
    assert "```" in est, "the equation must come in a verbatim block"

    assert "0.1966" in mtram.diagnose(caso)
    assert "0.027146" in mtram.impulse_response(caso)      # the gain

    fc = mtram.forecast(caso, 3)
    assert "82.0149" in fc and "81.6280" in fc and "82.4035" in fc
    assert "NO es simétrica" in fc, "the band's shape must be stated"


def test_the_forecast_band_is_the_asymmetric_one(caso):
    """A symmetric band from 1.96 s.e. would give [81.54, 82.49]. The right one
    is [81.63, 82.40] — formed in the transformed scale and mapped back."""
    mtram.estimate(caso)
    fc = mtram.forecast(caso, 1)
    assert "81.5421" not in fc and "82.4877" not in fc


# ── the plots ────────────────────────────────────────────────────────────────
def test_the_ccf_plot_is_drvarmas_with_the_lags_the_right_way_round():
    """mtram reuses `drvarma.plots.plot_ccf` so a CCF looks the same across the
    suite. The two libraries use OPPOSITE lag-sign conventions, though, so the
    arguments are swapped: `drvarma`'s k=+1 is `drtran`'s k=-1.

    Passing them unswapped draws the CCF MIRRORED — transfer on the feedback
    side — and the picture looks perfectly normal while saying the opposite of
    the truth. This pins the equivalence rather than the drawing.
    """
    import numpy as np
    from drvarma.diagnostics import ccf as dv_ccf

    from drtran.cast import Link, build_cast_spec
    from drtran.identify import ccf as dt_ccf
    from drtran.plots import prewhitened_pair

    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)],
                         links=[Link(0, 1, 0, 0, 0)])
    a, b = prewhitened_pair(cs, cs.links[0])

    mine = dt_ccf(a, b, 6)                       # drtran: k = 0..6
    theirs = np.asarray(dv_ccf(b, a, 6))         # SWAPPED, two-sided
    mid = len(theirs) // 2

    assert theirs[mid:mid + 4] == pytest.approx(mine[:4], abs=1e-9)
    back = dt_ccf(b, a, 6)                       # drtran: k = 0..-6
    assert theirs[mid - 1:mid - 4:-1] == pytest.approx(back[1:4], abs=1e-9)

    # and unswapped they do NOT agree — the trap is real, not hypothetical
    wrong = np.asarray(dv_ccf(a, b, 6))
    assert wrong[mid + 1] != pytest.approx(mine[1], abs=1e-6)


@pytest.mark.skipif(not os.path.exists("/tmp"), reason="no tmp")
def test_the_plot_tools_write_files(caso, tmp_path):
    pytest.importorskip("matplotlib")
    mtram.estimate(caso)
    for f, kw in ((mtram.plot_ccf, dict(input_index=1)),
                  (mtram.plot_impulse_response, {}),
                  (mtram.plot_forecast, dict(horizon=6))):
        p = f(caso, path=str(tmp_path / f"{f.__name__}.png"), **kw)
        assert os.path.exists(p) and os.path.getsize(p) > 5000


# ── the guided / autonomous split ────────────────────────────────────────────
def test_autonomous_reaches_the_canonical_model_by_revising_itself():
    """The whole point of the split working: the network scan proposes the wrong
    SHAPE (b=1, s=0, logL -756.92, adequacy p = 0.0000), node N6 catches it, and
    the run goes back to N1 with the finer instrument — the bivariate
    prewhitening — and lands on b=0, s=1.

    That is the canonical model: logL -718.287406, adequacy 0.1966, gain
    0.027146. Reached without anyone choosing anything.
    """
    mtram.load_pre("auto", f"{ES},{WTI}")
    txt = mtram.build_model("auto", horizon=3)

    assert "b=1 r=0 s=0  ->  b=0 r=0 s=1" in txt, "it must revise, and say so"
    assert "-756.916700" in txt and "-718.287406" in txt
    assert "0.1966" in txt and "0.027146" in txt


def test_autonomous_never_frees_a_covariance():
    """N3: freeing one is a claim about the world, and it costs the variance
    decomposition. An autonomous run makes no claims the data cannot make for
    it — but it must REPORT the correlation it found, or the analyst never
    learns there was a decision to make."""
    mtram.load_pre("auto2", f"{ES},{WTI}")
    txt = mtram.build_model("auto2", horizon=1)

    assert "NINGUNA liberada" in txt
    assert "r(0) = +0.504" in txt, "the correlation it declined must be visible"
    assert "ninguna" in txt.lower() and "restricciones" in txt.lower()

    # and the model really has it fixed: the decomposition still works
    f = mtram._FITS["auto2"]
    from drtran.forecast import forecast as _f
    from drtran.forecast import variance_decomposition as _v
    shares, why = _v(_f(f, L=3, embed=True), series=0)
    assert why is None and shares is not None


def test_the_two_modes_reach_the_same_model():
    """"They differ in who decides, never in what is computed." Guided, told to
    make the choices the autonomous run made, must land in the same place."""
    mtram.load_pre("m1", f"{ES},{WTI}")
    mtram.build_model("m1", horizon=1)
    auto = mtram._FITS["m1"]

    mtram.load_pre("m2", f"{ES},{WTI}")
    mtram.set_network("m2", json.dumps([{"out": 0, "inp": 1, "b": 0, "r": 0,
                                         "s": 1}]))
    mtram.estimate("m2")
    guided = mtram._FITS["m2"]

    assert auto.loglik == pytest.approx(guided.loglik, abs=1e-6)
    assert auto.x == pytest.approx(guided.x, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "m6")),
                    reason="the m6 data is missing")
def test_autonomous_stops_at_a_cycle_instead_of_pruning():
    """N2′ is a STOP, not a node. Pruning the weakest link to make the model
    estimable would invent a recursive structure the data does not support, and
    do it silently. On m6 the raw proposal IS cyclic."""
    m6 = os.path.join(DATA, "m6")
    names = ["EP", "EI", "EU", "EC", "EA", "P"]
    mtram.load_pre("cyc", ",".join(os.path.join(m6, f"M6_{n}.pre") for n in names))
    txt = mtram.build_model("cyc")

    assert "CICLO" in txt and "sima" in txt
    assert "SIMULTÁNEO" in txt
    assert "RESULTADO" not in txt, "it must not go on to produce a model"
    assert "cyc" not in mtram._FITS


# ── anomaly calibration ──────────────────────────────────────────────────────
def test_calibration_tells_a_bad_SHAPE_from_a_bad_OBSERVATION():
    """Node N6 has two causes that need OPPOSITE responses, and the whole point
    of calibrating is to tell them apart before revising anything.

    The wrong shape (b=1, s=0) fails adequacy and no single observation explains
    it — dropping any one leaves the verdict where it was. Verdict: `shape`, so
    re-identify. The well-specified model passes with nothing carrying it.
    """
    from drtran.cast import Link, build_cast_spec
    from drtran.calibrate import calibrate as _cal

    specs = [drtran.load_pre(ES), drtran.load_pre(WTI)]

    good = _cal(drtran.fit(build_cast_spec(specs,
                                           links=[Link(0, 1, b=0, r=0, s=1)]),
                           embed=True))
    assert good.verdict == "adequate"
    assert good.p_transfer == pytest.approx(0.1966, abs=1e-3)
    assert not good.anomalies, "this sample has no |z| > 3.5"

    bad = _cal(drtran.fit(build_cast_spec(specs,
                                          links=[Link(0, 1, b=1, r=0, s=0)]),
                          embed=True))
    assert bad.p_transfer < 0.05
    assert bad.verdict == "shape", "no observation explains it: re-identify"
    assert not bad.decisive


def test_an_injected_anomaly_is_found_and_its_distortion_measured():
    """Known answer: put one 6 % jump into the output and it must come back with
    a large z and a large share of the residual variance.

    And the finding worth reporting is counter-intuitive: adequacy goes UP, from
    0.1966 to 0.93. A big anomaly inflates the residual variance, which shrinks
    every correlation, which makes the portmanteau comfortable. An adequacy
    bought that way is evidence of an uncalibrated intervention, not of a good
    transfer — and the report has to say so, or the analyst reads 0.93 as good
    news.
    """
    import copy

    from drtran.calibrate import calibrate as _cal
    from drtran.calibrate import report_calibration
    from drtran.cast import Link, build_cast_spec
    from drtran.pre import PreSpec

    y, x = drtran.load_pre(ES), drtran.load_pre(WTI)
    m = copy.deepcopy(y.model)
    d = list(m.series.data)
    d[150] *= 1.06
    m.series.data = d
    hurt = PreSpec(ts=m.series, model=m, path=y.path)

    cal = _cal(drtran.fit(build_cast_spec([hurt, x],
                                          links=[Link(0, 1, b=0, r=0, s=1)]),
                          embed=True))
    assert cal.anomalies, "a 6 % jump must be visible"
    top = cal.anomalies[0]
    assert abs(top.z) > 8
    assert top.variance_fraction > 0.30, "one point, a third of the variance"
    assert cal.p_transfer > 0.5, "and the portmanteau gets MORE comfortable"

    # THE GLOBAL EFFECT, which is the primary one and the school's rule: by
    # inflating the residual variance -- the DIVISOR of every correlation -- an
    # anomaly flattens ALL the lags at once, not just the ones it touches.
    top = cal.anomalies[0]
    assert top.compression > 1.3, "removing it must lift every coefficient"
    import numpy as np
    m_with = float(np.mean(np.abs(cal.ccf)))
    m_without = float(np.mean(np.abs(top.ccf_without)))
    assert m_without / m_with == pytest.approx(top.compression, rel=1e-9)
    assert m_without > m_with

    # and the peak barely moves while the BULK does -- which is why the mean is
    # the right measurement here and the maximum is not
    assert abs(top.ccf_max_without - top.ccf_max) < 0.02

    txt = report_calibration(cal)
    assert "IS WHY IT PASSES SO WELL" in txt
    assert "uncalibrated" in txt and "art" in txt
    assert "CCF x" in txt and "VERIFICAR" in txt


def test_the_verification_plot_shows_both_ccfs():
    """The school's rule is a fact to VERIFY, and the verification is visual:
    take the point out and the coefficients come back. The plot must carry both
    CCFs, or it is an assertion rather than a check."""
    import copy

    pytest.importorskip("matplotlib")
    from drtran.calibrate import calibrate as _cal
    from drtran.calibrate import plot_calibration
    from drtran.cast import Link, build_cast_spec
    from drtran.pre import PreSpec

    y, x = drtran.load_pre(ES), drtran.load_pre(WTI)
    m = copy.deepcopy(y.model)
    d = list(m.series.data)
    d[150] *= 1.06
    m.series.data = d
    cal = _cal(drtran.fit(build_cast_spec([PreSpec(ts=m.series, model=m,
                                                   path=y.path), x],
                                          links=[Link(0, 1, b=0, r=0, s=1)]),
                          embed=True))
    fig = plot_calibration(cal)
    bars = [c for ax in fig.axes for c in ax.containers]
    assert len(bars) == 2, "both CCFs, with and without"
    labels = {b.get_label() for b in bars}
    assert any("sin" in l for l in labels) and any("con" in l for l in labels)


def test_the_calibrate_tool_is_wired(caso):
    mtram.estimate(caso)
    txt = mtram.calibrate(caso)
    assert "ANOMALY CALIBRATION" in txt
    assert "0.1966" in txt
