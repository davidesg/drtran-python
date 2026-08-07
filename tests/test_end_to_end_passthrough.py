"""The whole ladder, on real data: raw levels -> art -> fue -> mtram.

Every other test in this repo starts from a `.pre` that already exists. This one
starts from a CSV of ORIGINAL LEVELS and walks the entire chain autonomously, so
it is the only test that can fail because two *packages* disagree rather than
because one function does.

    levels_2002_2019.csv        WTI and IPC_ES, untransformed, as they come
        |  art.load_data        -> .inp
        |  art.batch_build      -> the univariate model per series (autonomous)
        |  fue <name>           -> .pre     ** the hop that has no MCP tool **
        |  mtram.load_pre       -> THE DIAGONAL GATE
        |  mtram.build_model    -> the transfer, autonomous
        v
    a gain, with a diagnosis behind it

**The load-bearing assertion is the gate**, not the gain. With a diagonal
structure the exact likelihood factorises, so the joint fit must reproduce the
sum of the univariate ones to numerical precision. If art's transformation,
differencing, deterministics or seeds did not cross into drtran intact, that
identity breaks — and it breaks *quietly*, which is why it is worth a test that
costs a minute.

It is a REGRESSION test for three packages at once, and the two halves do
different jobs. The first half checks the chain completes and the crossing is
exact. The second half pins NUMBERS — art's chosen lambda/d/seasonality, fue's
univariate likelihoods, drtran's diagonal identity, mtram's final order and
gain — one per package, so a failure says where to look before you open
anything.

Those numbers are tripwires, not truths. When art improves its
identification these will fail, and the right response is to read the diff,
satisfy yourself the new answer is better, and move the number here with the
reason. A tripwire nobody is allowed to move is just an obstacle.
"""
import os
import subprocess
import sys

import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("mcp", reason="needs the MCP extra: pip install 'drtran[mcp]'")
A = pytest.importorskip("art.mcp_server", reason="needs art-tseries")
pytest.importorskip("fue")

from drtran import mcp_server as M  # noqa: E402

CSV = ("/home/david/Dropbox/Nivel de Precios y Energia/passthrough_multiart"
       "/data/levels_2002_2019.csv")

pytestmark = [
    pytest.mark.skipif(not os.path.exists(CSV),
                       reason="the passthrough level series are missing"),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    """Walk the chain once; every test below reads its result."""
    d = str(tmp_path_factory.mktemp("passthrough"))

    # ── 1. THE ORIGINAL SERIES, IN LEVELS ─────────────────────────────────
    # The golden rule of the whole suite: load them untransformed. art applies
    # lambda and d itself, and a pre-differenced input makes the identification
    # wrong without making it fail.
    inps = {}
    for col in ("IPC_ES", "WTI"):
        p = os.path.join(d, f"{col}.inp")
        A.load_data(CSV, p, column=col, series_name=col,
                    freq=12, start_year=2002, start_period=2)
        inps[col] = p

    # ── 2. art, AUTONOMOUS ────────────────────────────────────────────────
    batch = "\n".join(getattr(c, "text", "") for c in
                      A.batch_build([inps["IPC_ES"], inps["WTI"]], d))

    # ── 3. .inp -> .pre. There is no MCP tool for this hop: art's autonomous
    #        batch writes the model as a fue input file, and fue turns it into
    #        the `.pre` that is mtram's entry contract. See the test below.
    pres = {}
    for col in ("IPC_ES", "WTI"):
        stem = os.path.join(d, f"{col}_auto")
        subprocess.run([sys.executable, "-m", "fue", stem],
                       cwd=d, capture_output=True, timeout=600)
        pres[col] = stem + ".pre"

    # ── 4 y 5. mtram: the gate, then the transfer ─────────────────────────
    gate = M.load_pre("E2E", f"{pres['IPC_ES']},{pres['WTI']}")
    auto = M.build_model("E2E")
    return dict(dir=d, batch=batch, pres=pres, gate=gate, auto=auto)


def test_art_builds_a_univariate_model_for_each_series(ladder):
    """From raw levels, with no orders given: art must decide lambda, d and the
    seasonality on its own for both series."""
    assert "IPC_ES" in ladder["batch"] and "WTI" in ladder["batch"]


def test_only_fue_produces_a_pre(ladder):
    """The fue step is not a missing tool in the chain — it IS the estimation.

    The file conventions carry the division of labour, and each extension is a
    different claim:

        .inp   a SPECIFICATION. art writes one, with every parameter at
               0.000000 — verified below. Identifying is deciding lambda, d and
               the orders; it is not estimating, and art does not claim to.
        .out   the full record of an estimation and its diagnosis.
        .pre   the same .inp with the estimates as new initial values — i.e.
               AN OPTIMUM, in re-runnable form. That is what makes the ladder
               climbable: each rung's output is the next rung's input in the
               same format, so you can edit and re-run toward a better model.

    So an MCP must NOT write a `.pre`. A `.pre` asserts that an estimation ran
    and converged here, and the file carries no mark of who wrote it: a
    fabricated one is indistinguishable downstream from a genuine one. Only the
    program that did the estimating can make that claim.
    """
    with open(os.path.join(ladder["dir"], "IPC_ES_auto.inp")) as fh:
        spec = fh.read()
    assert "0.000000" in spec, "art's .inp must be a specification, not estimates"
    for p in ladder["pres"].values():
        assert os.path.exists(p), "fue is what turns the specification into a .pre"


def test_a_genuine_pre_is_a_fixed_point(ladder):
    """The invariant behind the whole convention, and it is measurable: run fue
    on a `.pre` and the numbers do not move. That is what "an optimum, in
    re-runnable form" means operationally.

    Measured on this pipeline: 0.000000. It is the check that tells a real
    `.pre` from a file that merely has the extension — see `docs/BUGS.md`,
    BUG-3, where the block drtran writes back moves by 13.11 — which is
    why it is written as `.inp`.
    """
    import shutil
    import numpy as np
    from drtran.pre import load_pre

    d = ladder["dir"]
    src = ladder["pres"]["IPC_ES"]
    stem = os.path.join(d, "fixedpoint")
    shutil.copy(src, stem + ".inp")
    subprocess.run([sys.executable, "-m", "fue", "fixedpoint"],
                   cwd=d, capture_output=True, timeout=600)

    def vals(p):
        m = load_pre(p).model
        return np.array([o for it in m.interventions for o in it.omega]
                        + [c for f in m.ma for c in f], float)

    moved = float(np.max(np.abs(vals(src) - vals(stem + ".pre"))))
    assert moved < 1e-6, f"un .pre genuino debe ser punto fijo; se movió {moved}"


def test_the_crossing_is_exact(ladder):
    """THE assertion. With a diagonal structure the likelihood factorises, so
    the joint fit reproduces the sum of the univariate ones — and it does so
    only if the transformation, the differencing, the deterministics and the
    seeds all crossed from art intact. Measured here: -1.5e-07."""
    assert "✅" in ladder["gate"], ladder["gate"]
    assert "Coinciden" in ladder["gate"]


def test_the_two_series_are_declared_over_the_same_window(ladder):
    """BUG-2's premise, checked where it would actually bite. `cast.py` aligns
    the series at the END and states in a comment that the last observation is
    the same date — but nothing verifies it, and the whole fit goes through
    silently on series that share no calendar at all. This pipeline is the one
    that produces the pair, so it is the right place to check the premise
    holds."""
    from drtran.pre import load_pre
    ts = [load_pre(p).ts for p in
          (ladder["pres"]["IPC_ES"], ladder["pres"]["WTI"])]
    assert ts[0].start == ts[1].start
    assert ts[0].freq == ts[1].freq
    assert ts[0].nobs == ts[1].nobs


def test_mtram_reaches_a_transfer_with_a_diagnosis_behind_it(ladder):
    """The autonomous run must not just converge: it must end with an adequate
    model, having said which defaults it took. Here it revises once — b=1 r=0
    s=0 fails adequacy, and it goes back to N1 and comes out at b=0 r=0 s=1."""
    out = ladder["auto"]
    assert "RESULTADO" in out
    assert "ganancia" in out
    assert "adecuación p =" in out


def test_the_autonomous_run_lists_the_defaults_it_took(ladder):
    """What makes an autonomous result auditable. A number with no record of the
    choices behind it cannot be reviewed, and every node is a choice someone
    else might make differently."""
    out = ladder["auto"]
    for node in ("N0", "N2", "N3", "N4", "N5", "N6"):
        assert node in out, f"falta el nodo {node} en el informe autónomo"


# ── regression tripwires ───────────────────────────────────────────────────
# From here down the numbers ARE the test. Everything above checks that the
# chain runs; these check that it still gives the same answer, and each one
# belongs to a different package — so a failure says WHERE to look.
#
# They are tripwires, not truths. If art changes its identification on
# purpose, or drtran its optimiser, these fail and the right response is to
# read the diff, satisfy yourself the new number is better, and update it here
# with the reason. A tripwire nobody is allowed to move is just an obstacle.


def _num(txt, before, after="", cast=float):
    """Pull the number that follows `before` (and precedes `after`)."""
    import re
    pat = re.escape(before) + r"\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    m = re.search(pat, txt)
    assert m, f"no encuentro {before!r} en el informe"
    return cast(m.group(1))


def test_regression_art_still_identifies_the_same_two_models(ladder):
    """ART's surface. lambda, d and the seasonality are what art DECIDES from
    the raw levels, and they are the whole input to everything downstream.

    Recorded 2026-08-07: IPC_ES lambda=1 d=1 D=0 with 11 deterministics —
    seasonality as HARMONICS, i.e. deterministic, which is also the preferable
    specification for multivariate work; WTI lambda=0 d=1 D=0 with 3.
    """
    g = ladder["gate"]
    assert "IPC_ES: lambda=1 d=1 D=0 refactor=100 deterministas=11" in g, g
    assert "WTI: lambda=0 d=1 D=0 refactor=100 deterministas=3" in g, g


def test_regression_the_univariate_likelihoods_are_unchanged(ladder):
    """fue's surface, through art's choices. If these move, either art picked a
    different model or fue estimates it differently — and the diagonal identity
    below would still pass, because it compares drtran against whatever fue
    said. That is why this one is separate: it is the only check that fue's own
    answer has not drifted."""
    g = ladder["gate"]
    assert _num(g, "| IPC_ES |") == pytest.approx(-988.177767, abs=1e-3)
    assert _num(g, "| WTI |") == pytest.approx(-755.957815, abs=1e-3)


def test_regression_the_diagonal_identity_holds_to_precision(ladder):
    """drtran's surface, and the sharpest instrument in the suite: the joint
    fit must reproduce the sum to ~1e-7, not merely 'closely'. A change in the
    cast, the embedding or the seed handling shows up here first, and shows up
    as a number rather than as a wrong-looking model."""
    diff = abs(_num(ladder["gate"], "Diferencia con la suma: **"))
    assert diff < 1e-5, f"la identidad diagonal se degradó a {diff:.2e}"


def test_regression_mtram_lands_on_the_same_transfer(ladder):
    """mtram's surface: the ORDER it arrives at, which is the decision the
    autonomous mode exists to make. Recorded: it proposes b=1 r=0 s=0, fails
    adequacy, revises once at N6 and settles on b=0 r=0 s=1.

    The revision is the part worth pinning. A run that landed on the right
    order without it would be a different algorithm passing the same test.
    """
    out = ladder["auto"]
    assert "b=1 r=0 s=0  ->  b=0 r=0 s=1" in out, out
    assert _num(out, "adecuación p =") == pytest.approx(0.0, abs=1e-4)   # antes
    assert "exogeneidad p = 0.97" in out or "exogeneidad p = 0.98" in out


def test_regression_the_gain_is_unchanged(ladder):
    """The number a reader would act on, and the last thing in the chain: it
    depends on every step above it. Loose tolerance on purpose — this is the
    end of a long pipeline, and a change of 1e-6 in the fifth decimal of an
    optimiser is not a regression. A change in the second is.

    NOTE for whoever reads it: output and input carry DIFFERENT lambdas here
    (IPC_ES in levels, WTI in logs), so the gain is in units of IPC_ES level
    per log-unit of WTI. It is not a percentage pass-through.
    """
    assert _num(ladder["auto"], "ganancia nu(1) =") == pytest.approx(
        2.5190, abs=0.01)


def test_the_gate_accepts_a_specification_as_readily_as_an_optimum(ladder):
    """`.pre` and `.inp` are the same format, and mtram already takes either.

    That is not an accident of the parser — it follows from what `load_pre`
    does. It re-estimates each series with fue on the way in, so the stored
    values are seeds and nothing more: fed art's `.inp` with every parameter at
    zero, the gate lands on the same likelihoods as with fue's `.pre`
    (-1744.135582 both ways) and closes the same way.

    Which sharpens what the ladder's contract actually is. mtram needs a
    SPECIFICATION, and estimates the univariate optima itself; the `.pre` is
    welcome because it carries good seeds, not because it is required. So a
    `.pre` an analyst has edited — which by the convention has become an `.inp`
    again — can be handed straight back with nothing lost.
    """
    d = ladder["dir"]
    out = M.load_pre("SPEC", f"{d}/IPC_ES_auto.inp,{d}/WTI_auto.inp")
    assert "✅" in out, out
    assert "-1744.135582" in out


def test_a_diagonal_fit_reproduces_the_univariate_optima_exactly(ladder):
    """Why drtran COULD write a univariate `.pre`, and why it should not bother.

    With a diagonal structure the exact likelihood factorises, so the joint
    fit's univariate blocks ARE the univariate optima — not approximately.
    Measured here: max|difference| against fue's own file is 0.00000000.

    So a `.pre` written off a DIAGONAL fit would be legitimate: a fixed point,
    an optimum, everything the convention asks. It would also be a second copy
    of a file that already exists and is identical — no new information, and a
    second place for the two to drift apart.

    And off a fit WITH a transfer it would not be legitimate at all: those
    blocks are optimal for the joint model, hence not for the univariate one.
    That is BUG-3. Between "redundant" and "false" there is no third case in
    which drtran writing a univariate `.pre` would earn its keep.
    """
    import numpy as np
    from drtran.cast import build_cast_spec
    from drtran.pre import load_pre
    import drtran as D
    from drtran.slots import build_slots

    specs = [load_pre(ladder["pres"]["IPC_ES"]), load_pre(ladder["pres"]["WTI"])]
    cs = build_cast_spec(specs, links=[])          # DIAGONAL
    table = build_slots(cs)
    f = D.fit(cs, x0=D.x0_full(cs, table), embed=True, slots=table)
    assert not f.ifault

    # los deterministas de la salida, tal como los deja el ajuste diagonal
    # frente a los que escribió fue
    nombres = [n for n in table.names if n.startswith("omega_d1[")]
    x = np.asarray(f.x, float)
    conj = np.array([x[table.names.index(n)] for n in nombres])
    univ = np.array([o for it in specs[0].model.interventions for o in it.omega])
    assert np.max(np.abs(conj - univ[:len(conj)])) < 1e-6
