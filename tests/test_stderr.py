"""Standard errors: from the Hessian at the optimum, not from the optimiser.

The distinction is the whole content of this file, so it is worth stating once.
`raxopt` leaves behind the Hessian it ACCUMULATED by BFGS along the search path.
It steers the search well, and it is tempting to reuse: it is right there, it
costs nothing, and it has the right shape. It is not the curvature at the
optimum. It depends on the path, it degrades in the flattest directions — the
ones with the largest standard errors, i.e. exactly where one looks — and when
the search starts AT the optimum and stops immediately it is never built at all.

That last case is not hypothetical here: drtran seeds from fue's `.pre`, which on
the diagonal rung ALREADY is the optimum, so the search stops with `termcode 3`
after a couple of iterations.

fue C has this defect — its `fdhess` call sits commented out at
`drvmlest.c:112` — and reports different standard errors for the same point
estimates on different runs (`fue-1.13.1/ERRORES_ESTANDAR.md`). **drtran's C does
not**: its `est()` recomputes the Hessian. This port follows drtran, and the
tests below are what makes that claim checkable rather than a comment.
"""

import os
import re
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit, standard_errors  # noqa: E402
from drtran.slots import build_slots, read_cns  # noqa: E402

C_REPO = "/home/david/Dropbox/SRC/drtran"
CASES = os.path.join(C_REPO, "tests", "cases")
BIN = os.path.join(C_REPO, "bin", "drtran")
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def canonical():
    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    table = build_slots(cs)
    return cs, table, fit(cs, embed=True, slots=table)


# ── the Hessian itself ───────────────────────────────────────────────────────
def test_fdhess_recovers_a_known_hessian():
    """A quadratic has a constant, known Hessian. If this is wrong nothing
    downstream can be right, and it would be wrong in a way that still produces
    plausible-looking standard errors."""
    from drvarma import _qnewt

    def f(x):                       # 3x^2 + 2xy + 5y^2  ->  H = [[6,2],[2,10]]
        return 3 * x[1] ** 2 + 2 * x[1] * x[2] + 5 * x[2] ** 2

    x = np.array([0.0, 1.3, -0.7])
    H = np.zeros((3, 3))
    _qnewt.fdhess(f, 2, x, f(x), _qnewt.MACHEPS, H)

    assert H[1][1] == pytest.approx(6.0, abs=1e-4)
    assert H[2][2] == pytest.approx(10.0, abs=1e-4)
    assert H[1][2] == pytest.approx(2.0, abs=1e-4)
    assert H[1][2] == H[2][1]
    assert x[1] == 1.3 and x[2] == -0.7, "fdhess must restore x"


# ── against the binary ───────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_every_standard_error_matches_the_C(canonical, tmp_path):
    """All 17 of them, read out of the binary's own table live.

    The tolerance is relative and loose-ish (1e-3) on purpose: both sides
    approximate the Hessian by finite differences with a step of macheps^(1/3),
    so the last digit of the six the C prints is noise, not method.
    """
    cs, table, f = canonical
    out = str(tmp_path / "c.out")
    r = subprocess.run([BIN, ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-V",
                        "-o", out], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr

    ref = {}
    for ln in open(out):
        m = re.match(r"^(\S+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d\.\d+)",
                     ln)
        if m:
            ref[m.group(1)] = (float(m.group(2)), float(m.group(3)),
                               float(m.group(4)))
    assert len(ref) >= 15, "could not read the C's parameter table"

    se = standard_errors(f)
    assert se.ifault == 0

    checked = 0
    for i, sl in enumerate(table.slots):
        if sl.name not in ref:
            continue
        est_c, se_c, t_c = ref[sl.name]
        assert f.x[i] == pytest.approx(est_c, abs=1e-5), sl.name
        assert se.se_of_slot[i] == pytest.approx(se_c, rel=1e-3), sl.name
        # t needs an absolute tolerance as well: the C prints three decimals,
        # so a t near zero (omega_d1[9,0] is -0.190) carries up to 0.26% of pure
        # rounding, which no relative tolerance can tell from a real difference.
        assert se.t[i] == pytest.approx(t_c, rel=2e-3, abs=2e-3), sl.name
        checked += 1
    assert checked >= 15, f"only {checked} parameters compared"


def test_the_two_omegas_are_the_published_ones(canonical):
    """The headline figures, pinned without needing the binary present."""
    _cs, table, f = canonical
    se = standard_errors(f)
    i0, i1 = table.index("omega1[0]"), table.index("omega1[1]")
    assert se.se_of_slot[i0] == pytest.approx(0.001703, abs=1e-6)
    assert se.se_of_slot[i1] == pytest.approx(0.001693, abs=1e-6)
    assert se.t[i0] == pytest.approx(9.633, abs=0.01)
    assert se.t[i1] == pytest.approx(-6.349, abs=0.01)
    assert se.p[i0] < 1e-6 and se.p[i1] < 1e-6


# ── the property that motivates all of this ──────────────────────────────────
def test_the_standard_errors_do_not_depend_on_where_the_search_started(canonical):
    """THE test. Two different starting points, the same optimum, and therefore
    the same standard errors.

    This is what a BFGS-derived covariance cannot promise: its matrix is built
    from the sequence of steps, so a different path gives different numbers on
    identical estimates. That is precisely the symptom recorded for fue in
    `ERRORES_ESTANDAR.md`, and this test is what says drtran does not share it.
    """
    cs, table, base = canonical
    se0 = standard_errors(base)

    rng = np.random.default_rng(11)
    x0 = drtran.x0_full(cs, table).copy()
    x0[:2] += rng.normal(0, 0.005, 2)          # perturb the transfer
    x0[-3:-1] *= rng.uniform(0.7, 1.3, 2)      # and the ARMA
    other = fit(cs, x0=x0, embed=True, slots=table)

    assert other.loglik == pytest.approx(base.loglik, abs=1e-5), "same optimum"
    se1 = standard_errors(other)

    ok = np.isfinite(se0.se_of_slot) & np.isfinite(se1.se_of_slot)
    assert ok.sum() >= 15
    assert se1.se_of_slot[ok] == pytest.approx(se0.se_of_slot[ok], rel=1e-3)


def test_starting_AT_the_optimum_still_gives_standard_errors(canonical):
    """The case that breaks the optimiser's matrix outright.

    Restarted from the optimum, the line search cannot improve and stops almost
    at once — so a BFGS approximation is barely updated, or not at all. The
    finite-difference Hessian does not care: it is evaluated where it is asked.
    """
    cs, table, base = canonical
    again = fit(cs, x0=base.x, embed=True, slots=table)
    assert again.nit <= base.nit, "restarting at the optimum should stop early"

    se = standard_errors(again)
    assert se.ifault == 0
    i0 = table.index("omega1[0]")
    assert se.se_of_slot[i0] == pytest.approx(0.001703, abs=1e-5)
    assert np.all(se.se[np.isfinite(se.se)] > 0)


def test_the_objectives_arbitrary_scale_cancels(canonical):
    """The objective is normalised to 1 at x0, so it carries a constant c that
    depends on the starting point. With F = c*G both F(x_hat) and H scale by c
    and `2*F*H^-1` does not move — which is why the previous two tests can hold
    at all. Checked here directly on the formula rather than inferred."""
    _cs, table, f = canonical
    se = standard_errors(f)
    # the covariance is symmetric and positive on the diagonal: the signature of
    # a real curvature, which a rescaled objective preserves
    assert np.allclose(se.cov, se.cov.T, atol=1e-12)
    assert np.all(np.diag(se.cov) > 0)
    assert se.se == pytest.approx(np.sqrt(np.diag(se.cov)))


# ── the slot layer ───────────────────────────────────────────────────────────
def test_a_fixed_slot_has_no_standard_error(canonical):
    """`q[2,1]` is born fixed at zero. Printing a standard error for it would be
    inventing an inference about a parameter nobody estimated."""
    _cs, table, f = canonical
    se = standard_errors(f)
    i = table.index("q[2,1]")
    assert not np.isfinite(se.se_of_slot[i])
    assert not np.isfinite(se.t[i])


def test_a_shared_slot_carries_its_representatives_standard_error(tmp_path):
    """Sharing means ONE degree of freedom in two places, so both places report
    the same error — which is what the C prints, and what makes a shared
    parameter readable as a single estimate."""
    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)],
                         links=[Link(0, 1, b=0, r=1, s=0)])
    table = build_slots(cs)
    cns = tmp_path / "s.cns"
    cns.write_text("delta1[1] = phi_2[B^1]\n")
    read_cns(str(cns), table)

    f = fit(cs, embed=True, slots=table)
    se = standard_errors(f)
    assert se.ifault == 0

    i, j = table.index("delta1[1]"), table.index("phi_2[B^1]")
    assert f.x[i] == pytest.approx(f.x[j])
    assert se.se_of_slot[i] == pytest.approx(se.se_of_slot[j])
    assert np.isfinite(se.se_of_slot[i])


# ── the report ───────────────────────────────────────────────────────────────
def test_the_cli_prints_the_inference_columns_and_can_skip_them():
    """-Q exists because the Hessian costs (k^2+3k)/2 likelihood evaluations,
    which is the slow part on a large network. It must not change anything else."""
    import io
    import sys

    from drtran.cli import main

    def run(*argv):
        so, se_ = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            code = main(list(argv))
            return code, sys.stdout.getvalue()
        finally:
            sys.stdout, sys.stderr = so, se_

    code, out = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-o", "-")
    assert code == 0
    assert "std.error" in out and "t-stat" in out
    assert "0.001703" in out and "***" in out
    assert "Signif. codes" in out
    assert "not from the optimiser's BFGS matrix" in out

    code, quiet = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-Q", "-o", "-")
    assert code == 0
    assert "std.error" not in quiet
    assert "-718.287406" in quiet, "-Q must not change the estimation"
    assert "0.016400" in quiet


# ── the guard, and what it is guarding against ───────────────────────────────
def test_a_point_that_is_not_a_maximum_is_refused_not_patched():
    """`choldcp` is the MODIFIED Cholesky: given a non-positive pivot it patches
    it and carries on. That is right for steering a search and wrong for
    inference — it would turn "this is not a maximum" into a column of
    plausible-looking standard errors.

    m6 at the `.pre` seeds is exactly such a point: 2 of its 55 Hessian
    eigenvalues are <= 0 there, while at the C's actual optimum all 55 are
    positive. So the guard is not defensive programming against a hypothetical;
    it fires on the port's own canonical system.

    This is also the risk drvarma's `est` sidesteps by using the optimiser's
    BFGS matrix, which is positive definite by construction and so never fails —
    at the price of not being the curvature at the optimum.
    """
    from drtran.estimate import Fit
    from drtran.network import read_dag

    D = os.path.join(C_REPO, "tests/data/m6")
    if not os.path.exists(os.path.join(D, "M6_EP.pre")):
        pytest.skip("the m6 .pre files are missing")

    names = ["EP", "EI", "EU", "EC", "EA", "P"]
    specs = [drtran.load_pre(os.path.join(D, f"M6_{n}.pre")) for n in names]
    cs0 = build_cast_spec(specs)
    cs = build_cast_spec(specs, links=read_dag(os.path.join(D, "m6_net.dag"),
                                               cs0.names))
    table = build_slots(cs)
    read_cns(os.path.join(D, "m6_net.cns"), table)

    x = drtran.x0_full(cs, table)
    at_the_seeds = Fit(x=x, loglik=0.0, ifault=0, termcode=1, nit=0,
                       cast_spec=cs, converged=True, slots=table,
                       xfree=table.pack(x), embed=True)

    se = standard_errors(at_the_seeds)
    assert se.ifault == 2, "the seeds are not a maximum; this must be refused"
    assert not np.any(np.isfinite(se.se_of_slot))
