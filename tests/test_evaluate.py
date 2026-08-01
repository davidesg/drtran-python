"""The fixed window and the rolling origin: the empirical model comparison.

Everything else in this port compares the model against itself — a likelihood,
a theoretical variance, a portmanteau. This is the part that compares it against
what happened. The variances the model reports say what the model implies; out of
sample, parameter uncertainty and structural change have their say too, and the
ranking can come out the other way round.

The protocol is the C's: estimate ONCE on 1..E, hold the parameters FIXED, and
roll the origin forward one datum at a time to n-H. Holding them fixed is the
point — re-estimating at every origin measures a procedure, not a model.
"""

import csv
import os
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link  # noqa: E402
from drtran.evaluate import (fixed_window_fit, report_rolling,  # noqa: E402
                             rolling_evaluation, truncate,
                             write_rolling_csv)

C_REPO = "/home/david/Dropbox/SRC/drtran"
CASES = os.path.join(C_REPO, "tests", "cases")
BIN = os.path.join(C_REPO, "bin", "drtran")
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")
LINKS = [Link(0, 1, b=0, r=0, s=1)]

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


def _specs():
    return [drtran.load_pre(ES), drtran.load_pre(WTI)]


# ── truncation ───────────────────────────────────────────────────────────────
def test_truncating_does_not_touch_the_callers_series():
    """THE trap of this module. In fue `model.series` IS the spec's `ts`, the
    same object, so trimming one in place trims the other — and the evaluation
    would end up comparing a forecast against the very data it was made from,
    scoring suspiciously well."""
    specs = _specs()
    before = list(specs[0].ts.data)
    n0 = specs[0].ts.nobs

    cut = truncate(specs, 200)

    assert cut[0].ts.nobs == 200
    assert len(cut[0].model.series.data) == 200
    assert specs[0].ts.nobs == n0, "the caller's spec must keep the full sample"
    assert list(specs[0].ts.data) == before
    assert cut[0].model.series is not specs[0].model.series


def test_truncating_beyond_the_sample_is_a_no_op():
    specs = _specs()
    assert truncate(specs, 10_000)[0].ts.nobs == specs[0].ts.nobs


# ── against the binary ───────────────────────────────────────────────────────
def test_the_window_fit_matches_the_C():
    """Estimating on 1..200 must land where the C lands: it is the same model on
    the same 200 observations."""
    f, _cs = fixed_window_fit(_specs(), LINKS, window=200, embed=True)
    assert f.ifault == 0
    assert f.loglik == pytest.approx(-666.573252, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_rolling_forecasts_match_the_C_row_by_row(tmp_path):
    """All 66 rows of the per-origin CSV, against the binary run live."""
    c_csv = str(tmp_path / "c.csv")
    r = subprocess.run([BIN, ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-V",
                        "-estwin", "200", "-f", "6", "-C", c_csv,
                        "-o", str(tmp_path / "c.out")],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr
    ref = list(csv.DictReader(open(c_csv)))
    assert len(ref) == 66, "11 origins x 6 horizons"

    f, _cs = fixed_window_fit(_specs(), LINKS, window=200, embed=True)
    ev = rolling_evaluation(f.x, _specs(), LINKS, window=200, horizon=6)

    mine = {(o, h): (a, fo) for o, h, a, fo, _e in ev.rows}
    assert len(mine) == len(ref)
    for row in ref:
        key = (int(row["origin"]), int(row["horizon"]))
        act, fcst = mine[key]
        assert act == pytest.approx(float(row["actual"]), abs=1e-9), key
        assert fcst == pytest.approx(float(row["forecast"]), abs=1e-5), key


def test_the_summary_matches_the_C():
    """MAE, RMSE and MAPE by horizon, as the C's table publishes them."""
    f, _cs = fixed_window_fit(_specs(), LINKS, window=200, embed=True)
    ev = rolling_evaluation(f.x, _specs(), LINKS, window=200, horizon=6)

    c_mae = [0.150269, 0.253371, 0.326313, 0.430705, 0.447533, 0.466986]
    c_rmse = [0.178568, 0.346342, 0.463563, 0.541151, 0.562753, 0.569408]
    c_mape = [0.1830, 0.3086, 0.3989, 0.5268, 0.5475, 0.5685]

    assert ev.mae == pytest.approx(c_mae, abs=2e-6)
    assert ev.rmse == pytest.approx(c_rmse, abs=2e-6)
    assert ev.mape == pytest.approx(c_mape, abs=1e-4)


# ── the protocol itself ──────────────────────────────────────────────────────
def test_the_horizons_are_balanced():
    """Every origin runs to n-H, so every h is averaged over the SAME origins.
    Unbalanced columns cannot be compared with one another, which is the only
    thing anyone does with this table."""
    f, _cs = fixed_window_fit(_specs(), LINKS, window=200, embed=True)
    ev = rolling_evaluation(f.x, _specs(), LINKS, window=200, horizon=6)

    assert ev.n_origins == 11
    assert (ev.first_origin, ev.last_origin) == (200, 210)
    assert list(ev.count) == [11] * 6


def test_the_parameters_are_held_fixed():
    """Two evaluations with the same `x` give the same answer, and a DIFFERENT
    `x` gives a different one — i.e. the forecasts really do depend on the
    parameters passed and are not being re-estimated behind the caller's back."""
    specs = _specs()
    f, _cs = fixed_window_fit(specs, LINKS, window=200, embed=True)
    a = rolling_evaluation(f.x, specs, LINKS, window=200, horizon=3)
    b = rolling_evaluation(f.x, specs, LINKS, window=200, horizon=3)
    assert a.mae == pytest.approx(b.mae)

    x2 = np.asarray(f.x, float).copy()
    x2[0] *= 3.0                                    # move omega1[0]
    c = rolling_evaluation(x2, specs, LINKS, window=200, horizon=3)
    assert not np.allclose(c.mae, a.mae), "the parameters must actually be used"


def test_not_enough_data_is_an_error_not_an_average_over_two_origins():
    """Reporting a MAE over two origins as though it meant something is worse
    than refusing."""
    specs = _specs()
    f, _cs = fixed_window_fit(specs, LINKS, window=200, embed=True)
    with pytest.raises(ValueError, match="not enough data"):
        rolling_evaluation(f.x, specs, LINKS, window=214, horizon=6)


def test_the_csv_has_one_row_per_origin_and_horizon(tmp_path):
    f, _cs = fixed_window_fit(_specs(), LINKS, window=205, embed=True)
    ev = rolling_evaluation(f.x, _specs(), LINKS, window=205, horizon=4)
    path = write_rolling_csv(ev, str(tmp_path / "e.csv"))
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == ev.n_origins * ev.horizon
    assert set(rows[0]) == {"origin", "horizon", "actual", "forecast", "error"}
    for r in rows:
        assert float(r["error"]) == pytest.approx(
            float(r["forecast"]) - float(r["actual"]), abs=1e-9)


def test_the_report_says_the_variances_are_theoretical():
    """Not decoration: it is the reason the table exists, and a reader who
    misses it will compare these numbers with the model's own bands."""
    f, _cs = fixed_window_fit(_specs(), LINKS, window=205, embed=True)
    txt = report_rolling(rolling_evaluation(f.x, _specs(), LINKS,
                                            window=205, horizon=4))
    assert "THEORETICAL" in txt
    assert "held FIXED" in txt
    assert "MAE" in txt and "RMSE" in txt and "MAPE" in txt


# ── the command line ─────────────────────────────────────────────────────────
def _run(*argv):
    import io
    import sys

    from drtran.cli import main

    so, se = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        code = main(list(argv))
        return code, sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = so, se


def test_estwin_needs_a_horizon():
    code, _out, err = _run(ES, WTI, "-b", "0", "-r", "0", "-s", "1",
                           "-estwin", "200")
    assert code == 1 and "needs a horizon" in err


def test_a_window_that_leaves_nothing_out_of_sample_is_refused():
    code, _out, err = _run(ES, WTI, "-b", "0", "-r", "0", "-s", "1",
                           "-estwin", "216", "-f", "6")
    assert code == 1 and "no data out of sample" in err


def test_the_cli_reports_the_window_and_the_evaluation(tmp_path):
    csv_path = str(tmp_path / "roll.csv")
    code, out, err = _run(ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-V",
                          "-estwin", "200", "-f", "6", "-C", csv_path,
                          "-Q", "-o", "-")
    assert code == 0
    assert "estimation window 1..200" in err
    assert "-666.573252" in out, "the fit must use the window, not the full sample"
    assert "RECURSIVE FORECAST EVALUATION" in out
    assert "Origins  : 11" in out
    assert "0.150269" in out                    # MAE at h=1, the C's own
    assert os.path.exists(csv_path)
    assert len(list(csv.DictReader(open(csv_path)))) == 66
