"""The command line: the C's own options, and the numbers it must reproduce.

Two kinds of test live here. The cheap ones pin the PARSING — option letters,
comma-separated orders, the refusal of what is not ported — because a CLI's job
is to turn a command line into exactly one meaning, and every silent
mistranslation shows up as a plausible number nobody can trace back.

The expensive one runs the canonical case end to end and compares against the C
binary's own report. That is the only test that proves the wiring is right: each
piece is checked in its own module, and the CLI is what puts them in the right
order.
"""

import io
import os
import sys

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cli import (CliError, _dates, _orders,  # noqa: E402
                        _series_of, main)

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

needs_cases = pytest.mark.skipif(
    not os.path.exists(ES), reason="the canonical .pre files are missing")


class _Capture:
    """Run `main` with stdout/stderr captured; returns (code, out, err)."""

    def __call__(self, *argv):
        so, se = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            code = main(list(argv))
            return code, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = so, se


run = _Capture()


# ── parsing ──────────────────────────────────────────────────────────────────
def test_a_single_order_applies_to_every_input():
    """`-b 1` with three inputs means all three, not the first."""
    assert _orders("1", 3, "b") == [1, 1, 1]
    assert _orders("1,0,2", 3, "b") == [1, 0, 2]


def test_a_wrong_count_of_orders_is_rejected():
    with pytest.raises(CliError):
        _orders("1,0", 3, "b")
    with pytest.raises(CliError):
        _orders("x", 1, "b")
    with pytest.raises(CliError):
        _orders("-1", 1, "b")


def test_the_series_index_is_read_off_the_slot_name():
    """The bulk switches (-N/-X/-D/-E) select by series, and the only thing that
    says which series a slot belongs to is its name."""
    assert _series_of("phi_1[B^1]") == 1
    assert _series_of("theta_12[B^12]") == 12
    assert _series_of("omega_d2[3,0]") == 2
    assert _series_of("mu[3]") == 3
    assert _series_of("omega1[0]") == 0        # a TRANSFER, not a series
    assert _series_of("q[2,1]") == 0
    assert _series_of("log(var2/var1)") == 0


def test_the_transfer_omega_is_not_mistaken_for_a_deterministic_one():
    """`omega1[0]` is the transfer's; `omega_d1[1,0]` is a deterministic
    variable's. They differ by two characters and -D would fix the wrong one."""
    assert _series_of("omega1[0]") == 0
    assert _series_of("omega_d1[1,0]") == 1


def test_the_forecast_dates_continue_the_calendar():
    class _TS:
        start, freq, nobs = (2002, 1), 12, 216

    # 216 monthly observations from 1/2002 end in 12/2019
    assert _dates(_TS(), None, 3) == [" 1/2020", " 2/2020", " 3/2020"]
    # an earlier origin moves the whole block back
    assert _dates(_TS(), 210, 2) == [" 7/2019", " 8/2019"]

    class _Yearly:
        start, freq, nobs = (2000, 1), 1, 10

    assert _dates(_Yearly(), None, 2) == ["2010", "2011"]


def test_a_series_with_no_calendar_falls_back_to_the_index():
    class _TS:
        start, freq, nobs = None, 0, 50

    assert _dates(_TS(), None, 2) == ["51", "52"]


# ── the interface contract ───────────────────────────────────────────────────
def test_help_exits_clean():
    code, out, _err = run("-h")
    assert code == 0
    assert "Box-Jenkins transfer function" in out


def test_what_is_not_ported_is_refused_not_ignored():
    """The alternative is worse than an error: a script that passed -a to the C
    and gets no aggregates here would report a forecast that answers a different
    question, with no sign that anything was dropped."""
    code, _out, err = run(ES, WTI, "-a", "x.txt")
    assert code == 2 and "not ported" in err
    code, _out, err = run(ES, WTI, "-L")
    assert code == 2 and "not ported" in err


def test_estwin_is_the_same_option_as_R():
    """The C rewrites the token before getopt; so does this, or a command line
    written for one would silently mean something else in the other. Checked on
    the cheap failure path, so the test does not pay for two full estimations."""
    a = run(ES, WTI, "-estwin", "200")
    b = run(ES, WTI, "-R", "200")
    assert a[0] == b[0] == 1
    assert "needs a horizon" in a[2] and "needs a horizon" in b[2]


def test_prewhiten_only_reports_and_does_not_estimate():
    """-p is the cheap look before committing to a model. It is also the only
    path that reaches `identify.report`, and it broke once when that function's
    keyword was renamed and nothing exercised it."""
    code, out, _err = run(ES, WTI, "-p")
    assert code == 0
    assert "IDENTIFICATION" in out and "prewhitening" in out
    assert "b=0" in out and "s=1" in out                 # the C's own proposal
    assert "JOINT ESTIMATION" not in out, "-p must not estimate"


def test_the_options_may_follow_the_files():
    """gnu_getopt, not getopt: every real invocation puts the .pre files first,
    and plain getopt would stop parsing at the first one."""
    code, _out, err = run(ES, "-b", "0")
    assert code == 1
    assert "two .pre files" in err          # got as far as the file check


def test_missing_files_are_reported_by_name():
    code, _out, err = run("no_such.pre", WTI)
    assert code == 1 and "no_such.pre" in err


# ── end to end, against the binary ───────────────────────────────────────────
@needs_cases
def test_the_canonical_run_reproduces_the_C(tmp_path):
    """`drtran ES_CPI WTI -b 0 -r 0 -s 1 -V -f 6`, the case the whole port is
    homologated on. Every figure here is read off the C's own .out."""
    outfile = tmp_path / "canon.out"
    code, out, _err = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1",
                          "-V", "-f", "6", "-o", str(outfile))
    assert code == 0
    assert outfile.exists(), "-o must write the file, not only print"

    text = outfile.read_text()
    assert text == out

    assert "-718.287406" in text                       # the log-likelihood
    assert "0.016400" in text and "-0.010747" in text  # omega(B)

    # the forecast table, level and s.e., as the C publishes them. Compared as
    # NUMBERS: the C rounds to two decimals and this prints four, so `in` would
    # pass on 82.01 and fail on 83.44 for no reason but the rounding.
    blocks = text.split("FORECAST REPORT —")
    assert len(blocks) == 3, "one block per series"

    def read_row(block, date):
        """(level, s.e.) of a forecast row: DATE | LEVEL STD | PERIOD STD | ..."""
        for l in block.splitlines():
            if date in l and "|" in l:
                fields = l.replace("|", " ").split()
                if len(fields) >= 3 and fields[2] != "-":
                    return float(fields[1]), float(fields[2])
        raise AssertionError(date)

    for date, level, se in [(" 1/2020", 82.01, 0.24),
                             (" 2/2020", 82.02, 0.44),
                             (" 6/2020", 83.44, 0.95)]:
        v, e = read_row(blocks[1], date)                    # ES_CPI
        assert v == pytest.approx(level, abs=0.005)
        assert e == pytest.approx(se, abs=0.005)

    for date, level, se in [(" 1/2020", 60.76, 8.29),
                             (" 6/2020", 61.14, 27.10)]:
        v, e = read_row(blocks[2], date)                    # WTI, the input
        assert v == pytest.approx(level, abs=0.005)
        assert e == pytest.approx(se, abs=0.005)

    assert "ADEQUATE" in text                            # the diagnostics ran


@needs_cases
def test_dash_zero_is_the_homologation_with_fue(tmp_path):
    """-0 must land on the diagonal log-likelihood, which is fue run separately
    on each series. It is the gate the whole cast is built on."""
    code, out, _err = run(ES, WTI, "-0", "-o", "-")
    assert code == 0
    assert "-767.424341" in out
    assert "omega1[" not in out, "-0 means NO transfer at all"


@needs_cases
def test_the_bulk_switches_fix_the_right_series(tmp_path):
    """-N fixes the OUTPUT's ARMA and -M the means; the input's ARMA must stay
    free. Selecting by name is what makes this fragile, hence the test."""
    code, out, _err = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1",
                          "-N", "-M", "-o", "-")
    assert code == 0
    lines = {l.split()[0]: l for l in out.splitlines()
              if l.startswith("  ") and "[" in l}
    assert "(fixed)" in lines["phi_1[B^1]"]
    assert "(fixed)" in lines["mu[1]"]
    assert "(fixed)" not in lines["phi_2[B^1]"], "the INPUT's ARMA stays free"
    assert "(fixed)" not in lines["omega1[0]"], "the transfer stays free"


@needs_cases
def test_guided_writes_files_that_the_program_reads_back(tmp_path):
    """-g is only useful if its output is a valid input to -n/-c. The round trip
    is the whole point of the guided mode."""
    name = str(tmp_path / "guide")
    code, out, _err = run(ES, WTI, "-g", name, "-o", "-")
    assert code == 0
    assert os.path.exists(name + ".dag") and os.path.exists(name + ".cns")
    assert "ES_CPI <- WTI" in out

    code2, out2, _err2 = run(ES, WTI, "-n", name + ".dag",
                             "-c", name + ".cns", "-o", "-")
    assert code2 == 0
    assert "JOINT ESTIMATION" in out2


@needs_cases
def test_a_cycle_in_the_network_is_refused():
    """A cyclic system is simultaneous and cannot be cast as a triangular VARMA.
    Refusing is the only correct answer."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".dag", delete=False) as fh:
        fh.write("ES_CPI <- WTI   1 0 0\nWTI <- ES_CPI   1 0 0\n")
        ruta = fh.name
    try:
        code, _out, err = run(ES, WTI, "-n", ruta)
        assert code == 1
        assert "ciclo" in err or "cycle" in err.lower()
    finally:
        os.unlink(ruta)


def test_the_report_carries_the_variation_columns_and_the_decomposition():
    """The two sections the CLI was missing against the C: the PERIOD/ANNUAL
    variation columns with their standard errors, and the forecast error
    variance decomposition. Both checked against the binary's own figures."""
    code, out, _err = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-f", "6",
                          "-Q", "-o", "-")
    assert code == 0
    assert "PERIOD" in out and "ANNUAL" in out

    # 1/2020: level 82.01 (0.24), period -1.00 (0.24), annual 1.07 (0.24)
    es_block = out.split("FORECAST REPORT —")[1]     # ES_CPI's, not WTI's
    row = [l for l in es_block.splitlines()
           if l.strip().startswith("1/2020") and "|" in l
           and l.replace("|", " ").split()[1] != "-"]
    assert row, "the first forecast row is missing"
    nums = [float(x) for x in row[0].replace("|", " ").split()[1:]]
    assert nums[0] == pytest.approx(82.01, abs=0.005)
    assert nums[1] == pytest.approx(0.24, abs=0.005)
    assert nums[2] == pytest.approx(-1.00, abs=0.005)
    assert nums[3] == pytest.approx(0.24, abs=0.005)
    assert nums[4] == pytest.approx(1.07, abs=0.005)

    assert "FORECAST ERROR VARIANCE DECOMPOSITION" in out
    assert "own noise" in out
    assert "68.2%" in out and "31.8%" in out       # h=1, the C's own numbers
    assert "46.3%" in out and "53.7%" in out       # h=6
