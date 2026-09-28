"""Shea's exact likelihood (AS 242) as the C's -l: shea as the objective, both
as a check of elf at every point the optimizer visits (atsw-gui lib/lik; the C
battery's section 18 pins the same numbers)."""

import io
import os
import sys

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("drvarma._engine").marma_c                     # noqa: B018
from drtran.cli import CliError, main  # noqa: E402
from drtran.estimate import Fit, _ma_boundary_at, fit  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")
pytestmark = pytest.mark.skipif(not os.path.exists(ES),
                                reason="the canonical .pre files are missing")


def run(*argv):
    so, se = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        code = main(list(argv))
        return code, sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = so, se


def test_shea_is_the_homologation_with_fue():
    """-0 -l shea lands where elf does: the sum of fue's univariate logLs."""
    code, out, _ = run(ES, WTI, "-0", "-l", "shea", "-o", "-")
    assert code == 0
    assert "-767.4243" in out
    assert "likelihood    : exact, Shea (1989), AS 242" in out


def test_both_agrees_at_every_point():
    """As the C on the same run (1078 points, max 4.3e-07, 1.4e-12 at the
    optimum): far from the optimum the two differ by rounding in a large
    logL; at the optimum they are the same number."""
    code, out, _ = run(ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-l", "both", "-o", "-")
    assert code == 0
    line = next(l for l in out.splitlines() if "Shea check" in l)
    mx = float(line.split("max |dlogL| = ")[1].split(",")[0])
    at_opt = float(line.split("at the optimum ")[1].split(";")[0])
    assert mx < 1e-5 and at_opt < 1e-9
    assert "admissible for one only" not in line
    assert "-718.287406" in out


def test_elf_report_is_unchanged():
    code, out, _ = run(ES, WTI, "-0", "-o", "-")
    assert code == 0 and "likelihood    :" not in out and "Shea check" not in out


def test_dash_l_refuses_the_unknown():
    with pytest.raises(CliError):
        main([ES, WTI, "-0", "-l", "exact"])


def test_fit_refuses_the_unknown():
    with pytest.raises(ValueError):
        fit(None, lik="exact")


def test_a_stop_on_the_wall_is_not_convergence():
    f = Fit(x=np.zeros(1), loglik=0.0, ifault=0, termcode=1, nit=40,
            cast_spec=None, converged=False, ma_boundary=1, ma_nroots=3)
    assert f.status == "STOPPED AT THE MA INVERTIBILITY BOUNDARY"
    assert f.convergence_note.startswith("MA boundary: 1 of 3 inverse roots within 5e-5 of the unit circle")


def test_the_wall_has_one_tolerance_for_both_sides():
    """m6 EP <- EC: the port ends with theta_1 = 0.9999999926, the C with
    1.000048. Both are within 5e-5 of the unit circle, so both say so."""
    P = "/home/david/Dropbox/SRC/atsw-gui/engines/drtran/tests/data/m6/"
    if not os.path.exists(P + "M6_EP.pre"):
        pytest.skip("the m6 .pre files are missing")
    code, out, _ = run(P + "M6_EP.pre", P + "M6_EC.pre", "-b", "0", "-r", "0", "-s", "1", "-o", "-")
    assert code == 0
    assert "STOPPED AT THE MA INVERTIBILITY BOUNDARY" in out
    assert "MA boundary: 1 of 2 inverse roots within 5e-5 of the unit circle" in out
