"""Transfer adequacy: the check on the ESTIMATED model.

Two things are verified here, and only the second is about this port:

1. that the portmanteau reproduces the C's, digit for digit;
2. that it is fed the **structural** residuals, not the reduced-form ones.

The second is the trap. With a contemporaneous link (b=0) the embedded cast puts
omega_0 at lag zero, so Phi(0) != I and the reduced-form residuals come out
correlated **by construction** (Sigma_12 = omega_0 * sigma^2_X). Testing adequacy
on those measures the correlation the transfer itself creates and calls it
misspecification: p collapses to 0.0000 where the C reports 0.1966 — a correct
model condemned by its own transfer.
"""

import os
import re
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.diagnose import (chi_test, report_adequacy,  # noqa: E402
                             transfer_adequacy)
from drtran.estimate import fit  # noqa: E402

REPO = "/home/david/Dropbox/SRC/drtran"
CASES = os.path.join(REPO, "tests/cases")
BIN = os.path.join(REPO, "bin/drtran")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CASES, "ES_CPI_m10.pre")),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def ajuste():
    """The canonical case, fitted: ES_CPI <- WTI with (b, r, s) = (0, 0, 1)."""
    cs = build_cast_spec([drtran.load_pre(os.path.join(CASES, "ES_CPI_m10.pre")),
                          drtran.load_pre(os.path.join(CASES, "WTI_ar1.pre"))],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return fit(cs, embed=True)


# ── the statistic ────────────────────────────────────────────────────────────
def test_the_divisor_is_n_minus_i_plus_one():
    """`ChiTestC` divides by n−i+1, not n−i. It looks like a detail; it is the
    difference between matching the C and not matching it."""
    r = np.array([0.0, 0.3, -0.2, 0.1])
    n = 100
    Q, df = chi_test(r, n, first=1)
    esperado = n * (n + 2) * sum(r[i] ** 2 / (n - i + 1) for i in (1, 2, 3))
    assert df == 3
    assert Q == pytest.approx(esperado)


def test_the_transfer_test_includes_the_contemporaneous_lag():
    """k >= 0 for the transfer (lag zero belongs to it: that is where omega_0
    acts); k < 0 for exogeneity, skipping lag zero."""
    r = np.array([0.5, 0.1, 0.1])
    con, ncon = chi_test(r, 100, first=0)
    sin, nsin = chi_test(r, 100, first=1)
    assert ncon == 3 and nsin == 2
    assert con > sin, "with first=0 the contemporaneous lag must contribute"


# ── the residuals it is fed ──────────────────────────────────────────────────
def test_it_uses_the_structural_residuals_not_the_reduced_form(ajuste):
    """The trap of the module docstring, made into a test.

    With b=0 the two residual sets differ, and using the wrong one turns an
    adequate transfer into an inadequate one.
    """
    from drtran.netid import residuals

    red, i1 = residuals(ajuste.x, ajuste.cast_spec, structural=False)
    est, i2 = residuals(ajuste.x, ajuste.cast_spec, structural=True)
    assert i1 == 0 and i2 == 0
    assert not np.allclose(red, est), "with b=0 Phi(0) != I, so they must differ"

    # the contemporaneous correlation the transfer itself creates
    r_red = np.corrcoef(red[:, 0], red[:, 1])[0, 1]
    r_est = np.corrcoef(est[:, 0], est[:, 1])[0, 1]
    assert abs(r_red) > abs(r_est), (
        "the reduced-form residuals carry the correlation omega_0 puts there")


# ── homologation with the binary ─────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_adequacy_and_exogeneity_match_the_binary(ajuste, tmp_path):
    """Run live against the C rather than against frozen numbers."""
    out = str(tmp_path / "c.out")
    r = subprocess.run([BIN, os.path.join(CASES, "ES_CPI_m10.pre"),
                        os.path.join(CASES, "WTI_ar1.pre"),
                        "-b", "0", "-r", "0", "-s", "1", "-V", "-o", out],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    p_tr = float(re.search(r"Transfer adequacy\s*:\s*\w+\s*\(p = ([\d.]+)\)",
                           r.stdout).group(1))
    p_ex = float(re.search(r"Input exogeneity\s*:\s*\w+\s*\(p = ([\d.]+)\)",
                           r.stdout).group(1))

    ad = transfer_adequacy(ajuste)
    assert ad.p_transfer == pytest.approx(p_tr, abs=1e-4)
    assert ad.p_exog == pytest.approx(p_ex, abs=1e-4)
    assert ad.adequate and ad.exogenous


def test_the_verdict_comes_from_the_joint_test_not_from_isolated_peaks(ajuste):
    """With 5 % bands ~1 lag in 20 crosses by chance. Demanding zero significant
    lags would condemn a correct specification, so the report mentions them and
    the portmanteau decides."""
    ad = transfer_adequacy(ajuste)
    txt = report_adequacy(ad)
    assert ad.adequate
    if ad.significant_lags:
        assert "would be expected by chance" in txt
        assert "do not contradict the joint test" in txt
    assert "ADEQUATE" in txt and "Exogeneity" in txt
