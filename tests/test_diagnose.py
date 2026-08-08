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

import pytest
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
def fitted():
    """The canonical case, fitted: ES_CPI <- WTI with (b, r, s) = (0, 0, 1)."""
    cs = build_cast_spec([drtran.load_pre(os.path.join(CASES, "ES_CPI_m10.pre")),
                          drtran.load_pre(os.path.join(CASES, "WTI_ar1.pre"))],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return fit(cs, embed=True)


# ── the statistic ────────────────────────────────────────────────────────────
def test_the_divisor_is_n_minus_i_plus_one():
    """`ChiTestC` divides by n−i+1, not n−i. It looks like a detail; it is the
    difference between matching the C and not matching it.

    **Do not "fix" this to n−i.** It was reported as BUG-7 — the argument being
    that `ChiTestC` is 1-based with `corr[1]` contemporaneous, so lag k should
    divide by `n−k` — and the argument is wrong for the k<0 branch, which is
    the one `first=1` serves. Measured against the binary on the canonical case
    (embedded cast, the homologated configuration):

        divisors 215..192  (this code)   Q = 15.2377   <- the C prints 15.2377
        divisors 214..191  (the "fix")   Q = 15.3118

    `test_adequacy_and_exogeneity_match_the_binary` runs the C live and catches
    the change; this test says why before anyone gets there.
    """
    r = np.array([0.0, 0.3, -0.2, 0.1])
    n = 100
    Q, df = chi_test(r, n, first=1)
    expected = n * (n + 2) * sum(r[i] ** 2 / (n - i + 1) for i in (1, 2, 3))
    assert df == 3
    assert Q == pytest.approx(expected)


def test_the_transfer_test_includes_the_contemporaneous_lag():
    """k >= 0 for the transfer (lag zero belongs to it: that is where omega_0
    acts); k < 0 for exogeneity, skipping lag zero."""
    r = np.array([0.5, 0.1, 0.1])
    with_lag0, n_with = chi_test(r, 100, first=0)
    without, n_without = chi_test(r, 100, first=1)
    assert n_with == 3 and n_without == 2
    assert with_lag0 > without, "with first=0 the contemporaneous lag must contribute"


# ── the residuals it is fed ──────────────────────────────────────────────────
def test_it_uses_the_structural_residuals_not_the_reduced_form(fitted):
    """The trap of the module docstring, made into a test.

    With b=0 the two residual sets differ, and using the wrong one turns an
    adequate transfer into an inadequate one.
    """
    from drtran.netid import residuals

    reduced, i1 = residuals(fitted.x, fitted.cast_spec, structural=False)
    struct, i2 = residuals(fitted.x, fitted.cast_spec, structural=True)
    assert i1 == 0 and i2 == 0
    assert not np.allclose(reduced, struct), \
        "with b=0 Phi(0) != I, so they must differ"

    # the contemporaneous correlation the transfer itself creates
    r_red = np.corrcoef(reduced[:, 0], reduced[:, 1])[0, 1]
    r_est = np.corrcoef(struct[:, 0], struct[:, 1])[0, 1]
    assert abs(r_red) > abs(r_est), (
        "the reduced-form residuals carry the correlation omega_0 puts there")


# ── homologation with the binary ─────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_adequacy_and_exogeneity_match_the_binary(fitted, tmp_path):
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

    ad = transfer_adequacy(fitted)
    assert ad.p_transfer == pytest.approx(p_tr, abs=1e-4)
    assert ad.p_exog == pytest.approx(p_ex, abs=1e-4)
    assert ad.adequate and ad.exogenous


def test_the_verdict_comes_from_the_joint_test_not_from_isolated_peaks(fitted):
    """With 5 % bands ~1 lag in 20 crosses by chance. Demanding zero significant
    lags would condemn a correct specification, so the report mentions them and
    the portmanteau decides."""
    ad = transfer_adequacy(fitted)
    txt = report_adequacy(ad)
    assert ad.adequate
    if ad.significant_lags:
        assert "would be expected by chance" in txt
        assert "do not contradict the joint test" in txt
    assert "ADEQUATE" in txt and "Exogeneity" in txt


# ── the scale of the residuals ───────────────────────────────────────────────
def test_the_residuals_are_the_RAW_innovations(fitted):
    """Their sample variance must be Sigma_ii — that is what "innovation" means,
    and it is what makes them comparable with the series' own units.

    This was an open question for a while, because the port's residuals did not
    match the ones the C binary carries in `vf.a`, and the diagnostics could not
    settle it: the CCF is scale-invariant, so a per-series rescaling is exactly
    the error a portmanteau cannot see. The variance can.
    """
    from drtran.embed import cast_embedded
    from drtran.estimate import _f1f2
    from drtran.netid import residuals

    x = np.asarray(fitted.x, float)
    _phi, _th, _mu, w, Q, ifault = cast_embedded(x, fitted.cast_spec)
    assert ifault == 0
    a, ifa = residuals(x, fitted.cast_spec, embed=True)
    assert ifa == 0

    n, m = w.shape
    f1, _f2, _i = _f1f2(x, fitted.cast_spec, -1e-3, True)
    sigma2 = f1 / (n * m)

    # Var(a_i) = sigma2 * Q_ii = Sigma_ii, to sampling error
    for i in range(m):
        assert a[:, i].var() == pytest.approx(sigma2 * Q[i, i], rel=0.01)


def test_the_C_carries_the_STANDARDIZED_ones(fitted):
    """`vf.a` in the binary is `L^-1 a`, with L the Cholesky factor of Q — so its
    variance is sigma2 for EVERY series however different their scales (here they
    differ by a factor of 1180). Pinned because it is why the port's ERR column
    and the C's do not agree, and the difference is deliberate.

    The three values on the right are what instrumenting the binary printed for
    the last three observations of ES_CPI.
    """
    from drtran.embed import cast_embedded
    from drtran.netid import residuals

    x = np.asarray(fitted.x, float)
    _phi, _th, _mu, _w, Q, _i = cast_embedded(x, fitted.cast_spec)
    a, _ifa = residuals(x, fitted.cast_spec, embed=True)

    std = np.linalg.solve(np.linalg.cholesky(Q), a.T).T
    assert std[-3:, 0] == pytest.approx(
        [0.1268127395, -0.0890583542, -0.2053990514], abs=1e-7)

    # and the giveaway: one variance for both series, not two
    assert std[:, 0].var() == pytest.approx(std[:, 1].var(), rel=0.01)
