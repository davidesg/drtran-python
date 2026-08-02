"""Identifying (b, r, s) by prewhitening and the CCF, against the C binary.

Reference, from `./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/WTI_ar1.pre`:

    CCF BANDS  2.0/SQRT(N) =  0.13640
      0   0.492      1   0.310      2   0.025     -1  -0.077     -6  -0.128
    Exogeneity: Q(24) = 18.2969   p = 0.7884   [0 significant out of 24]
    PROPOSALS: [A] b=0 r=0 s=1     RECOMMENDED: b=0, r=0, s=1
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.identify import (identify, prewhiten,  # noqa: E402
                             report_identification)

C = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def ident():
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    return identify(cs, Link(out=0, inp=1))


def test_the_band_matches_the_C(ident):
    assert ident.threshold == pytest.approx(0.13640, abs=1e-5)


def test_the_ccf_matches_the_C(ident):
    mid = len(ident.ccf) // 2
    assert ident.ccf[mid] == pytest.approx(0.492, abs=1e-3)       # k=0
    assert ident.ccf[mid + 1] == pytest.approx(0.310, abs=1e-3)   # k=1
    assert ident.ccf[mid + 2] == pytest.approx(0.025, abs=1e-3)   # k=2
    assert ident.ccf[mid - 1] == pytest.approx(-0.077, abs=1e-3)  # k=-1
    assert ident.ccf[mid - 6] == pytest.approx(-0.128, abs=1e-3)  # k=-6


def test_the_proposal_matches_the_C(ident):
    assert (ident.b, ident.r, ident.s) == (0, 0, 1)


def test_the_exogeneity_portmanteau_matches_the_C(ident):
    """Q uses `diagnose.c:ChiTestC`'s divisor (n - i + 1), not (n - i)."""
    assert ident.Q_exogeneity == pytest.approx(18.2969, abs=1e-3)
    assert ident.p_exogeneity == pytest.approx(0.7884, abs=1e-3)
    assert ident.n_signif_negative == 0
    assert ident.exogenous is True


def test_only_the_contiguous_block_from_b_counts(ident):
    """There is a significant peak at lag 24 (sampling noise: with 5% bands one
    lag in 20 is expected outside). It must NOT enter the proposal: taking the
    last significant lag of the whole CCF sent s to 24."""
    mid = len(ident.ccf) // 2
    assert abs(ident.ccf[mid + 24]) > ident.threshold, "the distant peak is there"
    assert ident.s == 1, "but the proposal stays with the contiguous block 0..1"


def test_prewhitening_a_known_ar1():
    """Filtering an AR(1) by its own phi leaves white noise."""
    rng = np.random.default_rng(0)
    n, phi = 400, 0.6
    e = rng.normal(0, 1, n)
    w = np.zeros(n)
    for t in range(1, n):
        w[t] = phi * w[t - 1] + e[t]
    a = prewhiten(w, [phi], [])
    assert a[1:] == pytest.approx(e[1:], abs=1e-9)


def test_prewhitening_inverts_the_ma():
    """With theta, a[t] = u[t] + SUM theta_k a[t-k] undoes the MA."""
    w = np.array([1.0, 0.0, 0.0, 0.0])
    a = prewhiten(w, [], [0.5])
    assert a == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_it_detects_a_synthetic_transfer_with_a_delay():
    """A known truth: Y_t = 2*X_{t-3} + noise, with X white. b=3 must come out."""
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    y[3:] = 2.0 * x[:-3]
    y += rng.normal(0, 0.3, n)

    from drtran.identify import _ccf
    thr = 2.0 / np.sqrt(n)
    c = _ccf(x, y, 12)
    sig = [k for k in range(13) if abs(c[k]) > thr]
    assert sig and sig[0] == 3, f"the first significant lag must be 3: {sig}"


def test_the_report_mentions_the_essentials(ident):
    txt = report_identification(ident, ("WTI", "ES_CPI"))
    assert "b=0" in txt and "r=0" in txt and "s=1" in txt
    assert "18.2969" in txt or "18.297" in txt
    assert "exogenous" in txt


# ── what the original paper says (Haugh & Box 1977) ──────────────────────────
def test_the_haugh_box_band_removes_the_false_positive_at_lag_24():
    """Haugh & Box (1977) §1.4: var{r_xy(k)} ~ (N-k)^-1, not N^-1.

    The band must WIDEN with the lag. With N=215 the band at k=24 goes from
    0.1364 to 0.1447, and the peak of -0.1423 stops being significant. That peak
    is precisely the false positive that forced the C into the contiguous-block
    heuristic: the heuristic compensates for a badly calibrated band.
    """
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    const = identify(cs, Link(out=0, inp=1), band="constant")
    hb = identify(cs, Link(out=0, inp=1), band="haugh-box")

    assert 24 in const.significant_non_negative
    assert 24 not in hb.significant_non_negative
    assert hb.significant_non_negative == [0, 1]
    # the proposal does not change: the heuristic was already covering it up
    assert (const.b, const.r, const.s) == (hb.b, hb.r, hb.s) == (0, 0, 1)


def test_the_haugh_box_band_grows_with_the_lag():
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    hb = identify(cs, Link(out=0, inp=1), band="haugh-box")
    mid = len(hb.lags) // 2
    assert hb.bands[mid] == pytest.approx(2 / np.sqrt(215), abs=1e-6)
    assert hb.bands[mid + 24] == pytest.approx(2 / np.sqrt(215 - 24), abs=1e-6)
    assert hb.bands[mid + 24] > hb.bands[mid]
    # and it is symmetric
    assert hb.bands[mid - 24] == pytest.approx(hb.bands[mid + 24])


def test_the_default_still_homologates_with_the_C():
    """The default is the constant band, which is what the binary does."""
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    i = identify(cs, Link(out=0, inp=1))
    assert i.threshold == pytest.approx(0.13640, abs=1e-5)
