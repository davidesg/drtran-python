"""THE GATE: the diagonal joint fit must reproduce the sum of the univariate ones.

It is the design's validation criterion, and the entry condition to everything
else. With a diagonal structure (diagonal AR/MA and covariance, no transfer) the
exact likelihood factorises, so the joint fit has to give the SUM.

If it fails, the cast is wrong — **never** `elf`, which is used as it comes.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (build_cast_spec, cast_diagonal,  # noqa: E402
                         loglik_diagonal, x0_from_pre)

C = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")
TARGET = -767.424341

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def cs():
    return build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])


def test_the_structure_is_diagonal_and_carries_the_pre_arma(cs):
    phi, theta, mu, w, sigma, ifault = cast_diagonal(x0_from_pre(cs), cs)
    assert ifault == 0
    assert w.shape == (215, 2)          # 216 obs, d=1 in both
    assert phi.shape == (1, 2, 2) and theta.shape == (0, 2, 2)
    # a diagonal AR, with each .pre's phi in its own cell
    assert phi[0][0, 0] == pytest.approx(0.4028)
    assert phi[0][1, 1] == pytest.approx(0.2992)
    assert phi[0][0, 1] == 0.0 and phi[0][1, 0] == 0.0
    # means: Y's is free, X's is fixed at 0 by its .pre
    assert mu[0] == pytest.approx(0.154472)
    assert mu[1] == 0.0
    # normalised covariance: Q[0][0] = 1, sigma2 concentrates the scale
    assert sigma[0, 0] == pytest.approx(1.0)
    assert sigma[0, 1] == 0.0 and sigma[1, 0] == 0.0


def test_the_pre_seeds_already_are_the_diagonal_optimum(cs):
    """And that is why the C reports termcode 3 here: starting at the optimum,
    the line search cannot improve and stops. It is not a failure."""
    ll, ifault = loglik_diagonal(x0_from_pre(cs), cs)
    assert ifault == 0
    assert ll == pytest.approx(TARGET, abs=1e-4)


def test_the_variance_ratio_is_seeded_not_left_at_one(cs):
    """The scales differ by a factor of ~1098 (sigma2 0.0627 vs 68.84). Seeding
    log(var2/var1)=0 leaves the starting point at logL -1371 instead of -767."""
    x0 = x0_from_pre(cs)
    assert float(x0[-1]) == pytest.approx(np.log(68.8381 / 0.062666), abs=1e-3)

    bad = x0.copy()
    bad[-1] = 0.0
    ll_bad, _ = loglik_diagonal(bad, cs)
    assert ll_bad < TARGET - 100, "without seeding the ratio the start is far worse"


def test_omega_zero_splits_the_model_into_two_univariate_ones(cs):
    """The proof that the bridge is right (BRIDGE_DESIGN): with no transfer the
    VARMA factorises and the joint likelihood IS the sum of the two univariate
    ones."""
    import fue

    total = 0.0
    for p in (ES_CPI, WTI):
        _ts, m = fue.load(p)
        m.fit()
        total += float(m.loglik)

    ll, _ = loglik_diagonal(x0_from_pre(cs), cs)
    assert ll == pytest.approx(total, abs=1e-4)
