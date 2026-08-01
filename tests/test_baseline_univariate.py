"""The baseline the cast will be validated against.

drtran's validation criterion (the gate to everything else) is:

    **Diagonal joint estimation == fue run separately.**

With a diagonal structure (diagonal AR/MA and covariance, no transfer) the exact
likelihood factorises, so the joint fit must reproduce the SUM of the two
univariate ones. These tests pin that sum.

If one of them fails, it does not mean the cast is wrong: it means the reference
moved, and there is nothing left to validate anything against. That is why they
live apart.

Reference values: BRIDGE_DESIGN.md in the C repo, canonical case
ES_CPI_m10 (Y) <- WTI_ar1 (X), verified there on 2026-07-12 against the C.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")

C_EXAMPLES = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C_EXAMPLES, "ES_CPI_m10.pre")
WTI = os.path.join(C_EXAMPLES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI),
    reason="the canonical .pre files from the C repo are missing")

# BRIDGE_DESIGN.md, the canonical case's table
REF = {
    "ES_CPI": dict(phi=0.402839, mu=0.154472, loglik=-7.3917),
    "WTI": dict(phi=0.299193, mu=0.0, loglik=-760.0326),
}
JOINT_LOGLIK = -767.424341        # the diagonal joint estimation's target


def _fit(path):
    s = drtran.load_pre(path)
    s.model.fit()
    return s.model


def test_es_cpi_reproduces_the_reference():
    m = _fit(ES_CPI)
    assert float(np.atleast_1d(m.ar[0])[0]) == pytest.approx(REF["ES_CPI"]["phi"],
                                                             abs=1e-5)
    assert float(m.mu0) == pytest.approx(REF["ES_CPI"]["mu"], abs=1e-5)
    assert float(m.loglik) == pytest.approx(REF["ES_CPI"]["loglik"], abs=1e-3)


def test_wti_reproduces_the_reference():
    m = _fit(WTI)
    assert float(np.atleast_1d(m.ar[0])[0]) == pytest.approx(REF["WTI"]["phi"],
                                                             abs=1e-5)
    assert m.estimate_mu is False, "the .pre fixes mu at 0"
    assert float(m.loglik) == pytest.approx(REF["WTI"]["loglik"], abs=1e-3)


def test_the_sum_is_the_joint_fit_target():
    """THE test: the sum of the two univariate likelihoods is what the diagonal
    joint fit has to reproduce. Measured: -767.424341, differing by 7.6e-09."""
    total = float(_fit(ES_CPI).loglik) + float(_fit(WTI).loglik)
    assert total == pytest.approx(JOINT_LOGLIK, abs=1e-4)
