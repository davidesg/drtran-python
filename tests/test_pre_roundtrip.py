"""The `.pre` must arrive in Python intact.

The `.pre` is the ladder's continuity contract: fue leaves the best estimated
univariate model there and drtran takes it as a seed. If the reader loses a
field, the cast builds a VARMA structure different from the one fue estimated and
the homologation fails for no apparent reason — the worst possible failure mode.

These tests compare against the LITERAL values in the canonical file, not against
whatever the library returns, so that they keep their meaning if fue changes
internally.
"""

import os

import pytest

drtran = pytest.importorskip("drtran")

# The canonical .pre files live in the C repo; they are used from there while
# they are not copied into the Python one (they are the reference, not a copy).
C_EXAMPLES = "/home/david/Dropbox/SRC/drtran/examples"
ES_CPI = os.path.join(C_EXAMPLES, "work", "ES_CPI_m10.pre")
WTI = os.path.join(C_EXAMPLES, "work", "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI),
    reason="the canonical .pre files from the C repo are missing")


def test_it_reads_the_series_and_its_header():
    s = drtran.load_pre(ES_CPI)
    assert s.freq == 12
    assert s.nobs == 216
    assert s.ts.start == (2002, 1)
    assert s.name == "ES_CPI"
    assert len(s.ts.data) == 216
    assert s.ts.data[0] == pytest.approx(58.717)
    assert s.ts.data[-1] == pytest.approx(82.840)


def test_it_reads_the_transformation():
    """lambda, d, D and refactor. `refactor` is critical: at scale 1 the
    optimizer degrades (cdgrad's step ~6e-6 against a Delta-log of ~0.002)."""
    m = drtran.load_pre(ES_CPI).model
    assert m.boxlam == pytest.approx(0.0)
    assert m.d == 1
    assert m.D == 0
    assert m.refactor == pytest.approx(100.0)


def test_it_reads_the_arma_with_its_orders_and_flags():
    m = drtran.load_pre(ES_CPI).model
    assert len(m.ar) == 1                     # one regular AR operator
    assert len(m.ar[0]) == 1                  # of order 1
    assert float(m.ar[0][0]) == pytest.approx(0.4028)
    assert bool(m.ar_free[0][0]) is True      # flag 1 in the .pre = free
    # the other blocks are empty in this .pre
    assert len(m.ar_s) == 0 and len(m.ma) == 0 and len(m.ma_s) == 0
    assert len(m.ar_f) == 0 and len(m.ma_f) == 0


def test_it_reads_the_11_deterministics_with_their_omegas():
    m = drtran.load_pre(ES_CPI).model
    iv = m.interventions
    assert len(iv) == 11
    assert [v.type for v in iv] == (["cos", "sin"] * 5) + ["alter"]
    expected = [-0.161112, -0.166261, 0.289306, -0.577289, 0.086116, -0.033312,
                0.058589, -0.017333, -0.002610, 0.011549, 0.114945]
    assert [float(v.omega[0]) for v in iv] == pytest.approx(expected)
    # the fixed flags have to come through, or we will not know what is free
    assert all(hasattr(v, "omega_free") for v in iv)


def test_it_tells_a_free_mu_from_a_fixed_one():
    """The decisive test. ES_CPI carries a free mu; WTI_ar1 FIXES it at 0.

    If this were lost, the joint fit would free a parameter the method fixes and
    could not homologate with fue.
    """
    free = drtran.load_pre(ES_CPI).model
    assert free.mu0 == pytest.approx(0.154472)
    assert free.estimate_mu is True

    fixed = drtran.load_pre(WTI).model
    assert fixed.mu0 == pytest.approx(0.0)
    assert fixed.estimate_mu is False


def test_the_second_canonical_pre_in_full():
    s = drtran.load_pre(WTI)
    assert s.name == "WTI"
    assert s.nobs == 216
    assert float(s.model.ar[0][0]) == pytest.approx(0.2992)
    assert len(s.model.interventions) == 0
    assert s.model.refactor == pytest.approx(100.0)


def test_it_rejects_what_is_no_use_for_the_joint_fit():
    with pytest.raises(Exception):
        drtran.load_pre(os.path.join(C_EXAMPLES, "does_not_exist.pre"))


def test_it_warns_when_the_scale_degrades_the_optimizer():
    """refactor=100 must not warn; a raw scale must."""
    s = drtran.load_pre(ES_CPI)
    assert drtran.check_scale(s) is None

    class _Fake:
        model = type("M", (), {"refactor": 1.0})()
        name = "raw"

    warning = drtran.check_scale(_Fake())
    assert warning is not None and "refactor=1" in warning
