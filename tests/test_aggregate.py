"""Accounting identities and their band.

The identity itself is arithmetic on the answer. What is not trivial is the
band: the series' forecast errors are CORRELATED — they share innovations
through the network — so the aggregate's variance is `c'Vc`, not the sum of the
variances. And since the series are modelled transformed while the identity
lives in levels, the covariance has to be carried across by the delta method.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.aggregate import (Aggregate, forecast_aggregates,  # noqa: E402
                              read_aggregates, report_aggregates)
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit  # noqa: E402
from drtran.forecast import forecast, to_level  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def fc_cs():
    cs = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return forecast(fit(cs, embed=True), L=6), cs


# ── the file ─────────────────────────────────────────────────────────────────
def test_signs_attached_or_separate(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("TOTAL = + ES_CPI + WTI\n"
                 "SPREAD = +WTI -ES_CPI\n"
                 "# a comment\n\n")
    ags = read_aggregates(str(p), ["ES_CPI", "WTI"])
    assert [a.name for a in ags] == ["TOTAL", "SPREAD"]
    assert list(ags[0].coef) == [1.0, 1.0]
    assert list(ags[1].coef) == [-1.0, 1.0]


def test_an_unknown_series_is_an_error_not_a_dropped_term(tmp_path):
    """A dropped term turns an identity into a DIFFERENT identity that still
    adds up, which is the failure nobody notices."""
    p = tmp_path / "a.txt"
    p.write_text("TOTAL = + ES_CPI + NOTHING\n")
    with pytest.raises(ValueError, match="unknown series"):
        read_aggregates(str(p), ["ES_CPI", "WTI"])


def test_a_malformed_line_is_rejected(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("TOTAL + ES_CPI\n")
    with pytest.raises(ValueError, match="expected NAME"):
        read_aggregates(str(p), ["ES_CPI", "WTI"])

    p.write_text("EMPTY = \n")
    with pytest.raises(ValueError, match="no terms"):
        read_aggregates(str(p), ["ES_CPI", "WTI"])


# ── against the binary ───────────────────────────────────────────────────────
def test_the_aggregate_matches_the_C(fc_cs):
    """`TOTAL = ES_CPI + WTI`, which the C reports as
    142.7742 (5.1534), 143.0449 (8.5420), 143.4887 (11.2392)."""
    fc, cs = fc_cs
    a = forecast_aggregates(fc, cs, [Aggregate("TOTAL", np.array([1.0, 1.0]))])[0]
    assert a.level[:3] == pytest.approx([142.7742, 143.0449, 143.4887], abs=5e-4)
    assert a.sd[:3] == pytest.approx([5.1534, 8.5420, 11.2392], abs=5e-4)
    assert a.lower[0] == pytest.approx(132.6736, abs=5e-4)
    assert a.upper[0] == pytest.approx(152.8749, abs=5e-4)


def test_the_point_is_just_the_identity(fc_cs):
    """No modelling in it: the aggregate IS the linear combination of the level
    forecasts. Putting the identity in the model instead would add a series that
    is a linear combination of the others and make the likelihood singular."""
    fc, cs = fc_cs
    lv = np.column_stack([to_level(fc, cs, series=i) for i in range(2)])
    a = forecast_aggregates(fc, cs, [Aggregate("T", np.array([1.0, -1.0]))])[0]
    assert a.level == pytest.approx(lv[:, 0] - lv[:, 1])


# ── the band, which is the whole point ───────────────────────────────────────
def test_the_band_accounts_for_the_correlation(fc_cs):
    """Treating the series as independent understates the band — and in the
    direction that flatters the model. Measured here: 2 to 3 % too narrow, and
    growing with the horizon, on a two-series system with one link. On a real
    network of sectors moving together it is worse."""
    from drtran.aggregate import _jacobian

    fc, cs = fc_cs
    a = forecast_aggregates(fc, cs, [Aggregate("TOTAL", np.array([1.0, 1.0]))])[0]

    J = np.column_stack([_jacobian(to_level(fc, cs, series=i),
                                   cs.series[i].spec.model) for i in range(2)])
    naive = np.array([
        np.sqrt(sum(J[l - 1, i] ** 2 * fc.var_level[l][i, i] for i in range(2)))
        for l in range(1, fc.L + 1)])

    assert np.all(naive < a.sd), "ignoring the covariance must understate it"
    assert np.all(naive / a.sd < 0.99)
    assert a.sd[-1] / naive[-1] > a.sd[0] / naive[0], "and it worsens with h"


def test_the_delta_method_carries_the_transform(fc_cs):
    """The series are modelled in logs and the identity is in levels, so the
    covariance needs dz/db = level/refactor. Without it the band would be in the
    transformed scale and about two orders of magnitude too small here."""
    from drtran.aggregate import _jacobian

    fc, cs = fc_cs
    model = cs.series[0].spec.model
    lv = to_level(fc, cs, series=0)
    assert abs(model.boxlam) < 1e-8, "the canonical case is a log model"
    assert _jacobian(lv, model) == pytest.approx(lv / model.refactor)

    a = forecast_aggregates(fc, cs, [Aggregate("T", np.array([1.0, 0.0]))])[0]
    # a one-series aggregate must reproduce that series' own level band
    from drtran.forecast import level_band
    _lv, lo, hi = level_band(fc, cs, series=0)
    assert a.level == pytest.approx(lv)
    # the delta-method band is symmetric where level_band's is not, but they
    # must agree to first order
    assert a.sd[0] == pytest.approx((hi[0] - lo[0]) / (2 * 1.96), rel=0.01)


def test_the_report_says_where_the_band_comes_from(fc_cs):
    fc, cs = fc_cs
    txt = report_aggregates(
        forecast_aggregates(fc, cs, [Aggregate("TOTAL", np.array([1.0, 1.0]))]))
    assert "AFTER forecasting" in txt
    assert "c'Vc" in txt and "correlation" in txt
    assert "142.7742" in txt
