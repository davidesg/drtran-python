"""The SPS forecast report: fuf's own, adapted, not reimplemented.

fuf is the family's univariate forecast program and its Python port lives inside
fue (`fue.report_forecast`). drtran adapts its joint forecast into the objects
that writer already expects and calls it, so a univariate report from fuf and a
transfer-function report from drtran are literally the same page. The tests here
are mostly about the two things that do NOT carry over for free: the scales, and
the residuals.

Note the C's `-L` writes LaTeX while the Python ports write HTML. The port
follows the Python side; that is what "the same format for both ports" means.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("jinja2")
pytest.importorskip("matplotlib")

from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit  # noqa: E402
from drtran.report import (build_forecast_result,  # noqa: E402
                           write_forecast_report)

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def fitted():
    specs = [drtran.load_pre(ES), drtran.load_pre(WTI)]
    cs = build_cast_spec(specs, links=[Link(0, 1, b=0, r=0, s=1)])
    return specs, cs, fit(cs, embed=True)


def test_the_level_std_is_scaled_for_fufs_page(fitted):
    """fuf's `level_std` is divided by `refactor` because the page multiplies by
    100 to print a percentage. drtran's `se("level")` is already in the
    transformed scale, so it is divided and no more.

    Get this wrong and a relative standard error prints as index points: 0.24
    would become 24, or 0.0024. The check is against what the C's own column
    shows -- 0.24, 0.44, 0.60.
    """
    _specs, cs, f = fitted
    fr, _model = build_forecast_result(f, cs, series=0, horizon=6)
    printed = fr.level_std * 100.0
    assert printed[:3] == pytest.approx([0.24, 0.44, 0.60], abs=0.005)


def test_the_variations_use_fufs_metric(fitted):
    """100 * (difference of the transformed level) / refactor, which for a log
    model is exactly the percentage change. The C's row for 1/2020 is
    -1.00 (0.24) monthly and 1.07 (0.24) annual."""
    _specs, cs, f = fitted
    fr, _model = build_forecast_result(f, cs, series=0, horizon=6)
    assert fr.diff1[0] == pytest.approx(-1.00, abs=0.005)
    assert fr.diff1_std[0] == pytest.approx(0.24, abs=0.005)
    assert fr.seasonal_diff[0] == pytest.approx(1.07, abs=0.005)
    assert fr.seasonal_diff_std[0] == pytest.approx(0.24, abs=0.005)
    assert fr.level[0] == pytest.approx(82.0149, abs=0.001)


def test_the_residuals_are_attached_to_a_COPY(fitted):
    """The page's ERR column reads `model._result.residuals`, and drtran's model
    was not fitted by fue, so there is nothing there. Attaching them to the
    caller's model would be worse than useless: `model.series` IS the spec's
    `ts`, so the report would end up writing on the data it is reporting."""
    specs, cs, f = fitted
    before = specs[0].ts.nobs
    fr, model = build_forecast_result(f, cs, series=0, horizon=6)

    assert model is not specs[0].model
    assert model.series is not specs[0].model.series
    assert specs[0].ts.nobs == before

    res = np.asarray(model._result.residuals)
    assert len(res) == before - 1, "one observation lost to d=1"
    # fuf derives ornsop from this length, so it has to be the right one
    assert before - len(res) == 1


def test_the_model_details_come_from_the_JOINT_fit(fitted):
    """npar, AIC and BIC describe the model that produced the forecast — the
    joint one — not a univariate stand-in."""
    _specs, cs, f = fitted
    _fr, model = build_forecast_result(f, cs, series=0, horizon=6)
    r = model._result
    assert r.npar == len(f.x)
    assert r.aic == pytest.approx(-2 * f.loglik + 2 * r.npar)
    assert r.bic > r.aic, "BIC penalises more at n=215"


def test_it_writes_a_self_contained_page(fitted, tmp_path):
    _specs, cs, f = fitted
    path = str(tmp_path / "r.html")
    write_forecast_report(f, cs, path, series=0, horizon=12)

    assert os.path.exists(path)
    html = open(path).read()
    assert len(html) > 10_000
    assert "<svg" in html, "the chart must be embedded, not linked"

    # Self-contained means no resource LOADS. The page is full of http:// all
    # the same -- XML namespace URIs and matplotlib's own metadata inside the
    # embedded SVG -- so grepping for the scheme says nothing. What matters is
    # that nothing is fetched.
    import re
    for tag, attr in (("script", "src"), ("link", "href"), ("img", "src")):
        for m in re.finditer(rf"<{tag}\b[^>]*", html):
            ref = re.search(rf'{attr}\s*=\s*["\']([^"\']+)', m.group(0))
            assert ref is None or not ref.group(1).startswith("http"), m.group(0)
    assert "82.01" in html


def test_the_report_needs_a_horizon_on_the_command_line():
    import io
    import sys

    from drtran.cli import main

    so, se = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        code = main([ES, WTI, "-b", "0", "-r", "0", "-s", "1", "-L"])
        err = sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = so, se
    assert code == 1 and "needs a horizon" in err
