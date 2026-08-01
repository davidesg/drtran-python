"""The forecast report — fuf's, not one of our own.

fuf is the univariate forecast program of the family, and its Python port lives
inside fue (`fue.report_forecast`). It produces a self-contained HTML page:
table on the left, a two-panel chart on the right, PDF optional. This module does
not reimplement any of that. It **adapts** drtran's joint forecast into the
objects fuf's writer already expects and calls it.

That is the same decision as everywhere else in this port — `fue.load()` for the
`.pre`, `cast_us_py` for the stationary series, `elf` for the likelihood — and
here it buys something specific: a reader who gets a univariate report from fuf
and a transfer-function report from drtran sees **the same page**, and can
compare them without first working out which numbers mean the same thing.

Note the C's `-L` writes LaTeX and fuf's Python port writes HTML. The port
follows the Python side, because that is what "the same format for both ports"
means; the C's LaTeX is not reproduced.

What has to be adapted
----------------------
fuf's writer wants a `ForecastResult` and a fitted `fue.Model`. Two things do not
carry over for free:

* **The scales.** fuf's `level_std` is the standard error divided by `refactor`,
  which the page then multiplies by 100 to print a percentage; drtran's
  `se("level")` is already in the transformed scale, so it is divided by
  `refactor` and no more. Getting this wrong prints a relative error as if it
  were index points, which looks plausible and is out by a factor of 100.
* **The residuals.** The page's ERR column reads `model._result.residuals`, and
  drtran's model was not fitted by fue, so there is nothing there. The joint
  model's residuals for that series are attached to a copy — a copy, because
  `model.series` IS the spec's `ts` and writing on it would reach back into the
  caller's data.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np


def build_forecast_result(fit, cast_spec, series=0, horizon=12, origin=None):
    """drtran's joint forecast, in the shape fuf's writer expects.

    Returns `(fue.ForecastResult, fue.Model)`: the model is a COPY carrying the
    joint fit's residuals for this series, so the caller's specs stay intact.
    """
    from fue.forecast import ForecastResult

    from .estimate import _f1f2
    from .forecast import forecast, to_level
    from .netid import residuals as drtran_residuals

    sc = cast_spec.series[series]
    model = sc.spec.model
    freq = int(getattr(model.series, "freq", 1) or 1)
    refc = float(getattr(model, "refactor", 1.0) or 1.0)

    fc = forecast(fit, L=horizon, origin=origin, embed=fit.embed)
    level, star_f, star_h = to_level(fc, cast_spec, series=series,
                                     origin=origin, transformed=True)
    star = np.concatenate([star_h, star_f])
    nb = len(star_h)

    # the metric fuf uses for the variations: 100 * (transformed difference) /
    # refactor, which for a log model is exactly the percentage change
    vs = 100.0 / refc
    diff1 = np.array([vs * (star[nb + l] - star[nb + l - 1])
                      for l in range(horizon)])
    seas = np.array([vs * (star[nb + l] - star[nb + l - freq])
                     if nb + l - freq >= 0 else float("nan")
                     for l in range(horizon)])

    se_level = fc.se("level", series) / refc      # the page multiplies by 100
    se_diff = vs * fc.se("diff", series)
    se_seas = (vs * fc.se("annual", series) if freq > 1
               and fc.var_annual is not None else np.zeros(horizon))

    _phi, _th, _mu, w, _Q, ifa = _cast_of(fit, cast_spec)
    n, m = w.shape
    f1, _f2, _i = _f1f2(np.asarray(fit.x, float), cast_spec, -1e-3, fit.embed)
    sigma2 = float(f1) / (n * m) if f1 else float("nan")

    fr = ForecastResult(horizon=horizon, level=np.asarray(level, float),
                        level_std=np.asarray(se_level, float),
                        diff1=diff1, diff1_std=np.asarray(se_diff, float),
                        seasonal_diff=seas,
                        seasonal_diff_std=np.asarray(se_seas, float),
                        sigma2=sigma2)

    a, ifa_r = drtran_residuals(fit.x, cast_spec, embed=fit.embed)
    shim = copy.deepcopy(model)
    npar = len(fit.xfree) if fit.xfree is not None else len(fit.x)
    res = np.asarray(a[:, series], float) if not ifa_r else np.zeros(0)
    shim._result = SimpleNamespace(
        residuals=res,
        npar=npar,
        aic=-2.0 * fit.loglik + 2.0 * npar,
        bic=-2.0 * fit.loglik + npar * math.log(max(n, 2)),
    )
    if not getattr(shim, "_inp_stem", None):
        shim._inp_stem = sc.name
    return fr, shim


def _cast_of(fit, cast_spec):
    from .cast import cast_diagonal
    from .embed import cast_embedded

    x = np.asarray(fit.x, float)
    if fit.embed:
        return cast_embedded(x, cast_spec)
    return cast_diagonal(x, cast_spec)


def write_forecast_report(fit, cast_spec, path, series=0, horizon=12,
                          origin=None, title=None, source=None, sps_name=None,
                          narrative=None, pdf=False):
    """Write fuf's HTML forecast report for one series of the joint model.

    Needs `jinja2` and `matplotlib`, which is what fuf needs; the error says so
    rather than failing inside a template.
    """
    from fue.report_forecast import write_forecast_report as _fuf_write

    fr, model = build_forecast_result(fit, cast_spec, series=series,
                                      horizon=horizon, origin=origin)
    name = cast_spec.series[series].name
    _fuf_write(model, fr, path,
               title=title or f"A.{name}",
               source=source,
               sps_name=sps_name or "drtran — joint transfer function model",
               narrative=narrative, pdf=pdf)
    return path
