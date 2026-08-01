"""Out-of-sample evaluation: the fixed window and the rolling origin.

Port of `recursive_eval` (`drtran.c`) and of the `-estwin` mode it serves.

Why this exists at all is the point. The variances the model reports are
**theoretical**: they say what the model implies, not what happens out of
sample, where parameter uncertainty and structural change have their say. Two
specifications can be ranked by likelihood, by AIC, by the width of their
theoretical bands — and still be ranked the other way round by what they
actually got right. This is the only part of the program that answers the
empirical question, and the way to use it is to run it on two specifications
and compare.

The protocol, which is the C's:

1. estimate **once** on observations 1..E and hold the parameters FIXED;
2. roll the origin forward one datum at a time, e = E .. n-H;
3. at each origin re-truncate the sample to `e`, forecast H steps, and compare
   with what actually happened.

Holding the parameters fixed is deliberate, not a shortcut. Re-estimating at
every origin measures something else — a procedure rather than a model — and it
is expensive enough to discourage the exercise entirely. The horizons stay
**balanced**: every origin runs to `n-H`, so every h is averaged over the same
origins and the columns are comparable with one another.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np


def truncate(specs, n):
    """The same `.pre` specs with only their first `n` observations.

    A copy, not a view: the caller's specs keep the full sample, which is what
    the actual values are read from afterwards. The model is deep-copied because
    `model.series` IS the spec's `ts` in fue, so trimming one in place would trim
    the other and the evaluation would end up comparing a forecast against the
    data it was made from.
    """
    from .pre import PreSpec

    out = []
    for s in specs:
        ts = s.ts
        if n >= ts.nobs:
            out.append(s)
            continue
        model = copy.deepcopy(s.model)
        model.series.data = list(ts.data[:n])
        for attr in ("nobs", "n"):
            if hasattr(model.series, attr):
                try:
                    setattr(model.series, attr, n)
                except AttributeError:
                    pass                       # a read-only property: fine
        out.append(PreSpec(ts=model.series, model=model, path=s.path))
    return out


@dataclass
class RollingEval:
    """What the rolling origin found, by horizon."""

    horizon: int
    first_origin: int
    last_origin: int
    n_origins: int
    name: str = ""
    count: np.ndarray = None           # (H,) origins that reached horizon h
    mae: np.ndarray = None             # (H,)
    rmse: np.ndarray = None            # (H,)
    mape: np.ndarray = None            # (H,) in per cent
    rows: list = field(default_factory=list)   # (origin, h, actual, fcst, error)

    def __repr__(self):                                    # pragma: no cover
        return (f"RollingEval({self.name!r}, H={self.horizon}, "
                f"{self.n_origins} origins)")


def rolling_evaluation(x, specs, links, window, horizon, series=0,
                       embed=True, slots=None, xitol=-1e-3):
    """Roll the forecast origin over `window`..`n-horizon` with `x` held FIXED.

    `x` is the parameter vector estimated on the first `window` observations —
    see `fixed_window_fit`. `specs` are the FULL `.pre` specs; they are what the
    actual values are read from, and they are truncated afresh at every origin.

    Returns a `RollingEval`. Raises `ValueError` if there is not enough data,
    rather than reporting an average over two origins as though it meant
    something.
    """
    from .cast import build_cast_spec
    from .forecast import forecast, to_level

    specs = list(specs)
    nfull = specs[series].ts.nobs
    last = nfull - horizon
    if last < window:
        raise ValueError(
            f"not enough data for the rolling evaluation: window {window}, "
            f"horizon {horizon}, {nfull} observations (the last usable origin "
            f"would be {last})")

    actual_all = np.asarray(specs[series].ts.data, float)
    name = specs[series].name

    H = horizon
    sae = np.zeros(H)
    sse = np.zeros(H)
    sape = np.zeros(H)
    cnt = np.zeros(H, dtype=int)
    rows = []
    n_origins = 0

    for e in range(window, last + 1):
        cs = build_cast_spec(truncate(specs, e), links=links)
        try:
            fc = forecast(x, cs, L=H, embed=embed, xitol=xitol)
            level = to_level(fc, cs, series=series)
        except (RuntimeError, ValueError):
            # a truncated window the cast cannot handle: skipped, as the C does,
            # and visible in the origin count rather than silently averaged in
            continue
        n_origins += 1

        for l in range(1, H + 1):
            act = float(actual_all[e + l - 1])
            err = float(level[l - 1]) - act
            sae[l - 1] += abs(err)
            sse[l - 1] += err * err
            if abs(act) > 1e-12:
                sape[l - 1] += abs(err / act)
            cnt[l - 1] += 1
            rows.append((e, l, act, float(level[l - 1]), err))

    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(cnt > 0, sae / np.maximum(cnt, 1), np.nan)
        rmse = np.where(cnt > 0, np.sqrt(sse / np.maximum(cnt, 1)), np.nan)
        mape = np.where(cnt > 0, 100.0 * sape / np.maximum(cnt, 1), np.nan)

    return RollingEval(horizon=H, first_origin=window, last_origin=last,
                       n_origins=n_origins, name=name, count=cnt, mae=mae,
                       rmse=rmse, mape=mape, rows=rows)


def fixed_window_fit(specs, links, window, slots_of=None, **kw):
    """Estimate once on the first `window` observations. The `-estwin` mode.

    Returns `(fit, cast_spec_of_the_window)`. The parameters are then held fixed
    by everything downstream: the report, the forecast and the rolling
    evaluation. Re-estimating the SAME fixed window is deterministic and cheap,
    which is what lets the real-time exercise be repeated as data arrives
    without the parameters drifting underneath it.
    """
    from .cast import build_cast_spec
    from .estimate import fit as estimate

    cs = build_cast_spec(truncate(specs, window), links=links)
    slots = slots_of(cs) if slots_of is not None else None
    return estimate(cs, slots=slots, **kw), cs


def write_rolling_csv(ev, path):
    """The per-origin errors, one row per (origin, horizon)."""
    with open(path, "w") as f:
        f.write("origin,horizon,actual,forecast,error\n")
        for o, h, a, fo, e in ev.rows:
            f.write(f"{o},{h},{a:.6f},{fo:.6f},{e:.6f}\n")
    return path


def report_rolling(ev):
    """The evaluation table, in the C's layout."""
    L = ["=" * 61,
         "  RECURSIVE FORECAST EVALUATION (out of sample)",
         "=" * 61,
         "  The variances the model reports are THEORETICAL: they say what",
         "  the model implies, not what happens out of sample, where",
         "  parameter uncertainty and structural change have their say.",
         f"  Here the parameters are estimated ONCE on the first "
         f"{ev.first_origin}",
         "  observations, then held FIXED while the origin rolls forward",
         "  one datum at a time.",
         "",
         f"  Output   : {ev.name}",
         f"  Origins  : {ev.n_origins}  (from obs {ev.first_origin} to "
         f"{ev.last_origin})",
         f"  Horizon  : {ev.horizon}",
         "",
         "    h      n        MAE         RMSE        MAPE(%)",
         "  " + "-" * 49]
    for l in range(ev.horizon):
        if ev.count[l] == 0:
            continue
        L.append(f"  {l + 1:3d}  {ev.count[l]:5d}  {ev.mae[l]:11.6f}  "
                 f"{ev.rmse[l]:11.6f}  {ev.mape[l]:10.4f}")
    L.append("=" * 61)
    return "\n".join(L)
