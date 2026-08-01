"""Accounting identities, computed AFTER forecasting — and their band.

Port of the `-a` block of `drtran.c`. An identity like

    OCUPADOS = + EA + EP + EI + EU + EC
    PARADOS  = + ACTIVOS - EA - EP - EI - EU - EC

does **not** belong in the model: it is arithmetic on the answer, not a
restriction on the parameters, and the legacy computed it after forecasting for
exactly that reason. Putting it in would add a series that is a linear
combination of the others, and the likelihood would be singular.

What is not trivial is the **band**. The series' forecast errors are CORRELATED —
they share innovations through the network — so the variance of the aggregate is
NOT the sum of the variances. It is `c' V c` with V the full forecast-error
covariance. On a system where the sectors move together, treating them as
independent understates the band badly, and in the direction that flatters the
model.

There is a second wrinkle. The series are modelled TRANSFORMED (log, Box-Cox)
and the identity lives in LEVELS, so the covariance has to be carried across by
the delta method::

    dz/db = level / refactor                          (lambda = 0)
    dz/db = (lambda*b/refactor + 1)^(1/lambda - 1) / refactor   (otherwise)

and the variance is `sum_i sum_j c_i c_j J_i J_j V_ij(l)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Aggregate:
    """A named linear combination of the series."""

    name: str
    coef: np.ndarray                   # (m,) the c vector, mostly zeros

    def terms(self, names):
        out = []
        for i, c in enumerate(self.coef):
            if c:
                out.append(f"{'+' if c > 0 else '-'} {names[i]}")
        return " ".join(out)


@dataclass
class AggregateForecast:
    """The aggregate's path with its band."""

    name: str
    level: np.ndarray = None
    sd: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None
    terms: str = ""


def read_aggregates(path, names):
    """Read an aggregates file: `NAME = +A -B`, one per line, `#` for comments.

    Signs may be attached (`+EA`) or separate (`+ EA`), as in the C. An unknown
    series name is an error rather than a silently dropped term — a dropped term
    turns an identity into a different identity that still adds up.
    """
    names = list(names)
    out = []
    with open(path) as f:
        for nline, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"{path}:{nline}: expected NAME = +A -B: {line!r}")
            lhs, rhs = line.split("=", 1)
            name = lhs.strip()
            if not name:
                raise ValueError(f"{path}:{nline}: aggregate with no name")
            c = np.zeros(len(names))
            sign = 1.0
            for tok in rhs.split():
                if tok == "+":
                    sign = 1.0
                    continue
                if tok == "-":
                    sign = -1.0
                    continue
                if tok[0] == "+":
                    sign, tok = 1.0, tok[1:]
                elif tok[0] == "-":
                    sign, tok = -1.0, tok[1:]
                if not tok:
                    continue
                if tok not in names:
                    raise ValueError(f"{path}:{nline}: unknown series {tok!r}; "
                                     f"the ones loaded are {names}")
                c[names.index(tok)] += sign
                sign = 1.0
            if not np.any(c):
                raise ValueError(f"{path}:{nline}: {name!r} has no terms")
            out.append(Aggregate(name=name, coef=c))
    return out


def _jacobian(level, model):
    """`dz/db` at the forecast point: the delta-method factor to the level."""
    lam = float(getattr(model, "boxlam", 0.0) or 0.0)
    refc = float(getattr(model, "refactor", 1.0) or 1.0)
    if abs(lam) < 1e-8:
        return np.asarray(level, float) / refc
    # level = base^(1/lam), so dlevel/db = base^(1/lam - 1)/refactor
    return np.power(np.asarray(level, float), 1.0 - lam) / refc


def forecast_aggregates(fc, cast_spec, aggregates, origin=None, z=1.96):
    """The aggregates' paths and bands, `c' J V J c` per horizon."""
    from .forecast import to_level

    m = cast_spec.m
    L = fc.L
    levels = np.zeros((L, m))
    jac = np.zeros((L, m))
    for i in range(m):
        lv = to_level(fc, cast_spec, series=i, origin=origin)
        levels[:, i] = lv
        jac[:, i] = _jacobian(lv, cast_spec.series[i].spec.model)

    names = list(cast_spec.names)
    out = []
    for ag in aggregates:
        c = np.asarray(ag.coef, float)
        pnt = levels @ c
        sd = np.zeros(L)
        for l in range(1, L + 1):
            V = np.asarray(fc.var_level[l], float)
            g = c * jac[l - 1]                    # the delta-method vector
            var = float(g @ V @ g)
            sd[l - 1] = math.sqrt(var) if var > 0 else 0.0
        out.append(AggregateForecast(name=ag.name, level=pnt, sd=sd,
                                     lower=pnt - z * sd, upper=pnt + z * sd,
                                     terms=ag.terms(names)))
    return out


def report_aggregates(aggs):
    """The aggregates' tables, in the C's layout."""
    L = ["=" * 62,
         "  AGGREGATES  (accounting identities)",
         "=" * 62]
    for a in aggs:
        L += ["",
              f"  Aggregate: {a.name}  = {a.terms}",
              "  Computed AFTER forecasting; its band is c'Vc, so it accounts",
              "  for the correlation between the series' forecast errors.",
              "",
              "   l        LEVEL       sd         lower         upper",
              "  " + "-" * 56]
        for l in range(len(a.level)):
            L.append(f"  {l + 1:3d}  {a.level[l]:11.4f}  {a.sd[l]:8.4f}  "
                     f"{a.lower[l]:11.4f}  {a.upper[l]:11.4f}")
    L.append("=" * 62)
    return "\n".join(L)
