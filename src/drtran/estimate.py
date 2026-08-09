"""Joint estimation by exact maximum likelihood.

Minimises Mauricio's scaled objective (1995, §3 eq. 3.5) with the same factored
BFGS that fue and drvarma use (`raxopt`, Dennis & Schnabel A9.4.1). Neither the
optimizer nor the likelihood is reimplemented: they are only connected.

The objective
-------------
The concentrated likelihood is

    ll = C - 0.5*n*( m*log f1 + log f2 )

so maximising it is the same as minimising `f1^m * f2`. It is normalised to 1.0
at the starting point, as `objcfunc` does in the C and `objective` in fue:

    F(x) = (f1/f1_0)^m * (f2/f2_0)

A rejected point (ifault != 0, Q not positive definite, an AR pinned to the
circle...) returns 1.0: it does not improve on the start, so the optimizer moves
away from it. That is the strategy of the paper itself (§3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cast import cast_diagonal
from .cast import effective_embed
from .embed import cast_embedded


@dataclass
class Fit:
    """The result of the joint estimation."""

    x: np.ndarray
    loglik: float
    ifault: int
    termcode: int
    nit: int
    cast_spec: object
    converged: bool
    slots: object = None          # SlotTable, if the fit is constrained
    xfree: object = None          # what the optimizer saw (x is the full one)
    embed: bool = True            # which cast produced it; `standard_errors` needs it

    # The optimizer's termcode (raxopt / qnewtopt.c), with the classification
    # drtran settled on at its M1 milestone: 1-2 convergence, 3 stopped WITHOUT
    # improvement (normal when starting at the optimum, or on reaching it), 4-5
    # a real failure.
    _STATUS = {1: "CONVERGED (gradient)",
               2: "CONVERGED (step)",
               3: "stopped without improvement",
               4: "iteration limit",
               5: "steps of maximum length"}

    @property
    def status(self):
        return self._STATUS.get(self.termcode, f"termcode={self.termcode}")

    @property
    def convergence_note(self):
        """The interpretation of the termcode, or None when it is a clean
        gradient convergence.

        WHY it stopped matters as much as WHETHER it stopped, and drtran was
        reporting the code without saying what it means — drvarma already did
        (`drvarma/report.py:_convergence_block`). The wording here is drvarma's,
        unchanged, so the same situation reads the same way across the suite.

        termcode 3 is deliberately NOT flagged as a failure: drtran seeds from
        the `.pre`, which on the diagonal rung ALREADY is the optimum, so the
        search stops at once without improving. drvarma's report is harsher
        about 3; that disagreement is recorded in its TODO and is not settled
        here.
        """
        if self.termcode == 2:
            return ("stopped on steptol, NOT on the gradient: the step "
                    "collapsed while the gradient may still be appreciable. "
                    "Typical of an ill-conditioned likelihood (near "
                    "non-identification / common factors). Treat the standard "
                    "errors with caution and re-estimate from other starting "
                    "values or with a smaller order.")
        if self.termcode in (4, 5):
            return ("NOT a convergence: the optimiser gave up (%s). The "
                    "estimates are not a maximum; every criterion derived from "
                    "this fit is unreliable." % self.status)
        return None

    def __repr__(self):                                    # pragma: no cover
        return (f"Fit(logL={self.loglik:.6f}, {self.status}, "
                f"termcode={self.termcode}, nit={self.nit}, "
                f"npar={len(self.x)})")


def _f1f2(x, cast_spec, xitol, embed=False):
    """(f1, f2, ifault) of the cast at x, through drvarma's `elf`.

    `embed=True` uses the EMBEDDED cast (the default in the C), which puts the
    transfer inside the VARMA without subtracting anything, so there is no
    pre-sample truncation.
    """
    from drvarma._engine import elf_c

    embed = effective_embed(cast_spec, embed)
    build = cast_embedded if embed else cast_diagonal
    phi, theta, mu, w, sigma, ifault = build(x, cast_spec)
    if ifault:
        return None, None, int(ifault)
    # drvarma's COMPILED `elf`, exposed for this: the port needs to SCORE a
    # structure the cast builds, not to fit a free VARMA. Identical to the pure
    # Python one (1e-13) and ~100x faster.
    n, m = w.shape
    _lg, f1, f2, _a, ifa = elf_c(m, n, phi.shape[0], theta.shape[0],
                                 mu, phi, theta, sigma, w, 1.0, xitol, False)
    return float(f1), float(f2), int(ifa)


def loglik(x, cast_spec, xitol=-1e-3, embed=False):
    """Exact concentrated log-likelihood at x (drvmlest.c:est [4])."""
    f1, f2, ifa = _f1f2(x, cast_spec, xitol, embed)
    if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    build = cast_embedded if effective_embed(cast_spec, embed) else cast_diagonal
    _phi, _t, _m, w, _s, _i = build(x, cast_spec)
    n, m = w.shape
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def x0_full(cast_spec, slots):
    """The `.pre`'s seeds in the FULL space of the slot table.

    `x0_from_pre` goes as far as the variance ratios; the table appends the
    covariances behind them, and those start at zero — that is, at the
    diagonal-covariance model, which is the ladder's previous rung.
    """
    from .cast import x0_from_pre

    x0 = np.asarray(x0_from_pre(cast_spec), float)
    missing = len(slots) - len(x0)
    if missing < 0:
        raise ValueError(f"the table has {len(slots)} slots and the seeds "
                         f"{len(x0)}: is it this cast's table?")
    return np.concatenate([x0, np.zeros(missing)])


def fit(cast_spec, x0=None, xitol=-1e-3, maxits=500, grtol=1e-7,
        sptol=1e-7, embed=True, slots=None):
    """Estimate the joint model and return a `Fit`.

    `embed=True` (the default, as in the C) puts the transfer INSIDE the VARMA;
    `embed=False` subtracts it, which is the old cast (`-S`).

    `x0` defaults to the `.pre`'s seeds (fue's univariate estimates) with the
    transfers at zero — that is, the search starts on the diagonal rung and the
    optimizer is left to add the dynamics.

    `slots` is a `SlotTable` (see `drtran.slots`). With it the optimizer works in
    the space of the **free** parameters and every evaluation expands to the full
    structure: that is what makes it possible to fix, to share and to express
    some coefficients as functions of others. Without it the vector is the full
    one and everything is free, except whatever the `.pre` already declared
    fixed. `Fit.x` is always the full vector; `Fit.xfree` is what the optimizer
    saw.

    On `termcode`: 1-2 is convergence (gradient / step), **3 is stopped without
    improvement**, which here is NORMAL when starting at the optimum — the case
    of the diagonal rung, where the `.pre`'s seeds already are it. 4-5 is a real
    failure.
    """
    from drvarma import _qnewt

    from .cast import x0_from_pre

    if slots is None:
        x_ini = np.asarray(x0_from_pre(cast_spec) if x0 is None else x0, float)
        expand = lambda v: v                                      # noqa: E731
    else:
        xfull0 = np.asarray(x0_full(cast_spec, slots) if x0 is None else x0, float)
        if len(xfull0) != len(slots):
            raise ValueError(f"with a slot table, x0 is the FULL vector: "
                             f"expected {len(slots)}, got {len(xfull0)}")
        x_ini = slots.pack(xfull0)
        expand = slots.expand

    npar = len(x_ini)
    # The dispatch, resolved ONCE so `Fit.embed` records the cast that really
    # ran -- `standard_errors` reads it back, and a Fit that misreported its own
    # cast would be worse than no dispatch at all.
    embed = effective_embed(cast_spec, embed)

    def _ll(v):
        return loglik(expand(v), cast_spec, xitol, embed)

    def _pack(v, ll, ifa, termcode, nit):
        return Fit(x=np.asarray(expand(v), float), loglik=ll, ifault=int(ifa),
                   termcode=int(termcode), nit=int(nit), cast_spec=cast_spec,
                   converged=int(termcode) in (1, 2), slots=slots,
                   xfree=np.asarray(v, float), embed=embed)

    f1_0, f2_0, ifa0 = _f1f2(expand(x_ini), cast_spec, xitol, embed)
    if ifa0 or f1_0 is None or not (f1_0 > 0.0 and f2_0 > 0.0):
        return _pack(x_ini, float("-inf"), ifa0 or 5, 0, 0)

    m = cast_spec.m

    def objective(xv):
        f1, f2, ifa = _f1f2(expand(np.asarray(xv, float)), cast_spec, xitol, embed)
        if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
            return 1.0                       # rejected point: no improvement
        return (f1 / f1_0) ** m * (f2 / f2_0)

    if npar == 0:
        ll, ifa = _ll(x_ini)
        return _pack(x_ini, ll, ifa, 1, 0)

    # raxopt works on a 1-based vector (an unused slot at the front)
    xk = np.zeros(npar + 1)
    xk[1:] = x_ini

    def func1(xk1):
        return objective(xk1[1:npar + 1])

    _fk, _bfac, nit, termcode = _qnewt.raxopt(func1, npar, xk, maxits, grtol, sptol)
    x_hat = xk[1:npar + 1].copy()

    ll, ifa = _ll(x_hat)
    return _pack(x_hat, ll, ifa, termcode, nit)


@dataclass
class StdErrors:
    """Standard errors of a fit, in the FREE space and mapped onto the slots."""

    cov: np.ndarray               # (nfree, nfree) covariance of the free params
    se: np.ndarray                # (nfree,) its square root diagonal
    se_of_slot: np.ndarray        # (nslot,) NaN where a slot has no s.e.
    t: np.ndarray                 # (nslot,) t = estimate / s.e.
    p: np.ndarray                 # (nslot,) two-sided p-value
    ifault: int = 0


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def standard_errors(fit, xitol=-1e-3):
    """Standard errors from the Hessian recomputed AT the optimum.

    Port of `drvmlest.c:est` [2b]-[3] as drtran's copy has it::

        H   = fdhess(objective, x_hat)         # finite differences at the optimum
        cov = 2 * F(x_hat) * H^-1 / n

    **Not** the optimiser's matrix. `raxopt` leaves the Hessian accumulated by
    BFGS along the search path; it steers the search well but it is not the
    curvature at the optimum. It depends on the path — two different starting
    points give different standard errors on the same estimates — and it degrades
    in the flattest directions, which are exactly the ones with the largest
    errors. Worse, when the search starts AT the optimum and stops immediately
    (`termcode 3`, the normal case when the seeds come from the previous rung of
    the ladder) it is never built at all.

    That is a live defect elsewhere in the family, not a hypothetical: fue C
    computes its standard errors from the BFGS matrix — its `fdhess` call sits
    commented out at `drvmlest.c:112` — and reports different s.e. for the same
    point estimates on different runs. **drtran's C does not**: its `est()`
    recomputes the Hessian, and this port follows it.

    On the scaling: the objective is Mauricio's, normalised to 1 at x0, so it
    carries an arbitrary constant c. It cancels — with F = c*G, both `F(x_hat)`
    and `H` scale by c and `2*F*H^-1` does not move. The standard errors
    therefore do not depend on where the search started, which is the whole
    point.

    `ifault` in the result: 0 fine, 1 the Hessian could not be built, **2 the
    Hessian at that point is not positive definite** — which means the point is
    not a maximum, not that the arithmetic failed. Reporting standard errors
    there would be reporting curvature that does not exist.

    Cost: (k^2 + 3k)/2 likelihood evaluations for k free parameters, so this is
    computed on demand rather than inside `fit`.
    """
    from drvarma import _qnewt
    from drvarma._as311 import _chol_lower

    cast_spec = fit.cast_spec
    slots = fit.slots
    embed = getattr(fit, "embed", True)

    if slots is None:
        xfree = np.asarray(fit.x, float)
        expand = lambda v: v                                      # noqa: E731
    else:
        xfree = np.asarray(fit.xfree, float)
        expand = slots.expand
    k = len(xfree)
    nslot = len(slots) if slots is not None else k

    def objective(v):
        f1, f2, ifa = _f1f2(expand(np.asarray(v, float)), cast_spec, xitol, embed)
        if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
            return 1.0
        return (f1 ** cast_spec.m) * f2

    nan = float("nan")
    empty = StdErrors(cov=np.zeros((k, k)), se=np.full(k, nan),
                      se_of_slot=np.full(nslot, nan), t=np.full(nslot, nan),
                      p=np.full(nslot, nan), ifault=1)
    if k == 0:
        return empty

    # the sample length is the one elf sees: the rows of the stationary series
    build = cast_embedded if embed else cast_diagonal
    out = build(np.asarray(fit.x, float), cast_spec)
    if out[5]:
        return empty
    n = out[3].shape[0]

    x1 = np.zeros(k + 1)
    x1[1:] = xfree

    def func1(v1):
        return objective(v1[1:k + 1])

    fk = func1(x1)
    if not fk > 0.0:
        return empty

    H = np.zeros((k + 1, k + 1))
    _qnewt.fdhess(func1, k, x1, fk, _qnewt.MACHEPS, H)

    # POSITIVE DEFINITENESS, checked before trusting the factorisation.
    #
    # `choldcp` is the MODIFIED Cholesky: faced with a non-positive pivot it
    # patches it and carries on. That is right for steering a search and wrong
    # for reporting inference — it turns "this is not a maximum" into a column of
    # plausible-looking standard errors. And the Hessian at a point that is NOT
    # the optimum is under no obligation to be positive definite: measured on
    # m6, at the `.pre` seeds 2 of its 55 eigenvalues are <= 0, while at the C's
    # actual optimum all 55 are positive.
    #
    # This is the risk drvarma's `est` avoids by using the optimiser's BFGS
    # matrix instead, which is positive definite by construction and therefore
    # never fails — at the price of not being the curvature at the optimum. The
    # trade is: BFGS always answers, sometimes wrongly; fdhess answers correctly
    # or, with this guard, says it cannot.
    w = np.linalg.eigvalsh(0.5 * (H[1:, 1:] + H[1:, 1:].T))
    if w.min() <= 0.0:
        return StdErrors(cov=np.zeros((k, k)), se=np.full(k, nan),
                         se_of_slot=np.full(nslot, nan), t=np.full(nslot, nan),
                         p=np.full(nslot, nan), ifault=2)

    L, _detfac, ifa = _chol_lower(H, k)
    if ifa:
        return empty

    cov = np.zeros((k, k))
    for i in range(1, k + 1):
        e = np.zeros(k + 1)
        e[i] = 1.0
        _qnewt.cholsol(L, k, e)
        cov[:, i - 1] = (2.0 * fk * e[1:k + 1]) / n

    se = np.array([math.sqrt(cov[i, i]) if cov[i, i] > 0 else nan
                   for i in range(k)])

    # onto the slots: a FREE slot carries its own s.e.; an ALIAS carries its
    # representative's, because they are one degree of freedom in two places
    # (that is what sharing means, and it is what the C prints).
    se_slot = np.full(nslot, nan)
    t = np.full(nslot, nan)
    p = np.full(nslot, nan)
    xfull = np.asarray(fit.x, float)
    if slots is None:
        se_slot = se.copy()
    else:
        for i, sl in enumerate(slots.slots):
            j = slots.free_of_slot[i]
            if j < 0 and sl.kind == 2:            # ALIAS: follow to the source
                j = slots.free_of_slot[sl.pa]
            if j >= 0:
                se_slot[i] = se[j]
    for i in range(nslot):
        if se_slot[i] == se_slot[i] and se_slot[i] > 1e-15:
            t[i] = xfull[i] / se_slot[i]
            p[i] = 2.0 * (1.0 - _normal_cdf(abs(t[i])))

    return StdErrors(cov=cov, se=se, se_of_slot=se_slot, t=t, p=p, ifault=0)


def unpack(fit_or_x, cast_spec=None):
    """Split the estimated vector into its blocks, in `shootx` order.

    Returns a dict with `links` (a list of (omega, delta) per link), `series`
    (each series' univariate chunk, as fue understands it), `log_var_ratio` and
    `cov` (the lower-triangle covariances, empty if the vector does not carry
    them).
    """
    if hasattr(fit_or_x, "x"):
        x, cast_spec = fit_or_x.x, fit_or_x.cast_spec
    else:
        x = np.asarray(fit_or_x, float)
    idx = 0
    links = []
    for l in cast_spec.links:
        om = x[idx:idx + l.s + 1]; idx += l.s + 1
        de = x[idx:idx + l.r]; idx += l.r
        links.append((om, de))
    series = []
    for sc in cast_spec.series:
        series.append(x[idx:idx + sc.npar]); idx += sc.npar
    ratios = x[idx:idx + cast_spec.m - 1]; idx += cast_spec.m - 1
    return {"links": links, "series": series,
            "log_var_ratio": ratios, "cov": x[idx:]}
