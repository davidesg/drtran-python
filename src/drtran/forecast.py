"""Forecasting: point forecasts and their variances.

Port of `forecast.c`. Everything here follows from one object, the MA(infinity)
weights of the fitted VARMA::

    Psi(B) = Phi(B)^-1 Theta(B),    psi_l = sum_i Phi_i psi_(l-i) - Theta_l

The point forecast is the recursion of the model with future innovations set to
zero; the forecast error at horizon l is `sum_(j<l) psi_j a_(t+l-j)`, so its
covariance is `sum_(j<l) psi_j Sigma psi_j'`. Nothing else is needed.

Three variances, not one
------------------------
The model is fitted on the STATIONARY series `w` — after Box-Cox, differencing
and the deterministic part. A forecast is wanted for the **level**, and often for
its monthly and annual variation. Those are different linear filters of the same
innovations, so each has its own weights and its own variance:

===============  ==========================================================
level            psi*(B) = psi(B) / delta(B), undoing the differencing
                 delta(B) = (1-B)^d (1-B^s)^D
variation        psi*_l - psi*_(l-1)
annual variation psi*_l - psi*_(l-s)
===============  ==========================================================

Dividing by delta(B) is what turns the forecast error of a differenced series
into that of the level, and it is why the level variance grows without bound
while the variation's does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def psi_weights(phi, theta, L):
    """MA(infinity) weights `psi_0..psi_L`, shape (L+1, m, m).

    `phi` is (p, m, m) and `theta` (q, m, m), with `phi[k]` = Phi_(k+1) — the
    convention of the cast, where the leading identity is implicit.
    """
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    p = phi.shape[0]
    q = theta.shape[0]
    m = phi.shape[1] if p else theta.shape[1]

    psi = np.zeros((L + 1, m, m))
    psi[0] = np.eye(m)
    for l in range(1, L + 1):
        for i in range(1, min(l, p) + 1):
            psi[l] += phi[i - 1] @ psi[l - i]
        if l <= q:
            psi[l] -= theta[l - 1]
    return psi


def error_variance(psi, sigma, L):
    """`V_l = sum_(j<l) psi_j Sigma psi_j'`, shape (L+1, m, m); `V_0` unused."""
    psi = np.asarray(psi, float)
    sigma = np.asarray(sigma, float)
    m = sigma.shape[0]
    var = np.zeros((L + 1, m, m))
    acc = np.zeros((m, m))
    for l in range(1, L + 1):
        acc = acc + psi[l - 1] @ sigma @ psi[l - 1].T
        var[l] = acc
    return var


def _differencing_poly(d, D, s):
    """Coefficients of `delta(B) = (1-B)^d (1-B^s)^D`, with `delta[0] = 1`."""
    deg = d + D * s
    delta = np.zeros(deg + 1)
    delta[0] = 1.0
    for _ in range(d):
        for k in range(deg, 0, -1):
            delta[k] -= delta[k - 1]
    for _ in range(D):
        for k in range(deg, s - 1, -1):
            delta[k] -= delta[k - s]
    return delta


def integrated_weights(psi, d, D, s):
    """`psi*(B) = psi(B) / delta(B)` — the weights of the LEVEL.

        psi*_l = psi_l - sum_(k>=1) delta_k psi*_(l-k)

    Undoing the differencing is what turns the forecast error of `w` into that
    of the level, and why the level's variance grows without bound.
    """
    psi = np.asarray(psi, float)
    L = psi.shape[0] - 1
    delta = _differencing_poly(d, D, s)
    deg = len(delta) - 1
    out = np.zeros_like(psi)
    for l in range(L + 1):
        val = psi[l].copy()
        for k in range(1, min(l, deg) + 1):
            val -= delta[k] * out[l - k]
        out[l] = val
    return out


def forecast_mean(phi, theta, mu, w, a, L, origin=None):
    """Point forecasts `f[l]` for l = 1..L, shape (L, m).

    The model's own recursion with future innovations set to zero: the AR part
    uses observed `w` while it can and its own forecasts afterwards; the MA part
    uses the residuals still inside the window and dies out after lag q.

    `origin` is how many observations of `w` are used (default: all). Forecasting
    from an earlier origin is what an out-of-sample exercise needs, and it is why
    it is a parameter rather than always `n`.
    """
    phi = np.asarray(phi, float)
    theta = np.asarray(theta, float)
    w = np.asarray(w, float)
    a = np.asarray(a, float)
    mu = np.asarray(mu, float)
    p, q = phi.shape[0], theta.shape[0]
    n, m = w.shape
    if origin is None:
        origin = n
    if not 1 <= origin <= n:
        raise ValueError(f"origin outside the sample: {origin} of {n}")

    f = np.zeros((L, m))
    for l in range(1, L + 1):
        ar = np.zeros(m)
        for i in range(1, p + 1):
            if l > i:
                ar += phi[i - 1] @ (f[l - i - 1] - mu)
            else:
                ar += phi[i - 1] @ (w[origin - i + l - 1] - mu)
        ma = np.zeros(m)
        for j in range(1, q + 1):
            if l <= j:
                ma += theta[j - 1] @ a[origin - j + l - 1]
        f[l - 1] = mu + ar - ma
    return f


@dataclass
class Forecast:
    """Point forecasts and the three variances, per horizon."""

    f: np.ndarray                      # (L, m) point forecasts of `w`
    var_w: np.ndarray = None           # (L+1, m, m) of the stationary series
    var_level: np.ndarray = None       # (L+1, m, m) of the level
    var_diff: np.ndarray = None        # (L+1, m, m) of (1-B) level
    var_annual: np.ndarray = None      # (L+1, m, m) of (1-B^s) level
    names: list = field(default_factory=list)
    x: np.ndarray = None               # the fitted vector, for `to_level`
    psi_level: np.ndarray = None       # (L+1, m, m) integrated weights psi*
    sigma: np.ndarray = None           # (m, m) REDUCED-FORM innovation covariance
    phi0: np.ndarray = None            # Phi(0), to recover the STRUCTURAL form

    @property
    def L(self):
        return self.f.shape[0]

    def se(self, which="level", series=0):
        """Standard errors per horizon, `sqrt` of the diagonal."""
        v = {"w": self.var_w, "level": self.var_level,
             "diff": self.var_diff, "annual": self.var_annual}[which]
        if v is None:
            raise ValueError(f"the '{which}' variance was not computed")
        return np.array([math.sqrt(max(v[l][series, series], 0.0))
                         for l in range(1, self.L + 1)])

    def __repr__(self):                                    # pragma: no cover
        return f"Forecast(L={self.L}, m={self.f.shape[1]})"


def forecast(x, cast_spec=None, L=12, origin=None, embed=True, xitol=-1e-3):
    """Forecast the fitted model `L` periods ahead.

    `x` is the full parameter vector with its `cast_spec`, or a `Fit`. The
    differencing (d, D, s) is read per series from the `.pre`, which is where it
    lives — the cast never sees the level.
    """
    from .netid import residuals

    if hasattr(x, "x"):
        cast_spec = x.cast_spec
        x = x.x
    if cast_spec is None:
        raise TypeError("the cast_spec is required (or pass a Fit)")

    from .cast import cast_diagonal
    from .embed import cast_embedded

    if embed:
        phi, theta, mu, w, sigma, ifault, phi0 = cast_embedded(
            np.asarray(x, float), cast_spec, with_phi0=True)
    else:
        phi, theta, mu, w, sigma, ifault = cast_diagonal(np.asarray(x, float),
                                                         cast_spec)
        phi0 = np.eye(cast_spec.m)
    if ifault:
        raise RuntimeError(f"the cast failed: ifault={ifault}")

    a, ifa = residuals(x, cast_spec, embed=embed, xitol=xitol)
    if ifa:
        raise RuntimeError(f"cannot obtain the residuals: ifault={ifa}")

    # THE SCALE. The cast returns Q, not Sigma: the likelihood is CONCENTRATED
    # and sigma2 comes out separately (drvmlest.c:est). Without multiplying by
    # it, the variances come out with the right shape and the wrong magnitude --
    # which is exactly the error of reading Q as if it were Sigma.
    from drvarma._engine import elf_c
    n_, m_ = w.shape
    _lg, f1, _f2, _a2, ifa2 = elf_c(m_, n_, phi.shape[0], theta.shape[0],
                                    mu, phi, theta, sigma, w, 1.0, xitol, False)
    if ifa2 or not f1 > 0:
        raise RuntimeError(f"cannot concentrate sigma2: ifault={ifa2}")
    sigma2 = float(f1) / (n_ * m_)
    sigma = np.asarray(sigma, float) * sigma2

    f = forecast_mean(phi, theta, mu, w, a, L, origin)
    psi = psi_weights(phi, theta, L)
    var_w = error_variance(psi, sigma, L)

    # The level: EACH series' differencing is undone. d, D and s come from the
    # `.pre`, not from the cast — the cast always works on the stationary series.
    m0 = cast_spec.series[0].spec.model
    d = int(getattr(m0, "d", 0))
    D = int(getattr(m0, "D", 0))
    s = int(getattr(m0.series, "freq", 1) or 1)

    psis = integrated_weights(psi, d, D, s)
    var_level = error_variance(psis, sigma, L)

    dif = np.zeros_like(psis)
    dif[0] = psis[0]
    for l in range(1, L + 1):
        dif[l] = psis[l] - psis[l - 1]
    var_diff = error_variance(dif, sigma, L)

    if s > 1:
        ann = np.zeros_like(psis)
        for l in range(L + 1):
            ann[l] = psis[l] - (psis[l - s] if l >= s else 0.0)
        var_annual = error_variance(ann, sigma, L)
    else:
        var_annual = None

    return Forecast(f=f, var_w=var_w, var_level=var_level, var_diff=var_diff,
                    var_annual=var_annual, names=list(cast_spec.names),
                    x=np.asarray(x, float), psi_level=psis, sigma=sigma,
                    phi0=np.asarray(phi0, float))


def report_forecast(fc, series=0, which="w"):
    """The forecast of the STATIONARY series `w`, with its band.

    Deliberately not the level: `fc.f` holds `w`, and the standard errors this
    object carries are in the **transformed** scale — `se("level")` is the
    standard error of `100*log(level)`, i.e. a percentage, not a number of index
    points. Pairing a level with it and adding 1.96 of them would give a
    symmetric band in the wrong units; on the canonical case that is +/-0.47
    where the right answer is +/-0.39, because with a log model the level's band
    comes from exponentiating and is ASYMMETRIC.

    For the level, use `to_level` and `report_level`.
    """
    name = fc.names[series] if fc.names else f"series {series + 1}"
    se = fc.se(which, series)
    L = ["=" * 61,
         f"  FORECAST of w — {name}  (s.e. of the '{which}' filter)",
         "=" * 61,
         "   h        w        s.e.        95% interval",
         "  " + "-" * 55]
    for l in range(fc.L):
        v, e = fc.f[l, series], se[l]
        L.append(f"  {l + 1:2d}  {v:11.4f}  {e:9.4f}   "
                 f"[{v - 1.96 * e:10.4f}, {v + 1.96 * e:10.4f}]")
    L.append("=" * 61)
    return "\n".join(L)


def level_band(fc, cast_spec, series=0, origin=None, z=1.96):
    """The level forecast with its 95 % band, built the way the C builds it.

    The standard error lives in the TRANSFORMED scale, so the band is formed
    there and mapped back through the inverse Box-Cox. With a log model that
    makes it multiplicative and therefore asymmetric around the point forecast —
    which is the honest shape: a level cannot go negative, and a symmetric band
    on a log-modelled series pretends it can.

    Checked against the C's own table: 82.0149 -> [81.6280, 82.4035], where a
    symmetric +/-1.96*0.2412 would have given [81.5421, 82.4877].
    """
    from fue.cast_us import _boxcox
    from fue.forecast import _inv_boxcox

    model = cast_spec.series[series].spec.model
    lam, refc = model.boxlam, model.refactor
    level = to_level(fc, cast_spec, series=series, origin=origin)
    se = fc.se("level", series)

    lo = np.empty(len(level))
    hi = np.empty(len(level))
    for l in range(len(level)):
        zt = _boxcox(level[l], lam, refc)
        lo[l] = _inv_boxcox(zt - z * se[l], lam, refc)
        hi[l] = _inv_boxcox(zt + z * se[l], lam, refc)
    return level, lo, hi


# ── back to the level ────────────────────────────────────────────────────────
def variance_decomposition(fc, series=0):
    """How much of the l-step forecast error variance of the LEVEL of `series`
    comes from EACH source of innovation.

    Computed on the **STRUCTURAL** representation, not the reduced form. That is
    not a detail: with a contemporaneous transfer (b=0) the cast puts omega_0 at
    lag zero, `normalize_phi0` premultiplies by Phi(0)^-1 to give `elf` the
    Phi_0 = I it requires, and the reduced-form Sigma comes out CORRELATED by
    construction (Sigma_12 = omega_0*sigma2_X). Decomposing there would be
    impossible on principle -- and it is the same trap as in the diagnostics,
    where the reduced-form residuals condemn a correct model.

    Undoing the normalisation restores a diagonal Q::

        Q = Phi0 * Sigma * Phi0',      psi*_struct(t) = psi*(t) * Phi0^-1

    and the total variance is unchanged, because
    `psi Sigma psi' = (psi Phi0^-1) Q (psi Phi0^-1)'` identically. So the
    variances the report prints do not move; what changes is that the
    decomposition becomes well posed.

    With Q diagonal the answer is clean and unique::

        share_ij(l) = Q_jj * SUM_{t<l} psi*_struct,ij(t)^2  /  Var_i(l)

    **If Q is not diagonal the decomposition is NOT UNIQUE** and this returns
    `(None, reason)`. Someone has to be given the common part, and that requires
    an ORDERING whose answer changes with the order of the series. That is
    exactly the VAR's problem, and it is not solved here by picking an order
    quietly: it is avoided while Q stays diagonal, and declared when it does not.
    The C makes the same call, and it is why the covariances `q[i,j]` start out
    fixed at zero -- freeing one is a modelling decision that costs you this
    table.

    Returns `(shares, None)`, shape (L, m) with rows summing to 1, or
    `(None, reason)`.
    """
    if fc.psi_level is None or fc.sigma is None:
        return None, "the forecast does not carry psi* and Sigma"

    sigma = np.asarray(fc.sigma, float)
    phi0 = np.eye(sigma.shape[0]) if fc.phi0 is None else np.asarray(fc.phi0,
                                                                     float)
    Q = phi0 @ sigma @ phi0.T
    m = Q.shape[0]
    if np.max(np.abs(Q - np.diag(np.diag(Q)))) > 1e-10 * max(1.0,
                                                             np.max(np.abs(Q))):
        return None, ("the structural Q is not diagonal. With correlated "
                      "innovations the decomposition is NOT UNIQUE -- someone "
                      "has to be given the common part, and that requires an "
                      "ORDERING (Cholesky). That is exactly the VAR's problem. "
                      "It is not solved here; it is avoided while Q stays "
                      "diagonal, and declared when it does not.")

    inv = np.linalg.inv(phi0)
    psis = np.array([p @ inv for p in np.asarray(fc.psi_level, float)])
    L = fc.L
    shares = np.zeros((L, m))
    for l in range(1, L + 1):
        total = float(fc.var_level[l][series, series])
        if total <= 0.0:
            continue
        for j in range(m):
            c = sum(Q[j, j] * psis[t][series, j] ** 2 for t in range(l))
            shares[l - 1, j] = c / total
    return shares, None


def _fitted_deterministics(fc, cast_spec, series):
    """The deterministic coefficients **as estimated by the cast**, not as seeded.

    This is the one thing that cannot be taken from the `.pre` file. The cast
    re-estimates every free parameter jointly, and the deterministic ones move:
    on the canonical case the two `omega_d1` go from the univariate seeds to
    -0.040867 / -0.094588, which is what the C reports for the joint fit.

    Using the seeds instead produces a level forecast that is silently *the
    univariate one*: the stochastic part is right, the calendar effect is the
    one from before the transfer was there. It looks reasonable, it matches the
    C's `-0` run, and it is wrong.

    The free parameters live at the head of the series' univariate block, in
    `build_slots` order: every free omega of every intervention first, then
    every free delta. The fixed ones keep their declared value.
    """
    from .estimate import unpack

    model = cast_spec.series[series].spec.model
    if getattr(fc, "x", None) is None:
        raise ValueError("the forecast does not carry the estimated vector: the "
                         "fitted deterministics cannot be recovered")
    xs = unpack(np.asarray(fc.x, float), cast_spec)["series"][series]

    itv_omega = [list(i.omega) for i in model.interventions]
    itv_delta = [list(i.delta) for i in model.interventions]

    k = 0
    for iv, itv in enumerate(model.interventions):
        for j in range(len(itv.omega)):
            if itv.omega_free[j]:
                itv_omega[iv][j] = float(xs[k]); k += 1
    for iv, itv in enumerate(model.interventions):
        for j in range(len(itv.delta)):
            if itv.delta_free[j]:
                itv_delta[iv][j] = float(xs[k]); k += 1
    return itv_omega, itv_delta


def to_level(fc, cast_spec, series=0, origin=None, transformed=False):
    """Turn the forecast of `w` into a forecast of the LEVEL, in original units.

    The cast models `w`, which is the series after Box-Cox, differencing and
    with the deterministic part removed. Three things stand between that and a
    number a reader can use::

        w = delta(B) (z - xi),      z = boxcox(level),   xi = deterministic part

    so the level comes back by undoing them in order: integrate `w` against the
    observed history of `z - xi`, add the deterministic effect evaluated **at the
    forecast dates**, and invert the Box-Cox.

    The middle step is the one that is easy to forget and impossible to notice:
    without it the forecast decays to `mu` and looks perfectly reasonable, just
    without the seasonality. The C had that very bug filed once.

    The pieces come from `fue` — `_build_xi`, `_nonsop_coefs`, `_inv_boxcox` —
    rather than being rewritten here: the deterministic calendar, the individual
    seasonal factors and the Box-Cox with its `refactor` are exactly the kind of
    delicate detail that must have a single source of truth.
    """
    from fue.forecast import _build_xi, _inv_boxcox, _nonsop_coefs

    sc = cast_spec.series[series]
    model = sc.spec.model
    ts = model.series
    nobs = ts.nobs
    freq = ts.freq if ts.freq > 0 else 1
    L = fc.L

    itv_omega, itv_delta = _fitted_deterministics(fc, cast_spec, series)
    xi = _build_xi(model, nobs, freq, L, itv_omega, itv_delta)   # 1-indexado

    from fue.cast_us import _boxcox
    z = np.array([_boxcox(v, model.boxlam, model.refactor) for v in ts.data])

    # u = z - xi, the stochastic part of the LEVEL, which delta(B) acts on
    u = np.zeros(nobs + L)
    u[:nobs] = z - xi[1:nobs + 1]

    # delta(B): its coefficients come from fue, which already handles the
    # individual seasonal factors (ifadf). The convention is `rnsop`'s:
    # u_t = w_t + sum_k r_k u_(t-k).
    r = np.asarray(_nonsop_coefs(model.d, model.D, freq,
                                 ifadf=(model.ifadf or None)), float)
    o = nobs if origin is None else origin

    for l in range(1, L + 1):
        acc = fc.f[l - 1, series]
        for k in range(1, len(r) + 1):
            acc += r[k - 1] * u[o + l - 1 - k]
        u[o + l - 1] = acc

    z_f = np.array([u[o + l - 1] + xi[o + l] for l in range(1, L + 1)])
    level = np.array([_inv_boxcox(v, model.boxlam, model.refactor) for v in z_f])
    if not transformed:
        return level
    # `z` is the level in the TRANSFORMED scale (Box-Cox, before inverting it),
    # which is what the variation columns are computed on: the C's `ystar`.
    # Differencing there and not on the level is the point -- with lambda = 0 a
    # difference of 100*log IS the percentage change, exactly.
    return level, z_f, z[:o]
