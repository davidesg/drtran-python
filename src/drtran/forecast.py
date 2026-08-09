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


def _nu_of_links(x, cast_spec, kmax):
    """The impulse response of every link, as `compute_irf` gives it."""
    from .cast import compute_irf

    x = np.asarray(x, float)
    idx = 0
    out = []
    for l in cast_spec.links:
        om = x[idx:idx + l.s + 1]; idx += l.s + 1
        de = x[idx:idx + l.r]; idx += l.r
        out.append(compute_irf(om, de, l.b, kmax))
    return out


def _system_forecast(f, w, x, cast_spec, L, kmax):
    """Rebuild the OBSERVED series' forecast under the SUBTRACTION cast.

    Port of the `we` loop in `drtran.c:transfer_forecast`. With that cast series
    `i` of the VARMA is the NOISE, `N_t = w_i - SUM_k nu_k(B) w_inp(k)`, so
    `forecast_mean` returns the noise's path and the transfer has to be added
    back::

        we[i][n+l] = f[i][l] + SUM_{k: out=i} SUM_j nu_k[j] * we[inp(k)][n+l-j]

    In TOPOLOGICAL order, because an output's future needs its inputs' future
    first. With the embedded cast none of this happens — the transfer is already
    inside the VARMA and adding it again would count it twice, which in the C
    once inflated the standard deviation by 40 %.
    """
    from .embed import _topological_order

    n, m = w.shape
    nus = _nu_of_links(x, cast_spec, kmax)
    we = np.zeros((n + L, m))
    we[:n] = w
    for i in _topological_order(m, cast_spec.links):
        for l in range(1, L + 1):
            tt = n + l - 1                      # 0-based index into `we`
            acc = f[l - 1, i]
            for k, lk in enumerate(cast_spec.links):
                if lk.out != i:
                    continue
                top = min(tt + 1, len(nus[k]))
                for j in range(top):            # nus[k][j] weights lag j
                    acc += nus[k][j] * we[tt - j, lk.inp]
            we[tt, i] = acc
    return we[n:].copy()


def _system_psi(psi, x, cast_spec, L, kmax):
    """The psi weights of the SYSTEM, not of the VARMA. Port of the `pt` loop.

        Psi_ij(B) = d_ij psi_i(B) + SUM_{k: out=i} nu_k(B) Psi_{inp(k),j}(B)

    Series `i` responds to innovation `j` by two routes: its own noise (i == j)
    and everything reaching it through the network. That is why forecasting an
    output requires forecasting its inputs, and why the output's forecast error
    inherits every innovation upstream of it, each propagated through the nu(B)
    it crosses. Without this the variance under the subtraction cast is the
    NOISE's, which is smaller — an error in the flattering direction.
    """
    from .embed import _topological_order

    m = psi.shape[1]
    nus = _nu_of_links(x, cast_spec, kmax)
    pt = np.zeros_like(psi)
    for i in _topological_order(m, cast_spec.links):
        for j in range(m):
            for t in range(L + 1):
                acc = psi[t][i, i] if i == j else 0.0
                for k, lk in enumerate(cast_spec.links):
                    if lk.out != i:
                        continue
                    for v in range(min(t + 1, len(nus[k]))):
                        acc += nus[k][v] * pt[t - v][lk.inp, j]
                pt[t][i, j] = acc
    return pt


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

    # THE SUBTRACTION CAST NEEDS THE TRANSFER PUT BACK. There, series i of the
    # VARMA is the NOISE, so both the point forecast and the psi weights are the
    # noise's; the observed series is recovered by the topological recursions
    # above. With the embedded cast the transfer is already inside the VARMA and
    # doing this would count it twice.
    if not embed and cast_spec.links:
        kmax = max(l.b + l.s + 1 for l in cast_spec.links)
        if any(l.r for l in cast_spec.links):
            kmax = min(w.shape[0] + L, max(kmax, 200))
        n_used = w.shape[0] if origin is None else origin
        f = _system_forecast(f, w[:n_used], np.asarray(x, float), cast_spec,
                             L, kmax)
        psi = _system_psi(psi, np.asarray(x, float), cast_spec, L, kmax)

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

    # Same reason as in `to_level`: for a dispatched model the OUTPUT's level
    # variance is the by-parts one -- the noise's psi weights plus each input's
    # propagated through nu(B) -- not this integration of the joint psi.
    if cast_spec.links and getattr(cast_spec, "needs_subtracting", False):
        _l, _z, v_parts = forecast_by_parts(x, cast_spec, L=L, origin=origin,
                                            xitol=xitol)
        var_level = np.array(var_level, float)
        var_level[:, 0, 0] = v_parts

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


def _levels_of(x, cast_spec, series, L):
    """`u = boxcox(data) - xi`, the stochastic LEVEL the operator acts on.

    Plus the deterministic path and the operator's coefficients, all from
    `fue` so there is no second source of truth for the calendar.
    """
    from fue.cast_us import _boxcox
    from fue.forecast import _build_xi, _nonsop_coefs

    model = cast_spec.series[series].spec.model
    ts = model.series
    nobs = ts.nobs
    freq = ts.freq if ts.freq > 0 else 1
    shim = type("_X", (), {"x": np.asarray(x, float)})()
    io, idl = _fitted_deterministics(shim, cast_spec, series)
    xi = _build_xi(model, nobs, freq, L, io, idl)
    z = np.array([_boxcox(v, model.boxlam, model.refactor) for v in ts.data])
    u = np.zeros(nobs + L)
    u[:nobs] = z - xi[1:nobs + 1]
    r = np.asarray(_nonsop_coefs(model.d, model.D, freq,
                                 ifadf=(model.ifadf or None)), float)
    return u, xi, r, model, nobs


def _psi_scalar(phi, theta, i, L):
    """The psi weights of series `i`'s own ARMA, out of the block-diagonal cast.

    `psi_weights` works on the (p, m, m) form; a single series' block is scalar,
    so it goes in as (p, 1, 1) and comes back out of the corner.
    """
    ph = np.asarray(phi)[:, i, i].reshape(-1, 1, 1) if len(phi) else np.zeros((0, 1, 1))
    th = np.asarray(theta)[:, i, i].reshape(-1, 1, 1) if len(theta) else np.zeros((0, 1, 1))
    return psi_weights(ph, th, L)[:, 0, 0]


def _integrate_psi(psi, r, L):
    """`psi*(B) = psi(B)/op(B)`: the psi weights of the INTEGRATED process.

    Built from `r` rather than from `(d, D, s)` so the individual seasonal
    factors are carried too -- `integrated_weights` rebuilds the operator from
    the three integers and would drop `ifadf`.
    """
    out = np.zeros(L + 1)
    for k in range(L + 1):
        acc = psi[k] if k < len(psi) else 0.0
        for j in range(1, min(len(r), k) + 1):
            acc += r[j - 1] * out[k - j]
        out[k] = acc
    return out


def forecast_by_parts(x, cast_spec, L=12, origin=None, xitol=-1e-3):
    """TASTE's transfer forecast: the parts forecast apart, joined on LEVELS.

    The SUBTRACTING cast's forecast, and it is a different procedure from the
    embedded one rather than the same one with a correction -- see
    `docs/LEVEL_TRANSFER_PLAN.md`. With the embedded cast the transfer IS the
    VARMA, so forecasting the VARMA forecasts everything, in one recursion.
    Here the engine only ever sees the NOISE: the transfer was removed before it
    looked at anything, so it has to be put back, and putting it back means
    forecasting the inputs too. That is not a shortcoming of the cast; it is
    what "subtracting" means.

    `TFFO.PAS`, three problems joined at the end::

        FOR t := 1 TO L DO
           FOR j := 1 TO T DO
              FOR i := 0 TO lags1[j] DO
                 IF t > i THEN sum2 += NU[j][i] * fc[j][t-i]     { its FORECAST }
                 ELSE          sum2 += NU[j][i] * Data[..][N-B+t-i]  { its LEVEL }
           fc[0][t] := sum1 + fcN[t]                             { + the NOISE }

    1. each input is forecast univariately, by its own model;
    2. the noise is forecast by its own ARMA, on its own level;
    3. they are joined as `y = nu(B)x + N` on the LEVELS, each input observed
       where the index is in the past and forecast where it is not.

    **The question that produced BUG-8 never arises here.** No differencing
    appears in the recombination, so "which operator does the input carry" has
    no meaning: the input enters as a level, exactly as it does in `CalcNoise`.

    Returns `(level, z, var_level)`: the forecast on the original scale, on the
    Box-Cox scale, and the LEVEL forecast error variance per horizon.
    """
    from .cast import cast_diagonal
    from .netid import residuals

    phi, theta, mu, W, sigma, ifault = cast_diagonal(x, cast_spec)
    if ifault:
        raise ValueError(f"the cast failed with ifault={ifault}")
    a, ifa = residuals(x, cast_spec, embed=False, xitol=xitol)
    if ifa:
        raise RuntimeError(f"cannot obtain the residuals: ifault={ifa}")

    # THE SCALE, and the file already warns about getting it wrong: the cast
    # returns Q, not Sigma -- the likelihood is concentrated and sigma2 comes
    # out separately. Reading Q as Sigma gives the right shape and the wrong
    # magnitude, and it is easy to miss because the point forecast is unaffected.
    from drvarma._engine import elf_c
    _n, _m = W.shape
    _lg, _f1, _f2, _aa, _ifa = elf_c(_m, _n, phi.shape[0], theta.shape[0],
                                     mu, phi, theta, sigma, W, 1.0, xitol, False)
    if _ifa or not _f1 > 0:
        raise RuntimeError(f"cannot concentrate sigma2: ifault={_ifa}")
    sigma = np.asarray(sigma, float) * (float(_f1) / (_n * _m))

    m = cast_spec.m
    us, xis, rs, models, nobss = {}, {}, {}, {}, {}
    for i in range(m):
        us[i], xis[i], rs[i], models[i], nobss[i] = _levels_of(x, cast_spec, i, L)
    n0 = min(nobss.values())
    for i in range(m):                       # align at the END, as the cast does
        off = nobss[i] - n0
        if off:
            us[i] = np.concatenate([us[i][off:off + n0], np.zeros(L)])
            xis[i] = np.concatenate([xis[i][:1], xis[i][1 + off:]])

    kmax = max((l.b + l.s + 1) for l in cast_spec.links) if cast_spec.links else 1
    if any(l.r for l in cast_spec.links):
        kmax = max(kmax, min(n0 + L, 200))
    nus = _nu_of_links(x, cast_spec, kmax)

    # [1] The NOISE on levels: N = u_out - SUM_j nu_j(B) u_inp(j). This is
    #     `CalcNoise`, and it is where the transfer relates the LEVELS.
    N = np.zeros(n0 + L)
    N[:n0] = us[0][:n0]
    for k, lk in enumerate(cast_spec.links):
        if lk.out != 0:
            continue
        for t in range(n0):
            top = min(len(nus[k]), t + 1)
            N[t] -= float(np.dot(nus[k][:top], us[lk.inp][t::-1][:top]))

    # `origin` is in DATA indices, as in `to_level` -- NOT in stationary ones,
    # which is what `forecast_mean` wants. They differ by the operator's order,
    # and passing one where the other belongs is silent and wrong.
    o = n0 if origin is None else int(origin)
    f = forecast_mean(phi, theta, mu, W, a, L, o - (n0 - W.shape[0]))

    # [2] Integrate each part's DIFFERENCED forecast back to its own level: the
    #     noise by the OUTPUT's operator (W[:,0] = op_out(N)), each input by its
    #     own.
    def integra(dest, col, r):
        for l in range(1, L + 1):
            acc = f[l - 1, col]
            for k in range(1, len(r) + 1):
                acc += r[k - 1] * dest[o + l - 1 - k]
            dest[o + l - 1] = acc

    integra(N, 0, rs[0])
    for i in range(1, m):
        integra(us[i], i, rs[i])

    # [3] Join on the levels.
    uy = us[0]
    for l in range(1, L + 1):
        t = o + l - 1
        acc = N[t]
        for k, lk in enumerate(cast_spec.links):
            if lk.out != 0:
                continue
            top = min(len(nus[k]), t + 1)
            acc += float(np.dot(nus[k][:top], us[lk.inp][t::-1][:top]))
        uy[t] = acc

    from fue.forecast import _inv_boxcox
    mdl = models[0]
    z = np.array([uy[o + l - 1] + xis[0][o + l] for l in range(1, L + 1)])
    level = np.array([_inv_boxcox(v, mdl.boxlam, mdl.refactor) for v in z])

    # [4] The variance, split the same way (TFFO.PAS 336-360): the noise's own
    #     psi weights plus each input's propagated through nu(B), all INTEGRATED
    #     so they are level weights. Unlike TASTE this uses the full Sigma
    #     rather than the diagonal, so a freed covariance is not ignored.
    psi_n = _integrate_psi(_psi_scalar(phi, theta, 0, L), rs[0], L)
    coef = np.zeros((L + 1, m))
    coef[:, 0] = psi_n
    for k, lk in enumerate(cast_spec.links):
        if lk.out != 0:
            continue
        psi_i = _integrate_psi(_psi_scalar(phi, theta, lk.inp, L), rs[lk.inp], L)
        conv = np.convolve(nus[k], psi_i)[:L + 1]
        coef[:len(conv), lk.inp] += conv
    var = np.zeros(L + 1)
    for t in range(1, L + 1):
        acc = 0.0
        for j in range(t):
            acc += float(coef[j] @ sigma @ coef[j])
        var[t] = acc
    return level, z, var


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

    # A DISPATCHED model's level does not come from this recursion. There the
    # VARMA holds the NOISE, and the transfer has to be put back on the LEVELS
    # with each input forecast by its own model -- which is a different
    # procedure, not this one with a correction. Measured on the passthrough,
    # the one-step level RMSE falls 6.6-20.0 % by taking the other route.
    if (series == 0 and cast_spec.links
            and getattr(cast_spec, "needs_subtracting", False)):
        lvl, z_f, _var = forecast_by_parts(
            getattr(fc, "x", None), cast_spec, L=fc.L, origin=origin)
        if not transformed:
            return lvl
        from fue.cast_us import _boxcox as _bc
        m0 = cast_spec.series[0].spec.model
        o0 = m0.series.nobs if origin is None else origin
        zz = np.array([_bc(v, m0.boxlam, m0.refactor) for v in m0.series.data])
        return lvl, z_f, zz[:o0]

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
