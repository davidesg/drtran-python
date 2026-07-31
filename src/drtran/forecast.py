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
        raise ValueError(f"origen fuera de la muestra: {origin} de {n}")

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

    @property
    def L(self):
        return self.f.shape[0]

    def se(self, which="level", serie=0):
        """Standard errors per horizon, `sqrt` of the diagonal."""
        v = {"w": self.var_w, "level": self.var_level,
             "diff": self.var_diff, "annual": self.var_annual}[which]
        if v is None:
            raise ValueError(f"no se calculó la varianza '{which}'")
        return np.array([math.sqrt(max(v[l][serie, serie], 0.0))
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
        raise TypeError("hace falta el cast_spec (o pasa un Fit)")

    from .cast import cast_diagonal
    from .embed import cast_embedded

    hacer = cast_embedded if embed else cast_diagonal
    phi, theta, mu, w, sigma, ifault = hacer(np.asarray(x, float), cast_spec)
    if ifault:
        raise RuntimeError(f"el cast falló: ifault={ifault}")

    a, ifa = residuals(x, cast_spec, embed=embed, xitol=xitol)
    if ifa:
        raise RuntimeError(f"no se pueden obtener los residuos: ifault={ifa}")

    # LA ESCALA. El cast devuelve Q, no Sigma: la verosimilitud es CONCENTRADA
    # y sigma2 sale aparte (drvmlest.c:est). Sin multiplicar por el, las
    # varianzas salen con la forma correcta y la magnitud equivocada -- que es
    # exactamente el error que se comete al leer Q como si fuera Sigma.
    from drvarma._engine import elf_c
    n_, m_ = w.shape
    _lg, f1, _f2, _a2, ifa2 = elf_c(m_, n_, phi.shape[0], theta.shape[0],
                                    mu, phi, theta, sigma, w, 1.0, xitol, False)
    if ifa2 or not f1 > 0:
        raise RuntimeError(f"no se puede concentrar sigma2: ifault={ifa2}")
    sigma2 = float(f1) / (n_ * m_)
    sigma = np.asarray(sigma, float) * sigma2

    f = forecast_mean(phi, theta, mu, w, a, L, origin)
    psi = psi_weights(phi, theta, L)
    var_w = error_variance(psi, sigma, L)

    # El nivel: se deshace la diferenciación de CADA serie. d, D y s salen del
    # `.pre`, no del cast — el cast trabaja siempre sobre la serie estacionaria.
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
                    x=np.asarray(x, float))


def report_forecast(fc, serie=0, which="level"):
    """The forecast table for one series, with its 95 % band."""
    nombre = fc.names[serie] if fc.names else f"serie {serie + 1}"
    se = fc.se(which, serie)
    L = ["=" * 61,
         f"  FORECAST — {nombre}  ({which})",
         "=" * 61,
         "   h    forecast      s.e.        95% interval",
         "  " + "-" * 55]
    for l in range(fc.L):
        v, e = fc.f[l, serie], se[l]
        L.append(f"  {l + 1:2d}  {v:11.4f}  {e:9.4f}   "
                 f"[{v - 1.96 * e:10.4f}, {v + 1.96 * e:10.4f}]")
    L.append("=" * 61)
    return "\n".join(L)


# ── back to the level ────────────────────────────────────────────────────────
def _fitted_deterministics(fc, cast_spec, serie):
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

    model = cast_spec.series[serie].spec.model
    if getattr(fc, "x", None) is None:
        raise ValueError("la previsión no trae el vector estimado: no se pueden "
                         "recuperar los deterministas ajustados")
    xs = unpack(np.asarray(fc.x, float), cast_spec)["series"][serie]

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


def to_level(fc, cast_spec, serie=0, origin=None):
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

    sc = cast_spec.series[serie]
    model = sc.spec.model
    ts = model.series
    nobs = ts.nobs
    freq = ts.freq if ts.freq > 0 else 1
    L = fc.L

    itv_omega, itv_delta = _fitted_deterministics(fc, cast_spec, serie)
    xi = _build_xi(model, nobs, freq, L, itv_omega, itv_delta)   # 1-indexado

    from fue.cast_us import _boxcox
    z = np.array([_boxcox(v, model.boxlam, model.refactor) for v in ts.data])

    # u = z - xi, la parte estocastica del NIVEL, sobre la que actua delta(B)
    u = np.zeros(nobs + L)
    u[:nobs] = z - xi[1:nobs + 1]

    # delta(B): sus coeficientes vienen de fue, que ya trata los factores
    # estacionales individuales (ifadf). La convencion es la de `rnsop`:
    # u_t = w_t + sum_k r_k u_(t-k).
    r = np.asarray(_nonsop_coefs(model.d, model.D, freq,
                                 ifadf=(model.ifadf or None)), float)
    o = nobs if origin is None else origin

    for l in range(1, L + 1):
        acc = fc.f[l - 1, serie]
        for k in range(1, len(r) + 1):
            acc += r[k - 1] * u[o + l - 1 - k]
        u[o + l - 1] = acc

    z_f = np.array([u[o + l - 1] + xi[o + l] for l in range(1, L + 1)])
    return np.array([_inv_boxcox(v, model.boxlam, model.refactor) for v in z_f])
