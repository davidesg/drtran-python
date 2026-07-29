"""Estimación conjunta por máxima verosimilitud exacta.

Minimiza el objetivo escalado de Mauricio (1995, §3 ec. 3.5) con el mismo BFGS
factorizado que usan fue y drvarma (`raxopt`, Dennis & Schnabel A9.4.1). No se
reimplementa ni el optimizador ni la verosimilitud: sólo se conectan.

El objetivo
-----------
La verosimilitud concentrada es

    ll = C − 0.5·n·( m·log f1 + log f2 )

así que maximizarla equivale a minimizar `f1^m · f2`. Se normaliza a 1.0 en el
punto inicial, como hace `objcfunc` en el C y `objective` en fue:

    F(x) = (f1/f1₀)^m · (f2/f2₀)

Un punto rechazado (ifault ≠ 0, Q no definida positiva, AR pegado al círculo…)
devuelve 1.0: no mejora sobre el arranque, así que el optimizador se aleja de él.
Es la estrategia del propio artículo (§3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cast import cast_diagonal


@dataclass
class Fit:
    """Resultado de la estimación conjunta."""

    x: np.ndarray
    loglik: float
    ifault: int
    termcode: int
    nit: int
    cast_spec: object
    converged: bool

    def __repr__(self):                                    # pragma: no cover
        estado = "CONVERGIÓ" if self.converged else "SE DETUVO"
        return (f"Fit(logL={self.loglik:.6f}, {estado}, termcode={self.termcode}, "
                f"nit={self.nit}, npar={len(self.x)})")


def _f1f2(x, cast_spec, xitol):
    """(f1, f2, ifault) del cast en x, vía el `elf` de drvarma."""
    from drvarma.estimate_py import _elf_f1f2

    phi, theta, mu, w, sigma, ifault = cast_diagonal(x, cast_spec)
    if ifault:
        return None, None, int(ifault)
    f1, f2, ifa = _elf_f1f2(w, mu, phi, theta, sigma, xitol)
    return float(f1), float(f2), int(ifa)


def loglik(x, cast_spec, xitol=-1e-3):
    """Log-verosimilitud exacta concentrada en x (drvmlest.c:est [4])."""
    f1, f2, ifa = _f1f2(x, cast_spec, xitol)
    if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    _phi, _t, _m, w, _s, _i = cast_diagonal(x, cast_spec)
    n, m = w.shape
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def fit(cast_spec, x0=None, xitol=-1e-3, maxits=500, grtol=1e-7, sptol=1e-7):
    """Estima el modelo conjunto y devuelve un `Fit`.

    `x0` por defecto son las semillas del `.pre` (las estimaciones univariantes de
    fue) con las transferencias en cero — es decir, se arranca en el escalón
    diagonal y se deja que el optimizador añada la dinámica.

    Sobre `termcode`: 1-2 es convergencia (gradiente / paso), **3 es parada sin
    mejora**, que aquí es NORMAL cuando se arranca en el óptimo — el caso del
    escalón diagonal, donde las semillas del `.pre` ya lo son. 4-5 es fallo real.
    """
    from drvarma import _qnewt

    from .cast import x0_from_pre

    x0 = np.asarray(x0_from_pre(cast_spec) if x0 is None else x0, float)
    npar = len(x0)

    f1_0, f2_0, ifa0 = _f1f2(x0, cast_spec, xitol)
    if ifa0 or f1_0 is None or not (f1_0 > 0.0 and f2_0 > 0.0):
        return Fit(x=x0, loglik=float("-inf"), ifault=int(ifa0 or 5),
                   termcode=0, nit=0, cast_spec=cast_spec, converged=False)

    m = cast_spec.m

    def objetivo(xv):
        f1, f2, ifa = _f1f2(np.asarray(xv, float), cast_spec, xitol)
        if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
            return 1.0                       # punto rechazado: no mejora
        return (f1 / f1_0) ** m * (f2 / f2_0)

    if npar == 0:
        ll, ifa = loglik(x0, cast_spec, xitol)
        return Fit(x=x0, loglik=ll, ifault=ifa, termcode=1, nit=0,
                   cast_spec=cast_spec, converged=True)

    # raxopt trabaja sobre un vector 1-indexado (hueco inicial sin usar)
    xk = np.zeros(npar + 1)
    xk[1:] = x0

    def func1(xk1):
        return objetivo(xk1[1:npar + 1])

    _fk, _bfac, nit, termcode = _qnewt.raxopt(func1, npar, xk, maxits, grtol, sptol)
    x_hat = xk[1:npar + 1].copy()

    ll, ifa = loglik(x_hat, cast_spec, xitol)
    return Fit(x=x_hat, loglik=ll, ifault=int(ifa), termcode=int(termcode),
               nit=int(nit), cast_spec=cast_spec,
               converged=int(termcode) in (1, 2))


def unpack(fit_or_x, cast_spec=None):
    """Separa el vector estimado en sus bloques, en el orden de `shootx`.

    Devuelve un dict con `links` (lista de (omega, delta) por enlace), `series`
    (el trozo univariante de cada serie, tal como lo entiende fue) y
    `log_var_ratio`.
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
    return {"links": links, "series": series, "log_var_ratio": x[idx:]}
