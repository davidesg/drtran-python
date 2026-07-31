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
from .embed import cast_embedded


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
    slots: object = None          # SlotTable, si la estimación va restringida
    xfree: object = None          # lo que vio el optimizador (x es el completo)

    # termcode del optimizador (raxopt / qnewtopt.c), con la clasificación que
    # fijó drtran en su hito M1: 1-2 convergencia, 3 parada SIN MEJORA (normal si
    # se arranca en el óptimo, o si se llega a él), 4-5 fallo real.
    _ESTADO = {1: "CONVERGED (gradient)",
               2: "CONVERGED (step)",
               3: "stopped without improvement",
               4: "iteration limit",
               5: "steps of maximum length"}

    @property
    def estado(self):
        return self._ESTADO.get(self.termcode, f"termcode={self.termcode}")

    def __repr__(self):                                    # pragma: no cover
        return (f"Fit(logL={self.loglik:.6f}, {self.estado}, "
                f"termcode={self.termcode}, nit={self.nit}, "
                f"npar={len(self.x)})")


def _f1f2(x, cast_spec, xitol, embed=False):
    """(f1, f2, ifault) del cast en x, vía el `elf` de drvarma.

    `embed=True` usa el cast EMPOTRADO (el de por defecto en el C), que mete la
    transferencia dentro del VARMA sin restar nada, así que no hay truncamiento
    pre-muestral.
    """
    from drvarma._engine import elf_c

    hacer = cast_embedded if embed else cast_diagonal
    phi, theta, mu, w, sigma, ifault = hacer(x, cast_spec)
    if ifault:
        return None, None, int(ifault)
    # El `elf` COMPILADO de drvarma, expuesto para esto: el puerto necesita
    # PUNTUAR una estructura que construye el cast, no ajustar un VARMA libre.
    # Identico al de Python puro (1e-13) y ~100x mas rapido.
    n, m = w.shape
    _lg, f1, f2, _a, ifa = elf_c(m, n, phi.shape[0], theta.shape[0],
                                 mu, phi, theta, sigma, w, 1.0, xitol, False)
    return float(f1), float(f2), int(ifa)


def loglik(x, cast_spec, xitol=-1e-3, embed=False):
    """Log-verosimilitud exacta concentrada en x (drvmlest.c:est [4])."""
    f1, f2, ifa = _f1f2(x, cast_spec, xitol, embed)
    if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    hacer = cast_embedded if embed else cast_diagonal
    _phi, _t, _m, w, _s, _i = hacer(x, cast_spec)
    n, m = w.shape
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def x0_full(cast_spec, slots):
    """Semillas del `.pre` en el espacio COMPLETO de la tabla de slots.

    `x0_from_pre` llega hasta las razones de varianza; la tabla añade detrás las
    covarianzas, que arrancan en cero — o sea, en el modelo de covarianza
    diagonal, que es el escalón anterior de la escalera.
    """
    from .cast import x0_from_pre

    x0 = np.asarray(x0_from_pre(cast_spec), float)
    falta = len(slots) - len(x0)
    if falta < 0:
        raise ValueError(f"la tabla tiene {len(slots)} slots y las semillas "
                         f"{len(x0)}: ¿es la tabla de este cast?")
    return np.concatenate([x0, np.zeros(falta)])


def fit(cast_spec, x0=None, xitol=-1e-3, maxits=500, grtol=1e-7,
        sptol=1e-7, embed=True, slots=None):
    """Estima el modelo conjunto y devuelve un `Fit`.

    `embed=True` (por defecto, como en el C) mete la transferencia DENTRO del
    VARMA; `embed=False` la resta, que es el cast antiguo (`-S`).

    `x0` por defecto son las semillas del `.pre` (las estimaciones univariantes de
    fue) con las transferencias en cero — es decir, se arranca en el escalón
    diagonal y se deja que el optimizador añada la dinámica.

    `slots` es una `SlotTable` (ver `drtran.slots`). Con ella el optimizador
    trabaja en el espacio de los parámetros **libres** y cada evaluación expande
    a la estructura completa: es lo que permite fijar, compartir y expresar unos
    coeficientes en función de otros. Sin ella el vector es el completo y todo es
    libre, salvo lo que el `.pre` ya diera por fijo. `Fit.x` es siempre el vector
    completo; `Fit.xfree` lo que vio el optimizador.

    Sobre `termcode`: 1-2 es convergencia (gradiente / paso), **3 es parada sin
    mejora**, que aquí es NORMAL cuando se arranca en el óptimo — el caso del
    escalón diagonal, donde las semillas del `.pre` ya lo son. 4-5 es fallo real.
    """
    from drvarma import _qnewt

    from .cast import x0_from_pre

    if slots is None:
        x_ini = np.asarray(x0_from_pre(cast_spec) if x0 is None else x0, float)
        expandir = lambda v: v                                    # noqa: E731
    else:
        xfull0 = np.asarray(x0_full(cast_spec, slots) if x0 is None else x0, float)
        if len(xfull0) != len(slots):
            raise ValueError(f"con tabla de slots, x0 es el vector COMPLETO: "
                             f"esperaba {len(slots)}, recibí {len(xfull0)}")
        x_ini = slots.pack(xfull0)
        expandir = slots.expand

    npar = len(x_ini)

    def _ll(v):
        return loglik(expandir(v), cast_spec, xitol, embed)

    def _empaquetar(v, ll, ifa, termcode, nit):
        return Fit(x=np.asarray(expandir(v), float), loglik=ll, ifault=int(ifa),
                   termcode=int(termcode), nit=int(nit), cast_spec=cast_spec,
                   converged=int(termcode) in (1, 2), slots=slots,
                   xfree=np.asarray(v, float))

    f1_0, f2_0, ifa0 = _f1f2(expandir(x_ini), cast_spec, xitol, embed)
    if ifa0 or f1_0 is None or not (f1_0 > 0.0 and f2_0 > 0.0):
        return _empaquetar(x_ini, float("-inf"), ifa0 or 5, 0, 0)

    m = cast_spec.m

    def objetivo(xv):
        f1, f2, ifa = _f1f2(expandir(np.asarray(xv, float)), cast_spec, xitol, embed)
        if ifa or f1 is None or not (f1 > 0.0 and f2 > 0.0):
            return 1.0                       # punto rechazado: no mejora
        return (f1 / f1_0) ** m * (f2 / f2_0)

    if npar == 0:
        ll, ifa = _ll(x_ini)
        return _empaquetar(x_ini, ll, ifa, 1, 0)

    # raxopt trabaja sobre un vector 1-indexado (hueco inicial sin usar)
    xk = np.zeros(npar + 1)
    xk[1:] = x_ini

    def func1(xk1):
        return objetivo(xk1[1:npar + 1])

    _fk, _bfac, nit, termcode = _qnewt.raxopt(func1, npar, xk, maxits, grtol, sptol)
    x_hat = xk[1:npar + 1].copy()

    ll, ifa = _ll(x_hat)
    return _empaquetar(x_hat, ll, ifa, termcode, nit)


def unpack(fit_or_x, cast_spec=None):
    """Separa el vector estimado en sus bloques, en el orden de `shootx`.

    Devuelve un dict con `links` (lista de (omega, delta) por enlace), `series`
    (el trozo univariante de cada serie, tal como lo entiende fue),
    `log_var_ratio` y `cov` (las covarianzas del triángulo inferior, vacío si el
    vector no las trae).
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
    razones = x[idx:idx + cast_spec.m - 1]; idx += cast_spec.m - 1
    return {"links": links, "series": series,
            "log_var_ratio": razones, "cov": x[idx:]}
