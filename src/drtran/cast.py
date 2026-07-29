"""El cast: vector de parámetros → estructura VARMA.

Puerto de `tran_shootx.c` (`shootx`). Empieza por el caso **diagonal sin
transferencia**, que es la puerta de entrada del diseño: con estructura diagonal
la verosimilitud exacta se factoriza, así que la conjunta debe reproducir la SUMA
de las univariantes de fue. Si eso no cuadra, el cast está mal — nunca `elf`.

Qué se replica del C y qué no
-----------------------------
Se replica la SEMÁNTICA y las CONVENCIONES, no la ingeniería:

* El **orden del vector de parámetros** de `shootx`: transferencias (ω, δ) por
  enlace → ARMA por serie → deterministas → medias → covarianza. Mantenerlo no es
  un contrato externo (el `.cns` va por nombres, no por posición), pero hace que
  una discrepancia con el C se localice comparando posición a posición.
* La **normalización de la covarianza**: `Q[1][1] = 1` y `var_i = exp(x_i)` para
  i>1, con la escala concentrada en `sigma2`. Es una decisión deliberada, no un
  accidente: la verosimilitud concentrada de Mauricio (1995, ec. 3.1) depende de Q
  sólo a través de un producto invariante ante Q → cQ, así que dejar las m
  varianzas libres deja una dirección plana y un hessiano singular. El legacy y
  drvarma las dejan libres; aquí no, a propósito.

NO se replica la gestión de memoria, los tensores, los globals ni el 1-indexado.

Lo que NO se reimplementa
-------------------------
La serie estacionaria. `fue.cast_us.cast_us_py()` ya devuelve `w` con Box-Cox,
diferencias y deterministas restados — es lo que hace `build_stationary_series`
en el C. drtran usa el cast univariante de fue por serie y sólo **ensambla** el
VARMA. Reimplementarlo crearía una segunda fuente de verdad para la parte más
delicada del pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeriesCast:
    """El cast univariante de una serie, precomputado (lo fijo)."""

    spec: object            # PreSpec
    est_spec: object        # fue.cast_us.EstSpec
    npar: int               # nº de parámetros libres que consume del vector x
    name: str


@dataclass
class CastSpec:
    """Lo fijo del problema, precomputado una vez antes de optimizar.

    Equivale a `populate_globals` del C, pero sin globals: todo el estado vive
    aquí y se pasa explícitamente.
    """

    series: list = field(default_factory=list)      # list[SeriesCast]
    m: int = 0
    n_stat: int = 0                                  # longitud común de las w
    npar: int = 0

    @property
    def names(self):
        return [s.name for s in self.series]


def _npar_univariante(model):
    """Cuántos parámetros libres consume el cast univariante de fue.

    Se toma de la longitud del vector inicial de fue en vez de recontarlo aquí:
    el recuento y el orden son la misma cosa (`count_npar_build_par` en el C), y
    mantener una segunda copia del recuento es justo lo que acaba divergiendo.
    """
    from fue.cast_us import _build_initial_x

    return len(np.asarray(_build_initial_x(model), float))


def build_cast_spec(specs):
    """Precomputa el cast a partir de los `.pre` leídos (uno por serie).

    `specs[0]` es la SALIDA (la serie 1 del VARMA, la que recibiría las
    transferencias); el resto son las entradas.
    """
    from fue.cast_us import build_est_spec

    if len(specs) < 2:
        raise ValueError("la conjunta necesita al menos 2 series")

    cs = CastSpec()
    for s in specs:
        m = s.model
        cs.series.append(SeriesCast(spec=s, est_spec=build_est_spec(m),
                                    npar=_npar_univariante(m), name=s.name))
    cs.m = len(cs.series)
    # Parámetros: los univariantes de cada serie + la covarianza.
    # Covarianza: var[0] se fija en 1 (la escala la concentra sigma2), luego
    # log(var_i/var_1) para i>0. Las covarianzas nacen FIJAS en cero: sólo se
    # liberan si el .cns lo pide (todavía no implementado).
    cs.npar = sum(s.npar for s in cs.series) + (cs.m - 1)
    return cs


def cast_diagonal(x, cast_spec):
    """Vector de parámetros → estructura VARMA diagonal (sin transferencia).

    Orden de `x`, siguiendo a `shootx` (sin el bloque de transferencias, que aquí
    está vacío):

        1. ARMA + deterministas + media de cada serie (el cast univariante de fue,
           en el orden de `count_npar_build_par`)
        2. covarianza: log(var_i / var_1) para i = 2..m

    Devuelve `(phi, theta, mu, w, sigma, ifault)` listos para `elf_varma`:
    `phi` (p,m,m) y `theta` (q,m,m) diagonales por bloques, `w` (n,m) con las
    series estacionarias alineadas por el final, y `sigma` diagonal.
    """
    from fue.cast_us import cast_us_py

    x = np.asarray(x, float)
    m = cast_spec.m
    idx = 0
    ps, qs, phis, thetas, mus, ws = [], [], [], [], [], []

    for sc in cast_spec.series:
        xi = x[idx:idx + sc.npar]
        idx += sc.npar
        p, q, phi, theta, mu, w, ifault = cast_us_py(xi, sc.est_spec)
        if ifault:
            return None, None, None, None, None, int(ifault)
        ps.append(int(p)); qs.append(int(q))
        phis.append(np.asarray(phi, float)); thetas.append(np.asarray(theta, float))
        mus.append(float(mu)); ws.append(np.asarray(w, float))

    # Covarianza: Q[0][0] = 1, el resto exp(x) (positividad garantizada).
    var = np.ones(m)
    for i in range(1, m):
        var[i] = math.exp(x[idx]); idx += 1
    sigma = np.diag(var)

    # Alineación: si las series tienen distinto d/D sus w tienen distinta
    # longitud. Se alinean por el FINAL (la última observación es la misma fecha)
    # y se recorta a la más corta, que es lo que hace build_stationary_series.
    n = min(len(w) for w in ws)
    W = np.column_stack([w[len(w) - n:] for w in ws])

    p = max(ps) if ps else 0
    q = max(qs) if qs else 0
    PHI = np.zeros((p, m, m))
    THETA = np.zeros((q, m, m))
    for i in range(m):
        for k in range(ps[i]):
            PHI[k, i, i] = phis[i][k]
        for k in range(qs[i]):
            THETA[k, i, i] = thetas[i][k]

    return PHI, THETA, np.asarray(mus, float), W, sigma, 0


def loglik_diagonal(x, cast_spec, xitol=-1e-3):
    """Log-verosimilitud EXACTA CONCENTRADA de la conjunta diagonal.

    `est()` del C no estima la escala: descompone Σ = sigma2·Q con Q[1][1]=1 y
    **concentra** sigma2, que sale analíticamente de f1. Por eso `Q` sólo lleva
    las RAZONES de varianza. Evaluar `elf_varma` con un Σ absoluto en su lugar da
    una verosimilitud distinta — es el error de pasar la identidad como Σ cuando
    las varianzas reales difieren en un factor 1000.

    Se usa `_elf_f1f2` de drvarma (el mismo que usa su estimador) y su fórmula de
    la concentrada, sin tocar `elf`: cualquier discrepancia con fue es un bug de
    este cast.

    `xitol = -1e-3` selecciona la verosimilitud **exacta**, no la aproximada.
    """
    from drvarma.estimate_py import _elf_f1f2

    phi, theta, mu, w, sigma, ifault = cast_diagonal(x, cast_spec)
    if ifault:
        return float("-inf"), int(ifault)
    n, m = w.shape
    f1, f2, ifa = _elf_f1f2(w, mu, phi, theta, sigma, xitol)
    if ifa or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    # drvmlest.c:est [4] — verosimilitud concentrada
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def _sigma2_univariante(sc, x_i, xitol=-1e-3):
    """σ² de una serie en sus semillas, vía el mismo `elf` con m=1.

    No reestima nada: evalúa la verosimilitud univariante en las semillas del
    `.pre` y toma la varianza concentrada, σ² = f1/(n·m) con m=1.
    """
    from fue.cast_us import cast_us_py
    from drvarma.estimate_py import _elf_f1f2

    p, q, phi, theta, mu, w, ifault = cast_us_py(x_i, sc.est_spec)
    if ifault:
        return None
    w = np.asarray(w, float).reshape(-1, 1)
    phi = np.asarray(phi, float).reshape(-1, 1, 1) if p else np.zeros((0, 1, 1))
    theta = np.asarray(theta, float).reshape(-1, 1, 1) if q else np.zeros((0, 1, 1))
    f1, _f2, ifa = _elf_f1f2(w, np.array([float(mu)]), phi, theta,
                             np.ones((1, 1)), xitol)
    if ifa or not f1 > 0.0:
        return None
    return float(f1) / len(w)


def x0_from_pre(cast_spec):
    """Vector inicial: las semillas del `.pre`, que son las estimaciones de fue.

    Es el punto de partida natural — y explica el `termcode 3` del C: en el
    escalón diagonal estas semillas YA son el óptimo, así que la búsqueda lineal
    no puede mejorar y para. No es un fallo.

    Las RAZONES de varianza log(var_i/var_1) NO se siembran en cero: las escalas
    de las series pueden diferir en órdenes de magnitud (en el caso canónico,
    σ²=0.0627 frente a 68.84, razón 1098) y arrancar en 1 deja el punto inicial
    lejísimos — logL −1371 en vez de −767. Se calculan con el mismo `elf`, m=1,
    sobre las semillas de cada `.pre`.
    """
    from fue.cast_us import _build_initial_x

    partes, s2 = [], []
    for sc in cast_spec.series:
        xi = np.asarray(_build_initial_x(sc.spec.model), float)
        partes.append(xi)
        s2.append(_sigma2_univariante(sc, xi))

    razones = np.zeros(cast_spec.m - 1)
    if s2[0]:
        for i in range(1, cast_spec.m):
            if s2[i]:
                razones[i - 1] = math.log(s2[i] / s2[0])
    partes.append(razones)
    return np.concatenate(partes)
