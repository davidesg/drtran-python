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


@dataclass(frozen=True)
class Link:
    """Un enlace de transferencia: `out` recibe de `inp` con ω(B)/δ(B)·B^b.

    Copia de `struct Tlink` del C. Índices de serie 0-based (el C los usa
    1-based); `s` es el grado del numerador y `r` el del denominador, así que el
    enlace aporta (s+1) + r parámetros libres.
    """

    out: int
    inp: int
    b: int = 0
    r: int = 0
    s: int = 0

    @property
    def npar(self):
        return (self.s + 1) + self.r


def compute_irf(omega, delta, b, length):
    """Pesos ν del filtro racional ω(B)/δ(B) con retardo b. Puerto de `compute_irf`.

        ν[t] = ω[t-1-b] + Σ_{j=1..r} δ[j]·ν[t-j]

    `ν[j]` pesa la entrada en el retardo j−1 (1-based en el C; aquí `nu[0]` es el
    peso del retardo 0).

    **Convención de signo de Box-Jenkins**, la misma que `calcnu` de fue: el
    numerador es ω(B) = ω₀ − ω₁B − ω₂B² − …, o sea **el término líder suma y los
    demás restan**. Portar esto al revés es el tipo de error de un carácter que ya
    costó un bug en el propio C (el signo de Nyquist en `CalcNonsOp`).
    """
    omega = np.asarray(omega, float)
    delta = np.asarray(delta, float)
    s = len(omega) - 1
    r = len(delta)
    nu = np.zeros(length)
    for t in range(1, length + 1):
        lag = t - 1 - b
        acc = 0.0
        if 0 <= lag <= s:
            acc = omega[0] if lag == 0 else -omega[lag]
        for j in range(1, r + 1):
            if t > j:
                acc += delta[j - 1] * nu[t - j - 1]
        nu[t - 1] = acc
    return nu


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
    links: list = field(default_factory=list)       # list[Link]
    m: int = 0
    n_stat: int = 0                                  # longitud común de las w
    npar: int = 0

    @property
    def names(self):
        return [s.name for s in self.series]

    @property
    def npar_links(self):
        return sum(l.npar for l in self.links)


def _npar_univariante(model):
    """Cuántos parámetros libres consume el cast univariante de fue.

    Se toma de la longitud del vector inicial de fue en vez de recontarlo aquí:
    el recuento y el orden son la misma cosa (`count_npar_build_par` en el C), y
    mantener una segunda copia del recuento es justo lo que acaba divergiendo.
    """
    from fue.cast_us import _build_initial_x

    return len(np.asarray(_build_initial_x(model), float))


def build_cast_spec(specs, links=None):
    """Precomputa el cast a partir de los `.pre` leídos (uno por serie).

    `specs[0]` es la SALIDA (la serie 1 del VARMA, la que recibe las
    transferencias por defecto); el resto son las entradas. `links` es la lista de
    enlaces; sin enlaces el modelo es el diagonal, que es la puerta de validación.
    """
    from fue.cast_us import build_est_spec

    if len(specs) < 2:
        raise ValueError("la conjunta necesita al menos 2 series")

    cs = CastSpec(links=list(links or []))
    for s in specs:
        m = s.model
        cs.series.append(SeriesCast(spec=s, est_spec=build_est_spec(m),
                                    npar=_npar_univariante(m), name=s.name))
    cs.m = len(cs.series)
    for l in cs.links:
        if not (0 <= l.out < cs.m and 0 <= l.inp < cs.m):
            raise ValueError(f"enlace fuera de rango: {l}")
        if l.out == l.inp:
            raise ValueError(f"un enlace no puede ir de una serie a sí misma: {l}")
    # Orden del vector, siguiendo a shootx: transferencias → univariantes →
    # covarianza. Covarianza: var[0] se fija en 1 (la escala la concentra
    # sigma2), luego log(var_i/var_1) para i>0. Las covarianzas fuera de la
    # diagonal nacen FIJAS en cero: sólo se liberan si el .cns lo pide (todavía
    # no implementado).
    cs.npar = cs.npar_links + sum(s.npar for s in cs.series) + (cs.m - 1)
    return cs


def build_sigma(x, idx, m):
    """Bloque de covarianza del vector: `(Q, idx, ifault)`.

    Q lleva las RAZONES de varianza, no las varianzas: Q[0][0] = 1 y
    Q[i][i] = exp(x), con la escala concentrada en sigma2 (ver la nota de
    cabecera). Las covarianzas fuera de la diagonal van **crudas**, en el orden
    del triángulo inferior por filas — `q[2,1]`, `q[3,1]`, `q[3,2]`, … — que es
    el de la tabla de slots.

    Sólo se leen si el vector las trae: sin tabla de slots el vector termina en
    las razones y el modelo es de covarianza diagonal, que es el caso por
    defecto. Liberar una covarianza es una decisión del analista (`q[5,2] = free`
    en el `.cns`), no algo que se active en bloque.

    **Rechazo si Q no es definida positiva.** Las covarianzas crudas no están
    reparametrizadas, así que el optimizador puede pisar la región donde Q deja
    de serlo; allí la verosimilitud no existe. Se devuelve `ifault` y el objetivo
    responde 1.0, que es la estrategia del propio Mauricio (1995 §3): el punto no
    mejora sobre el arranque y la búsqueda se aleja de él. Es lo mismo que hace el
    C, y en los casos reales esa frontera nunca ha mordido — la correlación más
    fuerte de m6 (−0.41) converge sin acercarse.
    """
    var = np.ones(m)
    for i in range(1, m):
        var[i] = math.exp(x[idx]); idx += 1
    Q = np.diag(var)

    n_off = m * (m - 1) // 2
    if len(x) - idx >= n_off:
        for i in range(1, m):
            for j in range(i):
                Q[i, j] = Q[j, i] = x[idx]; idx += 1
        if n_off and np.any(Q[np.triu_indices(m, 1)] != 0.0):
            try:
                np.linalg.cholesky(Q)
            except np.linalg.LinAlgError:
                return None, idx, 1
    return Q, idx, 0


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

    # --- 1. Transferencias: ω[0..s] y δ[1..r] de cada enlace ---------------
    om, de = [], []
    for l in cast_spec.links:
        om.append(x[idx:idx + l.s + 1]); idx += l.s + 1
        de.append(x[idx:idx + l.r]); idx += l.r

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

    sigma, idx, ifa_q = build_sigma(x, idx, m)
    if ifa_q:
        return None, None, None, None, None, int(ifa_q)

    # Alineación: si las series tienen distinto d/D sus w tienen distinta
    # longitud. Se alinean por el FINAL (la última observación es la misma fecha)
    # y se recorta a la más corta, que es lo que hace build_stationary_series.
    n = min(len(w) for w in ws)
    W = np.column_stack([w[len(w) - n:] for w in ws])

    # --- Transferencias: cada enlace RESTA a su salida ---------------------
    # tr[o][t] = Σ_k ν_j[k]·w_in[t−k+1]; la serie 1 del VARMA pasa a ser el
    # RUIDO N_t = w_Y − Σ_j transferencia_j. Con ω = 0 no se resta nada y el
    # modelo se parte en univariantes independientes: la prueba del puente.
    if cast_spec.links:
        tr = np.zeros_like(W)
        for l, o_j, d_j in zip(cast_spec.links, om, de):
            nu = compute_irf(o_j, d_j, l.b, n)
            xin = W[:, l.inp]
            acc = np.zeros(n)
            for t in range(n):                      # Σ_{k=0..t} ν[k]·x[t−k]
                acc[t] = float(np.dot(nu[:t + 1], xin[t::-1]))
            tr[:, l.out] += acc
        W = W - tr

    p = max(ps) if ps else 0
    q = max(qs) if qs else 0
    PHI = np.zeros((p, m, m))
    THETA = np.zeros((q, m, m))
    for i in range(m):
        for k in range(ps[i]):
            PHI[k, i, i] = phis[i][k]
        for k in range(qs[i]):
            THETA[k, i, i] = thetas[i][k]

    # Restricción del C (shootx [12]): un AR(1) pegado al círculo unidad se
    # rechaza antes de llamar a elf, en vez de dejar que la verosimilitud explote.
    for i in range(m):
        if ps[i] >= 1 and abs(phis[i][0]) >= 0.999:
            return None, None, None, None, None, 1

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

    # Transferencias en CERO: es el modelo diagonal, que ya sabemos que homologa
    # con fue. Arrancar la red desde ahí es lo que hace la escalera — primero el
    # diagonal, luego la dinámica que la CCF de sus residuos sugiera.
    partes, s2 = [np.zeros(cast_spec.npar_links)], []
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
