"""Identificación de la RED: el `-p` del sistema entero.

Puerto de `identify_network` (`drtran.c`). Es el paso 3 de la escalera de la
escuela: estimado el modelo **diagonal**, se leen las CCF de sus RESIDUOS y de
ahí se proponen las relaciones dinámicas del sistema — quién mueve a quién, con
qué retardo — y las covarianzas contemporáneas (Muñoz Polo 2001, §2.6).

Convención, la del C::

    ccf_ij(k) = corr(a_i(t), a_j(t+k)),   con i < j,   |r| > 2/√n significativo

    k > 0   a_i antecede a a_j   ⇒  enlace  i → j
    k < 0   a_j antecede a a_i   ⇒  enlace  j → i
    k = 0   contemporáneo        ⇒  liberar la covarianza q[i,j]
    ambos lados significativos   ⇒  RETROALIMENTACIÓN

Es una GUÍA, no la red
----------------------
Lo dice el propio C y conviene repetirlo: esto propone CANDIDATOS. Hay que podar
por exogeneidad (a una serie exógena no le entra nada), por aciclicidad (un DAG
no admite ciclos) y por verosimilitud del retardo. La red identificada no es la
final: el punto de intervención del analista entre escalones es parte del método.

Dos decisiones heredadas, y por qué
-----------------------------------
* **La ventana de búsqueda se acota** a 2 periodos estacionales (u 8 sin
  estacionalidad), y a n/4 y a 12: una transferencia con retardo mayor es
  inverosímil, y cuantos más retardos se miran, más falsos positivos por
  contraste múltiple.
* **Con retroalimentación se toma el lado dominante**, avisando. No cabe en un
  DAG de una vía, y elegir en silencio sería peor que decirlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .identify import ccf

from .cast import Link


def residuals(x, cast_spec, embed=True, xitol=-1e-3, structural=False):
    """Los residuos `a` del modelo, (n, m). Es `elf` con `atf=True`.

    No se recalculan a mano: los devuelve el mismo `elf` que puntúa la
    verosimilitud, así que son los residuos EXACTOS del filtro, con su
    inicialización pre-muestral, y no una aproximación truncada.

    `structural=True` deshace la normalización de Φ(0): `a ← Φ(0)·a`. Lo piden
    los DIAGNÓSTICOS de la transferencia. Con un enlace contemporáneo (b=0) el
    cast empotrado mete ω₀ en el retardo cero, de modo que Φ(0) ≠ I y los
    residuos de la forma reducida salen correlacionados **por construcción**
    (Σ₁₂ = ω₀·σ²_X). Medir la adecuación sobre ésos es medir la correlación que
    la propia transferencia genera y llamarla mala especificación: el
    portmanteau se dispara y condena a un modelo correcto. Para leer las CCF de
    la identificación de red da igual, porque ahí no hay enlaces todavía.
    """
    from drvarma._engine import elf_c

    from .cast import cast_diagonal
    from .embed import cast_embedded

    phi0 = None
    if embed:
        out = cast_embedded(np.asarray(x, float), cast_spec, with_phi0=structural)
        if structural:
            phi, theta, mu, w, sigma, ifault, phi0 = out
        else:
            phi, theta, mu, w, sigma, ifault = out
    else:
        phi, theta, mu, w, sigma, ifault = cast_diagonal(np.asarray(x, float),
                                                         cast_spec)
    if ifault:
        return None, int(ifault)

    n, m = w.shape
    _ll, _f1, _f2, a, ifa = elf_c(m, n, phi.shape[0], theta.shape[0],
                                  mu, phi, theta, sigma, w, 1.0, xitol, True)
    if ifa:
        return None, int(ifa)
    res = np.asarray(a)[1:, 1:]
    if structural and phi0 is not None:
        res = res @ np.asarray(phi0, float).T
    return res, 0


def _bs_de_un_lado(c, nlags, thr):
    """(b, s, pico) del bloque significativo de un lado, o None. Puerto de
    `net_bs_from_side`.

    La estructura es el **bloque CONTIGUO** desde b: un pico aislado lejano es
    ruido, y con bandas al 5 % se espera uno de cada veinte fuera por azar.
    """
    b = last = -1
    pico = 0.0
    for k in range(1, nlags + 1):
        if abs(c[k]) > thr:
            if b < 0:
                b = k
            last = k
        elif b >= 0 and k > last + 1:
            break
        if abs(c[k]) > abs(pico):
            pico = c[k]
    if b < 0:
        return None
    last = b
    while last + 1 <= nlags and abs(c[last + 1]) > thr:
        last += 1
    return b, last - b, pico


@dataclass
class Candidato:
    """Un enlace propuesto, con el pico que lo sostiene."""

    out: int
    inp: int
    b: int
    s: int
    pico: float

    @property
    def link(self):
        return Link(out=self.out, inp=self.inp, b=self.b, r=0, s=self.s)


@dataclass
class RedIdentificada:
    """Lo que propone la lectura de las CCF residuales."""

    enlaces: list = field(default_factory=list)      # list[Candidato]
    covarianzas: list = field(default_factory=list)  # (i, j, r0), i < j
    retroalimentacion: list = field(default_factory=list)
    nlags: int = 0
    banda: float = 0.0
    nombres: list = field(default_factory=list)

    @property
    def links(self):
        """Los enlaces como `Link`, listos para `build_cast_spec`."""
        return [c.link for c in self.enlaces]

    @property
    def ciclo(self):
        """El ciclo del grafo propuesto, o `None`. Se espera que a veces lo haya.

        Leer las CCF par a par no impone aciclicidad, así que la propuesta puede
        salir cíclica — y entonces NO es estimable como está. No es un fallo de
        la identificación: es la parte del trabajo que le toca al analista, podar.
        """
        from .network import find_cycle

        return find_cycle(self.links, len(self.nombres))

    def __repr__(self):                                    # pragma: no cover
        return (f"RedIdentificada({len(self.enlaces)} enlaces, "
                f"{len(self.covarianzas)} covarianzas, banda {self.banda:.3f})")


def identify_network(cast_spec, x=None, nlags=None, embed=True):
    """Lee las CCF de los residuos del DIAGONAL y propone la red.

    `x` por defecto son las semillas del `.pre`. Lo natural es pasarle el óptimo
    del escalón diagonal (`fit(...).x`), que es lo que hace el C: la red se lee
    en los residuos del modelo diagonal ESTIMADO.
    """
    from .cast import x0_from_pre

    x = x0_from_pre(cast_spec) if x is None else np.asarray(x, float)
    a, ifault = residuals(x, cast_spec, embed=embed)
    if ifault:
        raise RuntimeError(f"no se pueden obtener los residuos: ifault={ifault}")

    n, m = a.shape
    thr = 2.0 / math.sqrt(n)

    if nlags is None:
        nlags = min(max(n // 4, 6), 12)
        freq = getattr(cast_spec.series[0].spec.model.series, "freq", 1) or 1
        tope = 2 * freq if freq > 1 else 8
        nlags = min(nlags, tope)

    red = RedIdentificada(nlags=nlags, banda=thr, nombres=list(cast_spec.names))

    for i in range(m):
        for j in range(i + 1, m):
            cpos = ccf(a[:, i], a[:, j], nlags)      # k ≥ 0: lado i → j
            cneg = ccf(a[:, j], a[:, i], nlags)      # k ≥ 0 de (j,i): lado j → i
            if not np.any(cpos) and not np.any(cneg):
                continue                              # serie degenerada

            if abs(cpos[0]) > thr:
                red.covarianzas.append((i, j, float(cpos[0])))

            pos = _bs_de_un_lado(cpos, nlags, thr)
            neg = _bs_de_un_lado(cneg, nlags, thr)

            if pos and neg:
                red.retroalimentacion.append((i, j, pos[0], pos[2], neg[0], neg[2]))
                # No cabe en un DAG de una via: se toma el dominante y se avisa.
                if abs(pos[2]) >= abs(neg[2]):
                    neg = None
                else:
                    pos = None
            if pos:
                red.enlaces.append(Candidato(out=j, inp=i, b=pos[0], s=pos[1],
                                             pico=pos[2]))
            if neg:
                red.enlaces.append(Candidato(out=i, inp=j, b=neg[0], s=neg[1],
                                             pico=neg[2]))

    # Los enlaces reales suelen ser los mas fuertes; los picos lejanos espurios
    # caen al fondo. Orden estable, para que empatados salgan en el orden de las
    # series y el informe sea reproducible.
    red.enlaces.sort(key=lambda c: -abs(c.pico))
    return red


def report_network(red):
    """El informe, en el formato del C, pero con las covarianzas por ÍNDICE.

    El `-i` del C imprime `q[EI,EU] = free` con NOMBRES bajo un rótulo que dice
    «paste into a -c file» — y su `.cns` **no lee nombres** en las `q`, sólo
    índices numéricos del triángulo inferior (`q[i,j]`, i > j), que es lo que sí
    escribe su modo guiado `-g`. Aquí se emite directamente lo que se puede
    pegar, con el nombre al lado en un comentario.
    """
    nb = red.nombres
    L = ["=" * 61,
         "  IDENTIFICACIÓN DE LA RED  (CCF de los residuos del diagonal)",
         "=" * 61,
         "  Muñoz Polo (2001) §2.6: las relaciones dinámicas del sistema se leen",
         "  en las CCF de los residuos del modelo DIAGONAL. Esto es una GUÍA de",
         "  candidatos, no la red final: podar por exogeneidad (a una serie",
         "  exógena no le entra nada), aciclicidad (el DAG no admite ciclos) y",
         "  verosimilitud del retardo.",
         f"  Búsqueda hasta k={red.nlags}; |r| > {red.banda:.3f} es significativo.",
         ""]

    for i, j, bp, pp, bn, pn in red.retroalimentacion:
        L.append(f"  [retroalimentación]  {nb[i]} <-> {nb[j]} : "
                 f"{nb[i]}→{nb[j]} k={bp}({pp:+.2f}), "
                 f"{nb[j]}→{nb[i]} k={bn}({pn:+.2f})  -> se toma el dominante")

    L.append("  CONTEMPORÁNEAS  (k=0; liberar la covarianza de las innovaciones):")
    if not red.covarianzas:
        L.append("    (ninguna por encima de la banda)")
    for i, j, r0 in red.covarianzas:
        L.append(f"    {nb[i]:<4s} - {nb[j]:<4s}   r(0) = {r0:+.3f}")

    L += ["", "  ENLACES DIRIGIDOS  (transferencias candidatas, de más a menos):"]
    if not red.enlaces:
        L.append("    (ninguno por encima de la banda)")
    for c in red.enlaces:
        L.append(f"    {nb[c.inp]:<4s} -> {nb[c.out]:<4s}   pico {c.pico:+.3f}"
                 f"   propuesta  b={c.b} r=0 s={c.s}")

    if red.enlaces:
        L += ["", "  RED PROPUESTA  (para un fichero -n / read_dag):"]
        for c in red.enlaces:
            L.append(f"    {nb[c.out]} <- {nb[c.inp]}   {c.b} 0 {c.s}")

    if red.covarianzas:
        L += ["", "  COVARIANZAS PROPUESTAS  (para un fichero -c / read_cns;",
              "  los índices son la posición en la línea de órdenes):"]
        for i, j, _r in red.covarianzas:
            L.append(f"    q[{j + 1},{i + 1}] = free      # {nb[i]} · {nb[j]}")

    if red.retroalimentacion:
        L += ["", f"  NOTA: {len(red.retroalimentacion)} par(es) con retroalimentación",
              "  (los dos sentidos). Un DAG de transferencias de una vía se quedó",
              "  con el lado dominante; conviene mirarlos a mano."]
    L.append("=" * 61)
    return "\n".join(L)


def write_guided(red, nombre):
    """Modo GUIADO (`-g`): escribe `NOMBRE.dag` y `NOMBRE.cns` listos para estimar.

    Se escriben tal cual salen, **sin podar**: la poda es del analista. Si la
    propuesta trae un ciclo se escribe igual, con el ciclo anotado en cabecera —
    esconderlo o podarlo por cuenta propia sería peor: el fichero es el borrador
    sobre el que se decide, y `read_dag` lo rechazará mientras el ciclo siga ahí.

    Devuelve las dos rutas.
    """
    from .network import write_dag

    dag, cns = f"{nombre}.dag", f"{nombre}.cns"
    write_dag(dag, red.links, red.nombres)
    c = red.ciclo
    if c is not None:
        ruta = " -> ".join(red.nombres[i] for i in c)
        with open(dag) as f:
            cuerpo = f.read()
        with open(dag, "w") as f:
            f.write(f"# OJO: la propuesta tiene un CICLO y asi no es estimable:\n"
                    f"#   {ruta}\n"
                    f"# Un DAG no admite ciclos. Poda uno de esos enlaces --el de\n"
                    f"# pico menor, o el que no tenga sentido -- antes de estimar.\n"
                    + cuerpo)
    with open(cns, "w") as f:
        f.write("# Covarianzas contemporáneas propuestas por la lectura de las\n"
                "# CCF residuales del modelo diagonal. Índices = posición en la\n"
                "# línea de órdenes; el triángulo inferior, i > j.\n")
        for i, j, r0 in red.covarianzas:
            f.write(f"q[{j + 1},{i + 1}] = free      "
                    f"# {red.nombres[i]} · {red.nombres[j]}, r(0) = {r0:+.3f}\n")
    return dag, cns
