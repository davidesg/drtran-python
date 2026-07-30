"""La tabla de slots: parámetros libres → estructura completa (`expand_params`).

Puerto de la tabla de slots de `drtran.c` (`add_slot` / `build_slots` /
`read_constraints` / `expand_params`). Es lo que convierte al cast en un pequeño
DSL: el vector que ve el optimizador **no** es el vector que ve el cast.

    xfree  ──expand──▶  xfull  ──cast──▶  Φ, Θ, μ, w, Σ  ──elf──▶  ℓ

Cada posición del vector completo es un **slot** con un nombre estable
(`omega1[0]`, `theta_2[B^1]`, `q[5,2]`, `log(var3/var1)`…) y una de cinco
naturalezas:

===========  =============================================================
`free`       lo estima el optimizador
`fixed`      valor constante  (`omega3[1] = 0.5`)
`alias`      COMPARTIDO con otro slot  (`delta1[1] = phi_2[B^1]`)
`product`    ES el producto de otros dos, con signo  (`x = -y * z`)
`lincomb`    combinación lineal de términos, cada uno slot o slot·slot
===========  =============================================================

Los tres últimos son lo que hacía a mano el `shootx` del legacy de Mauricio:
numeradores factorizados con parámetros compartidos entre la transferencia y el
ARMA de la entrada, y factores fijos como el (1−B) que impone ν_num(1) = 0.

El gradiente no necesita regla de la cadena
-------------------------------------------
`expand` se aplica DENTRO del objetivo, así que el optimizador ve una función de
los libres y `cdgrad` la deriva por diferencias finitas. Un producto o una
combinación lineal se propagan solos. Es la misma decisión del C, y es la razón
de que añadir expresiones al DSL no toque el optimizador.

El orden de los slots NO es el del C
------------------------------------
El C agrupa por CLASE (todos los ARMA, luego todos los deterministas, luego las
medias); aquí se agrupa por SERIE, porque el bloque univariante lo produce
`fue._build_initial_x` y el orden lo manda fue. Da igual: **el `.cns` va por
NOMBRES**, no por posiciones, y los nombres son los mismos. Lo que sí se
comprueba es que el total cuadre con `cast_spec.npar`, que es lo que ata las dos
enumeraciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FREE, FIXED, ALIAS, PRODUCT, LINCOMB = 0, 1, 2, 3, 4

_NOMBRE = {FREE: "free", FIXED: "fixed", ALIAS: "shared",
           PRODUCT: "product", LINCOMB: "lincomb"}

MAX_LC_TERMS = 6        # términos por combinación lineal; basta para m6 (como el C)


@dataclass
class Slot:
    """Una posición del vector completo."""

    name: str
    kind: int = FREE
    value: float = 0.0          # FIXED: el valor. PRODUCT: el signo (±1).
    pa: int = -1                # PRODUCT/ALIAS: primer operando (o el representante)
    pb: int = -1                # PRODUCT: segundo operando
    terms: tuple = ()           # LINCOMB: ((signo, a, b|-1), …); b = -1 ⇒ sin producto

    @property
    def kind_name(self):
        return _NOMBRE[self.kind]


def _fmt_freq(f):
    """`f=1`, no `f=1.0`: el C lo escribe con `%d` y el `.cns` de m6 lo usa así."""
    f = float(f)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def _nombres_univariantes(model, si):
    """Los nombres del bloque univariante de una serie, en el orden de fue.

    Espejo EXACTO de `fue.cast_us._build_initial_x`: si las dos enumeraciones se
    separan, los nombres dejan de corresponder a las posiciones y el `.cns`
    empieza a restringir el parámetro equivocado — en silencio. Por eso
    `build_slots` verifica el total contra `cast_spec.npar`.

    Los nombres son los del C (`add_arma_slots`), **colisiones incluidas**: dos
    factores AR regulares producen dos `phi_i[B^1]`, porque el C numera la
    potencia DENTRO del factor. `SlotTable.index` se queda con el primero, igual
    que `find_slot`. Es una limitación heredada, no un descuido; se avisa en
    `report()`.
    """
    sper = int(getattr(model.series, "freq", 1) or 1)
    out = []

    for iv, itv in enumerate(model.interventions, 1):
        for j in range(len(itv.omega)):
            if itv.omega_free[j]:
                out.append(f"omega_d{si}[{iv},{j}]")
    for iv, itv in enumerate(model.interventions, 1):
        for j in range(len(itv.delta)):
            if itv.delta_free[j]:
                out.append(f"delta_d{si}[{iv},{j + 1}]")

    def factores(facs, frees, sym, paso):
        for k, fac in enumerate(facs):
            libre = frees[k] if frees is not None else None
            for j in range(len(fac)):
                if libre is None or libre[j]:
                    out.append(f"{sym}_{si}[B^{(j + 1) * paso}]")

    factores(model.ar,   model.ar_free,   "phi",   1)
    factores(model.ar_s, model.ar_s_free, "phi",   sper)
    factores(model.ma,   model.ma_free,   "theta", 1)
    factores(model.ma_s, model.ma_s_free, "theta", sper)

    for ff in model.ar_f:
        if ff.free:
            out.append(f"phi_{si}[f={_fmt_freq(ff.freq)}]")
    for ff in model.ma_f:
        if ff.free:
            out.append(f"theta_{si}[f={_fmt_freq(ff.freq)}]")

    if model.estimate_mu:
        out.append(f"mu[{si}]")
    return out


class SlotTable:
    """El mapa completo: nombres, naturalezas y el `expand` que los resuelve."""

    def __init__(self, slots):
        self.slots = list(slots)
        self._by_name = {}
        for i, s in enumerate(self.slots):
            self._by_name.setdefault(s.name, i)     # el PRIMERO gana, como find_slot
        self._resolve()

    # ── consulta ────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.slots)

    @property
    def names(self):
        return [s.name for s in self.slots]

    def index(self, name):
        """Posición del slot, o -1. Espejo de `find_slot`."""
        return self._by_name.get(name.strip(), -1)

    def _idx_or_raise(self, name, ctx=""):
        i = self.index(name)
        if i < 0:
            raise KeyError(f"parámetro desconocido: {name!r}{ctx}")
        return i

    # ── mapas libre ↔ slot ──────────────────────────────────────────────────
    def _resolve(self):
        """`resolve_slots` del C: n_free es lo que ve el optimizador."""
        self.free_of_slot = [-1] * len(self.slots)
        self.slot_of_free = []
        for i, s in enumerate(self.slots):
            if s.kind == FREE:
                self.free_of_slot[i] = len(self.slot_of_free)
                self.slot_of_free.append(i)

    @property
    def n_free(self):
        return len(self.slot_of_free)

    # ── el corazón: expand_params ───────────────────────────────────────────
    def expand(self, xfree):
        """Vector libre → vector completo. Puerto de `expand_params`.

        Los alias se resuelven en UNA pasada porque las cadenas ya se aplanaron
        al leer la restricción (el C hace lo mismo: sigue la cadena hasta el
        representante final en `read_constraints`). Los productos y las
        combinaciones lineales se iteran hasta punto fijo, para admitir
        expresiones que dependan de otras expresiones.
        """
        xfree = np.asarray(xfree, float)
        if len(xfree) != self.n_free:
            raise ValueError(f"esperaba {self.n_free} libres, recibí {len(xfree)}")

        xfull = np.zeros(len(self.slots))
        for i, s in enumerate(self.slots):
            if s.kind == FREE:
                xfull[i] = xfree[self.free_of_slot[i]]
            elif s.kind == FIXED:
                xfull[i] = s.value
        for i, s in enumerate(self.slots):
            if s.kind == ALIAS:
                xfull[i] = xfull[s.pa]

        for _ in range(len(self.slots)):
            cambio = False
            for i, s in enumerate(self.slots):
                if s.kind == PRODUCT:
                    nv = s.value * xfull[s.pa] * xfull[s.pb]
                elif s.kind == LINCOMB:
                    nv = 0.0
                    for sg, a, b in s.terms:
                        nv += sg * xfull[a] * (xfull[b] if b >= 0 else 1.0)
                else:
                    continue
                if nv != xfull[i]:
                    xfull[i] = nv
                    cambio = True
            if not cambio:
                break
        return xfull

    def pack(self, xfull):
        """Vector completo → vector libre (para sembrar el optimizador)."""
        xfull = np.asarray(xfull, float)
        if len(xfull) != len(self.slots):
            raise ValueError(f"esperaba {len(self.slots)} slots, recibí {len(xfull)}")
        return np.array([xfull[i] for i in self.slot_of_free], float)

    # ── restricciones ───────────────────────────────────────────────────────
    def set_free(self, name):
        i = self._idx_or_raise(name)
        self.slots[i] = Slot(self.slots[i].name, FREE)
        self._resolve()

    def set_fixed(self, name, value):
        i = self._idx_or_raise(name)
        self.slots[i] = Slot(self.slots[i].name, FIXED, value=float(value))
        self._resolve()

    def set_shared(self, name, other):
        """`a = b`: un solo grado de libertad en dos sitios."""
        i = self._idx_or_raise(name)
        j = self._idx_or_raise(other, f" (a la derecha de {name!r})")
        while self.slots[j].kind == ALIAS:      # aplanar la cadena, como el C
            j = self.slots[j].pa
        if i == j:
            raise ValueError(f"{name!r} no puede compartirse consigo mismo")
        self.slots[i] = Slot(self.slots[i].name, ALIAS, pa=j)
        self._resolve()

    def set_product(self, name, a, b, sign=1.0):
        """`x = [-]y * z`: el coeficiente ES el producto de otros dos."""
        i = self._idx_or_raise(name)
        ia = self._idx_or_raise(a, f" (factor de {name!r})")
        ib = self._idx_or_raise(b, f" (factor de {name!r})")
        if i in (ia, ib):
            raise ValueError(f"{name!r} no puede ser factor de sí mismo")
        self.slots[i] = Slot(self.slots[i].name, PRODUCT, value=float(sign),
                             pa=ia, pb=ib)
        self._resolve()

    def set_lincomb(self, name, terms):
        """`x = [±]t1 [±]t2 …`, con cada término un slot o un producto slot·slot.

        `terms` es una lista de `(signo, nombre_a, nombre_b|None)`.
        """
        i = self._idx_or_raise(name)
        if not 1 <= len(terms) <= MAX_LC_TERMS:
            raise ValueError(f"{name!r}: entre 1 y {MAX_LC_TERMS} términos")
        res = []
        for sg, a, b in terms:
            ia = self._idx_or_raise(a, f" (término de {name!r})")
            ib = self._idx_or_raise(b, f" (término de {name!r})") if b else -1
            if i == ia or i == ib:
                raise ValueError(f"{name!r} no puede ser término de sí mismo")
            res.append((float(sg), ia, ib))
        self.slots[i] = Slot(self.slots[i].name, LINCOMB, terms=tuple(res))
        self._resolve()

    # ── informe ─────────────────────────────────────────────────────────────
    def report(self):
        """La tabla en texto, como la sección de parámetros del C."""
        n_dup = len(self.slots) - len(self._by_name)
        out = [f"Structural parameters: {len(self.slots)}   "
               f"(free: {self.n_free}, fixed/shared: {len(self.slots) - self.n_free})"]
        if n_dup:
            out.append(f"  ojo: {n_dup} nombre(s) repetido(s) — el `.cns` sólo "
                       f"alcanza el primero (limitación heredada del C)")
        for i, s in enumerate(self.slots):
            if s.kind == FREE:
                det = "free"
            elif s.kind == FIXED:
                det = f"= {s.value:.6f}"
            elif s.kind == ALIAS:
                det = f"= {self.slots[s.pa].name}"
            elif s.kind == PRODUCT:
                sg = "-" if s.value < 0 else ""
                det = f"= {sg}{self.slots[s.pa].name} * {self.slots[s.pb].name}"
            else:
                trs = []
                for sg, a, b in s.terms:
                    t = self.slots[a].name + (f" * {self.slots[b].name}" if b >= 0 else "")
                    trs.append(("- " if sg < 0 else "+ ") + t)
                det = "= " + " ".join(trs).lstrip("+ ")
            out.append(f"  {i:3d}  {s.name:<24s} {det}")
        return "\n".join(out)

    def __repr__(self):                                        # pragma: no cover
        return (f"SlotTable({len(self.slots)} slots, {self.n_free} libres)")


def build_slots(cast_spec):
    """La tabla de slots de un `CastSpec`. Puerto de `build_slots`.

    El orden es el del vector completo:

        1. transferencias: `omega{j}[k]` y `delta{j}[k]` de cada enlace (j 1-based)
        2. el bloque univariante de cada serie, en el orden de fue
        3. las razones de varianza `log(var{i}/var1)`, i = 2..m
        4. las covarianzas `q[i,j]`, i > j — **FIJAS EN CERO**

    Las covarianzas entran siempre al mapa pero nacen fijas: la covarianza
    diagonal es el caso por defecto y liberar una es una decisión del analista,
    no algo que se active en bloque. El legacy m6-1 no libera las 15 de su
    sistema: libera TRES. Se dice en el `.cns` con el mismo lenguaje que todo lo
    demás: `q[5,2] = free`.
    """
    slots = []

    for j, l in enumerate(cast_spec.links, 1):
        for k in range(l.s + 1):
            slots.append(Slot(f"omega{j}[{k}]"))
        for k in range(1, l.r + 1):
            slots.append(Slot(f"delta{j}[{k}]"))

    for si, sc in enumerate(cast_spec.series, 1):
        nombres = _nombres_univariantes(sc.spec.model, si)
        if len(nombres) != sc.npar:
            raise RuntimeError(
                f"serie {si} ({sc.name}): {len(nombres)} nombres para {sc.npar} "
                "parámetros. La enumeración de slots se ha separado de la de fue "
                "(_build_initial_x); los nombres ya no corresponden a las "
                "posiciones y el .cns restringiría el parámetro equivocado.")
        slots.extend(Slot(n) for n in nombres)

    for i in range(2, cast_spec.m + 1):
        slots.append(Slot(f"log(var{i}/var1)"))

    if len(slots) != cast_spec.npar:
        raise RuntimeError(f"{len(slots)} slots para npar={cast_spec.npar}")

    for i in range(2, cast_spec.m + 1):
        for j in range(1, i):
            slots.append(Slot(f"q[{i},{j}]", FIXED, value=0.0))

    return SlotTable(slots)


# ── el fichero de restricciones (.cns) ───────────────────────────────────────
def _parse_term(tok):
    """Un término de una combinación lineal: `slot` o `slot * slot`."""
    if "*" in tok:
        a, b = tok.split("*", 1)
        return a.strip(), b.strip()
    return tok.strip(), None


def read_cns(path, table):
    """Lee el fichero de restricciones y las aplica a `table`. Devuelve cuántas.

    Formato (el del C, `read_constraints`), con `#` para comentarios::

        NOMBRE = free            liberar (las q[i,j] nacen fijas en cero)
        NOMBRE = 0.5             fijar
        NOMBRE = OTRO            COMPARTIR: un grado de libertad en dos sitios
        NOMBRE = [-]OTRO * OTRO  PRODUCTO con signo
        NOMBRE = [±]t1 [±]t2 …   COMBINACIÓN LINEAL (ti = slot o slot*slot)

    La combinación lineal se detecta por un separador +/- **interno**: los
    nombres de slot no llevan signos. Es la misma heurística del C.
    """
    n = 0
    with open(path) as f:
        for nlin, linea in enumerate(f, 1):
            linea = linea.split("#", 1)[0].strip()
            if not linea:
                continue
            if "=" not in linea:
                raise ValueError(f"{path}:{nlin}: se esperaba NOMBRE = …: {linea!r}")
            lhs, rhs = linea.split("=", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            ctx = f" en {path}:{nlin}"
            if table.index(lhs) < 0:
                raise KeyError(f"parámetro desconocido: {lhs!r}{ctx}")

            # ¿combinación lineal? un +/- después del primer carácter
            cuerpo = rhs[1:] if rhs[:1] in "+-" else rhs
            if "+" in cuerpo or "-" in cuerpo:
                terms, sg, tok = [], 1.0, ""
                for ch in rhs:
                    if ch in "+-":
                        if tok.strip():
                            a, b = _parse_term(tok)
                            terms.append((sg, a, b))
                            tok = ""
                        sg = 1.0 if ch == "+" else -1.0
                    else:
                        tok += ch
                if tok.strip():
                    a, b = _parse_term(tok)
                    terms.append((sg, a, b))
                try:
                    table.set_lincomb(lhs, terms)
                except (KeyError, ValueError) as e:
                    raise type(e)(f"{e}{ctx}") from None
                n += 1
                continue

            if "*" in rhs:
                sign = 1.0
                cuerpo = rhs
                if cuerpo.startswith("-"):
                    sign, cuerpo = -1.0, cuerpo[1:]
                a, b = _parse_term(cuerpo)
                try:
                    table.set_product(lhs, a, b, sign)
                except (KeyError, ValueError) as e:
                    raise type(e)(f"{e}{ctx}") from None
                n += 1
                continue

            if rhs == "free":
                table.set_free(lhs)
                n += 1
                continue

            if table.index(rhs) >= 0:
                try:
                    table.set_shared(lhs, rhs)
                except ValueError as e:
                    raise ValueError(f"{e}{ctx}") from None
                n += 1
                continue

            try:
                table.set_fixed(lhs, float(rhs))
            except ValueError:
                raise ValueError(f"no sé interpretar '{lhs} = {rhs}'{ctx}") from None
            n += 1
    return n
