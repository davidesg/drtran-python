"""The slot table: free parameters -> full structure (`expand_params`).

Port of `drtran.c`'s slot table (`add_slot` / `build_slots` /
`read_constraints` / `expand_params`). It is what turns the cast into a small
DSL: the vector the optimizer sees is **not** the vector the cast sees.

    xfree  --expand-->  xfull  --cast-->  Phi, Theta, mu, w, Sigma  --elf-->  l

Every position of the full vector is a **slot** with a stable name
(`omega1[0]`, `theta_2[B^1]`, `q[5,2]`, `log(var3/var1)`...) and one of five
natures:

===========  =============================================================
`free`       the optimizer estimates it
`fixed`      a constant value  (`omega3[1] = 0.5`)
`alias`      SHARED with another slot  (`delta1[1] = phi_2[B^1]`)
`product`    IS the product of two others, with a sign  (`x = -y * z`)
`lincomb`    a linear combination of terms, each a slot or slot*slot
===========  =============================================================

The last three are what the `shootx` of Mauricio's legacy code did by hand:
factorized numerators with parameters shared between the transfer and the input's
ARMA, and fixed factors such as the (1-B) that imposes nu_num(1) = 0.

The gradient needs no chain rule
--------------------------------
`expand` is applied INSIDE the objective, so the optimizer sees a function of the
free parameters and `cdgrad` differentiates it by finite differences. A product
or a linear combination propagates on its own. It is the C's own decision, and it
is why adding expressions to the DSL does not touch the optimizer.

The slot order is NOT the C's
-----------------------------
The C groups by CLASS (all the ARMA, then all the deterministics, then the
means); here it groups by SERIES, because the univariate block is produced by
`fue._build_initial_x` and fue decides the order. It does not matter: **the
`.cns` goes by NAME**, not by position, and the names are the same. What IS
checked is that the total matches `cast_spec.npar`, which is what ties the two
enumerations together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FREE, FIXED, ALIAS, PRODUCT, LINCOMB = 0, 1, 2, 3, 4

_KIND_NAME = {FREE: "free", FIXED: "fixed", ALIAS: "shared",
              PRODUCT: "product", LINCOMB: "lincomb"}

MAX_LC_TERMS = 6        # terms per linear combination; enough for m6 (as in the C)


@dataclass
class Slot:
    """One position of the full vector."""

    name: str
    kind: int = FREE
    value: float = 0.0          # FIXED: the value. PRODUCT: the sign (+/-1).
    pa: int = -1                # PRODUCT/ALIAS: first operand (or the representative)
    pb: int = -1                # PRODUCT: second operand
    terms: tuple = ()           # LINCOMB: ((sign, a, b|-1), ...); b = -1 => no product

    @property
    def kind_name(self):
        return _KIND_NAME[self.kind]


def _fmt_freq(f):
    """`f=1`, not `f=1.0`: the C writes it with `%d` and m6's `.cns` uses it so."""
    f = float(f)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:g}"


def _univariate_names(model, si):
    """The names of a series' univariate block, in fue's order.

    An EXACT mirror of `fue.cast_us._build_initial_x`: if the two enumerations
    drift apart, the names stop matching the positions and the `.cns` starts
    constraining the wrong parameter — silently. That is why `build_slots` checks
    the total against `cast_spec.npar`.

    The names are the C's (`add_arma_slots`), **collisions included**: two
    regular AR factors produce two `phi_i[B^1]`, because the C numbers the power
    INSIDE the factor. `SlotTable.index` keeps the first, just like `find_slot`.
    It is an inherited limitation, not an oversight; `report()` warns about it.
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

    def factors(facs, frees, sym, step):
        for k, fac in enumerate(facs):
            free = frees[k] if frees is not None else None
            for j in range(len(fac)):
                if free is None or free[j]:
                    out.append(f"{sym}_{si}[B^{(j + 1) * step}]")

    factors(model.ar,   model.ar_free,   "phi",   1)
    factors(model.ar_s, model.ar_s_free, "phi",   sper)
    factors(model.ma,   model.ma_free,   "theta", 1)
    factors(model.ma_s, model.ma_s_free, "theta", sper)

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
    """The whole map: names, natures, and the `expand` that resolves them."""

    def __init__(self, slots):
        self.slots = list(slots)
        self._by_name = {}
        for i, s in enumerate(self.slots):
            self._by_name.setdefault(s.name, i)     # the FIRST wins, as find_slot
        self._resolve()

    # ── queries ─────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.slots)

    @property
    def names(self):
        return [s.name for s in self.slots]

    def index(self, name):
        """The slot's position, or -1. A mirror of `find_slot`."""
        return self._by_name.get(name.strip(), -1)

    def _idx_or_raise(self, name, ctx=""):
        i = self.index(name)
        if i < 0:
            raise KeyError(f"unknown parameter: {name!r}{ctx}")
        return i

    # ── free <-> slot maps ──────────────────────────────────────────────────
    def _resolve(self):
        """The C's `resolve_slots`: n_free is what the optimizer sees."""
        self.free_of_slot = [-1] * len(self.slots)
        self.slot_of_free = []
        for i, s in enumerate(self.slots):
            if s.kind == FREE:
                self.free_of_slot[i] = len(self.slot_of_free)
                self.slot_of_free.append(i)

    @property
    def n_free(self):
        return len(self.slot_of_free)

    # ── the heart of it: expand_params ──────────────────────────────────────
    def expand(self, xfree):
        """Free vector -> full vector. Port of `expand_params`.

        Aliases are resolved in ONE pass because the chains were already
        flattened when the constraint was read (the C does the same: it follows
        the chain to the final representative in `read_constraints`). Products
        and linear combinations are iterated to a fixed point, so that
        expressions may depend on other expressions.
        """
        xfree = np.asarray(xfree, float)
        if len(xfree) != self.n_free:
            raise ValueError(f"expected {self.n_free} free, got {len(xfree)}")

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
            changed = False
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
                    changed = True
            if not changed:
                break
        return xfull

    def pack(self, xfull):
        """Full vector -> free vector (to seed the optimizer)."""
        xfull = np.asarray(xfull, float)
        if len(xfull) != len(self.slots):
            raise ValueError(f"expected {len(self.slots)} slots, got {len(xfull)}")
        return np.array([xfull[i] for i in self.slot_of_free], float)

    # ── constraints ─────────────────────────────────────────────────────────
    def set_free(self, name):
        i = self._idx_or_raise(name)
        self.slots[i] = Slot(self.slots[i].name, FREE)
        self._resolve()

    def set_fixed(self, name, value):
        i = self._idx_or_raise(name)
        self.slots[i] = Slot(self.slots[i].name, FIXED, value=float(value))
        self._resolve()

    def set_shared(self, name, other):
        """`a = b`: a single degree of freedom in two places."""
        i = self._idx_or_raise(name)
        j = self._idx_or_raise(other, f" (on the right of {name!r})")
        while self.slots[j].kind == ALIAS:      # flatten the chain, as the C does
            j = self.slots[j].pa
        if i == j:
            raise ValueError(f"{name!r} cannot be shared with itself")
        self.slots[i] = Slot(self.slots[i].name, ALIAS, pa=j)
        self._resolve()

    def set_product(self, name, a, b, sign=1.0):
        """`x = [-]y * z`: the coefficient IS the product of two others."""
        i = self._idx_or_raise(name)
        ia = self._idx_or_raise(a, f" (factor of {name!r})")
        ib = self._idx_or_raise(b, f" (factor of {name!r})")
        if i in (ia, ib):
            raise ValueError(f"{name!r} cannot be a factor of itself")
        self.slots[i] = Slot(self.slots[i].name, PRODUCT, value=float(sign),
                             pa=ia, pb=ib)
        self._resolve()

    def set_lincomb(self, name, terms):
        """`x = [+-]t1 [+-]t2 ...`, each term a slot or a slot*slot product.

        `terms` is a list of `(sign, name_a, name_b|None)`.
        """
        i = self._idx_or_raise(name)
        if not 1 <= len(terms) <= MAX_LC_TERMS:
            raise ValueError(f"{name!r}: between 1 and {MAX_LC_TERMS} terms")
        res = []
        for sg, a, b in terms:
            ia = self._idx_or_raise(a, f" (term of {name!r})")
            ib = self._idx_or_raise(b, f" (term of {name!r})") if b else -1
            if i == ia or i == ib:
                raise ValueError(f"{name!r} cannot be a term of itself")
            res.append((float(sg), ia, ib))
        self.slots[i] = Slot(self.slots[i].name, LINCOMB, terms=tuple(res))
        self._resolve()

    # ── report ──────────────────────────────────────────────────────────────
    def report(self):
        """The table as text, like the C's parameter section."""
        n_dup = len(self.slots) - len(self._by_name)
        out = [f"Structural parameters: {len(self.slots)}   "
               f"(free: {self.n_free}, fixed/shared: {len(self.slots) - self.n_free})"]
        if n_dup:
            out.append(f"  note: {n_dup} repeated name(s) — the `.cns` only "
                       f"reaches the first (a limitation inherited from the C)")
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
        return (f"SlotTable({len(self.slots)} slots, {self.n_free} free)")


def build_slots(cast_spec):
    """A `CastSpec`'s slot table. Port of `build_slots`.

    The order is the full vector's:

        1. transfers: `omega{j}[k]` and `delta{j}[k]` of each link (j 1-based)
        2. each series' univariate block, in fue's order
        3. the variance ratios `log(var{i}/var1)`, i = 2..m
        4. the covariances `q[i,j]`, i > j — **FIXED AT ZERO**

    The covariances always enter the map but start out fixed: a diagonal
    covariance is the default case and freeing one is the analyst's decision, not
    something switched on in bulk. The legacy m6-1 does not free the 15 of its
    system: it frees THREE. It is said in the `.cns` in the same language as
    everything else: `q[5,2] = free`.
    """
    slots = []

    for j, l in enumerate(cast_spec.links, 1):
        for k in range(l.s + 1):
            slots.append(Slot(f"omega{j}[{k}]"))
        for k in range(1, l.r + 1):
            slots.append(Slot(f"delta{j}[{k}]"))

    for si, sc in enumerate(cast_spec.series, 1):
        names = _univariate_names(sc.spec.model, si)
        if len(names) != sc.npar:
            raise RuntimeError(
                f"series {si} ({sc.name}): {len(names)} names for {sc.npar} "
                "parameters. The slot enumeration has drifted from fue's "
                "(_build_initial_x); the names no longer match the positions and "
                "the .cns would constrain the wrong parameter.")
        slots.extend(Slot(n) for n in names)

    for i in range(2, cast_spec.m + 1):
        slots.append(Slot(f"log(var{i}/var1)"))

    if len(slots) != cast_spec.npar:
        raise RuntimeError(f"{len(slots)} slots for npar={cast_spec.npar}")

    for i in range(2, cast_spec.m + 1):
        for j in range(1, i):
            slots.append(Slot(f"q[{i},{j}]", FIXED, value=0.0))

    return SlotTable(slots)


# ── the constraints file (.cns) ──────────────────────────────────────────────
def _parse_term(tok):
    """One term of a linear combination: `slot` or `slot * slot`."""
    if "*" in tok:
        a, b = tok.split("*", 1)
        return a.strip(), b.strip()
    return tok.strip(), None


def read_cns(path, table):
    """Read the constraints file and apply them to `table`. Returns how many.

    Format (the C's, `read_constraints`), with `#` for comments::

        NAME = free            free it (the q[i,j] start out fixed at zero)
        NAME = 0.5             fix it
        NAME = OTHER           SHARE: one degree of freedom in two places
        NAME = [-]OTHER * OTHER  PRODUCT with a sign
        NAME = [+-]t1 [+-]t2 ...  LINEAR COMBINATION (ti = slot or slot*slot)

    A linear combination is detected by an **internal** +/- separator: slot names
    carry no signs. It is the C's own heuristic.
    """
    n = 0
    with open(path) as f:
        for nline, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{nline}: expected NAME = ...: {line!r}")
            lhs, rhs = line.split("=", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            ctx = f" in {path}:{nline}"
            if table.index(lhs) < 0:
                raise KeyError(f"unknown parameter: {lhs!r}{ctx}")

            # a linear combination? a +/- after the first character
            body = rhs[1:] if rhs[:1] in "+-" else rhs
            if "+" in body or "-" in body:
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
                body = rhs
                if body.startswith("-"):
                    sign, body = -1.0, body[1:]
                a, b = _parse_term(body)
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
                raise ValueError(f"cannot interpret '{lhs} = {rhs}'{ctx}") from None
            n += 1
    return n
