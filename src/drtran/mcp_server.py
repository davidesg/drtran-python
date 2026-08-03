"""mtram — MCP server for TRANSFER FUNCTION models and networks, with drtran.

The suite's three assistants, and why this one is separate (the full argument is
in `docs/ARCHITECTURE_MCP.md`):

    art     one series: ARIMA + interventions          engine: fue
    mtram   transfer functions and networks (a DAG)    engine: drtran   <- here
    sima    simultaneous VARMA                         engine: drvarma

`mtram` is ART's natural continuation: it starts from the `.pre` files ART
writes, so the univariate rung is already climbed and committed to a file the
analyst can read. `sima` is a different lineage — a classical symmetric VARMA —
and it makes a different claim: there everything is endogenous and the impulse
response is not identified without an ordering. Here exogeneity is **declared and
tested**, and that is what identifies nu(k) without ordering anything.

**The handoff.** If `identify_network` proposes a DAG with a CYCLE, the system
has no topological order, cannot be cast as a triangular VARMA, and is therefore
simultaneous — that is when the analyst should move to `sima`. It is a contrast,
not a preference.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:                                        # pragma: no cover
    raise ImportError("mtram needs the MCP extra: pip install 'drtran[mcp]'")

import drtran
from drtran.cast import Link, build_cast_spec


_INSTRUCTIONS = """
Eres mtram — asistente de modelos de FUNCIÓN DE TRANSFERENCIA y de REDES de
transferencias (motor drtran, máxima verosimilitud EXACTA). Eres el escalón que
sigue a ART en la escalera de la escuela Box-Jenkins-Treadway.

══════════════════════════════════════════════════════
IDIOMA / LANGUAGE
══════════════════════════════════════════════════════
Responde SIEMPRE en el idioma del usuario (inglés por defecto si es ambiguo).
Estas instrucciones y las salidas pueden venir en español: tradúcelas.
── Always respond in the user's language; translate tool output, never paste
Spanish at an English-speaking user.

══════════════════════════════════════════════════════
DE DÓNDE PARTES: LOS .pre, NO LAS SERIES CRUDAS
══════════════════════════════════════════════════════
mtram NO identifica modelos univariantes: eso es ART. Parte de ficheros `.pre`,
que son el CONTRATO de continuidad de la escalera — ART estima el mejor modelo
univariante de cada serie y lo deja escrito ahí.

Si el usuario no tiene `.pre`, NO improvises un modelo univariante: dile que
construya cada serie en ART primero. Un asistente de transferencia que se
inventa modelos univariantes es un asistente que tiene opiniones sobre ellos.

PREGUNTA INICIAL OBLIGATORIA:
  "¿Cuál de tus series es la SALIDA (la que quieres explicar) y cuáles las
   ENTRADAS? Y ¿cómo deseas proceder: 1) GUIADO paso a paso, o 2) AUTÓNOMO?"

══════════════════════════════════════════════════════
LA ESCALERA — Y DÓNDE TERMINA TU COMPETENCIA
══════════════════════════════════════════════════════
1. load_pre            carga los .pre. El PRIMERO es la salida.
2. identify_link       preblanqueo + CCF de un enlace -> propone (b, r, s)
   plot_ccf            EL INSTRUMENTO: enséñaselo y léelo CON el analista
   identify_network    CCF de los residuos del DIAGONAL -> propone el DAG entero
3. estimate            estima conjuntamente; presenta la ECUACIÓN
4. diagnose            adecuación (k>=0) y exogeneidad (k<0)
5. impulse_response    nu(k), acumulada y GANANCIA, con errores típicos
   plot_impulse_response
6. forecast            nivel + variación + anual, con bandas
   plot_forecast
7. evaluate            fuera de muestra: MAE/RMSE/MAPE por horizonte

⚠ SI identify_network PROPONE UN CICLO: el sistema es SIMULTÁNEO. No se puede
  expresar como VARMA triangular. DÍSELO al usuario y remítelo a `sima`, el
  asistente de VARMA simultáneo. No podes el ciclo por tu cuenta para que
  "funcione": la poda es juicio del analista, no aritmética.

══════════════════════════════════════════════════════
REGLAS QUE NO SE NEGOCIAN
══════════════════════════════════════════════════════
1. PRESENTA SIEMPRE LA ECUACIÓN QUE DEVUELVE EL TOOL, VERBATIM, dentro de su
   bloque de código. NUNCA construyas tu propia tabla de parámetros: los signos,
   los errores típicos y las convenciones son del motor, no tuyos.
2. DECLARA, NO ELIJAS. Si una herramienta se niega —descomposición de varianza
   con Q no diagonal, hessiano no definido positivo, DAG cíclico— transmite la
   negativa y su motivo. No busques la manera de obtener un número igualmente.
3. LA RED IDENTIFICADA ES UNA GUÍA, no un veredicto. Hay que podar por
   exogeneidad, aciclicidad y verosimilitud del retardo, y eso lo decide el
   analista. Presenta los candidatos con su evidencia y ESPERA.
4. Las bandas de previsión del NIVEL son ASIMÉTRICAS con modelos en logaritmos,
   y las columnas STD están en la escala TRANSFORMADA (son porcentajes). No
   construyas bandas sumando 1.96 desviaciones típicas a un nivel.
"""

mcp = FastMCP("mtram — Multivariate TRAnsfer Models (drtran)",
              instructions=_INSTRUCTIONS)

# ── session state ──────────────────────────────────────────────────────────
_SPECS: dict[str, list] = {}       # name -> [PreSpec, ...], first is the output
_LINKS: dict[str, list] = {}       # name -> [Link, ...]
_FITS: dict[str, object] = {}      # name -> Fit
_TABLES: dict[str, object] = {}    # name -> SlotTable


def _require(name: str):
    if name not in _SPECS:
        raise ValueError(f"no hay ningún caso llamado {name!r}; llama a load_pre "
                         f"primero (conocidos: {sorted(_SPECS)})")
    return _SPECS[name]


def _require_fit(name: str):
    # el caso PRIMERO: decir "no está estimado" de un caso que no existe manda
    # al analista a estimar algo que no ha cargado
    _require(name)
    if name not in _FITS:
        raise ValueError(f"el caso {name!r} no está estimado; llama a estimate")
    return _FITS[name]


def _cast(name: str):
    return build_cast_spec(_require(name), links=_LINKS.get(name, []))


def _png(name: str, kind: str, path: str = "") -> str:
    return path or os.path.join(tempfile.gettempdir(), f"mtram_{name}_{kind}.png")


# ── 1. the input: .pre files, the ladder's contract ────────────────────────
@mcp.tool()
def load_pre(name: str, paths: str) -> str:
    """Load the `.pre` files of a case. THE FIRST ONE IS THE OUTPUT.

    `paths` is a comma-separated list. These are ART's output: each carries a
    series and the univariate model ART estimated for it. mtram does not build
    univariate models — if one is missing, send the analyst to ART.
    """
    ps = [p.strip() for p in paths.split(",") if p.strip()]
    if len(ps) < 2:
        raise ValueError("hacen falta al menos dos .pre: una salida y una entrada")
    specs = []
    warn = []
    for p in ps:
        if not os.path.exists(p):
            raise ValueError(f"no encuentro {p}")
        s = drtran.load_pre(p)
        specs.append(s)
        w = drtran.check_scale(s)
        if w:
            warn.append(w)
    _SPECS[name] = specs
    _LINKS.pop(name, None)
    _FITS.pop(name, None)

    out = [f"Caso {name!r}: {len(specs)} series.",
           f"  SALIDA : {specs[0].name}  ({specs[0].nobs} obs, freq {specs[0].freq})"]
    for s in specs[1:]:
        out.append(f"  entrada: {s.name}  ({s.nobs} obs, freq {s.freq})")
    out.append("")
    for s in specs:
        m = s.model
        out.append(f"  {s.name}: lambda={m.boxlam:g} d={m.d} D={m.D} "
                   f"refactor={m.refactor:g} deterministas={len(m.interventions)}")
    if warn:
        out.append("")
        out += ["  AVISO: " + w for w in warn]
    return "\n".join(out)


# ── 2. identification ──────────────────────────────────────────────────────
@mcp.tool()
def identify_link(name: str, input_index: int = 1, band: str = "constant") -> str:
    """Identify (b, r, s) of ONE link by prewhitening and the CCF.

    Filters the input with ITS OWN ARMA and applies the SAME filter to the
    output, so r(k) estimates the impulse response weights directly. Proposes
    (b, r, s) from the CONTIGUOUS block, and judges exogeneity by a portmanteau
    over k < 0 — feedback there means a single-input transfer model does not
    hold, and you must say so.

    `band`: "constant" (2/sqrt(N), what the C does) or "haugh-box"
    (2/sqrt(N-|k|), what the original paper says).
    """
    specs = _require(name)
    if not 1 <= input_index < len(specs):
        raise ValueError(f"input_index fuera de rango (1..{len(specs)-1})")
    cs = build_cast_spec(specs, links=[Link(0, input_index, 0, 0, 0)])
    idt = drtran.identify(cs, cs.links[0], band=band)
    return drtran.report_identification(idt, names=(specs[input_index].name, specs[0].name))


@mcp.tool()
def identify_network(name: str, nlags: int = 0) -> str:
    """Propose the whole NETWORK from the residual CCFs of the DIAGONAL fit.

    Estimates the diagonal model (no transfers) and reads the cross-correlations
    of its residuals: who leads whom, at what lag, and which innovations move
    together contemporaneously.

    ⚠ IF THE PROPOSAL CONTAINS A CYCLE the system is SIMULTANEOUS: it has no
    topological order and cannot be cast as a triangular VARMA. Tell the analyst
    and route them to `sima`. Do not prune the cycle yourself to make it fit.
    """
    specs = _require(name)
    cs = build_cast_spec(specs)
    f0 = drtran.fit(cs, embed=True)
    net = drtran.identify_network(cs, x=f0.x, nlags=nlags or None)
    txt = drtran.report_network(net)

    cyc = net.cycle
    if cyc is not None:
        route = " -> ".join(net.names[i] for i in cyc)
        txt += ("\n\n  *** CICLO: " + route + "\n"
                "  La propuesta NO es un DAG: el sistema es SIMULTÁNEO y no se\n"
                "  puede expresar como VARMA triangular. Esto es el límite de\n"
                "  mtram. Remite al analista a `sima` (VARMA simultáneo), o\n"
                "  pídele que PODE uno de esos enlaces — la poda es su juicio.\n")
    return txt


@mcp.tool()
def set_network(name: str, links_json: str) -> str:
    """Fix the network to estimate: a JSON list of {out, inp, b, r, s}.

    `out` and `inp` are indices into the loaded `.pre` list (0 is the output).
    A cycle is refused here too — `read_dag`'s rule, in memory.
    """
    from drtran.network import check_acyclic

    specs = _require(name)
    spec = json.loads(links_json)
    links = [Link(out=int(d["out"]), inp=int(d["inp"]), b=int(d.get("b", 0)),
                  r=int(d.get("r", 0)), s=int(d.get("s", 0))) for d in spec]
    names = [s.name for s in specs]
    check_acyclic(links, len(specs), names)          # raises on a cycle
    _LINKS[name] = links
    _FITS.pop(name, None)
    return "\n".join([f"Red de {name!r}: {len(links)} enlace(s)."] +
                     [f"  {names[l.out]} <- {names[l.inp]}   b={l.b} r={l.r} s={l.s}"
                      for l in links])


@mcp.tool()
def plot_ccf(name: str, input_index: int = 1, lags: int = 0,
             path: str = "") -> str:
    """PLOT the prewhitened CCF of one link — the identification instrument.

    Show this to the analyst and read it WITH them; the numbers alone do not
    carry the shape. Right of zero the input leads, and the first significant bar
    is `b`; left of zero the OUTPUT leads, which is feedback and which a
    single-input transfer model assumes away. Bars on both sides mean the
    specification does not hold.

    It is drvarma's canonical CCF drawing, so it looks the same as in `sima` and
    as drvus drew it. Writes a PNG and returns its path.
    """
    from .plots import plot_ccf as _pc
    from .plots import prewhitened_pair, save

    specs = _require(name)
    if not 1 <= input_index < len(specs):
        raise ValueError(f"input_index fuera de rango (1..{len(specs)-1})")
    cs = build_cast_spec(specs, links=[Link(0, input_index, 0, 0, 0)])
    a, b = prewhitened_pair(cs, cs.links[0])
    freq = int(getattr(specs[0].model.series, "freq", 1) or 1)
    fig = _pc(a, b, freq=freq, lags=(lags or None),
              names=(specs[input_index].name, specs[0].name))
    return save(fig, _png(name, f"ccf{input_index}", path))


@mcp.tool()
def plot_impulse_response(name: str, link_index: int = 0, path: str = "") -> str:
    """PLOT nu(k) and its cumulative sum, each with a 95 % band.

    Left panel: the response to a ONE-OFF unit shock. Right: to a PERMANENT
    change, converging to the gain, which is drawn as a line because that
    convergence is the thing to look at. Writes a PNG and returns its path.
    """
    from .estimate import standard_errors
    from .irf import impulse_response as _irf
    from .plots import plot_irf, save

    f = _require_fit(name)
    se = standard_errors(f)
    ir = _irf(f, link_index=link_index, cov=(None if se.ifault else se.cov))
    return save(plot_irf(ir), _png(name, f"irf{link_index}", path))


@mcp.tool()
def plot_forecast(name: str, horizon: int = 12, series_index: int = 0,
                  path: str = "") -> str:
    """PLOT the level forecast with its band, over the recent history.

    The band is ASYMMETRIC under a log model — it is formed in the transformed
    scale and mapped back — so it is drawn as it really is, not as a symmetric
    ribbon. Writes a PNG and returns its path.
    """
    from .forecast import forecast as _fcast
    from .forecast import level_band
    from .plots import plot_forecast as _pf
    from .plots import save

    f = _require_fit(name)
    cs = f.cast_spec
    fc = _fcast(f, L=horizon, embed=f.embed)
    lvl, lo, hi = level_band(fc, cs, series=series_index)
    hist = cs.series[series_index].spec.ts.data
    fig = _pf(lvl, lo, hi, history=hist, name=cs.names[series_index])
    return save(fig, _png(name, f"fcst{series_index}", path))


# ── 3. estimation ──────────────────────────────────────────────────────────
@mcp.tool()
def estimate(name: str, embed: bool = True, cns_path: str = "") -> str:
    """Estimate every parameter JOINTLY by exact maximum likelihood.

    `embed=True` (default, as in the C) puts the transfer INSIDE the VARMA, so
    there is no pre-sample truncation. `embed=False` subtracts it instead — the
    old cast, and what TASTE does.

    `cns_path` is an optional constraints file (free / fixed / shared / product /
    linear combination).

    PRESENT THE EQUATION BLOCK VERBATIM. Do not rebuild the parameter table.
    """
    from drtran.cli import report_fit
    from drtran.estimate import standard_errors
    from drtran.slots import build_slots, read_cns

    specs = _require(name)
    cs = _cast(name)
    table = build_slots(cs)
    if cns_path:
        if not os.path.exists(cns_path):
            raise ValueError(f"no encuentro {cns_path}")
        read_cns(cns_path, table)
    x0 = drtran.x0_full(cs, table)
    f = drtran.fit(cs, x0=x0, embed=embed, slots=table)
    if f.ifault:
        raise ValueError(f"la verosimilitud no se pudo evaluar: ifault={f.ifault}")
    _FITS[name] = f
    _TABLES[name] = table

    se = standard_errors(f)
    head = ("[Claude: muestra el bloque ``` siguiente TAL CUAL, verbatim, "
            "antes de comentar nada]\n\n```\n")
    return head + report_fit(f, table, [s.name for s in specs], se) + "\n```"


# ── 4. diagnosis ───────────────────────────────────────────────────────────
@mcp.tool()
def diagnose(name: str, link_index: int = 0) -> str:
    """Is the transfer adequate, and is the input really exogenous?

    Two portmanteaus on the CCF between the estimated noise and the prewhitened
    input: k >= 0 tests the TRANSFER (lag zero belongs to it), k < 0 tests
    EXOGENEITY — significance there is feedback, and a single-input transfer
    model does not hold.

    Measured on the STRUCTURAL residuals: with a contemporaneous transfer the
    reduced-form ones are correlated by construction and would condemn a correct
    model.
    """
    f = _require_fit(name)
    return drtran.report_adequacy(
        drtran.transfer_adequacy(f, link_index=link_index, embed=f.embed))


# ── 5. structure ───────────────────────────────────────────────────────────
@mcp.tool()
def impulse_response(name: str, link_index: int = 0) -> str:
    """nu(k), its cumulative sum and the GAIN, with standard errors.

    nu_k is the response of the output k periods later to a ONE-OFF unit shock;
    the cumulative column is the response to a PERMANENT change and converges to
    the gain. This is usually the answer the analyst came for.

    Unlike a VAR's, this impulse response IS identified without an ordering —
    because the restrictions (exogenous input, diagonal Q) are declared and
    tested, not assumed.
    """
    from drtran.estimate import standard_errors
    from drtran.irf import impulse_response as _irf
    from drtran.irf import report_irf

    f = _require_fit(name)
    se = standard_errors(f)
    cov = None if se.ifault else se.cov
    return report_irf(_irf(f, link_index=link_index, cov=cov))


@mcp.tool()
def variance_decomposition(name: str, series_index: int = 0, horizon: int = 12) -> str:
    """How much of the forecast error comes from EACH source of innovation.

    REFUSES when the structural Q is not diagonal: with correlated innovations
    the decomposition is not unique, it needs an ordering, and that is the VAR's
    problem. Transmit the refusal — do not go looking for a number anyway.
    """
    from drtran.forecast import forecast as _fcast
    from drtran.forecast import variance_decomposition as _fevd

    f = _require_fit(name)
    cs = f.cast_spec
    fc = _fcast(f, L=horizon, embed=f.embed)
    shares, why = _fevd(fc, series=series_index)
    if shares is None:
        return ("NO SE PUEDE DESCOMPONER, y decirlo es la respuesta correcta:\n  "
                + why)
    nm = list(cs.names)
    head = "    l  " + "".join(
        f"{('ruido propio' if j == series_index else nm[j]):>14.14s}"
        for j in range(cs.m))
    out = [f"  {nm[series_index]} — % de la varianza del error de previsión del NIVEL",
           "", head, "  " + "-" * (5 + 14 * cs.m)]
    for l in range(horizon):
        out.append(f"  {l+1:3d}  " + "".join(f"{100*shares[l, j]:13.1f}%"
                                             for j in range(cs.m)))
    return "\n".join(out)


# ── 6. forecasting ─────────────────────────────────────────────────────────
@mcp.tool()
def forecast(name: str, horizon: int = 12, series_index: int = 0) -> str:
    """Forecast the LEVEL with its band, plus the period and annual variations.

    The band is formed in the TRANSFORMED scale and mapped back, so with a log
    model it is ASYMMETRIC. Never build it by adding 1.96 standard errors to a
    level: the STD columns are relative (percentages), not index points.
    """
    from drtran.forecast import forecast as _fcast
    from drtran.forecast import level_band

    f = _require_fit(name)
    cs = f.cast_spec
    fc = _fcast(f, L=horizon, embed=f.embed)
    lvl, lo, hi = level_band(fc, cs, series=series_index)
    se = fc.se("level", series_index)
    out = [f"  PREVISIÓN — {cs.names[series_index]}  (nivel, banda al 95 %)",
           "",
           "   h        nivel      inferior     superior    s.e.(%)",
           "  " + "-" * 52]
    for l in range(horizon):
        out.append(f"  {l+1:3d}  {lvl[l]:11.4f}  {lo[l]:11.4f}  {hi[l]:11.4f}"
                   f"  {se[l]:8.4f}")
    out.append("")
    out.append("  La banda NO es simétrica: se forma en la escala transformada")
    out.append("  y se deshace la Box-Cox. El s.e. es RELATIVO (un porcentaje).")
    return "\n".join(out)


@mcp.tool()
def evaluate(name: str, window: int, horizon: int = 6) -> str:
    """Out-of-sample evaluation: estimate ONCE on 1..window, then roll the origin.

    The only thing here that compares the model against WHAT HAPPENED instead of
    against itself — every other figure the suite prints is theoretical. Run it
    on two specifications to decide empirically which forecasts better.
    """
    from drtran.evaluate import (fixed_window_fit, report_rolling,
                                 rolling_evaluation)

    specs = _require(name)
    links = _LINKS.get(name, [])
    f, _cs = fixed_window_fit(specs, links, window=window, embed=True)
    ev = rolling_evaluation(f.x, specs, links, window=window, horizon=horizon)
    return report_rolling(ev)


# ── 7. back out ────────────────────────────────────────────────────────────
@mcp.tool()
def write_pre(name: str, outdir: str = ".") -> str:
    """Write the JOINTLY re-estimated univariate blocks back out as `.pre`.

    The blocks MOVE when the transfer is fitted beside them, and this is how
    those estimates leave mtram: into a modified specification, or back to ART
    and fuf, which read a `.pre`.

    NOT a better starting point — they are optimal WITH the transfer, so on the
    diagonal they evaluate worse, necessarily. The transfer is not written: a
    `.pre` is univariate.
    """
    from drtran.estimate import standard_errors
    from drtran.pre import next_pre_path
    from drtran.pre import write_pre as _wpre

    f = _require_fit(name)
    specs = _require(name)
    se = standard_errors(f)
    if se.ifault:
        raise ValueError("no hay errores típicos utilizables (el hessiano en este "
                         "punto no lo es); un .pre los lleva, así que reestima antes")
    written = []
    for i, s in enumerate(specs):
        tgt = os.path.join(outdir, os.path.basename(next_pre_path(s.path)))
        _wpre(f, series=i, path=tgt, std_errors=se)
        written.append(tgt)
    return ("Bloques univariantes reestimados escritos:\n  "
            + "\n  ".join(written)
            + "\n\n  La transferencia NO va en ellos: un .pre es univariante."
              "\n  Vuelve a declarar la red con set_network.")


def main():                                                # pragma: no cover
    mcp.run()


if __name__ == "__main__":                                 # pragma: no cover
    main()
