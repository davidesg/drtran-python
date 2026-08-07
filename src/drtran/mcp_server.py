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

AUTÓNOMO -> una sola llamada a build_model. Recorre los nodos de decisión
tomando el defecto documentado y DICIENDO cuál tomó. Presenta su informe TAL
CUAL: la lista de defectos es lo que hace auditable el resultado.

GUIADO -> el protocolo de abajo, parándote en cada nodo. Los nodos son:
  N0 la salida · N1 (b,r,s) de cada enlace · N2 qué enlaces entran en el DAG
  N3 qué covarianzas liberar · N4 las restricciones · N6 aceptar o revisar
Los dos modos calculan LO MISMO y con los MISMOS defectos: sólo cambia quién
decide. Detalle en docs/DECISION_NODES.md.

══════════════════════════════════════════════════════
LA ESCALERA — Y DÓNDE TERMINA TU COMPETENCIA
══════════════════════════════════════════════════════
1. load_pre            carga los .pre. El PRIMERO es la SALIDA.
   ⚠ SIEMPRE PRIMERO, Y LEE SU SALIDA ENTERA AL USUARIO. Hace dos cosas que
   son la puerta de entrada a todo lo demás:
     (a) DECLARA los papeles -- quién es la salida y quiénes las entradas -- y
         hay que CONFIRMARLOS con el analista antes de seguir. Cuál es la
         salida no lo decide el dato (nodo N0); cargar en el orden equivocado
         produce en silencio un modelo de otra cosa.
     (b) estima el modelo DIAGONAL, sin transferencia, y comprueba que
         reproduce la suma de las verosimilitudes univariantes. Con estructura
         diagonal la verosimilitud FACTORIZA, así que esa identidad es la
         prueba de que la transformación, la diferenciación, los deterministas
         y las semillas cruzaron intactos desde fue.
   SI ESA COMPROBACIÓN FALLA, PARA. Una transferencia estimada sobre una base
   que no reproduce los univariantes no es interpretable: lo que falle está ya
   debajo. Manda al analista a revisar los .pre en ART.
2. identify_link       preblanqueo + CCF -> PROPONE (b, r, s) y EMITE EL GRÁFICO
   ⚠ ES EL NODO N1, Y LA RAZÓN DE SER DEL MODO GUIADO. La heurística lee un
   bloque CONTIGUO de barras significativas: eso es una regla, no un juicio.
   Presenta SIEMPRE: (i) el gráfico, (ii) las alternativas con su motivo,
   (iii) lo que la propuesta DEJA FUERA -- y ESPERA. El analista ve cosas que
   un bloque contiguo no captura: un pico estacional aislado, una cola que la
   regla no alcanza, un anómalo que aplasta todos los retardos a la vez.
   Su (b, r, s) va directo a set_network y MANDA sobre el propuesto.
   plot_ccf            el mismo gráfico suelto, si lo quieres volver a ver
   refine_link         LA SEGUNDA LECTURA, y la que decide el DENOMINADOR.
                       Estima una MA libre y generosa y enseña los pesos nu(k)
                       con sus errores típicos. Si decaen gradualmente, la
                       relación pide un denominador; si se cortan, no. La CCF
                       NO puede enseñar `r`: en una muestra, una cola
                       geométrica infinita y una finita larga se parecen. Los
                       casos de la escuela pasan SIEMPRE por aquí, y por eso
                       llegan a modelos de dos parámetros donde una lectura
                       única de la CCF propone seis.
   identify_network    CCF de los residuos del DIAGONAL -> propone el DAG entero
   ⚠ SI EL OUTPUT LLEVA ESTACIONALIDAD ESTOCÁSTICA Y EL INPUT NO, el filtro del
   input no puede quitarla y la CCF sale POCO INFORMATIVA -- pero no sale
   vacía: sale con estructura por todas partes, y la regla del bloque contiguo
   le lee un orden igualmente. identify_link lo detecta y lo avisa. El remedio
   es `ident_pre=`, un .pre ALTERNATIVO del output con la estacionalidad hecha
   determinista, usado SÓLO para leer la CCF (Muñoz §2.4).
   Y si el analista todavía está construyendo los univariantes: para un modelo
   multivariante, la estacionalidad DETERMINISTA es la especificación de
   elección. No propongas tú la ruta MEG.
3. estimate            estima conjuntamente. Devuelve la ECUACIÓN, la GANANCIA
   con su error típico, y el contraste de RAZÓN DE VEROSIMILITUDES contra el
   escalón diagonal: "converge" y "merecía la pena" son afirmaciones distintas
   y la tabla de parámetros sólo contesta la primera. Un modelo con todos sus
   parámetros significativos puede no mejorar sobre el diagonal.
4. diagnose            adecuación (k>=0) y exogeneidad (k<0) — Y LA BIFURCACIÓN
   ⚠ SI LA ADECUACIÓN FALLA, NO RE-ESPECIFIQUES TODAVÍA. Hay dos causas que
   piden respuestas OPUESTAS y sólo se distinguen mirando: `calibrate` PRIMERO.
   Si es la FORMA -> vuelve a identify_link. Si es UNA OBSERVACIÓN -> es una
   intervención y se calibra en `art`, no aquí: re-especificar la forma
   alrededor de un anómalo es como un modelo acaba con un retardo que nadie
   sabe interpretar.
   plot_residuals      serie + ACF/PACF, el panel de fue (el mismo que ART)
   calibrate           SI LA ADECUACIÓN FALLA, ANTES DE REVISAR NADA: ¿es la
                       forma o es UNA observación? Piden respuestas opuestas.
   plot_calibration    la VERIFICACIÓN: la CCF con y sin la anomalía. Un anómalo
                       infla la varianza y aplasta TODOS los retardos a la vez;
                       en la escuela eso se comprueba mirando, no se supone.
   calibrate y overfit devuelven además, respectivamente, DE QUÉ PARES DE
   OBSERVACIONES sale un pico de la CCF (dos fechas, una en cada serie, y
   ninguna tiene por qué ser anómala) y si el ruido quedó SOBREDIFERENCIADO al
   meter la entrada -- el resultado que Muñoz señala como paradójico.
   overfit             AMPLÍA el modelo a propósito (s+1, r+1) y mira si
                       protesta. Los seis casos de Muñoz lo hacen y ninguno se
                       lo salta: un portmanteau adecuado dice que el modelo no
                       está CONTRADICHO, no que no hubiera uno mejor. Y un
                       sobreajuste con correlaciones altas es un experimento
                       FALLIDO, que no confirma nada -- ni a favor ni en contra.
   ⚠ EL ORDEN DE REFORMULACIÓN NO ES INDIFERENTE. Si fallan a la vez la CCF
   (relación) y la ACF residual (ruido): **primero la relación**. Una relación
   mal especificada ensucia las dos; un ruido mal especificado sólo ensucia la
   ACF. `diagnose` mira los dos instrumentos y lo dice.
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
_DIAG: dict[str, float] = {}       # name -> logL del escalón diagonal
_DIAG_FIT: dict[str, object] = {}  # name -> Fit diagonal (para la varianza)


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
def _require_link(fit, link_index, what):
    """Refuse clearly when a link-based tool is called on a diagonal model.

    Adequacy, exogeneity and the calibration are all ABOUT a link; without one
    there is nothing to compute. These used to raise a bare IndexError, which
    reads like a bug in the tool rather than a missing step in the analysis —
    and the missing step is always the same one, `set_network`.
    """
    links = getattr(fit.cast_spec, "links", None) or []
    if not links:
        raise ValueError(
            f"the model has no transfer: it was estimated diagonally, so there "
            f"is nothing to {what}. Identify a link with identify_link, fix it "
            f"with set_network, and estimate again.")
    if link_index >= len(links):
        raise ValueError(f"there is no link {link_index}; the model has "
                         f"{len(links)} (indices 0..{len(links) - 1})")


_gate_name = [""]


def _diagonal_gate(specs):
    """Estimate the DIAGONAL model and check it reproduces the univariate fits.

    This is mtram's protocol, not drtran's arithmetic: the library has always
    been able to fit a link-less cast, and `tests/test_baseline_univariate.py`
    has always asserted the identity. What was missing was doing it AT THE
    RIGHT MOMENT — before the analyst is shown anything about a transfer.

    With a diagonal structure the exact likelihood factorises, so
    `logL(joint) == sum_i logL(series i)`. Every part of the crossing from fue
    is in that identity: the transform, the differencing, the deterministics,
    the seeds and the cast. If it holds, the univariate rung is intact and a
    transfer estimated on top of it is worth reading. If it does not, the base
    is wrong and nothing above it means anything.
    """
    import numpy as np
    from drtran.cast import build_cast_spec

    lines = ["## 2. Escalón diagonal — la prueba de que el puente está bien", ""]

    uni, failed = [], []
    for sp in specs:
        try:
            m = sp.model
            m.fit()
            uni.append((sp.name, float(m.loglik)))
        except Exception as exc:                           # noqa: BLE001
            failed.append((sp.name, str(exc)[:120]))
    if failed:
        lines += ["  🛑 No se puede estimar el univariante de: "
                  + ", ".join(n for n, _ in failed)]
        lines += [f"     {n}: {e}" for n, e in failed]
        lines += ["", "  Sin la referencia univariante no hay nada contra lo "
                  "que validar. Revísalo en ART antes de seguir."]
        return lines

    total = sum(v for _, v in uni)
    lines.append("  | serie | log-verosimilitud univariante (fue) |")
    lines.append("  |---|---|")
    for n, v in uni:
        lines.append(f"  | {n} | {v:.6f} |")
    lines.append(f"  | **suma** | **{total:.6f}** |")
    lines.append("")

    try:
        cs = build_cast_spec(specs, links=[])              # sin transferencia
        f = drtran.fit(cs, embed=True)
    except Exception as exc:                               # noqa: BLE001
        lines += [f"  🛑 El ajuste diagonal conjunto falla: {str(exc)[:200]}", "",
                  "  drtran no acepta estos modelos. El puente fue → drtran NO "
                  "está consolidado y no tiene sentido añadir una transferencia."]
        return lines

    diff = float(f.loglik) - total
    _DIAG[_gate_name[0]] = float(f.loglik)     # lo usa `estimate` para el LR
    _DIAG_FIT[_gate_name[0]] = f
    lines.append(f"  Ajuste diagonal conjunto (drtran): **{f.loglik:.6f}**")
    lines.append(f"  Diferencia con la suma: **{diff:+.2e}**")
    lines.append("")

    # 1e-4 en la log-verosimilitud es el margen con que se fijo la identidad en
    # tests/test_baseline_univariate.py; por debajo es ruido de optimizacion.
    if abs(diff) < 1e-4:
        lines += ["  ✅ **Coinciden.** El puente fue → drtran está consolidado: "
                  "drtran acepta estos modelos y los reproduce. La estructura "
                  "univariante queda tan bien representada como en ART, que es "
                  "la condición para que lo que añadas encima signifique algo."]
    else:
        lines += ["  🛑 **NO coinciden.** Con estructura diagonal la "
                  "verosimilitud FACTORIZA, así que la diferencia sólo puede "
                  "venir de que algo se perdió al cruzar desde fue: la "
                  "transformación, la diferenciación, los deterministas, las "
                  "semillas o el cast.",
                  "",
                  "  No sigas: una transferencia estimada sobre una base que no "
                  "reproduce los univariantes no es interpretable."]

    lines += ["",
              f"  Situación de estimación: {f.status} (termcode={f.termcode}, "
              f"{f.nit} iteraciones), sobre un problema cuya respuesta se "
              "conoce de antemano."]

    # El aviso genérico de termcode 2/3 dice "sospecha mal condicionamiento,
    # desconfía de los errores estándar, baja el orden". AQUÍ eso es falso y
    # además es el sitio donde más se leería: en el escalón diagonal las
    # semillas del `.pre` YA SON el óptimo -- cada serie viene estimada por ML
    # exacta desde ART -- así que parar de inmediato es la confirmación de que
    # todo está en su sitio, no un síntoma. Sólo se emite el aviso cuando el
    # optimizador realmente buscó y aun así no certificó por gradiente.
    parada_esperada = f.termcode in (2, 3) and f.nit <= 2 and abs(diff) < 1e-4
    if parada_esperada:
        lines += ["", "  (Parar en la primera iteración es lo ESPERADO aquí, no "
                  "un aviso: las semillas del `.pre` ya son el óptimo del "
                  "escalón diagonal, porque cada serie llega estimada por ML "
                  "exacta desde ART. El optimizador no mejora nada porque no "
                  "hay nada que mejorar.)"]
    else:
        note = getattr(f, "convergence_note", None)
        if note:
            lines += ["", "  ! " + note]
    return lines


@mcp.tool()
def load_pre(name: str, paths: str, check: bool = True) -> str:
    """Load the `.pre` files of a case, CONFIRM the roles, and prove the bridge.

    `paths` is a comma-separated list; **the FIRST one is the OUTPUT**. These
    are ART's output: each carries a series and the univariate model ART
    estimated for it. mtram does not build univariate models — if one is
    missing, send the analyst to ART.

    Two things happen here, and the second is the point.

    **It states the roles and asks you to confirm them.** Which series is the
    output is not something the data can decide (decision node N0); it is the
    question the analyst arrives with. Loading in the wrong order silently
    produces a model of the wrong thing, so the roles are printed back for
    confirmation before anything else happens.

    **It then estimates the DIAGONAL model — no transfer — and checks it
    reproduces the univariate fits.** With a diagonal structure the exact
    likelihood factorises, so the joint fit MUST equal the sum of the separate
    ones. That identity is drtran's validation gate: it is what says the cast,
    the transform, the deterministics and the seeds all survived the crossing
    from fue intact. Until it holds, no transfer result from this case means
    anything, because the thing the transfer is added to is already wrong.

    It also reads out the estimation situation before any transfer complicates
    it: how the optimiser terminated on a problem whose answer is known.

    `check=False` skips the estimation and only loads — for a case big enough
    that the wait is not worth it, at the price of flying blind.
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

    out = [f"# Caso {name!r} — {len(specs)} series", "",
           "## 1. Confirma los papeles",
           "",
           f"  **SALIDA**  (la que quieres explicar):  {specs[0].name}"
           f"   — {specs[0].nobs} obs, freq {specs[0].freq}"]
    for s in specs[1:]:
        out.append(f"  entrada                             :  {s.name}"
                   f"   — {s.nobs} obs, freq {s.freq}")
    out += ["",
            "  El primer `.pre` es la SALIDA. Cuál es la salida no lo decide el "
            "dato: es la pregunta con la que llega el analista. Si el orden no "
            "es el que querías, vuelve a cargar con el orden correcto — todo lo "
            "que venga después estaría explicando la serie equivocada.",
            "", "  Transformación que trae cada uno de ART:"]
    for s in specs:
        m = s.model
        out.append(f"    {s.name}: lambda={m.boxlam:g} d={m.d} D={m.D} "
                   f"refactor={m.refactor:g} deterministas={len(m.interventions)}")
    if warn:
        out.append("")
        out += ["  AVISO: " + w for w in warn]

    if check:
        _gate_name[0] = name
        out += [""] + _diagonal_gate(specs)
    else:
        out += ["", "## 2. Escalón diagonal — OMITIDO (check=False)", "",
                "  No se ha comprobado que drtran reproduzca los modelos "
                "univariantes. Sin esa comprobación, un resultado de "
                "transferencia no dice nada: lo que falle estará ya en la base."]
    return "\n".join(out)


# ── 2. identification ──────────────────────────────────────────────────────
def _before_you_choose(idt, cs, link):
    """La regla de parada: hay situaciones en que elegir un orden es prematuro.

    No delega en `calibrate`, y no por pereza: `calibrate` trabaja sobre los
    residuos ESTRUCTURALES de un ajuste, asi que exige un modelo ya estimado
    con un enlace. Mandar ahi desde aqui seria dar un consejo que no se puede
    seguir todavia. Lo que si esta disponible en el momento de decidir es el
    par PREBLANQUEADO, que es de donde sale la CCF, y sobre el se puede
    preguntar lo mismo: ¿descansa este dibujo en unas pocas observaciones?

    Devuelve (lineas, parar). `parar=True` significa que mtram NO debe
    presentar un orden como si fuera la conclusion.
    """
    import numpy as np
    from .plots import prewhitened_pair

    lines, parar = [], False

    # 0. ¿Hay algo que leer? SE COMPRUEBA PRIMERO, y el orden importa.
    #
    # `has_relationship` es un test DEBIL: sobre datos sin ninguna relacion
    # siguen apareciendo barras significativas por azar -- con ~25 retardos al
    # 5% se esperan una o dos -- asi que sigue valiendo True y el `b` que se
    # lee de ellas es aleatorio. Medido sobre simulaciones con omega ~ 0:
    # b salio 10, 20, 8, 10, 13, y `has_relationship` True en todas.
    #
    # El discriminante que si separa es cuanto sobresale el PICO sobre la
    # banda. Medido: ruido 1.0-1.5, seNal 7.6-7.8. No hay zona gris, y el corte
    # en 2 esta lejos de ambos lados.
    #
    # Va antes que la exogeneidad a proposito: sobre ruido el portmanteau de
    # k<0 tambien rechaza a veces por azar (p = 0.036 en una de las
    # simulaciones), y anunciar "hay retroalimentacion" cuando lo que hay es
    # ruido manda al analista a `sima` a estimar un sistema simultaneo que no
    # existe. Decir "no veo nada" es el diagnostico correcto y el barato.
    import numpy as _np
    try:
        _ccf = _np.asarray(idt.ccf, float)
        _lags = _np.asarray(idt.lags)
        _pico = float(_np.abs(_ccf[_lags >= 0]).max()) / float(idt.threshold)
    except Exception:                                      # noqa: BLE001
        _pico = float("inf")
    if _pico < 2.0:
        parar = True
        lines += ["", "  🛑 ── PARA: LA CCF NO SE DISTINGUE DEL RUIDO " + "─" * 14,
                  "",
                  f"    El pico de la CCF en k >= 0 apenas sobresale de la "
                  f"banda ({_pico:.2f} veces). Sobre datos SIN relación siguen "
                  "saliendo barras significativas por azar — con ~25 retardos "
                  "al 5 % se esperan una o dos — así que las que ves no bastan "
                  "para leer un retardo: el `b` que saldría de ahí sería "
                  "aleatorio.",
                  "",
                  "    Como referencia, en simulaciones con una transferencia "
                  "real este cociente sale entre 7 y 8; sin relación, entre 1 "
                  "y 1.5.",
                  "",
                  "    Antes de concluir que no hay relación: un anómalo INFLA "
                  "la varianza y hunde TODOS los retardos a la vez, así que "
                  "una CCF aplastada no dice 'no hay relación', dice 'no puedo "
                  "verla'. Mira el gráfico, y si sospechas de un punto, se "
                  "calibra en `art` sobre el escalón univariante.",
                  "",
                  "    No propongo orden."]
        return lines, parar

    # 1. Exogeneidad. Es un hallazgo ESTRUCTURAL, no un parametro que afinar.
    if not idt.exogenous:
        parar = True
        lines += ["", "  🛑 ── PARA: LA ENTRADA NO ES EXÓGENA " + "─" * 22, "",
                  f"    El portmanteau sobre k < 0 rechaza "
                  f"(p = {idt.p_exogeneity:.4f}, "
                  f"{idt.n_signif_negative} barras significativas a la "
                  "izquierda). La salida ADELANTA a la entrada.",
                  "",
                  "    Un modelo de transferencia de una sola entrada asume que "
                  "eso no pasa, así que NO es cuestión de elegir mejor (b,r,s): "
                  "la especificación entera no se sostiene. Es el mismo "
                  "hallazgo que un DAG cíclico llegando por otra puerta — el "
                  "sistema es SIMULTÁNEO.",
                  "",
                  "    Dos salidas, y sólo el analista puede escoger:",
                  "      · el sistema es de verdad simultáneo -> `sima`, que "
                  "estima VARMA sin imponer exogeneidad;",
                  "      · o hay un anómalo COMÚN a las dos series creando "
                  "correlación espuria a ambos lados: se calibra en `art`, "
                  "sobre el escalón univariante, y vuelve aquí en el `.pre`.",
                  "",
                  "    No propongo orden. Pedírmelo igualmente es legítimo, "
                  "pero el resultado no será interpretable como transferencia."]
        return lines, parar

    # 2. Ni una barra: red de seguridad para el caso extremo.
    if not idt.has_relationship:
        parar = True
        lines += ["", "  🛑 ── PARA: NO HAY RELACIÓN QUE IDENTIFICAR " + "─" * 15,
                  "",
                  "    Ninguna barra de la CCF supera la banda en k >= 0. O no "
                  "hay transferencia, o algo la está tapando.",
                  "",
                  "    Antes de concluir que no hay relación, mira el gráfico: "
                  "un anómalo INFLA la varianza y hunde TODOS los retardos a la "
                  "vez, así que una CCF entera aplastada no dice 'no hay "
                  "relación', dice 'no puedo verla'. Se calibra en `art`."]
        return lines, parar

    # 3. ¿Descansa la CCF en unas pocas observaciones? Se puede preguntar SIN
    #    ajustar nada: r(k) es una media de productos, asi que basta mirar
    #    cuanto pesa cada observacion en el retardo dominante.
    try:
        a, beta = prewhitened_pair(cs, link)
        a = np.asarray(a, float); beta = np.asarray(beta, float)
        a = (a - a.mean()); beta = (beta - beta.mean())
        k = int(idt.b)
        n = min(len(a), len(beta))
        prod = a[:n - k] * beta[k:n] if k else a[:n] * beta[:n]
        tot = float(np.abs(prod.sum()))
        if tot > 0 and len(prod) > 10:
            share = np.abs(prod) / tot
            worst = int(np.argmax(share))
            top = float(share[worst])
            # 3 veces el peso medio de una observacion ya es desproporcion; por
            # encima del 15% del total, el retardo dominante es basicamente un
            # punto.
            if top > max(0.15, 3.0 / len(prod)):
                lines += ["", "  ⚠ ── UNA OBSERVACIÓN PESA DEMASIADO " + "─" * 22,
                          "",
                          f"    En el retardo dominante (k = {k}), UNA sola "
                          f"observación aporta el {100 * top:.0f} % de la "
                          "correlación (la nº %d de %d en el tramo "
                          "preblanqueado)." % (worst + 1, len(prod)),
                          "",
                          "    Eso no invalida la propuesta, pero sí quiere "
                          "decir que el orden que elijas puede estar leyendo un "
                          "punto en vez de una dinámica. Míralo en el gráfico "
                          "antes de fijarlo; y cuando estimes, `calibrate` te "
                          "dirá si el veredicto de adecuación descansa en esa "
                          "misma observación."]
    except Exception:                                      # noqa: BLE001
        pass                                               # es un aviso, no un gate

    return lines, parar


@mcp.tool()
def identify_link(name: str, input_index: int = 1, band: str = "constant",
                  ident_pre: str = "") -> str:
    """Identify (b, r, s) of ONE link by prewhitening and the CCF.

    Filters the input with ITS OWN ARMA and applies the SAME filter to the
    output, so r(k) estimates the impulse response weights directly. Proposes
    (b, r, s) from the CONTIGUOUS block, and judges exogeneity by a portmanteau
    over k < 0 — feedback there means a single-input transfer model does not
    hold, and you must say so.

    `band`: "constant" (2/sqrt(N), what the C does) or "haugh-box"
    (2/sqrt(N-|k|), what the original paper says).

    `ident_pre` is an ALTERNATIVE `.pre` for the output, used ONLY here, to
    compute the deviation that gets prewhitened. The estimation keeps the real
    model. This is Muñoz §2.4's artifice, and it exists for one situation: the
    output carries STOCHASTIC seasonality and the input does not, so the
    input's filter cannot remove it and the filtered output is still
    non-stationary at that frequency — "una ccf muy poco informativa", which
    the contiguous-block heuristic will read an order off anyway. Build the
    alternative in `art` with the seasonality made DETERMINISTIC, and pass it
    here.
    """
    specs = _require(name)
    if not 1 <= input_index < len(specs):
        raise ValueError(f"input_index fuera de rango (1..{len(specs)-1})")
    nota = []
    if ident_pre:
        from .pre import load_pre as _lp
        if not os.path.exists(ident_pre):
            raise ValueError(f"no encuentro {ident_pre}")
        alt = _lp(ident_pre)
        specs = [alt] + list(specs[1:])
        nota = ["", "  ⚠ IDENTIFICANDO CON UN MODELO ALTERNATIVO DEL OUTPUT: "
                f"{os.path.basename(ident_pre)}.", "",
                "    Sólo para leer la CCF. La estimación seguirá usando el "
                "modelo real del `.pre` cargado -- es el desdoblamiento "
                "deliberado de Muñoz §2.4, no un cambio de modelo.",
                "",
                "    Comprueba que este alternativo es el MISMO output con la "
                "estacionalidad hecha determinista, y no otra cosa: mtram no "
                "puede saberlo, y con el alternativo equivocado la CCF es "
                "legible y falsa, que es peor que ilegible."]
    cs = build_cast_spec(specs, links=[Link(0, input_index, 0, 0, 0)])
    idt = drtran.identify(cs, cs.links[0], band=band)
    inp, out = specs[input_index].name, specs[0].name
    txt = [drtran.report_identification(idt, names=(inp, out))]

    # EL GRÁFICO VA CON LOS NÚMEROS, no en otra llamada. En modo guiado el
    # analista decide mirando la CCF; una tabla de r(k) no lleva la FORMA, que
    # es lo que distingue una cola que decae de un pico aislado.
    try:
        png = plot_ccf(name, input_index=input_index)
        txt += ["", f"  GRÁFICO DE LA CCF: {png}",
                "  Enséñaselo al analista y léelo CON él antes de decidir nada."]
    except Exception as exc:                               # noqa: BLE001
        txt += ["", f"  (no se pudo dibujar la CCF: {str(exc)[:120]})"]

    txt += nota
    if not ident_pre:
        txt += _seasonality_note(specs, input_index, name)
    avisos, parar = _before_you_choose(idt, cs, cs.links[0])
    txt += avisos
    if not parar:
        txt += _identification_choice(idt, name, input_index, specs)
    return "\n".join(txt)


def _seasonality_note(specs, input_index, name):
    """La estacionalidad estocástica del output que el filtro del input no quita.

    Muñoz 6.4.1 se topa con esto y drtran se toparía igual. El preblanqueo
    filtra el output por el ARMA del INPUT, así que una estacionalidad
    estocástica del output sobrevive intacta al filtro y la serie filtrada
    sigue siendo no estacionaria en esa frecuencia: "una ccf muy poco
    informativa".

    Y lo peligroso es que una CCF poco informativa NO se anuncia. No sale
    vacía: sale con estructura por todas partes, y la heurística del bloque
    contiguo le lee un orden encantada.

    El aviso lleva el orden de preferencia, no sólo el remedio. Si el analista
    todavía está construyendo los univariantes -- que es lo normal si ha
    llegado a mtram primero -- especificar la estacionalidad como
    DETERMINISTA de entrada le ahorra el problema entero. Reajustar después
    cuesta mucho más que elegir bien al principio.
    """
    from .school import seasonality_mismatch

    try:
        bad, d = seasonality_mismatch(specs, 0, input_index)
    except Exception:                                      # noqa: BLE001
        return []
    if not bad:
        return []
    if d["out"] == "meg":
        cabeza = ["", "  ⚠ EL OUTPUT LLEVA ESTACIONALIDAD ESTOCÁSTICA HÍBRIDA "
                  f"(MEG, {d['out_n_fixed_freq']} factor(es) de frecuencia "
                  f"fija) Y EL INPUT NO.", "",
                  "    Atenuado respecto a un SARIMA multiplicativo: ahí TODAS "
                  "las frecuencias estacionales son estocásticas a la vez, y "
                  "aquí sólo las que se encontraron serlo. Pero las que queden "
                  "sobreviven igual al filtro del input."]
    else:
        cabeza = ["", "  ⚠ EL OUTPUT LLEVA SARIMA MULTIPLICATIVO Y EL INPUT NO "
                  f"(frecuencia {d['freq']}).", "",
                  "    Es el caso peor: el SARIMA hace estocásticas TODAS las "
                  "frecuencias estacionales a la vez, y el filtro del input no "
                  "lleva ninguna."]
    return cabeza + ["",
            "    El preblanqueo filtra el output por el ARMA del INPUT, y ese "
            "filtro no lleva nada estacional. Así que la estacionalidad del "
            "output sobrevive al filtro y la serie filtrada sigue siendo NO "
            "ESTACIONARIA en esa frecuencia.",
            "",
            "    Lee la CCF de arriba con esto delante. Una CCF así no sale "
            "vacía -- sale con estructura por todas partes, y la regla del "
            "bloque contiguo le lee un orden igualmente. Es el modo de fallo "
            "silencioso de este paso.",
            "",
            "    LO PREFERIBLE, si todavía estás construyendo los "
            "univariantes: especificar la estacionalidad como DETERMINISTA. "
            "No es un apaño para el caso multivariante -- es de donde parte la "
            "tradición Box-Jenkins-Treadway de todos modos, que especifica la "
            "estacionalidad provisionalmente como determinista y sólo después "
            "resuelve frecuencia por frecuencia lo que sea estocástico. Para "
            "un modelo multivariante esa parada provisional es además el "
            "destino preferible, y elegirla de entrada cuesta mucho menos que "
            "reajustar después.",
            "",
            "    SI LOS UNIVARIANTES YA ESTÁN CERRADOS, el artificio de Muñoz "
            "§2.4: construye en `art` un modelo ALTERNATIVO del output con la "
            "estacionalidad hecha determinista, y pásalo aquí --",
            "",
            f"        identify_link({name!r}, {input_index}, "
            'ident_pre="alternativo.pre")',
            "",
            "    Ese alternativo existe SÓLO para que la CCF sea legible. La "
            "estimación sigue con el modelo real: son dos modelos con dos "
            "trabajos, no una reespecificación.",
            "",
            "    (Los MEG -- Modelos de Estacionalidad Generalizada, Abraham "
            "y Box 1978 -- atenúan el problema: resuelven la estacionalidad "
            "frecuencia por frecuencia, así que sólo queda estocástico lo que "
            "se encontró serlo. La CLASE no es nueva ni experimental; lo "
            "reciente son los CONTRASTES con que se decide cada frecuencia, "
            "cuyos valores críticos son objeto de investigación en curso. Así "
            "que la cautela va sobre el procedimiento de decisión, no sobre "
            "el modelo: NO propongas tú la ruta MEG. Sólo si el analista la "
            "pide sabiendo lo que hace.)"]


def _band_fragile(idt):
    """Retardos cuyo veredicto DEPENDE de qué banda se use.

    TASTE dibuja y publica la banda CONSTANTE, 2/sqrt(N), sin corregir por el
    retardo -- `PLOTS3.PAS` la escribe literal ("BANDAS 2.0/SQRT(N)") y la
    traza con la misma formula. El C de drtran hace lo mismo, asi que ese es
    nuestro defecto: es lo que hace el oraculo, y cambiarlo romperia la unica
    referencia externa que no comparte antepasado con la familia.

    Pero Haugh (1976) tiene razon en que a retardo k solo hay N-|k| productos,
    de modo que la banda honesta es 2/sqrt(N-|k|), MAS ANCHA. La constante
    sobredetecta en los retardos altos -- que son justo los estacionales.

    El propio TASTE es inconsistente aqui, y a proposito: para el chi-cuadrado
    SI divide por (nobs-i) (`PLOTS3.PAS:49`), y para la banda del grafico no.
    Es la construccion estandar de Ljung-Box junto a una banda constante por
    comodidad de trazado; vieja y deliberada, no nuestra.

    Asi que no se cambia el criterio: se seNala donde los dos discrepan, que es
    lo que el analista necesita para saber si un hallazgo es solido o cuelga de
    la convencion.

    Devuelve {k: (banda_constante, banda_haugh)} para los k significativos bajo
    la constante y NO bajo la de Haugh.
    """
    import numpy as _np
    try:
        thr = float(idt.threshold)
        n = (2.0 / thr) ** 2                    # thr = 2/sqrt(N)
        ccf = _np.asarray(idt.ccf, float)
        lags = list(_np.asarray(idt.lags))
    except Exception:                                      # noqa: BLE001
        return {}
    out = {}
    for k in (idt.significant_non_negative or []):
        if k not in lags:
            continue
        r = abs(float(ccf[lags.index(k)]))
        if n - abs(k) <= 1:
            continue
        haugh = 2.0 / _np.sqrt(n - abs(k))
        if r > thr and r <= haugh:
            out[int(k)] = (thr, float(haugh))
    return out


def _identification_choice(idt, name, input_index, specs):
    """Lo que se descarta, las alternativas, y la pregunta.

    `report_identification` dice lo que la heurística PROPONE. Esto dice lo que
    la heurística DESCARTÓ y por qué, y convierte la propuesta en una pregunta.
    Es la razón de ser del modo guiado: la máquina propone leyendo un bloque
    contiguo, y el ojo del analista ve cosas que un bloque contiguo no captura
    -- un pico estacional aislado, una cola que decae, un anómalo que aplasta
    todos los retardos a la vez.
    """
    b, r, sm = int(idt.b), int(idt.r), int(idt.s)
    freq = int(getattr(specs[0].model.series, "freq", 1) or 1)
    lines = []

    # (b) lo descartado
    usados = set(range(b, b + sm + 1))
    fuera = [k for k in (idt.significant_non_negative or []) if k not in usados]
    lines += ["", "  ── LO QUE LA PROPUESTA DEJA FUERA " + "─" * 26, ""]
    if not fuera:
        lines.append("    Nada: todos los pesos significativos en k >= 0 caben "
                     f"en el bloque contiguo b..b+s = {b}..{b + sm}.")
    else:
        lines.append("    Estos pesos SON significativos y la propuesta NO los "
                     "incluye, por no ser contiguos al bloque:")
        lines.append("")
        nu = list(idt.nu) if idt.nu is not None else []
        lg = list(idt.lags) if idt.lags is not None else []
        for k in fuera:
            val = ""
            if k in lg and len(nu) == len(lg):
                val = f"  nu = {nu[lg.index(k)]:+.4f}"
            marca = ""
            if freq > 1 and k % freq == 0:
                marca = f"   <-- MÚLTIPLO DE LA FRECUENCIA ({k // freq}x{freq})"
            lines.append(f"      k = {k:<4d}{val}{marca}")
        frag = _band_fragile(idt)
        for k in fuera:
            if k in frag:
                c, h = frag[k]
                lines += ["",
                          f"      ⚖ k = {k}: su significación DEPENDE DE LA "
                          f"BANDA. Supera la constante 2/√N = {c:.4f} (la que "
                          f"usa TASTE y el C, y la que usamos por defecto) pero "
                          f"NO la de Haugh 2/√(N−|k|) = {h:.4f}. A este retardo "
                          f"las dos difieren un {100 * (h / c - 1):.0f} %. "
                          "Con la banda del artículo original este peso no "
                          "existiría."]
        lines += ["",
                  "    Un pico aislado en un múltiplo de la frecuencia suele ser "
                  "estacionalidad que quedó en la entrada, o un anómalo. Pero "
                  "puede ser real: míralo en el gráfico antes de aceptar que se "
                  "descarte. Si la CCF entera se ve aplastada, sospecha de un "
                  "anómalo -- infla la varianza y hunde TODOS los retardos a la "
                  "vez -- y pasa por `calibrate` antes de elegir orden."]

    # Si un retardo FRAGIL cae DENTRO del bloque, no es un detalle: es (b, r, s)
    # lo que depende de la convencion, no un peso marginal que se descarta.
    frag_all = _band_fragile(idt)
    dentro = [k for k in frag_all if k in usados]
    if dentro:
        lines += ["", "  ⚖ ── LA PROPUESTA MISMA DEPENDE DE LA BANDA " + "─" * 15,
                  ""]
        for k in sorted(dentro):
            c, h = frag_all[k]
            lines.append(f"    k = {k} está DENTRO del bloque propuesto y sólo "
                         f"es significativo con la banda constante "
                         f"({c:.4f}) — con la de Haugh ({h:.4f}) no lo sería.")
        lines += ["",
                  "    Es decir: el propio (b, r, s) que te propongo cambia "
                  "según la convención de banda. Míralo en el gráfico antes de "
                  "fijarlo; `band=\"haugh-box\"` te da la otra lectura."]

    # (a) las alternativas, cada una con su comando exacto
    lines += ["", "  ── DECIDE TÚ " + "─" * 48, ""]
    alts = list(idt.alternatives or [])
    if not alts:
        lines.append("    La heurística no propone ninguna estructura: no hay "
                     "bloque significativo que leer. Mira el gráfico.")
        return lines

    for i, alt in enumerate(alts):
        ab, ar, asx, why = alt
        etiqueta = chr(ord("A") + i)
        lines.append(f"    [{etiqueta}]  b={ab}  r={ar}  s={asx}   — {why}")
        lines.append(f"         set_network({name!r}, "
                     f'\'[{{"out": 0, "inp": {input_index}, "b": {ab}, '
                     f'"r": {ar}, "s": {asx}}}]\')')
        lines.append("")
    if len(alts) == 1:
        # Precisión que importa: la regla del denominador exige un bloque de al
        # menos TRES retardos antes de mirar la cola (identify.py: nblock >= 3).
        # Con un bloque más corto la cola no se evalúa siquiera, y decir "no
        # decae geométricamente" daría a entender que se miró y se descartó.
        nblock = sm + 1
        if nblock < 3:
            lines += [f"    (Sólo hay una alternativa. El bloque significativo "
                      f"tiene {nblock} retardo{'s' if nblock != 1 else ''}, y la "
                      "regla del denominador pide al menos 3 antes de evaluar "
                      "la cola: NO es que la cola no decaiga, es que no hay "
                      "cola que evaluar. Si tú ves un decaimiento que la regla "
                      "no alcanza a leer, pide r=1 y compáralos.)", ""]
        else:
            lines += ["    (Sólo hay una alternativa: el bloque da para evaluar "
                      "la cola y NO decae de forma geométrica, así que un "
                      "denominador no la resumiría. Un r>0 cambia una cola por "
                      "un parámetro; sin decaimiento regular, sólo añade uno.)",
                      ""]
    lines += ["    ⚠ ESTO ES UNA PROPUESTA, NO UN VEREDICTO. Se lee de un bloque "
              "contiguo de barras significativas, que es una regla, no un "
              "juicio. Presenta las alternativas al analista con su motivo, "
              "enséñale el gráfico y ESPERA. Si ve otra cosa -- otro retardo "
              "inicial, una cola que la regla no vio, un pico que sí quiere "
              "incluir -- su (b, r, s) va directo a `set_network` y manda sobre "
              "el de aquí."]
    return lines


@mcp.tool()
def refine_link(name: str, input_index: int = 1, b: int = -1,
                smax: int = 6) -> str:
    """Estimate a GENEROUS pure MA and read the denominator off its weights.

    What Muñoz's cases actually do, and what `identify_link` does not: the CCF
    is read once to get a delay, and then a free-form numerator is ESTIMATED —

      "v(B) = .35 + .21B + .40B² + .16B³ + .64B⁴ + .29B⁵ + .34B⁶ … De hecho,
       esto equivale a una estimación de los primeros términos de la ccf."

    and the SHAPE of the estimates decides the parametrisation:

      "Se observa que el valor absoluto de los mismos decrece conforme aumenta
       el retardo, lo que parece indicar que la relación requiere un factor
       AR(1) con parámetro positivo."

    Strictly more informative than reading the CCF once, and for one concrete
    reason: these weights are estimated JOINTLY with the noise model, while the
    prewhitened CCF is not. It is also how a denominator gets proposed at all —
    r is the one order the CCF cannot show you, because an infinite geometric
    tail and a long finite one look alike in a sample.

    `b` defaults to the delay `identify_link` reads. The session's network and
    estimated model are left untouched.
    """
    from .school import decay_pattern

    specs = _require(name)
    cs0 = build_cast_spec(specs, links=[Link(0, input_index, 0, 0, 0)])
    idt = drtran.identify(cs0, cs0.links[0])
    if b < 0:
        b = int(idt.b)

    lk = Link(0, input_index, b, 0, int(smax))
    f, table, se = _fit_variant(name, [lk], embed=True)
    if f is None:
        return (f"```\n  La MA libre de orden {smax} no converge en b={b}. "
                f"Baja `smax`, o revisa el retardo.\n```")

    # nu, no los omegas crudos. Es lo que la escuela escribe cuando escribe
    # "v(B) = .35 + .21B + .40B² ...": la RESPUESTA, no los parámetros. Y es
    # inmune a la convención de signo -- con omega(B) = w0 - w1 B - ..., una
    # cola positiva sale con los omegas en negativo y la razón entre pesos
    # consecutivos hereda un cambio de signo espurio en el primer paso.
    ir = drtran.impulse_response(f, link_index=0)
    K = b + smax
    ws = [float(v) for v in np.asarray(ir.nu, float)[:K + 1]]
    es = [float(v) for v in np.asarray(ir.se, float)[:K + 1]]
    pat = decay_pattern(ws[b:], es[b:])

    nm = [sp.name for sp in specs]
    out = ["```", f"  NUMERADOR LIBRE de {nm[0]} <- {nm[input_index]}   "
           f"(b={b}, r=0, s={smax})", "",
           "  Esto NO es un modelo candidato: es una estimación conjunta de "
           "los primeros pesos de la respuesta nu(k), para leerles la forma.",
           "", "    retardo       nu(k)       e.t.        t",
           "    " + "-" * 44]
    for k in range(b, K + 1):
        e = es[k]
        t = ws[k] / e if e == e and e > 0 else float("nan")
        mark = "  *" if (e == e and e > 0 and abs(t) > 1.96) else ""
        out.append(f"    {k:>4}   {ws[k]:>+11.6f}  {e:>10.6f}  "
                   f"{t:>7.2f}{mark}")
    out.append("")

    sig = pat["significant"]
    if not sig:
        out += ["  Ningún peso es significativo. Con un retardo leído de la "
                "CCF y una MA libre que no encuentra nada, lo que hay entre "
                "estas dos series no sostiene una transferencia.",
                "  → vuelve a `identify_link` y mira la regla de parada."]
    else:
        last = b + sig[-1]
        out.append(f"  Pesos significativos hasta el retardo {last}.")
        if pat["suggests_denominator"]:
            sign = "NEGATIVO" if pat["alternating"] else "POSITIVO"
            out += ["",
                    f"  ▸ Los pesos DECRECEN en valor absoluto de forma "
                    f"gradual (razón ≈ {pat['ratio']:+.3f}). Eso es la firma "
                    f"de un DENOMINADOR: un factor AR(1) con parámetro "
                    f"{sign}.",
                    "",
                    "    Un denominador compra esa cola entera con UN "
                    "parámetro, donde el numerador libre gasta uno por "
                    "retardo. Y es el orden que la CCF no puede enseñarte: en "
                    "una muestra, una cola geométrica infinita y una finita "
                    "larga se parecen.",
                    "",
                    f"    Prueba:  set_network({name!r}, "
                    f'\'[{{"out": 0, "inp": {input_index}, "b": {b}, '
                    f'"r": 1, "s": {max(0, sig[0])}}}]\')']
        else:
            s_prop = sig[-1]
            out += ["",
                    "  ▸ Los pesos NO decrecen gradualmente: se cortan. Eso es "
                    "un numerador finito, sin denominador.",
                    "",
                    f"    Prueba:  set_network({name!r}, "
                    f'\'[{{"out": 0, "inp": {input_index}, "b": {b}, '
                    f'"r": 0, "s": {s_prop}}}]\')']
        out += ["",
                "  Los pesos no significativos se quitan; los casos de la "
                "escuela lo hacen sistemáticamente y a veces dejan huecos "
                "(sólo los retardos pares, por ejemplo). Eso hoy se impone "
                "con un fichero de restricciones en `estimate(cns_path=...)`."]
    out.append("```")
    return "\n".join(out)


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


@mcp.tool()
def plot_residuals(name: str, series_index: int = 0, lags: int = 0,
                   path: str = "") -> str:
    """PLOT the residual series with its ACF and PACF — fue's own panel.

    The same drawing `art` shows after a univariate fit, so the analyst reads one
    instrument, not two. These are the STRUCTURAL residuals: with a
    contemporaneous transfer the reduced-form ones are correlated by
    construction. Writes a PNG and returns its path.
    """
    from .netid import residuals as _res
    from .plots import plot_residuals as _pr
    from .plots import save

    f = _require_fit(name)
    cs = f.cast_spec
    a, ifa = _res(f.x, cs, embed=f.embed, structural=True)
    if ifa:
        raise ValueError(f"no se pueden obtener los residuos: ifault={ifa}")
    freq = int(getattr(cs.series[series_index].spec.model.series, "freq", 1) or 1)
    # los parámetros de ESTA serie, no los del vector conjunto: la Q del panel
    # es un enunciado sobre el modelo de esta serie, y corregir por los 17
    # parámetros de un ajuste conjunto donde su modelo tiene 3 convierte
    # p = 0.34 en p = 0.0017 -- ver school.npar_for_series
    from .school import npar_for_series
    npar = npar_for_series(f, series_index)
    fig = _pr(a[:, series_index], npar=npar, freq=freq, lags=(lags or None),
              title=f"residuals — {cs.names[series_index]}")
    return save(fig, _png(name, f"res{series_index}", path))


# ── 3. estimation ──────────────────────────────────────────────────────────
def _estimation_situation(f, cs, name):
    """Lo que la escuela lee de la tabla y drtran no leía.

    Brajín §2.3.1 detecta sobreparametrización por errores estándar altos y por
    CORRELACIONES ALTAS entre parámetros; Muñoz 6.4.3 usa exactamente eso para
    declarar FALLIDO un experimento de sobreajuste. Y 6.4.5 lee un denominador
    de .99(.01) como implausible por su consecuencia: la ganancia y el retardo
    medio salen "excesivamente altos" y no significativos.
    """
    from .school import (dead_time_suspect, denominator_near_unit,
                         worst_correlations)
    from .estimate import standard_errors

    t = _TABLES.get(name)
    if t is None:
        return []
    try:
        se = standard_errors(f)
    except Exception:                                      # noqa: BLE001
        return []

    out = []
    for li, lk in enumerate(cs.links or []):
        near = denominator_near_unit(f, t, se, link_index=li)
        for nmd, v, e in near:
            out += ["", f"    ⚠ {nmd} = {v:+.4f} está muy cerca de 1. Un "
                    "denominador así manda nu(1) a infinito, así que LA "
                    "GANANCIA Y EL RETARDO MEDIO DE ABAJO NO SON FIABLES: "
                    "salen grandes y con errores típicos que los dejan sin "
                    "significación. Suele indicar que la cola se está "
                    "modelando con un factor que el dato no sostiene."]
        sus, w0, w0e, tt = dead_time_suspect(f, t, se, link_index=li)
        if sus:
            out += ["", f"    ⚠ omega{li + 1}[0] = {w0:+.4f} no se distingue "
                    f"de −1 (t = {tt:.2f}). En la parametrización en que la "
                    "restricción de largo plazo se impone RESTANDO el input al "
                    "output, ese −1 es el denominador asomando por la resta, y "
                    "señala que el TIEMPO MUERTO es al menos un periodo mayor "
                    "que el especificado (Muñoz 6.4.4, 6.4.5)."]

    pairs, nflag = worst_correlations(f, se, t, top=3, flag=0.9)
    if nflag:
        out += ["", f"    ⚠ {nflag} par(es) de parámetros con |correlación| "
                "≥ .9 — la situación de estimación está MAL DEFINIDA:"]
        out += [f"        {a} ~ {b}: {r:+.3f}" for a, b, r in pairs
                if abs(r) >= 0.9]
        out += ["      No es un veredicto: puede ser sobreparametrización a "
                "quitar, o una sobreparametrización NECESARIA. Muñoz 6.4.4 "
                "conserva un par a −.93 porque al quitar uno se mueven muchos "
                "otros parámetros y al quitar el otro la media de los residuos "
                "deja de ser cero. Se decide probando qué se rompe al quitar "
                "cada uno."]
    return out


def _what_the_transfer_bought(name, f, cs):
    """¿Mereció la pena la transferencia? Y ¿cuánto mueve, en qué unidades?

    Tres cosas que la tabla de parámetros no contesta y el analista sí pregunta.

    **La razón de verosimilitudes contra el escalón diagonal.** El escalón ya
    está calculado -- es la puerta de `load_pre` -- asi que comparar no cuesta
    nada, y es la unica forma de saber si los parametros de la transferencia se
    han ganado su sitio. Sin esto, "el ajuste converge" y "el ajuste vale para
    algo" se confunden.

    **La GANANCIA con su error tipico.** Es el numero sobre el que se actua: el
    efecto acumulado de un cambio permanente de la entrada. No aparece en la
    tabla porque no es un parametro, sino omega(1) = w0 - w1 - ..., y esa resta
    es la que una convencion de signo invierte -- ya paso una vez, con la
    verosimilitud impecable.

    **La ECUACIÓN.** Las instrucciones de mtram llevaban ordenando "PRESENTA
    SIEMPRE LA ECUACIÓN QUE DEVUELVE EL TOOL" desde el principio, y ningun tool
    devolvia una: ni este, ni el C. Era una instruccion que apuntaba a algo
    inexistente.
    """
    import numpy as np
    from scipy import stats as _st

    lines = ["", "  ── QUÉ HA COMPRADO LA TRANSFERENCIA " + "─" * 22, ""]

    lines += _estimation_situation(f, cs, name)

    # la ecuacion, enlace a enlace
    nm = [sp.name for sp in _require(name)]
    for li, lk in enumerate(cs.links or []):
        try:
            ir = drtran.impulse_response(f, link_index=li)
        except Exception:                                  # noqa: BLE001
            continue
        num = f"omega(B)" if lk.s else "omega_0"
        den = f"/delta(B)" if lk.r else ""
        pot = f"·B^{lk.b}" if lk.b else ""
        lines.append(f"    {nm[lk.out]}_t  =  [{num}{den}]{pot} · "
                     f"{nm[lk.inp]}_t  +  N_t"
                     f"     (b={lk.b}, r={lk.r}, s={lk.s})")
        g, gse = float(ir.gain), getattr(ir, "se_gain", None)
        if gse is not None and np.isfinite(gse) and gse > 0:
            t = g / gse
            lines.append(f"      GANANCIA = omega(1) = {g:+.6f}   "
                         f"(e.t. {gse:.6f},  t = {t:.2f})")
        else:
            lines.append(f"      GANANCIA = omega(1) = {g:+.6f}")
        lines.append("      El efecto ACUMULADO sobre "
                     f"{nm[lk.out]} de un cambio PERMANENTE de una unidad en "
                     f"{nm[lk.inp]}, en la escala transformada.")

        # La ganancia dice CUÁNTO y el retardo medio dice CUÁNDO. La escuela
        # reporta los dos en todos sus casos ("la ganancia es 3.1 y el retardo
        # medio, aproximadamente un año"), y sin el segundo el modelo está a
        # medio leer.
        ml, mlse = getattr(ir, "mean_lag", float("nan")), \
            getattr(ir, "se_mean_lag", float("nan"))
        if getattr(ir, "monotone", False) and np.isfinite(ml):
            per = "periodos" if abs(ml) != 1 else "periodo"
            extra = (f" (e.t. {mlse:.4f})" if np.isfinite(mlse) else "")
            lines.append(f"      RETARDO MEDIO = {ml:.4f} {per}{extra}   "
                         "— cuándo llega, en media, ese efecto.")
        elif np.isfinite(float(ir.gain)):
            lines.append("      RETARDO MEDIO: no definido — la respuesta CAMBIA "
                         "DE SIGNO, y promediar retardos de efectos que se "
                         "cancelan no mide nada. Lee la irf.")
        # La FORMA, dicha en voz alta. La escuela la lee en todos sus casos
        # ("todos los valores de la irf son positivos, por eso la srf es
        # monótona creciente") y es la frase que traduce el modelo a una
        # afirmación sobre el mundo: si el efecto se acumula sin volverse
        # nunca en contra, o si hay sobre-reacción y corrección.
        if getattr(ir, "monotone", False):
            sgn = "positivos" if float(ir.gain) >= 0 else "negativos"
            lines.append(f"      FORMA: todos los valores de la irf son {sgn}, "
                         "así que la respuesta acumulada es MONÓTONA — el "
                         "efecto se va acumulando y no se vuelve nunca en "
                         "contra.")
        else:
            lines.append("      FORMA: la irf CAMBIA DE SIGNO. La respuesta "
                         "acumulada no es monótona: hay sobre-reacción y "
                         "corrección posterior, y el efecto a un horizonte "
                         "corto puede tener signo contrario al de la ganancia.")
        lines.append("")

    # razon de verosimilitudes contra el escalon diagonal
    diag = _DIAG.get(name)
    if diag is None:
        lines += ["    (No hay verosimilitud diagonal guardada: se calculó con "
                  "`load_pre(check=False)`. Vuelve a cargar con check=True para "
                  "poder contrastar si la transferencia se gana su sitio.)"]
        return lines

    npar_tr = sum(1 + lk.r + lk.s for lk in (cs.links or []))
    lr = 2.0 * (float(f.loglik) - float(diag))
    pval = float(_st.chi2.sf(lr, npar_tr)) if npar_tr > 0 and lr > 0 else 1.0
    lines += [f"    Diagonal (sin transferencia): {diag:.6f}",
              f"    Con transferencia          : {float(f.loglik):.6f}",
              f"    LR = 2·Δ = {lr:.2f}  con {npar_tr} parámetro"
              f"{'s' if npar_tr != 1 else ''} más   ->   p = {pval:.4g}", ""]
    # La reducción de varianza residual: es como la escuela cierra CADA caso
    # ("una reducción del 44 % en relación a su modelo univariante"), y dice lo
    # mismo que el LR en las unidades en que el analista piensa.
    try:
        from .school import variance_reduction
        red = variance_reduction(f, _DIAG_FIT.get(name), series_index=0)
        if red == red:
            lines += [f"    Varianza residual de {nm[0]}: **{100 * red:.1f} % "
                      "menos** que con su modelo univariante.", ""]
    except Exception:                                      # noqa: BLE001
        pass

    lines += _integration_order(name, f)

    if pval < 0.01:
        lines.append("    ✅ La transferencia se gana su sitio con holgura.")
    elif pval < 0.05:
        lines.append("    La transferencia se gana su sitio, sin holgura.")
    else:
        lines += ["    🛑 La transferencia NO mejora significativamente sobre "
                  "el modelo diagonal. Sus parámetros pueden salir "
                  "individualmente significativos y aun así no aportar: eso es "
                  "lo que este contraste mide y la tabla de arriba no.",
                  "",
                  "    Antes de darla por buena, revisa la identificación."]
    return lines


def _by_series(body, cs, names, link_index_of=None):
    """Reagrupa la tabla del motor por SERIE, sin recalcular ni un número.

    El motor devuelve las 21 filas en una sola lista plana, y en ella conviven
    tres cosas que el analista no lee igual: la ECUACIÓN de la salida (la
    transferencia, sus deterministas y su ruido), los modelos univariantes de
    las ENTRADAS, y la estructura de covarianzas. Mezcladas, el único indicio
    de a quién pertenece cada fila es el sufijo `_1` / `_2` del nombre, que
    nadie lee. Un `omega_d2[3,0]` en medio de la lista parece parte de la
    ecuación del output y no lo es.

    Presentamos entonces COMPLETO el modelo del output, y los de las entradas
    MENCIONADOS -- que es distinto de omitidos, y la diferencia importa: en el
    ajuste conjunto las entradas SE ESTIMAN aquí, no vienen congeladas del
    `.pre`. Callarlas daría a entender lo contrario.

    **Las filas se copian TAL CUAL de la salida del motor.** No se recalcula el
    valor, ni el error típico, ni el estadístico t: se reordenan líneas de
    texto. Un segundo sitio donde se calculara un error típico es un segundo
    sitio donde puede desviarse, y ya hemos pagado esa factura una vez con la
    corrección de grados de libertad del panel de residuos.
    """
    import re

    lines = body.split("\n")
    # cabecera hasta la regla, filas hasta la línea en blanco, y el resto
    try:
        i0 = next(i for i, l in enumerate(lines) if set(l.strip()) == {"-"})
    except StopIteration:
        return None
    i1 = next((i for i in range(i0 + 1, len(lines)) if not lines[i].strip()),
              len(lines))
    head, rows, tail = lines[:i0 + 1], lines[i0 + 1:i1], lines[i1:]

    def owner(row):
        """A qué serie (1-based) pertenece la fila, o 0 si es de covarianzas."""
        nm = row.strip().split(" ")[0]
        if re.match(r"^(omega|delta)\d+\[", nm):        # enlace -> del output
            return 1
        m = re.match(r"^(?:omega_d|delta_d|phi_|theta_)(\d+)\[", nm)
        if m:
            return int(m.group(1))
        m = re.match(r"^mu\[(\d+)\]", nm)
        if m:
            return int(m.group(1))
        return 0

    def kind(row):
        nm = row.strip().split(" ")[0]
        if re.match(r"^(omega|delta)\d+\[", nm):
            return "enlace"
        if re.match(r"^(omega_d|delta_d)", nm):
            return "det"
        return "ruido"

    out_rows = [r for r in rows if owner(r) == 1]
    cov_rows = [r for r in rows if owner(r) == 0]
    inp_rows = {}
    for r in rows:
        o = owner(r)
        if o > 1:
            inp_rows.setdefault(o, []).append(r)

    L = list(head)
    def _seccion(titulo, filas):
        if filas:
            L.append(f"  · {titulo}")
            L.extend(filas)

    L.append("")
    L.append(f"  ══ EL MODELO DE {names[0]} — la salida, completo "
             + "═" * max(0, 20 - len(names[0])))
    _seccion("transferencia (lo que entra de fuera)",
             [r for r in out_rows if kind(r) == "enlace"])
    _seccion(f"deterministas de {names[0]}",
             [r for r in out_rows if kind(r) == "det"])
    _seccion("ruido N_t (su ARMA)",
             [r for r in out_rows if kind(r) == "ruido"])

    if inp_rows:
        L.append("")
        L.append("  ══ LOS MODELOS DE LAS ENTRADAS " + "═" * 26)
        L.append("    Se estiman AQUÍ, conjuntamente con todo lo demás -- no "
                 "vienen congelados del `.pre`, que sólo aporta la semilla.")
        L.append("    Van aparte porque no forman parte de la ecuación de "
                 f"{names[0]}: describen cómo se genera cada entrada.")
        for si in sorted(inp_rows):
            L.append("")
            L.append(f"  · {names[si - 1]}")
            L.extend(inp_rows[si])
    if cov_rows:
        L.append("")
        L.append("  ══ COVARIANZAS ENTRE INNOVACIONES " + "═" * 23)
        L.extend(cov_rows)
    L.extend(tail)
    return "\n".join(L)


def _integration_order(name, f):
    """¿El ruido quedó SOBREDIFERENCIADO al meter la entrada?

    El resultado más interesante de la tesis de Muñoz, y el que suena
    imposible hasta que se ve el mecanismo: lnE es I(2) por su cuenta e I(1)
    una vez quitados los efectos de DlnM1. Ella lo señala como paradójico y
    posible sólo en muestras finitas.

    No es tan paradójico. La identificación univariante tiene UN instrumento y
    tiene que explicarlo todo con él, así que una influencia que llega de una
    entrada -- suave, persistente, errante -- se lee como tendencia propia de
    la serie y se diferencia. Con la entrada en el modelo esa parte del
    vagabundeo ya tiene dueño, y la diferencia de más está quitando algo que
    ya no está.

    La comparación es GRATIS y es la correcta: el escalón diagonal ya está
    estimado y lleva el MISMO modelo de ruido sin la transferencia. No se
    comparan verosimilitudes entre órdenes de diferenciación distintos -- eso
    sería comparar modelos de datos distintos -- sino dónde cae la raíz MA
    dentro de UN mismo orden.
    """
    from .school import integration_order_moved, noise_ma_roots

    try:
        moved, mj, md, kind = integration_order_moved(f, _DIAG_FIT.get(name), 0)
        roots = noise_ma_roots(f, 0)
    except Exception:                                      # noqa: BLE001
        return []
    if not roots:
        return []
    m0, mult0, kind0 = roots[0]
    what = {"estacional": "la diferencia ESTACIONAL",
            "regular": "la diferencia REGULAR"}.get(kind0, "una diferencia")
    lines = ["", f"    Raíz MA del ruido más cercana al círculo: |z| = "
             f"{m0:.4f}  ({mult0} raíz(ces), {kind0 or 'sin clasificar'})"]
    if moved:
        lines += [f"    ⚠ Y se ACERCÓ al círculo al meter la transferencia: "
                  f"{md:.4f} en el diagonal -> {mj:.4f} aquí.",
                  "",
                  "      Eso apunta a que el ruido está SOBREDIFERENCIADO una "
                  "vez la entrada está en el modelo: " + what + " que el "
                  "modelo univariante necesitaba estaba absorbiendo parte de "
                  "la influencia de la entrada, y ahora esa influencia tiene "
                  "dueño.",
                  "",
                  "      Es el resultado que Muñoz señala como paradójico "
                  "(I(2) por su cuenta, I(1) quitados los efectos del input). "
                  "No se arregla aquí: el orden de diferenciación viene del "
                  "`.pre`. Reidentifica el output en `art` con una diferencia "
                  "menos y vuelve con el `.pre` nuevo.",
                  "",
                  "      ⚠ NO compares las dos verosimilitudes para decidir: "
                  "con órdenes de diferenciación distintos son modelos de "
                  "DATOS distintos y el número no significa nada."]
    elif m0 < 1.02:
        lines += ["    ⚠ Está muy cerca del círculo, pero NO se movió al meter "
                  "la transferencia: ya estaba así en el diagonal. Entonces "
                  "es una cuestión del modelo univariante del output, no de "
                  "la transferencia -- revísalo en `art`."]
    return lines


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
    nombres = [s.name for s in specs]
    body = report_fit(f, table, nombres, se)
    if cs.links:
        agrupado = _by_series(body, cs, nombres)
        if agrupado:
            body = agrupado
    extra = ""
    if cs.links:
        try:
            extra = "\n" + "\n".join(_what_the_transfer_bought(name, f, cs))
        except Exception as exc:                           # noqa: BLE001
            extra = f"\n  (no se pudo contrastar la transferencia: {str(exc)[:120]})"
    return head + body + extra + "\n```"


# ── 4. diagnosis ───────────────────────────────────────────────────────────
def _what_now(ad, name, link_index):
    """El nodo N6: qué hacer con el veredicto. Las ramas piden cosas OPUESTAS.

    Un diagnóstico que dice "adecuado / no adecuado" y calla deja al analista
    con la parte difícil. Y en el caso de fallo, la respuesta correcta depende
    de algo que todavía no se ha mirado: si el veredicto lo sostiene la FORMA o
    una sola OBSERVACIÓN. Re-especificar (b,r,s) alrededor de un anómalo es
    exactamente como un modelo adquiere un retardo que nadie sabe interpretar.
    """
    lines = ["", "  ── QUÉ HACER AHORA " + "─" * 39, ""]

    if not ad.exogenous:
        lines += ["    🛑 LA EXOGENEIDAD FALLA (p = %.4f). Eso NO se arregla "
                  "cambiando (b, r, s): un modelo de transferencia de una sola "
                  "entrada supone que la salida no adelanta a la entrada, y "
                  "aquí lo hace. Es el mismo hallazgo que un DAG cíclico por "
                  "otra puerta -- el sistema es SIMULTÁNEO." % ad.p_exog,
                  "",
                  "      · si es real -> `sima`, que estima VARMA sin imponer "
                  "exogeneidad;",
                  "      · si sospechas de un anómalo COMÚN a las dos series, "
                  "se calibra en `art` y vuelve en el `.pre`.",
                  "",
                  "    Lo de abajo sobre la adecuación es secundario mientras "
                  "esto no se resuelva."]

    if ad.adequate:
        if ad.exogenous:
            lines += ["    ✅ El modelo pasa las dos pruebas: la forma de la "
                      "transferencia es adecuada y la entrada se comporta como "
                      "exógena.",
                      "",
                      "      Siguiente: `impulse_response` (nu(k), acumulada y "
                      "ganancia con sus bandas), `forecast`, y `evaluate` si "
                      "quieres error fuera de muestra.",
                      "",
                      "      Antes de darlo por cerrado, dos miradas baratas: "
                      "`plot_residuals` (serie + ACF/PACF, el panel de fue) "
                      "para ver lo que un portmanteau agregado no ve, y "
                      "`calibrate` para saber si el veredicto de adecuación "
                      "descansa en una sola observación. Un modelo adecuado "
                      "GRACIAS a un punto es tan frágil como uno inadecuado "
                      "POR un punto."]
    else:
        lines += ["    🛑 LA ADECUACIÓN FALLA (p = %.4f): queda estructura en "
                  "la CCF a k >= 0, así que la FORMA de la transferencia no "
                  "recoge todo lo que hay." % ad.p_transfer,
                  "",
                  "      ⚠ NO re-especifiques todavía. Hay DOS causas y piden "
                  "respuestas OPUESTAS, y sólo se distinguen mirando:",
                  "",
                  f"      1. `calibrate({name!r}, link_index={link_index})` "
                  "PRIMERO. Dice si el veredicto se sostiene en la forma o en "
                  "una sola observación.",
                  "      2. · si es la FORMA -> vuelve a `identify_link` y "
                  "revisa (b, r, s) con la CCF delante;",
                  "         · si es una OBSERVACIÓN -> es una INTERVENCIÓN, y "
                  "se calibra en `art`, sobre el escalón univariante, no aquí. "
                  "Re-especificar la forma alrededor de un anómalo es como un "
                  "modelo acaba con un retardo que nadie sabe interpretar.",
                  "",
                  "      `plot_calibration` enseña la CCF con y sin la "
                  "observación sospechosa: en la escuela eso se comprueba "
                  "mirando, no se supone."]

    lines += _reformulation_order(ad, name, link_index)

    if ad.significant_lags is not None and len(ad.significant_lags):
        lines += ["",
                  "    (Los cruces individuales de arriba son orientativos: con "
                  "~25 retardos al 5 % se espera alguno por azar. El contraste "
                  "conjunto es el portmanteau, no el recuento de barras.)"]
    return lines


def _reformulation_order(ad, name, link_index):
    """En qué ORDEN se arregla lo que está mal. Muñoz §2.6 p.42.

    Este es el enunciado más codificable de todo el capítulo metodológico, y
    mtram no lo decía en ninguna parte:

      "La especificación inadecuada de la relación v(B) puede generar la
       apariencia (en acf/pacf residuales) de especificación inadecuada del
       ruido theta(B) A LA VEZ QUE una ccf que requiere reformulación de la
       relación. Sin embargo, la especificación inadecuada del ruido NO puede
       dar la impresión en ccf de especificación inadecuada de la relación.
       Por estas razones, se reformula v(B) hasta que parezca adecuada ANTES
       de reformular theta(B)."

    La contaminación va en UNA dirección: una relación mal especificada
    ensucia la ACF residual **y** la CCF; un ruido mal especificado sólo
    ensucia la ACF. Así que cuando los dos instrumentos fallan a la vez, el
    orden de reparación no es cuestión de gusto -- está determinado, y el
    analista que empieza por el ruido persigue un síntoma y puede iterar
    mucho tiempo sin converger.

    Para decirlo hay que MIRAR los dos instrumentos, y hasta ahora `diagnose`
    sólo miraba uno.
    """
    f = _FITS.get(name)
    if f is None:
        return []
    try:
        from .school import noise_adequacy
        Q, p, k, df = noise_adequacy(f, series_index=0)
    except Exception:                                      # noqa: BLE001
        return []
    if not (p == p):                                       # NaN
        return []

    noise_bad, rel_bad = p < 0.05, not ad.adequate
    out = ["", "  ── EL RUIDO, Y EN QUÉ ORDEN " + "─" * 30, "",
           f"    Ljung-Box sobre la ACF del residuo del output: "
           f"Q({k}) = {Q:.2f},  g.l. = {df},  p = {p:.4f}"]

    if rel_bad and noise_bad:
        out += ["",
                "    ⚠ FALLAN LOS DOS -- y el orden de reparación NO es "
                "indiferente: **primero la RELACIÓN, después el RUIDO**.",
                "",
                "      La contaminación va en una sola dirección. Una relación "
                "v(B) mal especificada ensucia la ACF residual Y la CCF a la "
                "vez; un ruido mal especificado sólo puede ensuciar la ACF, "
                "nunca la CCF (Muñoz §2.6). Así que la ACF sucia que ves "
                "puede ser un síntoma de la relación, mientras que la CCF "
                "sucia no puede venir del ruido.",
                "",
                "      Reformula (b, r, s) hasta que la CCF quede limpia y "
                "VUELVE A MIRAR la ACF: puede haberse arreglado sola. Empezar "
                "por el ruido es perseguir un síntoma, y se puede iterar mucho "
                "tiempo sin converger."]
    elif noise_bad and not rel_bad:
        out += ["",
                "    La CCF está limpia y la ACF no: la RELACIÓN se sostiene y "
                "lo que falta es estructura en el RUIDO. Eso no se toca aquí "
                "-- el modelo del ruido viene del `.pre` del output; "
                "reidentifícalo en `art` y vuelve con el `.pre` nuevo.",
                "",
                "      (Este es el único caso en que el ruido es la primera "
                "parada, y es limpio precisamente porque la CCF descarta la "
                "otra causa.)"]
    elif rel_bad:
        out += ["",
                "    La ACF del ruido está limpia, así que lo que falla es la "
                "RELACIÓN y sólo la relación. No hay ambigüedad de orden aquí."]
    else:
        out += ["", "    Los dos instrumentos limpios: la relación y el ruido "
                "se sostienen por separado."]
    return out


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
    _require_link(f, link_index, "test for adequacy or exogeneity")
    ad = drtran.transfer_adequacy(f, link_index=link_index, embed=f.embed)
    return (drtran.report_adequacy(ad)
            + "\n" + "\n".join(_what_now(ad, name, link_index)))


def _fit_variant(name, links, embed=None):
    """Ajusta una variante de la red SIN tocar el estado de la sesión.

    Un experimento de sobreajuste que deja el caso estimado en el modelo
    ampliado es una trampa: el analista lo rechaza, sigue trabajando, y todo
    lo que hace después sale del modelo que acaba de rechazar.
    """
    from drtran.estimate import standard_errors
    from drtran.slots import build_slots

    cs = build_cast_spec(_require(name), links=links)
    table = build_slots(cs)
    f = drtran.fit(cs, x0=drtran.x0_full(cs, table),
                   embed=(f_embed(name) if embed is None else embed),
                   slots=table)
    if f.ifault:
        return None, None, None
    return f, table, standard_errors(f)


def f_embed(name):
    f = _FITS.get(name)
    return True if f is None else bool(f.embed)


@mcp.tool()
def overfit(name: str, link_index: int = 0) -> str:
    """Overfit ON PURPOSE: enlarge the transfer and see whether it protests.

    Every one of Muñoz's six cases does this, and none of them skips it even
    when nothing looks wrong — an adequate portmanteau says the model is not
    contradicted, and that is not the same as saying nothing better was
    available. Brajín §2.3.1 states the doctrine: a model is confirmed by
    enlarging it and finding the extra parameters non-significant.

    Two enlargements, one at a time: one more numerator weight (s+1) and one
    denominator (r+1). Each is refitted jointly and compared by likelihood
    ratio with 1 degree of freedom.

    **A failed experiment is not a confirmation.** Muñoz 6.4.3 abandons an
    overfit because "la situación de estimación está mal definida (altas
    correlaciones entre muchos de los parámetros de relación), por lo que este
    experimento puede considerarse FALLIDO" — the enlarged model says nothing
    about the small one, in either direction. That distinction is reported.

    The session's estimated model is left untouched.
    """
    from scipy import stats as _st

    from .school import worst_correlations

    f0 = _require_fit(name)
    _require_link(f0, link_index, "overfit")
    links = list(f0.cast_spec.links)
    lk = links[link_index]
    ll0 = float(f0.loglik)

    out = ["```", f"  SOBREAJUSTE de {name!r}, enlace {link_index} "
           f"(b={lk.b}, r={lk.r}, s={lk.s})", "",
           f"  Modelo actual:  logL = {ll0:.6f}", ""]

    trials = [("s+1  (un peso más en el numerador)",
               Link(lk.out, lk.inp, lk.b, lk.r, lk.s + 1),
               f"omega{link_index + 1}[{lk.s + 1}]"),
              ("r+1  (un denominador)",
               Link(lk.out, lk.inp, lk.b, lk.r + 1, lk.s),
               f"delta{link_index + 1}[{lk.r + 1}]")]   # delta_0 = 1 va implícito

    verdicts = []
    for label, newlk, slot in trials:
        v = list(links); v[link_index] = newlk
        f1, t1, se1 = _fit_variant(name, v)
        if f1 is None:
            out += [f"  {label}", "    ✗ no converge — eso ya es información: "
                    "el parámetro extra no tiene dónde ir.", ""]
            verdicts.append(("no converge", label)); continue
        ll1 = float(f1.loglik)
        lr = 2.0 * (ll1 - ll0)
        p = float(1.0 - _st.chi2.cdf(max(lr, 0.0), 1))
        val, err = school_slot(f1, t1, se1, slot)
        pairs, nflag = worst_correlations(f1, se1, t1, top=3)
        out += [f"  {label}",
                f"    logL = {ll1:.6f}   LR = {lr:.3f}  (1 g.l.)  p = {p:.4f}"]
        if val is not None and err and err > 0:
            out.append(f"    {slot} = {val:+.6f}  (e.t. {err:.6f}, "
                       f"t = {val / err:.2f})")
        elif val is not None:
            out.append(f"    {slot} = {val:+.6f}  (sin error típico)")
        if nflag:
            worst = pairs[0] if pairs else ("", "", float("nan"))
            out += [f"    ⚠ EXPERIMENTO FALLIDO: {nflag} pareja(s) de "
                    "parámetros con |correlación| >= .9 "
                    f"({worst[0]} / {worst[1]}: {worst[2]:+.3f}). La "
                    "situación de estimación del modelo ampliado está mal "
                    "definida, así que NO dice nada sobre el pequeño -- ni a "
                    "favor ni en contra (Muñoz 6.4.3)."]
            # Un caso concreto que NO es mala suerte: el último peso del
            # numerador contra el denominador nuevo. Los dos describen la misma
            # cola, así que sobre un modelo que no necesita denominador la
            # colinealidad es la respuesta ESPERADA, no un accidente. Decirlo
            # evita que el analista lea "fallido" como "mal dato".
            tail = {f"omega{link_index + 1}[{lk.s}]",
                    f"delta{link_index + 1}[{lk.r + 1}]"}
            if newlk.r > lk.r and {worst[0], worst[1]} == tail:
                out += ["      (Y aquí la pareja es el último peso del "
                        "numerador contra el denominador nuevo: los dos "
                        "describen la misma cola. Sobre un modelo que NO "
                        "necesita denominador esa colinealidad es lo que "
                        "cabe esperar -- el experimento no fracasa por los "
                        "datos, fracasa porque la ampliación es redundante.)"]
            verdicts.append(("fallido", label))
        elif p < 0.05:
            out += ["    → el parámetro extra SÍ aporta. El modelo actual se "
                    "queda corto por ese lado."]
            verdicts.append(("aporta", label))
        else:
            out += ["    → no aporta. Por este lado el modelo actual se "
                    "sostiene."]
            verdicts.append(("no aporta", label))
        out.append("")

    kinds = [k for k, _ in verdicts]
    clean = [lb for k, lb in verdicts if k == "no aporta"]
    failed = [lb for k, lb in verdicts if k in ("fallido", "no converge")]
    if "aporta" in kinds:
        out += ["  El modelo NO está cerrado: alguna ampliación aporta. "
                "Cámbiala con `set_network` y vuelve a `estimate` y "
                "`diagnose` -- una ampliación significativa cambia también el "
                "veredicto de adecuación, no sólo la verosimilitud."]
    elif clean:
        out += ["  ✅ CONFIRMADO por %d de %d ampliaciones: no significativas "
                "y con la estimación bien definida. Esto es más que un "
                "portmanteau adecuado -- un portmanteau dice que el modelo no "
                "está CONTRADICHO; esto dice que ampliarlo no compra nada."
                % (len(clean), len(verdicts))]
        if failed:
            out += ["  Las otras (%s) no llegaron a veredicto, y eso no cuenta "
                    "ni a favor ni en contra." % "; ".join(failed)]
    else:
        out += ["  Sin veredicto: NINGUNA ampliación salió bien condicionada, "
                "y un sobreajuste mal condicionado se descarta, no se "
                "interpreta. El modelo no queda confirmado ni desmentido."]
    out += ["", f"  (El modelo estimado de {name!r} sigue siendo el de antes: "
            "estas variantes no se han guardado.)", "```"]
    return "\n".join(out)


def school_slot(fit, table, se, name):
    """`school._slot` por la puerta principal, para los tools de arriba."""
    from .school import _slot
    return _slot(fit, table, se, name)


@mcp.tool()
def calibrate(name: str, link_index: int = 0, threshold: float = 3.5) -> str:
    """Which observations are BENDING the instruments — and what to do about it.

    Leave-one-out over the CCF and the adequacy portmanteau. Its verdict is the
    branch to take at node N6 when adequacy fails, and the two branches need
    OPPOSITE responses:

    * **shape** — no single observation explains the failure → re-identify
      (b, r, s);
    * **observation** — the verdict rests on one point → that is an
      INTERVENTION. Re-specifying the shape around it is how a model acquires a
      lag nobody can interpret. Interventions are calibrated in `art`, on the
      univariate rung, and travel here in the `.pre`.

    This is NOT ART's scan run again: an anomaly in the output's univariate
    residuals may be explained by the INPUT once the transfer is in the model.
    What survives the joint fit is the genuine one.
    """
    from .calibrate import calibrate as _cal
    from .calibrate import report_calibration

    f = _require_fit(name)
    _require_link(f, link_index, "calibrate")
    txt = report_calibration(_cal(f, link_index=link_index,
                                  threshold=threshold))
    return txt + "\n" + "\n".join(_which_pairs(f, link_index))


def _which_pairs(f, link_index):
    """El pico de la CCF residual, trazado a PARES de observaciones.

    `calibrate` pregunta "qué observación tuerce los instrumentos" y contesta
    con leave-one-out sobre una serie. Los casos de Muñoz preguntan algo un
    paso más fino, y la respuesta es un PAR:

      "se justifica por distorsión negativa entre el ruido preblanqueado y los
       residuos del input en II/94 y II/92, y III/86 y III/84" (6.4.2, ret. 8)

    Dos fechas, una en cada serie, separadas por el retardo. Y no es otra
    manera de decir "un anómalo": ninguna de las dos observaciones tiene por
    qué ser extrema por su cuenta. El coeficiente de la CCF es una suma de
    PRODUCTOS, y un producto es grande cuando los dos factores son
    medianamente grandes A LA VEZ y del mismo signo. Un par de observaciones
    corrientes que se alinean puede cargar con un coeficiente entero, y ningún
    barrido sobre una sola serie lo enseñará jamás.

    Distinguirlo importa porque las dos cosas piden respuestas distintas: un
    anómalo es una intervención; una coincidencia de dos valores moderados a
    una distancia fija es o una dinámica real que el modelo no recoge, o nada.
    """
    from .school import ccf_pairs

    try:
        lag, r_k, pairs = ccf_pairs(f, link_index)
    except Exception:                                      # noqa: BLE001
        return []
    if not pairs:
        return []
    names = f.cast_spec.names
    li = f.cast_spec.links[link_index]
    out = ["```", f"  DE QUÉ PARES SALE EL PICO — retardo {lag}, "
           f"r = {r_k:+.4f}", "",
           f"    {names[li.inp]:>12s}   {names[li.out]:>12s}    "
           "aporta al coeficiente", "    " + "-" * 52]
    for dx, dn, c in pairs:
        out.append(f"    {dx:>12s}   {dn:>12s}   {100 * c:>10.1f} %")
    top = 100 * abs(pairs[0][2])
    out.append("")
    if top > 15.0:
        out += [f"    ⚠ Un solo par carga con el {top:.0f} % del coeficiente. "
                "Míralo antes de tratar ese retardo como estructura.",
                "",
                "      Ojo a qué NO dice esto: no dice que ninguna de las dos "
                "observaciones sea anómala. El coeficiente es una suma de "
                "productos, así que dos valores moderados que se alinean "
                "bastan, y un barrido sobre una sola serie no los ve. Si son "
                "dos fechas con una historia común, es un suceso; si no, es "
                "coincidencia y el retardo no significa nada."]
    else:
        out += [f"    Ningún par domina (el mayor aporta el {top:.0f} %): el "
                "coeficiente lo sostiene la muestra entera, no un suceso."]
    out.append("```")
    return out


@mcp.tool()
def plot_calibration(name: str, link_index: int = 0, path: str = "") -> str:
    """PLOT the CCF **with and without** the dominant anomaly — the verification.

    In the school's teaching this is a fact to VERIFY, not to infer, and it is
    easy to see: an anomaly inflates the residual variance, which is the divisor
    of every correlation, so it flattens ALL the lags at once — not only the ones
    it touches. Take the point out and the coefficients come back.

    Show this beside `calibrate`'s table. The number (`CCF x`) states the claim;
    the picture is what lets the analyst check it.
    """
    from .calibrate import calibrate as _cal
    from .calibrate import plot_calibration as _pcal
    from .plots import save

    f = _require_fit(name)
    _require_link(f, link_index, "plot the calibration")
    cal = _cal(f, link_index=link_index)
    if not cal.anomalies:
        raise ValueError("no hay ninguna anomalía por encima del umbral: "
                         "nada que verificar")
    return save(_pcal(cal), _png(name, f"cal{link_index}", path))


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


# ── autonomous: the same nodes, the documented defaults, said out loud ─────
@mcp.tool()
def build_model(name: str, horizon: int = 12) -> str:
    """AUTONOMOUS run: walk the ladder taking the documented default at every
    decision node, and REPORT WHICH ONE IT TOOK at each.

    The nodes are in `docs/DECISION_NODES.md`. The two rules this obeys:

    * **it never makes a claim the data cannot make for it** — it does not free a
      covariance, does not invent a constraint, and does not prune a cycle;
    * **it differs from guided only in who decides**, never in what is computed.
      A guided run that made the same choices reaches the same model.

    It STOPS, rather than choosing, at two points: a cyclic network (the system
    is simultaneous — `sima`) and a failed exogeneity test (a single-input
    transfer model does not hold). Both are findings, not obstacles.
    """
    from .estimate import standard_errors
    from .forecast import forecast as _fcast
    from .forecast import level_band
    from .irf import impulse_response as _irf
    from .slots import build_slots

    specs = _require(name)
    log = [f"RUN AUTÓNOMO — {name!r}", "=" * 62, ""]

    # N0 — the output
    log += [f"N0  salida = {specs[0].name} (el primer .pre). No es deducible de "
            f"los datos:", "    es la pregunta que trae el analista.", ""]

    # N2 — the network, from the residual CCFs of the diagonal
    cs0 = build_cast_spec(specs)
    f0 = drtran.fit(cs0, embed=True)
    net = drtran.identify_network(cs0, x=f0.x)

    cyc = net.cycle
    if cyc is not None:
        route = " -> ".join(net.names[i] for i in cyc)
        log += ["N2′ CICLO EN LA RED PROPUESTA: " + route,
                "",
                "    Sin orden topológico no hay VARMA triangular: el sistema es",
                "    SIMULTÁNEO. Podar el enlace más débil para que 'funcione'",
                "    sería inventar una estructura recursiva que los datos no",
                "    sostienen. Esto es un HALLAZGO, no un obstáculo.",
                "",
                "    → El asistente que corresponde es `sima` (VARMA simultáneo).",
                "    → O poda tú uno de esos enlaces: la poda es tu juicio."]
        return "\n".join(log)

    if not net.candidates:
        log += ["N2  ningún enlace por encima de la banda: no hay transferencia",
                "    que identificar. El modelo es el DIAGONAL.",
                f"    logL = {f0.loglik:.6f}"]
        return "\n".join(log)

    links = net.links
    log += ["N2  red (candidatos tal como salen, sin podar):"]
    for c in net.candidates:
        log.append(f"      {net.names[c.inp]} -> {net.names[c.out]}   "
                   f"pico {c.peak:+.3f}   b={c.b} r=0 s={c.s}")
    log += ["", "N3  covarianzas contemporáneas: NINGUNA liberada."]
    if net.covariances:
        for i, j, r0 in net.covariances:
            log.append(f"      {net.names[i]} · {net.names[j]}  r(0) = {r0:+.3f}"
                       "  ← detectada, NO liberada")
        log += ["    Liberarlas es una afirmación sobre el mundo, y además cuesta",
                "    la descomposición de la varianza. Decisión del analista."]
    log += ["", "N4  restricciones: ninguna. Codifican teoría; un run autónomo",
            "    no tiene ninguna.",
            "", "N5  cast: EMPOTRADO (el defecto; no trunca la muestra).", ""]

    _LINKS[name] = links
    cs = build_cast_spec(specs, links=links)
    table = build_slots(cs)
    f = drtran.fit(cs, x0=drtran.x0_full(cs, table), embed=True, slots=table)
    if f.ifault:
        return "\n".join(log + [f"La verosimilitud no se pudo evaluar: "
                                f"ifault={f.ifault}. Me detengo."])
    _FITS[name] = f
    _TABLES[name] = table
    log += [f"    estimado: logL = {f.loglik:.6f}   ({f.status}, "
            f"{f.nit} iteraciones)", ""]

    # N6 — diagnosis, with ONE revision loop back to N1 if the shape is wrong
    log += ["N6  diagnosis:"]
    stop = False
    revised = False
    for _pass in (0, 1):
        bad = []
        for k in range(len(links)):
            ad = drtran.transfer_adequacy(f, link_index=k, embed=True)
            nm = f"{net.names[links[k].inp]} -> {net.names[links[k].out]}"
            log.append(f"      {nm}:  adecuación p = {ad.p_transfer:.4f}   "
                       f"exogeneidad p = {ad.p_exog:.4f}")
            if ad.p_transfer < 0.05:
                bad.append(k)
        if not bad or revised:
            break
        # La adecuación falla, y hay DOS causas que piden respuestas opuestas.
        # Antes de revisar la forma hay que descartar que sea UNA observación:
        # reespecificar alrededor de un anómalo es como un modelo acaba con un
        # retardo que nadie sabe interpretar.
        from .calibrate import calibrate as _cal
        por_obs = []
        for k in list(bad):
            cal = _cal(f, link_index=k)
            if cal.verdict == "observation":
                d = cal.decisive[0]
                por_obs.append((k, d))
                bad.remove(k)
                nm = f"{net.names[links[k].inp]} -> {net.names[links[k].out]}"
                log += ["",
                        f"    ⚠ {nm}: la adecuación NO falla por la forma, falla",
                        f"      por UNA observación ({d.date}): sin ella la p pasa",
                        f"      de {cal.p_transfer:.4f} a {d.p_transfer_without:.4f}.",
                        "      Eso es una INTERVENCIÓN, y se calibra en `art`, en",
                        "      el escalón univariante, no reespecificando aquí.",
                        "      NO reviso (b,r,s) de este enlace."]
        if not bad:
            break
        log += ["", "    La adecuación falla y ninguna observación suelta lo",
                "    explica: la FORMA es la equivocada. Vuelvo a N1 con el",
                "    preblanqueo bivariante, que es el instrumento fino. Una",
                "    sola revisión."]
        for k in bad:
            lk = links[k]
            cs_k = build_cast_spec(specs, links=[Link(lk.out, lk.inp, 0, 0, 0)])
            idt = drtran.identify(cs_k, cs_k.links[0])
            links[k] = Link(lk.out, lk.inp, b=idt.b, r=idt.r, s=idt.s)
            log.append(f"      {net.names[lk.inp]} -> {net.names[lk.out]}:  "
                       f"b={lk.b} r={lk.r} s={lk.s}  ->  "
                       f"b={idt.b} r={idt.r} s={idt.s}")
        _LINKS[name] = links
        cs = build_cast_spec(specs, links=links)
        table = build_slots(cs)
        f = drtran.fit(cs, x0=drtran.x0_full(cs, table), embed=True, slots=table)
        if f.ifault:
            return "\n".join(log + [f"    la reestimación falló: ifault={f.ifault}"])
        _FITS[name] = f
        _TABLES[name] = table
        revised = True
        log += [f"    reestimado: logL = {f.loglik:.6f}   ({f.status}, "
                f"{f.nit} iteraciones)", ""]

    for k in range(len(links)):
        ad = drtran.transfer_adequacy(f, link_index=k, embed=True)
        nm = f"{net.names[links[k].inp]} -> {net.names[links[k].out]}"
        if ad.p_exog < 0.05:
            stop = True
            log += ["",
                    f"    ⚠ LA EXOGENEIDAD FALLA en {nm}. La entrada no es exógena,",
                    "      así que un modelo de transferencia de una sola entrada",
                    "      NO se sostiene. No es un problema de ajuste fino: es el",
                    "      mismo hallazgo que un DAG cíclico por otra vía.",
                    "      → `sima`. Me detengo aquí y no reespecifico alrededor."]
    if stop:
        return "\n".join(log)

    # the answer
    se = standard_errors(f)
    cov = None if se.ifault else se.cov
    log += ["", "RESULTADO", "-" * 62]
    for k in range(len(links)):
        ir = _irf(f, link_index=k, cov=cov)
        g = (f"ganancia nu(1) = {ir.gain:.6f}"
             + (f"  (s.e. {ir.se_gain:.6f}, t = {ir.gain/ir.se_gain:.2f})"
                if ir.se_gain == ir.se_gain and ir.se_gain > 1e-15 else ""))
        log.append(f"  {ir.inp_name} -> {ir.out_name}:  {g}")

    fc = _fcast(f, L=horizon, embed=True)
    lvl, lo, hi = level_band(fc, cs, series=0)
    log += ["", f"  previsión de {cs.names[0]} ({horizon} periodos, nivel, 95 %):"]
    for l in range(min(horizon, 6)):
        log.append(f"    h={l+1:2d}  {lvl[l]:11.4f}   [{lo[l]:.4f}, {hi[l]:.4f}]")
    if horizon > 6:
        log.append(f"    … hasta h={horizon}")
    log += ["", "Todos los defectos tomados están enumerados arriba. Para",
            "revisarlos uno a uno, usa el modo guiado."]
    return "\n".join(log)


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
