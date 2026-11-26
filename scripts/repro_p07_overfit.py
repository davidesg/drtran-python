"""P07: `overfit` arranca cada ampliación de las semillas por defecto y sin el .cns.

Rodaje P07 (noviembre 2026), dos síntomas de una misma causa:

  1. LR NEGATIVO. Italia, forma (0,1,1): la ampliación r+1 da
     logL = −644,98 frente a −643,06 del modelo actual, LR = −3,835. Un modelo
     que anida al actual no puede tener menos verosimilitud en su óptimo: lo
     que se compara es un óptimo LOCAL peor. El informe lo tapa con
     «EXPERIMENTO FALLIDO» por la correlación delta1/delta2.
  2. EL .cns SE PIERDE. Con pesos fijados a cero por restricción (Italia con
     pesos en 0, 1 y 4), la ampliación los libera todos y el LR se lee «con
     1 g.l.» cuando la diferencia es de 3.

Causa: `_fit_variant` (mcp_server.py:2737) construye una tabla nueva con
`build_slots(cs)` —sin las restricciones que `estimate` guardó en _TABLES— y
arranca de `drtran.x0_full(cs, table)`, las semillas univariantes con la
transferencia a cero, en vez del óptimo del modelo actual con el parámetro
nuevo a cero. Arrancar del óptimo actual garantiza logL(ampliado) >= logL(actual).

No es defecto (comprobado aquí mismo): el LR = 0,000 de España en s+1 y r+1.
Con el WTI sin escalón de 2008, delta sale ≈ −0,01 desde cualquier semilla
(0, 0,18, 0,5, −0,3): es el óptimo de esos univariantes, no una semilla que no
se mueve. La referencia (delta = 0,18, LR 1,06) usa otro WTI.

Uso:  python3 scripts/repro_p07_overfit.py        (≈ 1 min)
Sale con 1 mientras el defecto exista.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import numpy as np                              # noqa: E402

import drtran                                   # noqa: E402
import drtran.mcp_server as M                   # noqa: E402
from drtran.cast import build_cast_spec         # noqa: E402
from drtran.network import Link                 # noqa: E402
from drtran.slots import build_slots            # noqa: E402

D = os.path.join(RAIZ, "tests", "data", "p07")
PATHS = f"{D}/ipc_det_ar2.pre,{D}/wti_ar1.pre"
fallos = []

# ── 1. LR negativo en r+1 ─────────────────────────────────────────────────
M.load_pre("it", PATHS, check=False)
M.set_network("it", '[{"out":0,"inp":1,"b":0,"r":1,"s":1}]')
M.estimate("it")
f0, t0 = M._FITS["it"], M._TABLES["it"]
ll0 = float(f0.loglik)
txt = M.overfit("it")
lr = [ln for ln in txt.splitlines() if "LR =" in ln]
print(f"(0,1,1) logL actual = {ll0:.6f}")
for ln in lr:
    print("  overfit:", ln.strip())
lr_r = float(lr[1].split("LR =")[1].split()[0])

# el mismo modelo ampliado, arrancando del óptimo actual con delta2 = 0
cs1 = build_cast_spec(M._SPECS["it"], links=[Link(0, 1, 0, 2, 1)])
t1 = build_slots(cs1)
x = drtran.x0_full(cs1, t1)
for i, nm in enumerate(t0.names):
    j = t1.index(nm)
    if j >= 0:
        x[j] = f0.x[i]
f1 = drtran.fit(cs1, x0=x, embed=True, slots=t1)
print(f"  r+1 desde el óptimo actual: logL = {f1.loglik:.6f}  "
      f"LR = {2 * (f1.loglik - ll0):.3f}")
if lr_r < 0:
    fallos.append(f"overfit r+1 da LR = {lr_r} < 0 (desde el óptimo actual "
                  f"sale {2 * (f1.loglik - ll0):.3f})")

# ── 2. el .cns se ignora ──────────────────────────────────────────────────
M.set_network("it", '[{"out":0,"inp":1,"b":0,"r":0,"s":4}]')
with tempfile.NamedTemporaryFile("w", suffix=".cns", delete=False) as fh:
    fh.write("omega1[2] = 0\nomega1[3] = 0\n")
    cns = fh.name
M.estimate("it", cns_path=cns)
fa, ta = M._FITS["it"], M._TABLES["it"]
print(f"\n(0,4,0) con omega1[2] = omega1[3] = 0: logL = {fa.loglik:.6f}, "
      f"{ta.n_free} libres")
fv, tv, _ = M._fit_variant("it", [Link(0, 1, 0, 0, 5)])
w = {nm: fv.x[tv.index(nm)] for nm in ("omega1[2]", "omega1[3]")}
print(f"  variante s+1 de overfit: {tv.n_free} libres; "
      + ", ".join(f"{k} = {v:+.6f}" for k, v in w.items()))
print(f"  libres de más: {tv.n_free - ta.n_free} (overfit dice 1 g.l.)")
if any(abs(v) > 0 for v in w.values()) or tv.n_free - ta.n_free != 1:
    fallos.append("overfit libera los pesos fijados por el .cns: "
                  f"{tv.n_free - ta.n_free} parámetros más, leídos con 1 g.l.")
os.unlink(cns)

print()
for f in fallos:
    print("BUG:", f)
print("BUG PRESENTE" if fallos else "ok")
sys.exit(1 if fallos else 0)
