#!/usr/bin/env python3
"""Rodaje P07: `evaluate` corta cada serie por sus PRIMERAS `window`
observaciones, no por fecha, y reestima en esa ventana tambien el modelo de
la entrada sin decirlo.

`evaluate.truncate(specs, n)` se queda con `ts.data[:n]` de cada `.pre`. Si
las series empiezan en fechas distintas y acaban en la misma (Portugal
desde 2010:01 con el WTI desde 2002:01, lo que el curso pide), `load_pre`,
`estimate` y `diagnose` funcionan --el cast alinea por el final--, pero la
ventana de `evaluate` termina en fechas distintas para cada serie y el cast
se niega («the series do NOT end on the same date»): `evaluate` no se puede
usar sin rehacer el `.pre` del WTI desde 2010 (G12 tuvo que escribir
`datos/ipc_wti_2010_2025.csv`). Y dentro de `rolling_evaluation` ese mismo
ValueError se traga en silencio en cada origen.

Ademas, `fixed_window_fit` reestima TODO el vector en la ventana --tambien el
AR del WTI-- y el informe no lo dice.

Caso: ES_CPI_m10.1.pre recortado a 2010:01-2019:12 (120 obs) frente a
WTI_ar1.1.pre (2002:01-2019:12, 216). `estimate` va; `evaluate(window=96)`
debe ventanear por fecha (2010:01-2017:12 en las dos).

Sale con 1 mientras `evaluate` falle con este caso.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, tempfile, warnings
warnings.filterwarnings("ignore")
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
os.chdir(RAIZ)
import drtran.mcp_server as M

lin = open(ES).read().split("\n")
i = next(k for k, l in enumerate(lin) if l.startswith("** Time series"))
datos = [l for l in lin[i + 1:] if l.strip()]
assert len(datos) == 216
cab = "\n".join(lin[:i + 1]).replace(" 216  1 2002 ES_CPI", " 120  1 2010 ES_CPI")
tmp = tempfile.mkdtemp()
pre = os.path.join(tmp, "ES_CPI_2010.inp")
open(pre, "w").write(cab + "\n" + "\n".join(datos[96:]) + "\n")

M.load_pre("EV", f"{pre},{WTI}", check=False)
M.set_network("EV", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
est = M.estimate("EV")
print("estimate con las dos series:", "ℓ =" + re.search(r"ℓ = (-?[\d.]+)", est).group(1))
try:
    out = M.evaluate("EV", window=96, horizon=6)
    print(out[:600])
    ok = True
except Exception as exc:
    print("evaluate(window=96) ->", type(exc).__name__, str(exc)[:300])
    ok = False
print("esperado: la ventana de 96 obs sobre la muestra comun por fecha "
      "(2010:01-2017:12 en las dos), y el informe dice que tambien se "
      "reestima el modelo del WTI")
sys.exit(0 if ok else 1)
