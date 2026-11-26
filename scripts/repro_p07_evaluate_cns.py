"""P07: `evaluate` de mtram descarta las restricciones y no deja fichero.

Rodaje P07 (noviembre 2026). El grupo 3 adoptó Irlanda con pesos fijados a
cero por un .cns y lo evaluó fuera de muestra: `evaluate` no tiene
`cns_path` y reestima la red con TODOS los pesos libres, así que lo que
evaluó es otro modelo que el adoptado, sin un aviso. Y el resultado sólo
existe en el texto devuelto: no escribe CSV ni nada que se pueda entregar.

La librería ya sabe hacer las dos cosas: `evaluate.fixed_window_fit(...,
slots_of=)`, `rolling_evaluation(..., slots=)` y `write_rolling_csv`. La
herramienta no las usa (mcp_server.py:3117-3133): ni recibe un .cns ni toma
la tabla que `estimate` dejó en _TABLES.

Uso:  python3 scripts/repro_p07_evaluate_cns.py     (≈ 1 min)
Sale con 1 mientras el defecto exista.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import inspect
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import drtran.evaluate as E                     # noqa: E402
import drtran.mcp_server as M                   # noqa: E402

D = os.path.join(RAIZ, "tests", "data", "p07")
M.load_pre("ev", f"{D}/ipc_det_ar2.pre,{D}/wti_ar1.pre", check=False)
M.set_network("ev", '[{"out":0,"inp":1,"b":0,"r":0,"s":4}]')
with tempfile.NamedTemporaryFile("w", suffix=".cns", delete=False) as fh:
    fh.write("omega1[2] = 0\nomega1[3] = 0\n")
M.estimate("ev", cns_path=fh.name)
t_adoptado = M._TABLES["ev"]
print(f"adoptado: omega1[2] = omega1[3] = 0, {t_adoptado.n_free} libres")

visto = {}
_orig = E.fixed_window_fit


def espia(*a, **k):
    f, cs = _orig(*a, **k)
    visto["f"] = f
    return f, cs


E.fixed_window_fit = espia
antes = set(os.listdir(os.getcwd()))
M.evaluate("ev", window=204, horizon=1)
E.fixed_window_fit = _orig
f = visto["f"]
n_libres = len(f.xfree) if f.xfree is not None else len(f.x)
nombres = t_adoptado.names
w2 = f.x[nombres.index("omega1[2]")]
w3 = f.x[nombres.index("omega1[3]")]
print(f"evaluate estimó: {n_libres} libres, omega1[2] = {w2:+.6f}, "
      f"omega1[3] = {w3:+.6f}")
print("evaluate admite cns_path:",
      "cns_path" in inspect.signature(M.evaluate).parameters)
nuevos = set(os.listdir(os.getcwd())) - antes
print("ficheros escritos por evaluate:", sorted(nuevos) or "ninguno")
os.unlink(fh.name)

fallo = (w2 != 0 or w3 != 0) or not nuevos
print("BUG PRESENTE" if fallo else "ok")
sys.exit(1 if fallo else 0)
