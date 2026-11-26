"""P07: del modelo conjunto de mtram no queda fichero, y `write_inp` pisa.

Rodaje P07 (noviembre 2026). El enunciado pide los .out de los modelos; con
mtram, las cifras de la transferencia (pesos, ganancia, LR, adecuación) sólo
quedaron copiadas a mano en los registros y no se pueden verificar (grupos
10, 12, Irlanda del 9). La CLI sí escribe NAME.out (cli.py:794); ninguna
herramienta de mtram lo hace.

Y `write_inp`, la única que escribe, (a) no lleva la red ni los pesos, y
(b) nombra con `next_inp_path`, que pone SIEMPRE el sufijo .1
(pre.py:296-314) sin mirar si existe: dos formas candidatas del mismo caso,
o dos países que comparten el WTI, escriben el mismo `WTI....1.inp` y el
segundo borra el primero sin aviso (grupo 5). El docstring dice que nunca
sobrescribe la fuente; sobrescribe lo que escribió antes.

Uso:  python3 scripts/repro_p07_ficheros_conjunto.py     (≈ 20 s)
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

import drtran.mcp_server as M                   # noqa: E402

PATHS = f"{ES},{WTI}"
out = tempfile.mkdtemp()
os.chdir(out)

M.load_pre("f", PATHS, check=False)
escritos = {}
for red in ('[{"out":0,"inp":1,"b":0,"r":0,"s":1}]',
            '[{"out":0,"inp":1,"b":0,"r":1,"s":0}]'):
    M.set_network("f", red)
    M.estimate("f")
    M.diagnose("f")
    txt = M.write_inp("f", outdir=out)
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln.endswith(".inp"):
            escritos.setdefault(ln, []).append(open(ln).read())

pisados = [p for p, v in escritos.items() if len(v) > 1 and v[0] != v[1]]
print("ficheros en", out, ":", sorted(os.listdir(out)))
print("escritos dos veces con contenido distinto (el segundo pisó al primero):")
for p in pisados:
    print("  ", os.path.basename(p))
con_red = any("omega" in v[-1].lower() or "transfer" in v[-1].lower()
              for v in escritos.values())
print("¿algún fichero lleva la transferencia?", con_red)
outs = [n for n in os.listdir(out) if n.endswith(".out")]
print(".out del modelo conjunto:", outs or "ninguno")

fallo = bool(pisados) or not outs
print("BUG PRESENTE" if fallo else "ok")
sys.exit(1 if fallo else 0)
