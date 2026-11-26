"""P07: el arreglo de BUG-70 no llegó a `impulse_response` ni a su figura.

Rodaje P07 (noviembre 2026). El IPC de España se quedaba en steptol con
factor 1; mtram pide factor 100 para ESA serie, el analista lo hace y el WTI
sigue a 1. `estimate` ya da la ganancia en las unidades de cada serie
(BUG-70, 0.2.7), pero `impulse_response` y `plot_impulse_response`, que es
donde el analista va a buscar la respuesta («un 10 % de petróleo, cuánto
IPC»), siguen en las del motor: ganancia 2,71 donde es 0,027, sin un aviso.
Grupos 2 y 9 leyeron la cifra ×100.

Código: mcp_server.py:3004 (`impulse_response`) y :1655
(`plot_impulse_response`) pasan el ajuste a `irf.report_irf` / `plots.plot_irf`
sin `fue.scale`; irf.py no conoce los factores.

Uso:  python3 scripts/repro_p07_escala_irf.py      (≈ 10 s)
Sale con 1 mientras el defecto exista.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os
import re
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import drtran.mcp_server as M                   # noqa: E402

# El mismo WTI con factor 1 (el que escribe la GUI del taller).
src = open(WTI).read()
assert "\n 0.00 100.00\n" in src
tmp = tempfile.mkdtemp()
wti1 = os.path.join(tmp, "wti_f1.pre")
open(wti1, "w").write(src.replace("\n 0.00 100.00\n", "\n 0.00 1.00\n"))

M.load_pre("mx", f"{ES},{wti1}", check=False)
M.set_network("mx", '[{"out":0,"inp":1,"b":0,"r":0,"s":1}]')
est = M.estimate("mx")
irf = M.impulse_response("mx")

g_series = float(re.search(r"EN LAS UNIDADES DE CADA SERIE: ([+-][\d.]+)", est).group(1))
g_irf = float(re.search(r"gain nu\(1\) = ([+-]?[\d.]+)", irf).group(1))
avisa = bool(re.search(r"unidades|units|factor|escala|scale", irf, re.I))
print(f"estimate, en las unidades de cada serie: {g_series:+.6f}")
print(f"impulse_response, ganancia:              {g_irf:+.6f}   "
      f"¿dice en qué unidades? {avisa}")
fallo = abs(g_irf - g_series) > 1e-4 and not avisa
print("BUG PRESENTE" if fallo else "ok")
sys.exit(1 if fallo else 0)
