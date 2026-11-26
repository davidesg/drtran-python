#!/usr/bin/env python3
"""Rodaje P07: `identify_link` informa de barras hasta k = 24 y la CCF que
dibuja junto a esos numeros sólo llega a ±12. Ademas no acepta `path`.

La identificacion (`identify.py`) usa max(10, min(n // 4, 24)) retardos: en
el caso de clase lista k = 24 como significativo y dice «miralo en el
grafico antes de aceptar que se descarte». La figura (`plots.plot_ccf`) usa
`ccf_default_lags(12)` = 12: el retardo 24 no esta. Lo mismo con k = 14 y
-16 en Francia, Italia y Alemania (G5, G11, G12). `identify_link` no tiene
`path`, asi que la figura incrustada no se puede guardar donde pide la
practica; hay que llamar aparte a `plot_ccf(lags=24, path=...)`, que el
asistente no hizo en G5.

Sale con 1 mientras el texto cite un retardo que la figura no dibuja, o
`identify_link` no acepte `path`.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import inspect, os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import drtran.mcp_server as M
from drtran.plots import plot_ccf, prewhitened_pair
from drtran.cast import build_cast_spec
from drtran.network import Link

M.load_pre("CC", f"{ES},{WTI}", check=False)
res = M.identify_link("CC", 1)
txt = "\n".join(getattr(r, "text", "") for r in res if hasattr(r, "text")) \
    if isinstance(res, list) else str(res)
citados = sorted({int(k) for k in re.findall(r"^\s+(-?\d+)\s+-?0\.\d+\s+-?0\.\d+\s+\*",
                                             txt, re.M)})
specs = M._require("CC")
cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 0)])
a, b = prewhitened_pair(cs, cs.links[0])
fig = plot_ccf(a, b, freq=12)
ax = fig.axes[0]
xmax = max(abs(v) for v in ax.get_xlim())
fuera = [k for k in citados if abs(k) > xmax]
print("retardos significativos que cita identify_link:", citados)
print(f"la figura llega a ±{int(xmax)}; citados y no dibujados: {fuera}")
print("identify_link acepta path:", "path" in inspect.signature(M.identify_link).parameters)
print("esperado: la figura cubre los retardos del texto (±24) y se puede guardar")
sys.exit(1 if (fuera or "path" not in inspect.signature(M.identify_link).parameters) else 0)
