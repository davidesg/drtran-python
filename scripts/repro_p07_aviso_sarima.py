#!/usr/bin/env python3
"""Rodaje P07: «EL OUTPUT LLEVA SARIMA MULTIPLICATIVO Y EL INPUT NO» salta con
un IPC de armonicos deterministas y un AR estacional ESTACIONARIO.

`school.seasonality_mismatch` clasifica como «sarima» cualquier modelo con
D > 0, `ar_s` o `ma_s`. Un (1 - 0.3 B^12) con D = 0 no deja ninguna raiz
unitaria estacional: el filtro del input no tiene nada que quitar, y el
aviso («la serie filtrada sigue siendo NO ESTACIONARIA en esa frecuencia»,
«es el caso peor») es falso. Lo vieron G4 (Paises Bajos, Phi = 0.64) y G5
(Portugal, que demostro con `ident_pre` que la CCF no cambiaba).

Caso: ES_CPI_m10.1.pre con un AR(1) anual de 0.3 anadido (D = 0, sus once
armonicos intactos), frente al WTI.

Sale con 1 mientras `identify_link` de el aviso para ese modelo.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, tempfile, warnings
warnings.filterwarnings("ignore")
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
os.chdir(RAIZ)
import drtran.mcp_server as M

txt = open(ES).read()
nuevo = txt.replace("** Number and orders of annual AR operators:\n0\n",
                    "** Number and orders of annual AR operators:\n1 1\n**\n0.3 1\n")
assert nuevo != txt
tmp = tempfile.mkdtemp()
pre = os.path.join(tmp, "ES_CPI_sar.inp")
open(pre, "w").write(nuevo)

M.load_pre("SA", f"{pre},{WTI}", check=False)
m = M._require("SA")[0].model
print(f"modelo del output: D = {m.D}, ar_s = {m.ar_s}")
res = M.identify_link("SA", 1)
out = "\n".join(getattr(r, "text", "") for r in res) if isinstance(res, list) else str(res)
aviso = re.search(r"⚠ EL OUTPUT LLEVA SARIMA MULTIPLICATIVO.*", out)
print("aviso:", aviso.group(0) if aviso else "(no sale)")
print("esperado: sin aviso; con D = 0 y un AR estacional estacionario no queda "
      "ninguna no estacionariedad estacional que el filtro del input deje pasar")
sys.exit(1 if aviso else 0)
