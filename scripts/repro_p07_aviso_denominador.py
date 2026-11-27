#!/usr/bin/env python3
"""Rodaje P07: con denominador, `estimate` avisa «la suma de los coeficientes
no es la ganancia: revisa el signo», y rotula la ganancia «omega(1)».

`_splice_transfer` (mcp_server.py) compara la suma de los coeficientes de
omega(B) con `impulse_response(...).gain`. Sin denominador son lo mismo; con
r >= 1 la ganancia es omega(1)/delta(1) y la diferencia es la que tiene que
haber. El aviso salta en TODO modelo con denominador y manda a buscar un
error de signo que no existe. Y `_what_the_transfer_bought` escribe
«GANANCIA = omega(1)» tambien cuando es omega(1)/delta(1). Lo vieron G3 y
G11.

Caso de clase con (b, s, r) = (0, 0, 1): omega0 = 0.0168, delta = 0.456,
ganancia = 0.0168/(1 - 0.456) = 0.0309.

Sale con 1 mientras salga el aviso o el rotulo «omega(1)» con r >= 1, o
falte «omega(1)/delta(1)» con g = w0/(1 - d1). Arreglado 2026-11-27.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import drtran.mcp_server as M

M.load_pre("DN", f"{ES},{WTI}", check=False)
M.set_network("DN", '[{"out": 0, "inp": 1, "b": 0, "r": 1, "s": 0}]')
txt = M.estimate("DN")
w0 = float(re.search(r"omega1\[0\]\s+(-?[\d.]+)", txt).group(1))
d1 = float(re.search(r"delta1\[1\]\s+(-?[\d.]+)", txt).group(1))
aviso = re.search(r"⚠ la suma de los coeficientes.*", txt)
rotulo = re.search(r"GANANCIA = omega\(1\) = ([+-][\d.]+)", txt)
print(f"omega0 = {w0:.6f}, delta1 = {d1:.6f}, omega0/(1-delta1) = {w0/(1-d1):.6f}")
print(f"aviso  : {aviso.group(0) if aviso else '(no sale)'}")
print(f"rotulo : {rotulo.group(0) if rotulo else '(no sale)'}")
bien = re.search(r"GANANCIA = omega\(1\)/delta\(1\) = ([+-][\d.]+)", txt)
print(f"rotulo correcto: {bien.group(0) if bien else '(no sale)'}")
print("esperado: sin aviso (la suma de omega no es la ganancia cuando r >= 1, "
      "y no es un error), y la ganancia rotulada omega(1)/delta(1)")
# con el convenio de Box y Jenkins, omega(B) = w0 y delta(B) = 1 - d1 B: g = w0/(1 - d1)
ok_g = bool(bien) and abs(float(bien.group(1)) - w0 / (1 - d1)) < 5e-6
sys.exit(1 if (aviso or rotulo or not ok_g) else 0)
