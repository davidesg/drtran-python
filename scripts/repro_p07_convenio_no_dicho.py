#!/usr/bin/env python3
"""Rodaje P07: mtram no dice en ningun sitio visible el convenio de signo de omega.

drtran usa el de Box y Jenkins, omega(B) = w0 - w1 B - w2 B^2 - ...: la tabla
del motor da omega1[1] = -0.0107 y la ecuacion de `estimate` escribe el mismo
polinomio desarrollado, +0.0107 B. Es coherente, pero el convenio solo esta en
comentarios del codigo (mcp_server._omega_poly, cast.py), no en las
instrucciones del servidor ni en la salida. Un agente o un alumno que ve los
dos signos piensa que hay un error (cinco grupos lo leyeron mal; un asistente
lo llamo «inconsistencia de mtram»).

Sale con 1 mientras ni las instrucciones ni la salida de `estimate` digan el
convenio junto a la tabla.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import drtran.mcp_server as M

M.load_pre("CV", f"{ES},{WTI}", check=False)
M.set_network("CV", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
txt = M.estimate("CV")
tab = float(re.search(r"omega1\[1\]\s+(-?[\d.]+)", txt).group(1))
pat = re.compile(r"Box.?Jenkins.{0,120}omega|ω0\s*[−-]\s*ω1|w0\s*-\s*w1|omega0\s*[−-]\s*omega1|convenio de signo|sign convention", re.I | re.S)
en_salida = bool(pat.search(txt))
en_instr = bool(pat.search(M._INSTRUCTIONS))
print(f"tabla: omega1[1] = {tab:+.6f}; la ecuacion escribe {-tab:+.4f}·B")
print(f"convenio dicho en la salida de estimate: {en_salida}")
print(f"convenio dicho en las instrucciones del servidor: {en_instr}")
sys.exit(0 if (en_salida and en_instr) else 1)
