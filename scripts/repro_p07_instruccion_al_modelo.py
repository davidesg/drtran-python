#!/usr/bin/env python3
"""Rodaje P07: la salida de `estimate` empieza con una orden a un modelo de
lenguaje concreto, «[Claude: muestra el bloque ``` siguiente TAL CUAL ...]».

Es parte del valor devuelto (mcp_server.py, `estimate`, variable `head`), no
una instruccion del servidor. Los asistentes guardan la salida en ficheros y
la orden acaba en la entrega: G11, linea 1 de los tres
`*_estimacion_mtram.txt`. Y va dirigida a «Claude» cuando el curso deja usar
cualquier asistente (Codex, CodeWhale...).

Sale con 1 mientras el texto de `estimate` lleve una linea «[Claude: ...]».
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import drtran.mcp_server as M

M.load_pre("IM", f"{ES},{WTI}", check=False)
M.set_network("IM", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
txt = M.estimate("IM")
m = re.search(r"^\[Claude:[^\]]*\]", txt, re.M)
print("primera linea de estimate:", txt.splitlines()[0])
print("esperado: la salida empieza por el modelo; la orden de presentacion, "
      "si hace falta, en las instrucciones del servidor y sin nombrar un modelo")
sys.exit(1 if m else 0)
