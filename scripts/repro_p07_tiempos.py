#!/usr/bin/env python3
"""Rodaje P07: los alumnos dieron `load_pre(check=True)` de mas de 10 minutos
(G4, Paises Bajos), y `refine_link(smax=6)` y `evaluate` con MA estacional
por encima de los 60 s del cliente MCP (G9, Italia y Francia, refactor 1).

Este script mide las cuatro llamadas en el caso de clase, en el mismo caso
con un MA anual anadido y con refactor 1, y en el modelo de Paises Bajos de
G4 si se le pasa (`--nl NL_IPC_m03.pre WTI_m01.pre`). Sale con 1 si alguna
pasa de 60 s.

Medido 2026-11-26 en Linux, drtran 0.2.7, motores compilados: todo entre 0,2
y 4,7 s, tambien el caso de G4 (load_pre 1,2 s; refine 3,4 s; estimate 3,2 s;
evaluate 1,4 s), y tambien con DRVARMA_NO_ENGINE=1. NO REPRODUCIDO aqui.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, re, sys, tempfile, time, warnings
warnings.filterwarnings("ignore")
RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ, "src"))
os.chdir(RAIZ)
import drtran.mcp_server as M

tmp = tempfile.mkdtemp()
es = open(ES).read()
sma = es.replace("** Number and orders of anual MA operators:\n0\n",
                 "** Number and orders of anual MA operators:\n1 1\n**\n0.5 1\n")
r1 = lambda t: re.sub(r"(reescaling factor: \n\s*\S+\s+)100\.00", r"\g<1>1.00", t)
open(f"{tmp}/sma.inp", "w").write(sma)
open(f"{tmp}/sma_r1.inp", "w").write(r1(sma))
open(f"{tmp}/wti_r1.inp", "w").write(r1(open(WTI).read()))
casos = [("clase", ES, WTI),
         ("MA anual", f"{tmp}/sma.inp", WTI),
         ("MA anual, refactor 1", f"{tmp}/sma_r1.inp", f"{tmp}/wti_r1.inp")]
if "--nl" in sys.argv:
    i = sys.argv.index("--nl")
    casos.append(("G4 Paises Bajos", sys.argv[i + 1], sys.argv[i + 2]))

peor = 0.0
for k, (nom, out, inp) in enumerate(casos):
    n = f"T{k}"
    t = time.time(); M.load_pre(n, f"{out},{inp}", check=True); a = time.time() - t
    t = time.time(); M.refine_link(n, 1, smax=6); b = time.time() - t
    M.set_network(n, '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
    t = time.time(); M.estimate(n); c = time.time() - t
    t = time.time(); M.evaluate(n, window=180, horizon=6); d = time.time() - t
    peor = max(peor, a, b, c, d)
    print(f"{nom:22s} load_pre {a:5.1f} s  refine_link {b:5.1f} s  "
          f"estimate {c:5.1f} s  evaluate {d:5.1f} s")
print(f"la mas lenta: {peor:.1f} s (limite del cliente MCP: 60 s)")
sys.exit(1 if peor > 60 else 0)
