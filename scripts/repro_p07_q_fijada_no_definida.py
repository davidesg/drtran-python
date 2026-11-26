"""P07 (rodaje 2026-11): fijar en el .cns una covarianza de innovaciones q[i,j]
con un valor que deja la matriz Q no definida positiva no se rechaza al leer
el .cns. El C sale «NO CONVERGENCE AFTER 0 ITERATIONS / unknown /
ifault = 1», escribe logL = 0,000000, errores tipicos 0 y p = 1, sigue con la
caracterizacion de la transferencia como si hubiera un modelo, y muere con
SEGMENTATION FAULT en File_PlotSer (diagnose). El taller guardo esas corridas
como normales.

Q esta en unidades relativas: Q[1,1] = 1, Q[2,2] = exp(log(var2/var1)). Con el
WTI como serie 1 y el IPC como serie 2, Q[2,2] ~ 0,001, de modo que cualquier
|q[2,1]| > ~0,03 es imposible. Un grupo escribio q[2,1] = 1 (tres corridas).
Con los ficheros en el otro orden el mismo .cns estima sin protestar: el
significado de un numero en q[i,j] depende del orden de los ficheros.

Uso: python3 scripts/repro_p07_q_fijada_no_definida.py   Sale con 1 mientras exista.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from _p07 import binario, pre

tmp = tempfile.mkdtemp()
for f in ("WTI_ar1.1.pre", "ES_CPI_m10.1.pre"):
    shutil.copy(pre(f), os.path.join(tmp, f))
bug = False
for q in ("1", "0.05", "0.01"):
    with open(os.path.join(tmp, "q.cns"), "w") as fh:
        fh.write(f"q[2,1] = {q}\n")
    r = subprocess.run([binario(), "WTI_ar1.1.pre", "ES_CPI_m10.1.pre", "-b", "0",
                        "-r", "0", "-s", "1", "-c", "q.cns", "-m", "q"],
                       cwd=tmp, capture_output=True, text=True)
    out = open(os.path.join(tmp, "q.out")).read()
    ll = re.search(r"Log-likelihood = (\S+)", out)
    conv = re.search(r"\*\*\*\* (.*ITERATIONS.*)", out)
    lv = re.search(r"log\(var2/var1\)\s+(\S+)", out)
    lim = (float(lv.group(1)) and __import__("math").exp(float(lv.group(1)) / 2)) if lv else float("nan")
    print(f"q[2,1] = {q:5s}  rc = {r.returncode:4d}  logL = {ll.group(1) if ll else '-':>14s}"
          f"  {conv.group(1).strip() if conv else ''}   (|q| maximo admisible ~ {lim:.3f})")
    if r.returncode != 0 or (ll and float(ll.group(1)) == 0.0):
        bug = True
print("BUG PRESENTE: un q[i,j] imposible no se rechaza; logL = 0 y/o segfault"
      if bug else "ok")
shutil.rmtree(tmp)
sys.exit(1 if bug else 0)
