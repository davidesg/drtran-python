"""P07 (rodaje 2026-11), BUG-75: fijar en el .cns una covarianza de
innovaciones q[i,j] con un valor que deja la matriz Q no definida positiva no
se rechazaba al leer el .cns. El C salia «NO CONVERGENCE AFTER 0 ITERATIONS /
unknown / ifault = 1», escribia logL = 0,000000, errores tipicos 0 y p = 1,
seguia con la caracterizacion de la transferencia como si hubiera un modelo, y
moria con SEGMENTATION FAULT en File_PlotSer (diagnose). El puerto devolvia
loglik = -inf sin mensaje. El taller guardo esas corridas como normales.

Q esta en unidades relativas: Q[1,1] = 1, Q[2,2] = var2/var1. Con el WTI como
serie 1 y el IPC como serie 2, Q[2,2] ~ 0,001, de modo que cualquier
|q[2,1]| > ~0,03 es imposible. Un grupo escribio q[2,1] = 1 (tres corridas).

Arreglado (2026-11-27): el C rechaza la Q de partida antes de estimar, con el
rango admisible y la advertencia de que q va relativa a la serie 1, rc = 12 y
sin tabla; el puerto lanza StartNotAdmissible con el mismo texto. Un q dentro
del rango estima igual en los dos.

Uso: python3 scripts/repro_p07_q_fijada_no_definida.py
Sale con 1 mientras exista el defecto; con 2 si no puede correr.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

from _p07 import ES, WTI, binario, pre

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

tmp = tempfile.mkdtemp()
for f in ("WTI_ar1.1.pre", "ES_CPI_m10.1.pre"):
    shutil.copy(pre(f), os.path.join(tmp, f))
bug = False
ll_c = {}
print("C:")
for q in ("1", "0.05", "0.01"):
    with open(os.path.join(tmp, "q.cns"), "w") as fh:
        fh.write(f"q[2,1] = {q}\n")
    if os.path.exists(os.path.join(tmp, "q.out")):
        os.remove(os.path.join(tmp, "q.out"))
    r = subprocess.run([binario(), "WTI_ar1.1.pre", "ES_CPI_m10.1.pre", "-b", "0",
                        "-r", "0", "-s", "1", "-c", "q.cns", "-m", "q"],
                       cwd=tmp, capture_output=True, text=True)
    out = open(os.path.join(tmp, "q.out")).read() if os.path.exists(os.path.join(tmp, "q.out")) else ""
    ll = re.search(r"Log-likelihood = (\S+)", out)
    rechazo = "not positive definite" in r.stderr and "RELATIVE TO SERIES 1" in r.stderr
    print(f"  q[2,1] = {q:5s}  rc = {r.returncode:4d}  logL = {ll.group(1) if ll else '-':>14s}"
          f"  {'rechazada con el rango admisible' if rechazo else ''}")
    if q == "0.01":
        if r.returncode != 0 or not ll:
            bug = True
        else:
            ll_c[q] = float(ll.group(1))
    elif r.returncode != 12 or ll or not rechazo:
        bug = True
shutil.rmtree(tmp)

print("puerto:")
import drtran                       # noqa: E402
import drtran.mcp_server as M       # noqa: E402

M.load_pre("QF", f"{WTI},{ES}", check=False)
M.set_network("QF", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
tmp = tempfile.mkdtemp()
for q in ("1", "0.05", "0.01"):
    cns = os.path.join(tmp, "q.cns")
    with open(cns, "w") as fh:
        fh.write(f"q[2,1] = {q}\n")
    try:
        M.estimate("QF", cns_path=cns)
        f = M._FITS["QF"]
        print(f"  q[2,1] = {q:5s}  logL = {f.loglik:.6f}")
        if q != "0.01":
            bug = True
        elif abs(f.loglik - ll_c.get(q, f.loglik + 1)) > 1e-3:
            print(f"    ... el C da {ll_c.get(q)}: no homologado")
            bug = True
    except drtran.StartNotAdmissible as e:
        ok = "RELATIVE TO SERIES 1" in str(e) and "<- outside" in str(e)
        print(f"  q[2,1] = {q:5s}  StartNotAdmissible{' (con el rango admisible)' if ok else ''}")
        if q == "0.01" or not ok:
            bug = True
shutil.rmtree(tmp)
print("BUG PRESENTE: un q[i,j] imposible no se rechaza con su mensaje" if bug else "ok")
sys.exit(1 if bug else 0)
