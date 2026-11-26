"""P07 (rodaje 2026-11): la cabecera del .out del drtran en C rotula
«Output (Y)» al PRIMER fichero de la linea de ordenes y «Input (X1)» al
segundo, aunque la red (-n .dag) diga lo contrario. Con la red
ES_CPI <- WTI y los ficheros en el orden WTI, ES_CPI, la cabecera dice que la
salida es el WTI, y dos lineas mas abajo la red dice que es el IPC.

Un grupo leyo la cabecera, dio su caso de Espana por «invertido» y lo saco de
la comparacion, cuando el modelo estaba bien (g = 0,027).

Uso: python3 scripts/repro_p07_cabecera_out.py     Sale con 1 mientras exista.
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
with open(os.path.join(tmp, "red.dag"), "w") as fh:
    fh.write("ES_CPI <- WTI   0 0 1\n")
subprocess.run([binario(), "WTI_ar1.1.pre", "ES_CPI_m10.1.pre", "-n", "red.dag",
                "-m", "cab"], cwd=tmp, capture_output=True, text=True)
out = open(os.path.join(tmp, "cab.out")).read()
cab_y = re.search(r"Output \(Y\)\s*:\s*(\S+)", out).group(1)
red = re.search(r"^\s*(\S+)\s+<-\s+(\S+)", out, re.M)
print(f"cabecera: Output (Y) = {cab_y}")
print(f"red     : {red.group(1)} <- {red.group(2)}")
bug = not cab_y.startswith(red.group(1))
print("BUG PRESENTE: la cabecera contradice la red" if bug else "ok")
shutil.rmtree(tmp)
sys.exit(1 if bug else 0)
