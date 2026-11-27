"""P07 (rodaje 2026-11): la cabecera del .out del drtran en C rotula
«Output (Y)» al PRIMER fichero de la linea de ordenes y «Input (X1)» al
segundo, aunque la red (-n .dag) diga lo contrario. Con la red
ES_CPI <- WTI y los ficheros en el orden WTI, ES_CPI, la cabecera dice que la
salida es el WTI, y dos lineas mas abajo la red dice que es el IPC.

Un grupo leyo la cabecera, dio su caso de Espana por «invertido» y lo saco de
la comparacion, cuando el modelo estaba bien (g = 0,027).

Arreglado (2026-11-27, atsw-gui): con -n la cabecera dice «File k : ruta» y la
red («Network : ES_CPI <- WTI»); sin -n no cambia.

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
cab = out[:out.index("Frequency")]
cab_y = re.search(r"Output \(Y\)\s*:\s*(\S+)", cab)
cab_red = re.findall(r"^Network\s*:\s*(\S+)\s+<-\s+(\S+)", cab, re.M)
red = re.search(r"^\s*(\S+)\s+<-\s+(\S+)\s+b=", out, re.M)
print(f"cabecera: Output (Y) = {cab_y.group(1) if cab_y else '-'}   "
      f"Network = {' ; '.join(a + ' <- ' + b for a, b in cab_red) or '-'}")
print(f"red     : {red.group(1)} <- {red.group(2)}")
# Con -n la cabecera no debe rotular un fichero como salida por su orden, y si
# lista la red, tiene que ser la misma que la del informe.
bug = (cab_y is not None and not cab_y.group(1).startswith(red.group(1))) \
    or (red.group(1), red.group(2)) not in cab_red
print("BUG PRESENTE: la cabecera contradice la red" if bug else "ok")
shutil.rmtree(tmp)
sys.exit(1 if bug else 0)
