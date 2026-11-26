"""P07 (rodaje 2026-11): el diagnostico de residuos del drtran en C fecha los
residuos con un calendario RELATIVO que empieza en 1/1 (diagnose.c, donde se
rellena el Tseries de cada residuo: begtime = begyear = 1). El primer residuo
de una serie diferenciada que empieza en 1/2002 es de 2/2002; el .out dice
«215 observations: from 1/1 to 11/18», y el desplome del WTI (obs. 81,
10/2008) sale «at 9/7». Se lee como un mes antes y en un ano que no existe.
El fichero _res.txt, en cambio, fecha bien (obs. 81 = 10/2008).

Uso: python3 scripts/repro_p07_fechas_residuos.py    Sale con 1 mientras exista.
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
subprocess.run([binario(), "ES_CPI_m10.1.pre", "WTI_ar1.1.pre", "-b", "0", "-r", "0",
                "-s", "1", "-m", "fe"], cwd=tmp, capture_output=True, text=True)
out = open(os.path.join(tmp, "fe.out")).read()
rango = re.search(r"(\d+) observations: from (\S+) to (\S+)", out)
mins = re.findall(r"Minimum:\s+\S+ at\s+(\S+) \(observation\s+(\d+)\)", out)
print(f"el .out: {rango.group(0)}   (esperado: from 2/2002 to 12/2019)")
for d, o in mins:
    print(f"  minimo at {d} (observation {o})")
bug = rango.group(2) != "2/2002"
print("BUG PRESENTE: fechas relativas y desplazadas" if bug else "ok")
shutil.rmtree(tmp)
sys.exit(1 if bug else 0)
