"""BUG-89: el C y el puerto siembran log(var_i/var_1) de forma distinta.

El C usa la varianza muestral de cada serie estacionaria; el puerto, la de las
innovaciones del modelo univariante de cada .pre. La Q[2,2] de partida se lee
del C en el mensaje de rechazo del BUG-75 (q[2,1] = 1 fijado) y del puerto en
`x0_full`.

Uso: python3 scripts/repro_semilla_razon_varianzas.py
Sale con 1 mientras las dos semillas difieran; con 2 si no puede correr.
"""
import math
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
with open(os.path.join(tmp, "q.cns"), "w") as fh:
    fh.write("q[2,1] = 1\n")
r = subprocess.run([binario(), "WTI_ar1.1.pre", "ES_CPI_m10.1.pre", "-b", "0", "-r", "0",
                    "-s", "1", "-c", "q.cns", "-m", "q"], cwd=tmp, capture_output=True, text=True)
shutil.rmtree(tmp)
m = re.search(r"Q\[2,2\] = var2/var1 = (\S+)", r.stderr)
if not m:
    sys.stderr.write("el C no informa la Q de partida (¿binario anterior al BUG-75?)\n")
    sys.exit(2)
q_c = float(m.group(1))

import drtran                       # noqa: E402
from drtran.slots import build_slots  # noqa: E402

specs = [drtran.load_pre(WTI), drtran.load_pre(ES)]
cs = drtran.build_cast_spec(specs, [drtran.Link(out=0, inp=1, b=0, r=0, s=1)])
table = build_slots(cs)
x0 = drtran.x0_full(cs, table)
q_p = math.exp(x0[table.index("log(var2/var1)")])
print(f"Q[2,2] de partida (WTI, ES_CPI): C {q_c:g}   puerto {q_p:g}")
diff = abs(q_c - q_p) > 1e-6 * max(abs(q_c), 1e-12)
print("BUG PRESENTE: las semillas difieren" if diff else "ok")
sys.exit(1 if diff else 0)
