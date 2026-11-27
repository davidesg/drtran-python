"""P07 (rodaje 2026-11): un .pre con mu FIJADA a 0 entra en drtran y el motor
la respeta -- eso es correcto: lo fijado en el .pre es una decision del
analista -- pero no avisa cuando la media de los residuos de esa serie es
significativa. Aqui el IPC de Espana con mu = 0 fija: la media residual sale
~0,067 con error tipico ~0,014 (t ~ 4,8) y el .out no dice nada.

Veredicto: POLITICA (no es un error de calculo). Arreglado (2026-11-27): con
mu fijada y d + D > 0, si |media/e.t.| > 2, avisan el .out del C y el
`diagnose` de mtram, con el mismo texto del motor.
Sale con 1 mientras falte el aviso en alguno de los dos.

Uso: python3 scripts/repro_p07_media_fijada_sin_aviso.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from _p07 import binario, pre

tmp = tempfile.mkdtemp()
shutil.copy(pre("WTI_ar1.1.pre"), os.path.join(tmp, "WTI_ar1.1.pre"))
txt = open(pre("ES_CPI_m10.1.pre")).read()
txt = re.sub(r"(\*\* Mean parameter \(mu\):\n)\S+ 1\n", r"\g<1>0.000000 0\n", txt)
open(os.path.join(tmp, "ES_mu0.pre"), "w").write(txt)
subprocess.run([binario(), "ES_mu0.pre", "WTI_ar1.1.pre", "-b", "0", "-r", "0", "-s", "1",
                "-m", "mu0"], cwd=tmp, capture_output=True, text=True)
out = open(os.path.join(tmp, "mu0.out")).read()
fij = re.search(r"mu\[1\].*fixed.*", out)
m = float(re.search(r"Mean:\s+(\S+)", out).group(1))
se = float(re.search(r"Standard error of mean:\s+(\S+)", out).group(1))
print(fij.group(0) if fij else "mu[1] no aparece fijada")
print(f"residuos del IPC: media = {m:.4f}, e.t. = {se:.4f}, t = {m / se:.2f}")
aviso = re.search(r"(?i)warning.*mean|mean.*significant|media.*significativ", out)
print("aviso en el .out:", aviso.group(0) if aviso else "NINGUNO")

import warnings                                       # noqa: E402
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import drtran.mcp_server as M                         # noqa: E402

M.load_pre("MU0", f"{os.path.join(tmp, 'ES_mu0.pre')},{os.path.join(tmp, 'WTI_ar1.1.pre')}",
           check=False)
M.set_network("MU0", '[{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]')
M.estimate("MU0")
dg = M.diagnose("MU0", 0)
aviso_py = re.search(r"WARNING: series 1 \(ES_CPI\): the residual mean is significant.*", dg)
print("aviso en el diagnose de mtram:", aviso_py.group(0) if aviso_py else "NINGUNO")
shutil.rmtree(tmp)
sys.exit(0 if (aviso and aviso_py) else 1)
