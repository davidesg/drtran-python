"""P07 (rodaje 2026-11): un .pre con mu FIJADA a 0 entra en drtran y el motor
la respeta -- eso es correcto: lo fijado en el .pre es una decision del
analista -- pero no avisa cuando la media de los residuos de esa serie es
significativa. Aqui el IPC de Espana con mu = 0 fija: la media residual sale
~0,067 con error tipico ~0,014 (t ~ 4,8) y el .out no dice nada.

Veredicto: POLITICA (no es un error de calculo). Propuesta: un aviso en el
diagnostico de residuos cuando la serie tiene mu fijada y |media/e.t.| > 2.
Sale con 1 mientras no haya aviso.

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
shutil.rmtree(tmp)
sys.exit(0 if aviso else 1)
