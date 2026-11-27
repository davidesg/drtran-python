"""P07 (rodaje 2026-11): los pesos FIJADOS a cero por el .cns cuentan como
grados de libertad en la Q de adecuacion de la transferencia y en el LR contra
el modelo sin enlace.

Caso: IPC de Espana <- WTI con s = 3 y omega1[2] = omega1[3] = 0 en el .cns.
Libres en el enlace: 2 (omega1[0], omega1[1]). drtran descuenta 4 en la Q de
adecuacion (diagnose.transfer_adequacy: npar_tr = (s+1)+r) y mtram usa 4 g.l.
en el LR contra el diagonal (mcp_server.py: npar_tr = sum(1+r+s)).

Uso:  python3 scripts/repro_p07_gl_pesos_fijados.py      (desde la raiz del repo)
Sale con 1 mientras el bug exista.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os
import sys
import tempfile

import drtran as D
from drtran.cast import Link, build_cast_spec
from drtran.diagnose import transfer_adequacy
from drtran.slots import build_slots, read_cns

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Y = D.load_pre(ES)
X = D.load_pre(WTI)

cs = build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=3)])
table = build_slots(cs)
with tempfile.NamedTemporaryFile("w", suffix=".cns", delete=False) as fh:
    fh.write("omega1[2] = 0\nomega1[3] = 0\n")
    cns = fh.name
read_cns(cns, table)
os.unlink(cns)

libres_enlace = sum(1 for sl in table.slots
                    if sl.name.startswith("omega1[") and sl.kind == 0)
f = D.fit(cs, x0=D.x0_full(cs, table), embed=True, slots=table)
ad = transfer_adequacy(f)
k = ad.nlags + 1
descontados_q = k - ad.df_transfer
# el LR de mtram, tal como lo escribe `estimate` (con el certificado diagonal)
import re                                                     # noqa: E402

import drtran.mcp_server as M                                 # noqa: E402

with tempfile.NamedTemporaryFile("w", suffix=".cns", delete=False) as fh:
    fh.write("omega1[2] = 0\nomega1[3] = 0\n")
    cns2 = fh.name
M.load_pre("gl", f"{ES},{WTI}", check=True)
M.set_network("gl", '[{"out":0,"inp":1,"b":0,"r":0,"s":3}]')
txt = M.estimate("gl", cns_path=cns2)
os.unlink(cns2)
m_lr = re.search(r"LR = 2·Δ = \S+\s+con (\d+) parámetro", txt)
descontados_lr = int(m_lr.group(1)) if m_lr else -1

print(f"logL = {f.loglik:.4f}   termcode = {f.termcode}")
print(f"slots del enlace: {[ (sl.name, sl.kind) for sl in table.slots if sl.name.startswith('omega1[')]}")
print(f"parametros LIBRES del enlace: {libres_enlace}")
print(f"Q de adecuacion: {k} retardos, descuenta {descontados_q}  "
      f"(esperado {libres_enlace}) -> Q({ad.df_transfer}) = {ad.Q_transfer:.3f}, "
      f"p = {ad.p_transfer:.4f}")
print(f"LR contra el diagonal (mtram): {descontados_lr} g.l. (esperado {libres_enlace})")

bug = descontados_q != libres_enlace or descontados_lr != libres_enlace
print("BUG PRESENTE" if bug else "ok")
sys.exit(1 if bug else 0)
