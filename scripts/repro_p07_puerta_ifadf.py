"""P07: la puerta diagonal (N0) no cuadra cuando la salida lleva factores ifadf.

Rodaje P07 (noviembre 2026): con un IPC con raíces unitarias estacionales
sueltas (ifadf) y el WTI con d = 1, `load_pre` da «NO coinciden» y obliga a
cambiar el univariante. Cifras de los alumnos: +1,87 (Alemania, ifadf en f1 y
f2), hasta +11,4.

Causa: el arreglo de 2026-08-12 (963c335) hizo que `_muestra_comun` contara el
grado del polinomio de diferenciación, pero `_en_muestra_comun` sigue
calculando las pérdidas como `d + D·freq` (mcp_server.py:604). Con
∇·(1−√3B+B²)·(1−B+B²) el grado es 5 y la función cree que es 1: recorta la
salida a n_comun + 1 = 212 observaciones cuando su muestra útil ya era la
común (216 − 5 = 211). fue puntúa entonces 207 observaciones de la salida y el
conjunto 211.

Segundo defecto, visible en la misma salida: cuando hay recorte, `m.fit()`
reestima una COPIA, así que `_certificado` compara los coeficientes del fichero
consigo mismos y dice siempre «un ÓPTIMO ... reestimarlos no movió nada», a la
vez que el hueco de optimalidad dice «Aquí no lo eran».

Uso:  python3 scripts/repro_p07_puerta_ifadf.py      (≈ 5 s)
Sale con 1 mientras el defecto exista.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import copy
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))

import drtran                                   # noqa: E402
import drtran.mcp_server as M                   # noqa: E402
from drtran.cast import differencing_poly       # noqa: E402

OUT = os.path.join(RAIZ, "tests", "data", "p07", "ipc_ifadf_f1f2.pre")
INP = WTI

out, inp = drtran.load_pre(OUT), drtran.load_pre(INP)
grados = [len(differencing_poly(s.model)) - 1 for s in (out, inp)]
n = M._muestra_comun([out, inp])
m_out = M._en_muestra_comun(out, n)
util = m_out.series.nobs - grados[0]
print(f"grado de diferenciación  salida={grados[0]}  entrada={grados[1]}")
print(f"muestra común = {n};  la salida queda con {m_out.series.nobs} obs "
      f"-> {util} útiles   (esperado: {n})")

texto = M.load_pre("repro", f"{OUT},{INP}")
dif = [ln for ln in texto.splitlines() if "Diferencia con la suma" in ln][0]
print("puerta:", dif.strip())

# La identidad con la muestra bien contada:
m = copy.deepcopy(out.model); m.fit(); a = float(m.loglik)
m = M._en_muestra_comun(inp, n); m.fit(); b = float(m.loglik)
f = drtran.fit(drtran.build_cast_spec([out, inp], links=[]), embed=True)
print(f"con la salida entera: suma fue = {a + b:.6f}  diagonal = "
      f"{f.loglik:.6f}  dif = {f.loglik - (a + b):+.2e}")

contradice = ("Aquí no lo eran" in texto and "reestimarlos no movió nada" in texto)
print("certificado contradictorio (hueco > 0 y 'no movió nada'):", contradice)

fallo = util != n or "NO coinciden" in texto or contradice
print("BUG PRESENTE" if fallo else "ok")
sys.exit(1 if fallo else 0)
