#!/usr/bin/env python3
"""Rodaje P07: `refine_link` dice «se cortan» de unos pesos que no se cortan.

`school.decay_pattern` sólo propone denominador si los |pesos| significativos
bajan ESTRICTAMENTE en todo el tramo. Si no, `refine_link` escribe «Los pesos
NO decrecen gradualmente: se cortan. Eso es un numerador finito, sin
denominador». Hay un tercer caso que la regla no contempla: una cola larga
y ruidosa, que ni decae limpia ni se corta.

Caso real (G3, Irlanda, `registro_IE.md` N1), pesos de `refine_link` s = 6:
  nu = .00926 .00720 .00430 .00655 .00360 -.00199 .00609
  t  = 4.39  3.46  2.08  3.09  1.72  -0.95  2.84
Cinco significativos y positivos hasta k = 6, y el texto dice «se cortan».
`overfit` le lleva la contraria: r+1 con t = 9.15.

Y un caso sintetico: una cola geometrica nu(k) = 0.01·0.7^k con e.t. 0.001
y un solo peso intermedio movido por el ruido (k = 3 un 50 % arriba, 1,7 e.t.) deja de
"decrecer" y tambien sale «se cortan».

Sale con 1 mientras alguno de los dos se lea como corte.
"""
from _p07 import ES, WTI, pre  # noqa: F401,E402 (datos y codigo de salida)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drtran.school import decay_pattern

g3_w = [.009256, .007196, .004304, .006553, .003604, -.001989, .006090]
g3_e = [.002107, .002078, .002074, .002124, .002091, .002096, .002145]
geo = [0.01 * 0.7 ** k for k in range(7)]
geo[3] *= 1.5
geo_e = [0.001] * 7

malos = 0
for nombre, w, e in (("G3 Irlanda", g3_w, g3_e), ("geometrica + ruido", geo, geo_e)):
    p = decay_pattern(w, e)
    pat = p.get("pattern", "decay" if p["suggests_denominator"] else "cut")
    lee = {"decay": "denominador", "cut": "«se cortan»",
           "unclear": "«no se lee: compara r = 0 y r = 1 con overfit»"}.get(pat, pat)
    print(f"{nombre:20s} significativos {p['significant']}  ->  {lee}")
    if pat == "cut" and len(p["significant"]) >= 3 \
            and p["significant"][-1] - p["significant"][0] >= 3:
        malos += 1
print("esperado: con 3 o mas pesos significativos en un tramo largo, no "
      "«se cortan»: o denominador, o «no se lee; compara r = 0 y r = 1 con overfit»")
sys.exit(1 if malos else 0)
