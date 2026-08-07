"""BUG-2: las series se emparejan POR INDICE, no por fecha, y nadie avisa.

Cada `.pre` declara su fecha inicial en la cabecera y `fue` la lee
(`spec.ts.start`). Pero:

  * `pre.load_pre` lee cada fichero por separado y nunca compara `start` entre
    series;
  * `mcp_server.load_pre` recorre la lista, comprueba que el fichero existe y
    llama a `check_scale`, y punto: ni compara `start`, ni `freq`, ni siquiera
    `nobs`. Imprime "(N obs, freq F)" -- la fecha no aparece por ningun lado;
  * `cast.py:252` alinea "al FINAL (la ultima observacion es la misma fecha)" y
    recorta a la mas corta. Ese comentario describe el caso para el que se
    escribio -- d/D distintos sobre la MISMA ventana --, pero nada comprueba la
    premisa cuando las ventanas son de calendarios distintos.

Resultado: se pueden cruzar dos series que no comparten ni un solo anho y el
ajuste sale adelante sin una sola queja.

Este script lo demuestra por invariancia: se estima el mismo par dos veces,
cambiando SOLO la fecha declarada de una de las series. Si las fechas contaran
para algo, la segunda pasada tendria que dar otra cosa (o negarse).

Ejecutar:  python3 scripts/repro_alineacion_por_indice.py
"""
import copy
import os

import numpy as np

import drtran
from drtran.cast import Link, build_cast_spec
from drtran.identify import identify
from drtran.pre import PreSpec

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cargar(nombre, desplaza_anhos=0):
    """El .pre, opcionalmente con su fecha inicial corrida N anhos."""
    s = drtran.load_pre(os.path.join(AQUI, nombre))
    if desplaza_anhos:
        m = copy.deepcopy(s.model)
        anho, per = m.series.start
        m.series.start = (anho + desplaza_anhos, per)
        return PreSpec(ts=m.series, model=m, path=s.path)
    return s


def identifica(desplaza):
    salida = cargar("ES_CPI_m10.1.pre")
    entrada = cargar("WTI_ar1.1.pre", desplaza)
    cs = build_cast_spec([salida, entrada], [Link(out=0, inp=1, b=0, r=0, s=0)])
    return salida, entrada, identify(cs, cs.links[0])


print(__doc__)

for desplaza in (0, 50):
    sal, ent, ident = identifica(desplaza)
    solapan = "si" if sal.ts.start[0] == ent.ts.start[0] else "NINGUNO"
    r = np.asarray(ident.ccf, float)
    k = np.asarray(ident.lags, int)
    n_usada = int(round((2.0 / ident.threshold) ** 2))
    print(f"--- entrada declarada en {ent.ts.start}  "
          f"(salida en {sal.ts.start}; anhos en comun: {solapan})")
    print(f"    banda              : {ident.threshold:.6f}"
          f"  => n = (2/banda)^2 = {n_usada}")
    print(f"    r(0..4)            : {np.round(r[(k >= 0) & (k <= 4)], 6)}")
    print(f"    suma |r|           : {np.abs(r).sum():.10f}")
    print(f"    (b, r, s) propuesto: {(ident.b, ident.r, ident.s)}")
    print(f"    Q exogeneidad      : {ident.Q_exogeneity:.6f} "
          f"(p={ident.p_exogeneity:.6f})")

print("""
Las dos pasadas son IDENTICAS hasta el ultimo decimal. En la segunda, las dos
series no comparten un solo anho -- y aun asi se estima el mismo modelo, porque
la fecha no interviene en ningun punto del calculo.

Como se detecta desde fuera (asi salio en un caso real): la banda que imprime la
identificacion es 2/sqrt(n), luego delata la n que se ha usado. Cargando un
precio de 1700-1896 (197 obs) contra una lluvia de 1766-2024 (259) la banda
salio 0.1429 = 2/sqrt(196), cuando el solapamiento real 1766-1896 son 131
observaciones. El emparejamiento era por posicion: precio de 1700 contra lluvia
de 1766, 66 anhos de desfase. La identificacion propuso b=18 con toda seriedad.

Es el mas peligroso de los tres porque NUNCA da error: entrega un resultado
plausible.

ARREGLO: en `load_pre` (o al construir el caso) intersecar por fechas -- o, como
minimo, rechazar la carga si `start`/`freq`/`nobs` no son compatibles. Y sacar la
fecha en el resumen que imprime `mcp_server.load_pre`, que hoy solo dice cuantas
observaciones hay.
""")
