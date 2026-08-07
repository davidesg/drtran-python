"""BUG-1 -- CORREGIDO 2026-08-07. Este script era la reproduccion; ahora es la
VERIFICACION. Antes toda fila con phi1 >= 0.999 devolvia ifault=1; hoy todas las
estacionarias devuelven 0. Lo que sigue describe el defecto que habia.

BUG-1: la guarda del circulo unidad se aplica a TODO orden AR, no solo a AR(1).

`cast.py:286` y `embed.py:224` llevan, identica:

    if ps[i] >= 1 and abs(phis[i][0]) >= 0.999:
        return None, None, None, None, None, 1

El comentario declara la intencion -- "an AR(1) pinned to the unit circle" --, y
para p=1 es correcta: ahi phi[0] ES la raiz. Pero la condicion es `ps[i] >= 1`, o
sea todo orden. En un AR(2) phi[0] no es una raiz: la region estacionaria es el
triangulo |phi2|<1, phi2+phi1<1, phi2-phi1<1, donde phi1 llega hasta 2. Un
AR(2) de raices COMPLEJAS con phi1>1 es perfectamente estacionario y drtran lo
rechaza con ifault=1.

Ejecutar:  python3 scripts/repro_ar2_phi1_bound.py
"""
import copy
import os

import numpy as np

import drtran
from drtran.cast import build_cast_spec, cast_diagonal, x0_from_pre
from drtran.embed import cast_embedded
from drtran.pre import PreSpec

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def con_ar2(path, phi1, phi2):
    """El mismo .pre con su AR sustituido por un AR(2) = (phi1, phi2)."""
    s = drtran.load_pre(os.path.join(AQUI, path))
    m = copy.deepcopy(s.model)
    m.ar = [[float(phi1), float(phi2)]]
    m.ar_free = [[1, 1]]
    return PreSpec(ts=m.series, model=m, path=s.path)


def raices(phi1, phi2):
    """Modulo de las raices EN B de 1 - phi1*B - phi2*B^2.

    Estacionario <=> todas fuera del circulo unidad, |B| > 1.
    """
    return np.abs(np.roots([-phi2, -phi1, 1.0]))


def evalua(phi1, phi2):
    """(ifault del cast diagonal, ifault del embebido) sembrando ese AR(2)."""
    specs = [con_ar2("ES_CPI_m10.1.pre", phi1, phi2),
             con_ar2("WTI_ar1.1.pre", phi1, phi2)]
    cs = build_cast_spec(specs, [])
    x = x0_from_pre(cs)
    return cast_diagonal(x, cs)[-1], cast_embedded(x, cs)[-1]


print(__doc__)
print(f"{'phi1':>7} {'phi2':>7} {'|raices|':>18} {'estacionario':>13} "
      f"{'ifault diag':>12} {'ifault emb':>11}")
print("-" * 76)

for phi1, phi2 in [(0.30, 0.00),      # AR(1) benigno
                   (0.95, -0.52),     # AR(2) complejo, phi1 < 1
                   (0.99, -0.52),
                   (1.0020, -0.5425),  # Estrasburgo B + intervencion 1847
                   (1.0354, -0.5184),  # Londres, precio del trigo era B
                   (1.60, -0.80)]:     # complejo, muy persistente, estacionario
    mod = raices(phi1, phi2)
    est = np.all(mod > 1.0)
    d, e = evalua(phi1, phi2)
    print(f"{phi1:7.4f} {phi2:7.4f} {str(np.round(mod, 3)):>18} "
          f"{'SI' if est else 'no':>13} {d:12d} {e:11d}")

print("""
Todas las filas con phi1 >= 0.999 son ESTACIONARIAS (modulo de las raices > 1).
ANTES devolvian ifault=1 y la verosimilitud ni se llegaba a evaluar; ahora
devuelven 0.

Consecuencia mas grave que el propio ifault: cuando el punto de partida esta por
DEBAJO de 0.999, el optimizador no falla -- avanza hasta la barrera y se queda
clavado, porque `estimate.py` devuelve 1.0 (no mejora) en cada punto rechazado y
la busqueda lineal nunca la cruza. Con el precio de Londres era B eso da
phi1 = 0.998998 con error estandar 1e-6 y t = 1.04e6, cuando el maximo real que
da fue es 1.0354. Es decir: un resultado con aspecto publicable, fijado contra
una pared invisible.

El sesgo no es aleatorio -- excluye justamente los ciclos persistentes.

ARREGLO APLICADO: `cast.ar_is_stationary` comprueba la estacionariedad por las
RAICES del polinomio, para cualquier orden. La guarda se queda -- rechazar antes
de llamar a elf es mas barato que evaluar la verosimilitud -- pero rechaza ahora
la region correcta, igual que `chekma` hace para el MA tres lineas mas abajo en
el propio C.

Y resulto ser mas ancha de lo que decia este informe: tambien sobraba en p=1.
Medido con la guarda desactivada, todo AR(1) estacionario evalua limpio hasta
phi=0.9999 (ifault 0); phi=1 exacto da ifault 2 y phi>1 da ifault 3. Es decir,
elf ya rechazaba por su cuenta el conjunto correcto, y el umbral 0.999 tiraba
una franja viva de espacio paramétrico por debajo de la frontera de verdad.

PENDIENTE: la misma correccion en el C (`tran_shootx.c:629`).
""")
