"""BUG-1: the unit-circle guard used to reject stationary AR(2)s.

`tran_shootx.c:629` — and its two Python ports — carried

    if (p_ord[i] >= 1 && fabs(phi[i][1]) >= 0.999) { *ifaultx = 1; ... }

The comment states the intent, "an AR(1) pinned to the unit circle", and the
condition is `>= 1`, i.e. EVERY order. For p >= 2, `phi[0]` is not a root: in an
AR(2) the stationary region is the triangle |phi2| < 1, phi2 + phi1 < 1,
phi2 - phi1 < 1, in which **phi1 reaches 2**. Every AR(2) with complex roots and
phi1 > 1 is stationary, and every one of them was refused before the likelihood
was even evaluated.

Not an exotic corner. That region is exactly where the PERSISTENT CYCLES live,
which is what a whole class of study is about — and the damage was not a visible
`ifault`. Starting below the barrier the optimiser never failed: it walked up to
0.998998 and pinned there, standard error 1e-06, t = 1.04e+06. A
publishable-looking estimate pressed against an invisible wall.

The guard is kept, because rejecting before calling `elf` is cheaper than a
likelihood evaluation and gives the line search a clean refusal. Only the region
it rejects changed. See `drtran.cast.ar_is_stationary`.
"""
import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")

from drtran.cast import ar_is_stationary, build_cast_spec, cast_diagonal  # noqa: E402
from drtran.cast import x0_from_pre  # noqa: E402
from drtran.embed import cast_embedded  # noqa: E402
from drtran.pre import PreSpec  # noqa: E402

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ES, WTI = "ES_CPI_m10.1.pre", "WTI_ar1.1.pre"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(AQUI, ES)),
    reason="the repo-root .pre files are missing")


def _con_ar(path, coefs):
    """El mismo `.pre` con su AR sustituido."""
    import copy

    s = drtran.load_pre(os.path.join(AQUI, path))
    m = copy.deepcopy(s.model)
    m.ar = [[float(c) for c in coefs]]
    m.ar_free = [[1] * len(coefs)]
    return PreSpec(ts=m.series, model=m, path=s.path)


def _ifaults(coefs):
    cs = build_cast_spec([_con_ar(ES, coefs), _con_ar(WTI, coefs)], [])
    x = x0_from_pre(cs)
    return cast_diagonal(x, cs)[-1], cast_embedded(x, cs)[-1]


# ── la función, contra la definición ───────────────────────────────────────
@pytest.mark.parametrize("coefs,estacionario", [
    ([0.4028], True),                 # el canónico
    ([0.9990], True),                 # justo EN el viejo umbral, y estacionario
    ([0.9999], True),                 # |raíz| = 1.0001
    ([1.0000], False),                # sobre el círculo
    ([1.0010], False),
    ([0.9500, -0.5200], True),
    ([1.0354, -0.5184], True),        # el máximo real del trigo de Londres
    ([1.6000, -0.8000], True),        # |raíz| = 1.118, ciclo persistente
    ([2.1000, -1.0500], False),       # |raíz| = 0.782, de verdad explosivo
])
def test_stationarity_is_read_off_the_roots(coefs, estacionario):
    assert ar_is_stationary(coefs) is estacionario, (
        f"phi={coefs}, |raíces| = {np.abs(np.roots(np.r_[-np.asarray(coefs, float)[::-1], 1.0]))}")


# ── los casts, que es donde se rechazaba ───────────────────────────────────
@pytest.mark.parametrize("coefs", [
    [0.9990], [0.9999],               # p = 1: el umbral 0.999 también sobraba
    [1.0020, -0.5425], [1.0354, -0.5184], [1.6000, -0.8000],
])
def test_a_stationary_ar_is_accepted_by_both_casts(coefs):
    """Todas éstas devolvían ifault=1 y ni se evaluaba la verosimilitud."""
    d, e = _ifaults(coefs)
    assert (d, e) == (0, 0), f"phi={coefs} rechazado: diag={d}, emb={e}"


@pytest.mark.parametrize("coefs", [[1.0], [1.05], [2.1, -1.05]])
def test_a_non_stationary_ar_is_still_refused(coefs):
    """La otra mitad. Una guarda que dejara pasar lo explosivo sería peor que
    la que había: el punto de la corrección es mover la frontera al sitio
    correcto, no quitarla."""
    d, e = _ifaults(coefs)
    assert d != 0 and e != 0, f"phi={coefs} aceptado: diag={d}, emb={e}"


def test_the_boundary_sits_where_elf_puts_it():
    """Por qué 1.0 y no 0.999: `elf` ya rechaza exactamente el conjunto
    correcto por su cuenta. Medido con la guarda desactivada, todo AR(1)
    estacionario evalúa limpio hasta phi=0.9999 (ifault 0); phi=1 da ifault 2 y
    phi>1 da ifault 3. El umbral viejo tiraba una franja viva de espacio
    paramétrico por debajo de la frontera de verdad."""
    assert ar_is_stationary([0.9999]) and not ar_is_stationary([1.0])


def test_the_guard_still_fires_before_the_likelihood():
    """No se ha sustituido por "que lo cace elf". La guarda se queda porque
    rechazar antes de llamar a elf es más barato que evaluar la verosimilitud y
    le da a la búsqueda lineal una negativa limpia; lo único que cambió es la
    región que rechaza."""
    d, _ = _ifaults([2.1, -1.05])
    assert d == 1, "el rechazo debe venir de la guarda (ifault=1), no de elf"
