"""LA PUERTA: la conjunta diagonal debe reproducir la suma de las univariantes.

Es el criterio de validación del diseño, y la condición de entrada a todo lo
demás. Con estructura diagonal (AR/MA y covarianza diagonales, sin transferencia)
la verosimilitud exacta se factoriza, así que la conjunta tiene que dar la SUMA.

Si falla, el cast está mal — **nunca** `elf`, que se usa tal cual.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (build_cast_spec, cast_diagonal,  # noqa: E402
                         loglik_diagonal, x0_from_pre)

C = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")
DIANA = -767.424341

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI), reason="faltan los .pre canónicos del repo C")


@pytest.fixture(scope="module")
def cs():
    return build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])


def test_la_estructura_es_diagonal_y_lleva_los_arma_del_pre(cs):
    phi, theta, mu, w, sigma, ifault = cast_diagonal(x0_from_pre(cs), cs)
    assert ifault == 0
    assert w.shape == (215, 2)          # 216 obs, d=1 en ambas
    assert phi.shape == (1, 2, 2) and theta.shape == (0, 2, 2)
    # AR diagonal, con el phi de cada .pre en su casilla
    assert phi[0][0, 0] == pytest.approx(0.4028)
    assert phi[0][1, 1] == pytest.approx(0.2992)
    assert phi[0][0, 1] == 0.0 and phi[0][1, 0] == 0.0
    # medias: la de Y libre, la de X fijada en 0 por su .pre
    assert mu[0] == pytest.approx(0.154472)
    assert mu[1] == 0.0
    # covarianza normalizada: Q[0][0] = 1, la escala la concentra sigma2
    assert sigma[0, 0] == pytest.approx(1.0)
    assert sigma[0, 1] == 0.0 and sigma[1, 0] == 0.0


def test_las_semillas_del_pre_ya_son_el_optimo_diagonal(cs):
    """Y por eso el C reporta termcode 3 aquí: arrancando en el óptimo, la
    búsqueda lineal no puede mejorar y para. No es un fallo."""
    ll, ifault = loglik_diagonal(x0_from_pre(cs), cs)
    assert ifault == 0
    assert ll == pytest.approx(DIANA, abs=1e-4)


def test_la_razon_de_varianzas_se_siembra_no_se_deja_en_uno(cs):
    """Las escalas difieren en un factor ~1098 (σ² 0.0627 vs 68.84). Sembrar
    log(var2/var1)=0 deja el punto inicial en logL −1371 en vez de −767."""
    x0 = x0_from_pre(cs)
    assert float(x0[-1]) == pytest.approx(np.log(68.8381 / 0.062666), abs=1e-3)

    malo = x0.copy()
    malo[-1] = 0.0
    ll_malo, _ = loglik_diagonal(malo, cs)
    assert ll_malo < DIANA - 100, "sin sembrar la razón, el arranque es muy peor"


def test_omega_cero_parte_el_modelo_en_dos_univariantes(cs):
    """La prueba de que el puente es correcto (BRIDGE_DESIGN): sin transferencia
    el VARMA se factoriza y la conjunta ES la suma de las dos univariantes."""
    import fue

    total = 0.0
    for p in (ES_CPI, WTI):
        _ts, m = fue.load(p)
        m.fit()
        total += float(m.loglik)

    ll, _ = loglik_diagonal(x0_from_pre(cs), cs)
    assert ll == pytest.approx(total, abs=1e-4)
