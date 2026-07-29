"""La línea base contra la que se validará el cast.

El criterio de validación de drtran (puerta de entrada a todo lo demás) es:

    **Estimación conjunta diagonal ≡ fue por separado.**

Con estructura diagonal (AR/MA y covarianza diagonales, sin transferencia) la
verosimilitud exacta se factoriza, así que la conjunta debe reproducir la SUMA de
las dos univariantes. Estos tests fijan esa suma.

Si alguno falla, no es que el cast esté mal: es que la referencia se movió, y no
hay contra qué validar nada. Por eso van aparte.

Valores de referencia: BRIDGE_DESIGN.md del repo C, caso canónico
ES_CPI_m10 (Y) ← WTI_ar1 (X), verificado allí el 2026-07-12 contra el C.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")

C_EXAMPLES = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C_EXAMPLES, "ES_CPI_m10.pre")
WTI = os.path.join(C_EXAMPLES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI), reason="faltan los .pre canónicos del repo C")

# BRIDGE_DESIGN.md, tabla del caso canónico
REF = {
    "ES_CPI": dict(phi=0.402839, mu=0.154472, loglik=-7.3917),
    "WTI": dict(phi=0.299193, mu=0.0, loglik=-760.0326),
}
LOGLIK_CONJUNTA = -767.424341        # objetivo de la estimación conjunta diagonal


def _fit(path):
    s = drtran.load_pre(path)
    s.model.fit()
    return s.model


def test_es_cpi_reproduce_la_referencia():
    m = _fit(ES_CPI)
    assert float(np.atleast_1d(m.ar[0])[0]) == pytest.approx(REF["ES_CPI"]["phi"],
                                                             abs=1e-5)
    assert float(m.mu0) == pytest.approx(REF["ES_CPI"]["mu"], abs=1e-5)
    assert float(m.loglik) == pytest.approx(REF["ES_CPI"]["loglik"], abs=1e-3)


def test_wti_reproduce_la_referencia():
    m = _fit(WTI)
    assert float(np.atleast_1d(m.ar[0])[0]) == pytest.approx(REF["WTI"]["phi"],
                                                             abs=1e-5)
    assert m.estimate_mu is False, "el .pre fija mu en 0"
    assert float(m.loglik) == pytest.approx(REF["WTI"]["loglik"], abs=1e-3)


def test_la_suma_es_la_diana_de_la_conjunta():
    """LA prueba: la suma de las dos univariantes es lo que la conjunta diagonal
    tiene que reproducir. Medido: -767.424341 con diferencia 7.6e-09."""
    total = float(_fit(ES_CPI).loglik) + float(_fit(WTI).loglik)
    assert total == pytest.approx(LOGLIK_CONJUNTA, abs=1e-4)
