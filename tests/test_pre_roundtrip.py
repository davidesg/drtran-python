"""El `.pre` debe llegar íntegro a Python.

El `.pre` es el contrato de continuidad de la escalera: fue deja ahí el mejor
modelo univariante estimado y drtran lo toma como semilla. Si la lectura pierde
un campo, el cast construye una estructura VARMA distinta de la que fue estimó y
la homologación falla sin motivo aparente — el peor modo de fallo posible.

Estos tests comparan contra los valores LITERALES del fichero canónico, no contra
lo que devuelva la librería, para que sigan valiendo si fue cambia por dentro.
"""

import os

import pytest

drtran = pytest.importorskip("drtran")

EJEMPLOS = os.path.join(os.path.dirname(__file__), "data")
# Los .pre canónicos viven en el repo del C; se usan desde ahí mientras no se
# copien al de Python (son la referencia, no una copia).
C_EXAMPLES = "/home/david/Dropbox/SRC/drtran/examples"
ES_CPI = os.path.join(C_EXAMPLES, "work", "ES_CPI_m10.pre")
WTI = os.path.join(C_EXAMPLES, "work", "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI), reason="faltan los .pre canónicos del repo C")


def test_lee_la_serie_y_su_cabecera():
    s = drtran.load_pre(ES_CPI)
    assert s.freq == 12
    assert s.nobs == 216
    assert s.ts.start == (2002, 1)
    assert s.name == "ES_CPI"
    assert len(s.ts.data) == 216
    assert s.ts.data[0] == pytest.approx(58.717)
    assert s.ts.data[-1] == pytest.approx(82.840)


def test_lee_la_transformacion():
    """λ, d, D y refactor. `refactor` es crítico: a escala 1 el optimizador se
    degrada (paso de cdgrad ~6e-6 frente a Δlog ~0.002)."""
    m = drtran.load_pre(ES_CPI).model
    assert m.boxlam == pytest.approx(0.0)
    assert m.d == 1
    assert m.D == 0
    assert m.refactor == pytest.approx(100.0)


def test_lee_el_arma_con_sus_ordenes_y_flags():
    m = drtran.load_pre(ES_CPI).model
    assert len(m.ar) == 1                     # un operador AR regular
    assert len(m.ar[0]) == 1                  # de orden 1
    assert float(m.ar[0][0]) == pytest.approx(0.4028)
    assert bool(m.ar_free[0][0]) is True      # flag 1 en el .pre = libre
    # el resto de bloques están vacíos en este .pre
    assert len(m.ar_s) == 0 and len(m.ma) == 0 and len(m.ma_s) == 0
    assert len(m.ar_f) == 0 and len(m.ma_f) == 0


def test_lee_los_11_deterministas_con_sus_omegas():
    m = drtran.load_pre(ES_CPI).model
    iv = m.interventions
    assert len(iv) == 11
    assert [v.type for v in iv] == (["cos", "sin"] * 5) + ["alter"]
    esperado = [-0.161112, -0.166261, 0.289306, -0.577289, 0.086116, -0.033312,
                0.058589, -0.017333, -0.002610, 0.011549, 0.114945]
    assert [float(v.omega[0]) for v in iv] == pytest.approx(esperado)
    # los flags de fijado tienen que venir, o no sabremos qué está libre
    assert all(hasattr(v, "omega_free") for v in iv)


def test_distingue_mu_libre_de_mu_fijado():
    """La prueba decisiva. ES_CPI trae μ libre; WTI_ar1 lo FIJA en 0.

    Si esto se perdiera, la conjunta liberaría un parámetro que el método fija y
    no podría homologar con fue.
    """
    libre = drtran.load_pre(ES_CPI).model
    assert libre.mu0 == pytest.approx(0.154472)
    assert libre.estimate_mu is True

    fijo = drtran.load_pre(WTI).model
    assert fijo.mu0 == pytest.approx(0.0)
    assert fijo.estimate_mu is False


def test_el_segundo_pre_canonico_completo():
    s = drtran.load_pre(WTI)
    assert s.name == "WTI"
    assert s.nobs == 216
    assert float(s.model.ar[0][0]) == pytest.approx(0.2992)
    assert len(s.model.interventions) == 0
    assert s.model.refactor == pytest.approx(100.0)


def test_rechaza_lo_que_no_sirve_para_la_conjunta():
    with pytest.raises(Exception):
        drtran.load_pre(os.path.join(C_EXAMPLES, "no_existe.pre"))


def test_avisa_cuando_la_escala_degrada_al_optimizador():
    """refactor=100 no debe avisar; una escala cruda sí."""
    s = drtran.load_pre(ES_CPI)
    assert drtran.check_scale(s) is None

    class _Fake:
        model = type("M", (), {"refactor": 1.0})()
        name = "crudo"

    aviso = drtran.check_scale(_Fake())
    assert aviso is not None and "refactor=1" in aviso
