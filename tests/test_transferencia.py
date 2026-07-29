"""Transferencias: los pesos ν, la identidad con ω=0, y la estimación conjunta."""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (Link, build_cast_spec, compute_irf,  # noqa: E402
                         loglik_diagonal, x0_from_pre)
from drtran.estimate import fit, unpack  # noqa: E402

C = "/home/david/Dropbox/SRC/drtran/examples/work"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")
DIANA = -767.424341

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI), reason="faltan los .pre canónicos del repo C")


# ── los pesos del filtro racional ────────────────────────────────────────────
def test_irf_convencion_box_jenkins():
    """ω(B) = ω₀ − ω₁B − ω₂B²…: el líder SUMA y los demás RESTAN.

    Portar este signo al revés es un error de un carácter que ya costó un bug en
    el C (el signo de Nyquist en CalcNonsOp), así que va fijado explícitamente.
    """
    assert compute_irf([2.0, 0.5], [], b=0, length=5) == pytest.approx(
        [2.0, -0.5, 0.0, 0.0, 0.0])


def test_irf_retardo_b_desplaza():
    assert compute_irf([2.0, 0.5], [], b=2, length=5) == pytest.approx(
        [0.0, 0.0, 2.0, -0.5, 0.0])


def test_irf_denominador_da_decaimiento_geometrico():
    """1/(1−δB) con δ=0.5 ⇒ ν = 1, .5, .25, .125, …"""
    assert compute_irf([1.0], [0.5], b=0, length=5) == pytest.approx(
        [1.0, 0.5, 0.25, 0.125, 0.0625])


def test_irf_es_cero_si_omega_es_cero():
    assert compute_irf([0.0, 0.0], [0.7], b=1, length=6) == pytest.approx(
        np.zeros(6))


# ── la identidad que prueba que el puente es correcto ────────────────────────
@pytest.fixture(scope="module")
def dos():
    return drtran.load_pre(ES_CPI), drtran.load_pre(WTI)


def test_omega_cero_es_exactamente_el_diagonal(dos):
    """BRIDGE_DESIGN: «Con ω = 0 el modelo se parte en dos univariantes
    independientes — y ahí está la prueba de que el puente es correcto.»"""
    Y, X = dos
    cs0 = build_cast_spec([Y, X])
    cs1 = build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=0)])
    ll0, _ = loglik_diagonal(x0_from_pre(cs0), cs0)
    ll1, _ = loglik_diagonal(x0_from_pre(cs1), cs1)   # x0 pone ω = 0
    assert ll1 == pytest.approx(ll0, abs=1e-10)
    assert ll0 == pytest.approx(DIANA, abs=1e-4)


def test_el_enlace_anade_exactamente_sus_parametros(dos):
    Y, X = dos
    base = build_cast_spec([Y, X]).npar
    for (b, r, s) in ((0, 0, 0), (1, 0, 2), (2, 1, 1)):
        cs = build_cast_spec([Y, X], links=[Link(0, 1, b, r, s)])
        assert cs.npar == base + (s + 1) + r


def test_rechaza_enlaces_imposibles(dos):
    Y, X = dos
    with pytest.raises(ValueError):
        build_cast_spec([Y, X], links=[Link(out=0, inp=0)])      # a sí misma
    with pytest.raises(ValueError):
        build_cast_spec([Y, X], links=[Link(out=0, inp=5)])      # fuera de rango


# ── estimación conjunta ──────────────────────────────────────────────────────
def test_el_diagonal_estimado_se_queda_en_la_diana(dos):
    Y, X = dos
    f = fit(build_cast_spec([Y, X]))
    assert f.ifault == 0
    assert f.loglik == pytest.approx(DIANA, abs=1e-4)


def test_la_transferencia_mejora_significativamente(dos):
    """Y←X con un solo ω: la conjunta gana ~61 puntos de LR sobre el diagonal."""
    Y, X = dos
    f0 = fit(build_cast_spec([Y, X]))
    f1 = fit(build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=0)]))

    assert f1.ifault == 0
    assert f1.loglik > f0.loglik
    lr = 2.0 * (f1.loglik - f0.loglik)
    assert lr > 10.0, "el pass-through WTI→IPC debe ser claramente significativo"

    omega = unpack(f1)["links"][0][0]
    assert omega[0] > 0.0, "un alza del crudo sube el IPC"
    assert omega[0] == pytest.approx(0.016, abs=5e-3)


# ── el ciclo completo, y la robustez al arranque ─────────────────────────────
def test_ciclo_completo_identificar_estimar_contrastar(dos):
    """Box-Jenkins de punta a punta sin decirle los órdenes."""
    from drtran.identify import identify

    Y, X = dos
    idn = identify(build_cast_spec([Y, X]), Link(out=0, inp=1))
    assert (idn.b, idn.r, idn.s) == (0, 0, 1)
    assert idn.exogena

    cs = build_cast_spec([Y, X], links=[Link(0, 1, idn.b, idn.r, idn.s)])
    f = fit(cs)
    f0 = fit(build_cast_spec([Y, X]))
    assert f.ifault == 0
    assert 2.0 * (f.loglik - f0.loglik) > 50.0        # LR medido: 98.3 con 2 gl


def test_el_optimo_es_un_atractor_no_un_eco_del_arranque(dos):
    """El test de robustez del hito M1 de drtran.

    `termcode=3` es "parada sin mejora", y sólo es un fallo si el punto no es el
    óptimo. La forma de dirimirlo es perturbar el arranque: si todos los caminos
    caen en el mismo sitio, la parada es legítima.
    """
    Y, X = dos
    cs = build_cast_spec([Y, X], links=[Link(0, 1, 0, 0, 1)])
    base = fit(cs)
    om_base = unpack(base)["links"][0][0]

    rng = np.random.default_rng(0)
    for _ in range(3):
        x0 = x0_from_pre(cs).copy()
        x0[:2] += rng.normal(0, 0.01, 2)
        x0[2:6] *= rng.uniform(0.5, 1.5, 4)
        f = fit(cs, x0=x0)
        assert f.loglik == pytest.approx(base.loglik, abs=1e-5)
        assert unpack(f)["links"][0][0] == pytest.approx(om_base, abs=1e-5)


# ── la especificación de la media ────────────────────────────────────────────
def test_la_media_es_la_media_no_un_intercepto():
    """Box-Jenkins escribe el modelo en DESVIACIONES de la media:

        (w_Y − μ_Y) = ν(B)·(w_X − μ_X) + N_t   ⇒   E[w_Y] = μ_Y

    así que la media de la salida NO hereda nada de la entrada. Multiplicando
    por δ(B), la fila 1 es Φ(B)(w − μ) = Θ(B)a con μ = (μ_Y, μ_X): sin término
    adicional. La alternativa (w_Y = c + ν(B)w_X + N) trata μ como INTERCEPTO.

    Las dos son la misma familia MIENTRAS μ_Y sea libre; divergen cuando está
    FIJADA. La coherencia con fue decide: el μ del `.pre` es la media que fue
    estimó, no un intercepto.
    """
    from drtran.embed import cast_embedded

    C = "/home/david/Dropbox/SRC/drtran/tests/cases"
    # salida con μ FIJA en 0, entrada con μ libre: el caso que discrimina
    Y = drtran.load_pre(os.path.join(C, "WTI_ar1.pre"))
    X = drtran.load_pre(os.path.join(C, "ES_CPI_m10.pre"))
    assert Y.model.estimate_mu is False and X.model.estimate_mu is True

    cs = build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=1)])
    x = x0_from_pre(cs)
    x[0], x[1] = 17.0, 5.97           # ω no triviales, para que el término pese
    _phi, _th, mu, _w, _s, ifault = cast_embedded(x, cs)

    assert ifault == 0
    assert mu[0] == pytest.approx(0.0), "μ de la salida es 0: no hereda de la entrada"
    assert mu[1] == pytest.approx(0.154472), "μ de la entrada es la del .pre"


def test_con_media_de_salida_libre_la_parametrizacion_da_igual(dos):
    """Prueba de que ajustar o no la media es una REPARAMETRIZACIÓN: con μ_Y
    libre, cualquiera de las dos alcanza el mismo óptimo. Sólo importa cuando
    μ_Y está fijada, y ahí manda la convención de desviaciones."""
    C = "/home/david/Dropbox/SRC/drtran/tests/cases"
    Y = drtran.load_pre(os.path.join(C, "ES_CPI_m10.pre"))   # μ libre
    X = drtran.load_pre(os.path.join(C, "WTI_ar1.pre"))
    cs = build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=1)])
    f = fit(cs, embed=True)
    assert f.ifault == 0
    # homologa con el binario, que en este caso (μ de la ENTRADA = 0) coincide
    assert f.loglik == pytest.approx(-718.287406, abs=1e-4)
