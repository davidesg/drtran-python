"""Identificación de (b, r, s) por preblanqueo y CCF, contra el binario C.

Referencia, de `./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/WTI_ar1.pre`:

    CCF BANDS  2.0/SQRT(N) =  0.13640
      0   0.492      1   0.310      2   0.025     -1  -0.077     -6  -0.128
    Exogeneity: Q(24) = 18.2969   p = 0.7884   [0 significant out of 24]
    PROPOSALS: [A] b=0 r=0 s=1     RECOMMENDED: b=0, r=0, s=1
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.identify import identify, prewhiten, report  # noqa: E402

C = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES_CPI = os.path.join(C, "ES_CPI_m10.pre")
WTI = os.path.join(C, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES_CPI), reason="faltan los .pre canónicos del repo C")


@pytest.fixture(scope="module")
def ident():
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    return identify(cs, Link(out=0, inp=1))


def test_la_banda_coincide_con_el_c(ident):
    assert ident.threshold == pytest.approx(0.13640, abs=1e-5)


def test_la_ccf_coincide_con_el_c(ident):
    mid = len(ident.ccf) // 2
    assert ident.ccf[mid] == pytest.approx(0.492, abs=1e-3)      # k=0
    assert ident.ccf[mid + 1] == pytest.approx(0.310, abs=1e-3)  # k=1
    assert ident.ccf[mid + 2] == pytest.approx(0.025, abs=1e-3)  # k=2
    assert ident.ccf[mid - 1] == pytest.approx(-0.077, abs=1e-3)  # k=-1
    assert ident.ccf[mid - 6] == pytest.approx(-0.128, abs=1e-3)  # k=-6


def test_la_propuesta_coincide_con_el_c(ident):
    assert (ident.b, ident.r, ident.s) == (0, 0, 1)


def test_el_portmanteau_de_exogeneidad_coincide_con_el_c(ident):
    """Q usa el divisor (n − i + 1) de `diagnose.c:ChiTestC`, no (n − i)."""
    assert ident.Q_exogeneidad == pytest.approx(18.2969, abs=1e-3)
    assert ident.p_exogeneidad == pytest.approx(0.7884, abs=1e-3)
    assert ident.n_signif_negativos == 0
    assert ident.exogena is True


def test_solo_cuenta_el_bloque_contiguo_desde_b(ident):
    """Hay un pico significativo en el lag 24 (ruido muestral: con bandas al 5 %
    se espera 1 de cada 20 fuera). NO debe entrar en la propuesta: tomar el
    último significativo de toda la CCF disparaba s a 24."""
    mid = len(ident.ccf) // 2
    assert abs(ident.ccf[mid + 24]) > ident.threshold, "el pico lejano existe"
    assert ident.s == 1, "pero la propuesta se queda en el bloque contiguo 0..1"


def test_preblanqueo_de_un_ar1_conocido():
    """Filtrar un AR(1) por su propio φ deja ruido blanco."""
    rng = np.random.default_rng(0)
    n, phi = 400, 0.6
    e = rng.normal(0, 1, n)
    w = np.zeros(n)
    for t in range(1, n):
        w[t] = phi * w[t - 1] + e[t]
    a = prewhiten(w, [phi], [])
    assert a[1:] == pytest.approx(e[1:], abs=1e-9)


def test_preblanqueo_invierte_el_ma():
    """Con θ, a[t] = u[t] + Σ θ_k a[t−k] deshace el MA."""
    w = np.array([1.0, 0.0, 0.0, 0.0])
    a = prewhiten(w, [], [0.5])
    assert a == pytest.approx([1.0, 0.5, 0.25, 0.125])


def test_detecta_una_transferencia_sintetica_con_retardo():
    """Verdad conocida: Y_t = 2·X_{t−3} + ruido, con X blanco. Debe salir b=3."""
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    y[3:] = 2.0 * x[:-3]
    y += rng.normal(0, 0.3, n)

    from drtran.identify import _ccf
    thr = 2.0 / np.sqrt(n)
    c = _ccf(x, y, 12)
    sig = [k for k in range(13) if abs(c[k]) > thr]
    assert sig and sig[0] == 3, f"el primer retardo significativo debe ser 3: {sig}"


def test_el_informe_menciona_lo_esencial(ident):
    txt = report(ident, ("WTI", "ES_CPI"))
    assert "b=0" in txt and "r=0" in txt and "s=1" in txt
    assert "18.2969" in txt or "18.297" in txt
    assert "exógena" in txt


# ── lo que dice el artículo original (Haugh & Box 1977) ──────────────────────
def test_banda_de_haugh_box_elimina_el_falso_positivo_del_lag_24():
    """Haugh & Box (1977) §1.4: var{r_xy(k)} ≈ (N−k)⁻¹, no N⁻¹.

    La banda debe ENSANCHARSE con el retardo. Con N=215 el umbral en k=24 pasa de
    0.1364 a 0.1447, y el pico de −0.1423 deja de ser significativo. Ese pico es
    justamente el falso positivo que obligó al C a la heurística del bloque
    contiguo: la heurística compensa una banda mal calibrada.
    """
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    const = identify(cs, Link(out=0, inp=1), banda="constante")
    hb = identify(cs, Link(out=0, inp=1), banda="haugh-box")

    assert 24 in const.significativos_no_negativos
    assert 24 not in hb.significativos_no_negativos
    assert hb.significativos_no_negativos == [0, 1]
    # la propuesta no cambia: la heurística ya lo estaba tapando
    assert (const.b, const.r, const.s) == (hb.b, hb.r, hb.s) == (0, 0, 1)


def test_la_banda_haugh_box_crece_con_el_retardo():
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    hb = identify(cs, Link(out=0, inp=1), banda="haugh-box")
    mid = len(hb.lags) // 2
    assert hb.umbral[mid] == pytest.approx(2 / np.sqrt(215), abs=1e-6)
    assert hb.umbral[mid + 24] == pytest.approx(2 / np.sqrt(215 - 24), abs=1e-6)
    assert hb.umbral[mid + 24] > hb.umbral[mid]
    # y es simétrica
    assert hb.umbral[mid - 24] == pytest.approx(hb.umbral[mid + 24])


def test_el_defecto_sigue_homologando_con_el_c():
    """El defecto es la banda constante, que es lo que hace el binario."""
    cs = build_cast_spec([drtran.load_pre(ES_CPI), drtran.load_pre(WTI)])
    i = identify(cs, Link(out=0, inp=1))
    assert i.threshold == pytest.approx(0.13640, abs=1e-5)
