"""La muestra común de la puerta diagonal cuenta `ifadf`, no sólo `d + D·s`.

`_muestra_comun` calculaba las observaciones perdidas como `d + D·freq`, lo que
**ignora `ifadf`**. Un factor de frecuencia consume observaciones igual que ∇:
con `ifadf[3]=1` el operador lleva un `(1+B²)` de grado 2 que nadie descontaba.
La puerta pedía entonces a fue que puntuara dos observaciones más de las que usa
el ajuste conjunto, y los dos lados dejaban de comparar los mismos datos.

Es **BUG-5 reabierto por otra puerta**: aquel se diagnosticó y arregló para `D`,
el motor ya tenía `differencing_poly` —que sí incluye `ifadf`— y esta función no
la usaba. La lección de BUG-8 otra vez: comparar polinomios, no tuplas `(d, D)`.

Medido sobre el caso reportado (IPC_ES con f=3 estocástica frente a WTI con d=1),
que reproduce exactamente las cifras de la ficha BUG-0017 de art:

    sin el arreglo   suma fue = -772.025418   diagonal = -764.493984   dif +7.53
    con el arreglo   suma fue = -765.017984   diagonal = -764.493984   dif +0.524

El arreglo cierra el 93 % del hueco. El residuo queda anotado en la ficha: el
ajuste conjunto termina con `termcode=3` (stopped without improvement).
"""
import pytest


class _Serie:
    def __init__(self, nobs, freq=12):
        self.nobs = nobs
        self.freq = freq
        self.start = (2002, 1)


class _Modelo:
    """Lo mínimo que `differencing_poly` y `_muestra_comun` miran."""
    def __init__(self, d=0, D=0, ifadf=None, nobs=216, freq=12):
        self.d = d
        self.D = D
        self.ifadf = ifadf or [0] * (freq // 2 + 1)
        self.series = _Serie(nobs, freq)


class _Spec:
    def __init__(self, model):
        self.model = model


def _muestra(models):
    from drtran.mcp_server import _muestra_comun

    return _muestra_comun([_Spec(m) for m in models])


def _grado(m):
    from drtran.cast import differencing_poly

    return len(differencing_poly(m)) - 1


# ── el defecto ─────────────────────────────────────────────────────────────

def test_an_interior_ifadf_factor_costs_two_observations():
    """`(1 + B²)` en f=3 es de grado 2, y ese grado se descuenta."""
    m = _Modelo(d=1, ifadf=[0, 0, 0, 1, 0, 0, 0])
    assert _grado(m) == 3          # ∇ (grado 1) · (1+B²) (grado 2)
    assert _muestra([m]) == 216 - 3


def test_the_nyquist_factor_costs_one():
    m = _Modelo(d=1, ifadf=[0, 0, 0, 0, 0, 0, 1])
    assert _grado(m) == 2          # ∇ · (1+B)
    assert _muestra([m]) == 216 - 2


def test_the_pair_takes_the_shortest():
    """El caso del bug: output con ifadf, input sin él."""
    out = _Modelo(d=1, ifadf=[0, 0, 0, 1, 0, 0, 0])
    inp = _Modelo(d=1)
    assert (_grado(out), _grado(inp)) == (3, 1)
    assert _muestra([out, inp]) == 213      # antes daba 215


# ── y lo que ya funcionaba, como guardia ───────────────────────────────────

@pytest.mark.parametrize("d,D,esperado", [(0, 0, 216), (1, 0, 215),
                                          (2, 0, 214), (1, 1, 203)])
def test_the_plain_d_and_D_cases_are_unchanged(d, D, esperado):
    """Sin `ifadf`, el grado del polinomio ES `d + D·s`, así que el arreglo no
    puede mover estos casos — incluido el D=1 que arregló BUG-5."""
    m = _Modelo(d=d, D=D)
    assert _grado(m) == d + D * 12
    assert _muestra([m]) == esperado


def test_it_is_the_polynomial_that_decides_not_the_tuple():
    """`d=2, D=0, ifadf=[0,1,1]` y `d=1, D=1` dan el MISMO operador en freq=4 —
    la identidad de BUG-8. La muestra común tiene que coincidir."""
    a = _Modelo(d=2, D=0, ifadf=[0, 1, 1], nobs=100, freq=4)
    b = _Modelo(d=1, D=1, nobs=100, freq=4)
    # ∇∇₄ = (1−B)²(1+B)(1+B²): grado 2+1+2 = 5, no 4. El operador lleva orden
    # DOS en la frecuencia cero, que es la parte que se cuenta mal a ojo.
    assert _grado(a) == _grado(b) == 5
    assert _muestra([a]) == _muestra([b]) == 95
