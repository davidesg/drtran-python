"""Homologación con el `drtran` de C: la referencia externa del puerto.

Los valores de `REF_C` salen de ejecutar el binario compilado sobre los mismos
`.pre`, y son la fuente de verdad más fuerte que tiene el puerto — por encima de
cualquier invariante interno. Se generaron con:

    ./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/WTI_ar1.pre \\
        -b B -r R -s S -S -o /tmp/out

(`-S` = cast por RESTA, que es el que el puerto implementa hoy; `-V`, el
empotrado, es el DEFECTO del C pero está bloqueado en Python por el bug de Φ_p
singular de drvarma — ver `drvarma/bench/repro_phi_p_singular.py`.)

La batería del C (`test_battery.sh`, 292 PASS) cubre lo demás: verdad sintética,
pass-through con Y=X, red m6 y round-trip del driver guiado.
"""

import os
import shutil
import subprocess

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.estimate import fit, unpack  # noqa: E402

C_REPO = "/home/david/Dropbox/SRC/drtran"
CASOS = os.path.join(C_REPO, "tests", "cases")
BIN = os.path.join(C_REPO, "bin", "drtran")
Y_PRE = os.path.join(CASOS, "ES_CPI_m10.pre")
X_PRE = os.path.join(CASOS, "WTI_ar1.pre")

# logL del binario C, cast por resta (-S). Diferencia medida con el puerto: ~1e-7.
REF_C = {
    (0, 0, 0): -736.774158,
    (0, 1, 0): -721.720197,
    (0, 0, 1): -718.183933,
    (1, 1, 1): -756.528944,
}
REF_DIAGONAL = -767.424341          # -0 (sin transferencia); también la suma de fue
REF_OMEGA0 = 0.016002               # ± 0.001935, t = 8.27 (errores estándar del C)

pytestmark = pytest.mark.skipif(
    not os.path.exists(Y_PRE), reason="falta el repo C de drtran")


@pytest.fixture(scope="module")
def dos():
    return drtran.load_pre(Y_PRE), drtran.load_pre(X_PRE)


@pytest.mark.parametrize("brs", sorted(REF_C))
def test_el_puerto_reproduce_el_binario_c(dos, brs):
    b, r, s = brs
    Y, X = dos
    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=b, r=r, s=s)]))
    assert f.ifault == 0
    assert f.loglik == pytest.approx(REF_C[brs], abs=1e-5)


def test_el_diagonal_coincide_con_el_c_y_con_fue(dos):
    """Doble ancla: el `-0` del C y la suma de las univariantes de fue dan el
    mismo número."""
    Y, X = dos
    f = fit(build_cast_spec([Y, X]))
    assert f.loglik == pytest.approx(REF_DIAGONAL, abs=1e-4)


def test_omega_coincide_con_el_estimado_por_el_c(dos):
    Y, X = dos
    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=0, r=0, s=0)]))
    omega0 = unpack(f)["links"][0][0][0]
    assert omega0 == pytest.approx(REF_OMEGA0, abs=1e-5)
    # el C reporta s.e. 0.001935 → t = 8.27; el ω está muy lejos de cero
    assert omega0 > 5 * 0.001935


@pytest.mark.skipif(not os.path.exists(BIN), reason="no hay binario compilado")
def test_contra_el_binario_en_vivo(dos, tmp_path):
    """No se fía de los valores tabulados: vuelve a ejecutar el C y compara.

    Si el C cambia, este test lo detecta en vez de arrastrar una referencia
    obsoleta.
    """
    Y, X = dos
    b, r, s = 0, 1, 0
    out = tmp_path / "c.out"
    res = subprocess.run(
        [BIN, Y_PRE, X_PRE, "-b", str(b), "-r", str(r), "-s", str(s),
         "-S", "-o", str(out)],
        capture_output=True, text=True, timeout=300, cwd=C_REPO)
    linea = [l for l in res.stdout.splitlines() if "Log-likelihood" in l]
    if not linea:
        pytest.skip(f"el binario no dio log-likelihood: {res.stdout[-300:]}")
    ll_c = float(linea[0].split(":")[1])

    f = fit(build_cast_spec([Y, X], links=[Link(0, 1, b=b, r=r, s=s)]))
    assert f.loglik == pytest.approx(ll_c, abs=1e-5)
