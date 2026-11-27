"""Bloque de identificación del rodaje P07: BUG-82 (tres lecturas de los
pesos), BUG-83 (un AR estacional estacionario no es un SARIMA), BUG-84 (la CCF
con los retardos del texto) y BUG-85 (la Q del ruido a s, 2s y 3s + 3)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("repro", ["repro_p07_refine_se_cortan.py",     # BUG-82
                                   "repro_p07_aviso_sarima.py",         # BUG-83
                                   "repro_p07_ccf_12_frente_a_24.py",   # BUG-84
                                   "repro_p07_q_del_ruido.py"])         # BUG-85
def test_repro_cerrado(repro):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", repro)],
                       capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_decay_pattern_tres_lecturas():
    from drtran.school import decay_pattern

    assert decay_pattern([0.01, 0.0], [0.001, 0.001])["pattern"] == "cut"
    assert decay_pattern([0.01 * 0.6 ** k for k in range(6)], [0.0005] * 6)["pattern"] == "decay"
    assert decay_pattern([0.01, 0.0, 0.0, 0.01], [0.001] * 4)["pattern"] == "unclear"


def test_ar_estacional_estacionario_no_es_sarima():
    from drtran.school import _seasonal_ar_unit_root

    class _M:
        pass
    m = _M()
    m.ar_s = [[0.3]]
    assert not _seasonal_ar_unit_root(m)
    m.ar_s = [[1.0]]
    assert _seasonal_ar_unit_root(m)
