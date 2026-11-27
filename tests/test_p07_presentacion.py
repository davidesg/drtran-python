"""Bloque de presentación del rodaje P07 en mtram: BUG-78 (la irf en las
unidades de cada serie), BUG-80 (la ganancia con denominador), BUG-81 (sin
órdenes a un asistente en la salida) y BUG-88 (el convenio de Box y Jenkins,
dicho). Cada repro sale con 0 cuando el defecto está arreglado."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("repro", [
    "repro_p07_escala_irf.py",            # BUG-78
    "repro_p07_aviso_denominador.py",     # BUG-80
    "repro_p07_instruccion_al_modelo.py",  # BUG-81
    "repro_p07_convenio_no_dicho.py",     # BUG-88
])
def test_repro_cerrado(repro):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", repro)],
                       capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
