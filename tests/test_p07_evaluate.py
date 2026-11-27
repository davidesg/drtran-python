"""BUG-76 (evaluate evalúa el modelo adoptado, con su .cns, y escribe los
errores) y BUG-77 (la ventana por fecha, con series de inicio distinto)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("repro", ["repro_p07_evaluate_cns.py",       # BUG-76
                                   "repro_p07_evaluate_ventana.py"])  # BUG-77
def test_repro_cerrado(repro):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", repro)],
                       capture_output=True, text=True, timeout=1500,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_truncate_por_fecha():
    import warnings

    import drtran
    from drtran.evaluate import truncate

    warnings.filterwarnings("ignore")
    d = os.path.join(ROOT, "tests", "data", "p07")
    es = drtran.load_pre(os.path.join(d, "es_cpi_m10_clase.pre"))
    wti = drtran.load_pre(os.path.join(d, "wti_ar1_clase.pre"))
    es120 = truncate([wti, es], 120, ref=0)            # misma muestra: por posición
    assert [s.ts.nobs for s in es120] == [120, 120]
