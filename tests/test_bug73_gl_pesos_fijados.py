"""BUG-73: los pesos fijados no cuentan como grados de libertad, ni en la Q
de adecuación (como el C, `link_npar_free`) ni en el LR de mtram."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_repro_closed():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "repro_p07_gl_pesos_fijados.py")],
                       capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_free_params_count():
    from drtran.cast import Link
    from drtran.diagnose import link_free_params, transfer_free_params
    from drtran.slots import FIXED, Slot, SlotTable

    lk = Link(out=0, inp=1, b=0, r=1, s=2)
    assert link_free_params(lk, 0) == 4                      # nominal
    t = SlotTable([Slot("omega1[0]"), Slot("omega1[1]", FIXED), Slot("omega1[2]"),
                   Slot("delta1[1]"), Slot("omega_d1[1,0]"), Slot("omega10[0]")])
    assert link_free_params(lk, 0, t) == 3
    assert transfer_free_params([lk], t) == 3
