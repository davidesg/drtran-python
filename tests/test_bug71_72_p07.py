"""BUG-71 (la puerta N0 con factores ifadf) y BUG-72 (overfit desde el óptimo
actual y con el .cns), del rodaje de la práctica 7: sus repros salen con 0."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("repro", ["repro_p07_puerta_ifadf.py", "repro_p07_overfit.py"])
def test_repro_closed(repro):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", repro)],
                       capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_the_variant_inherits_constraints_and_optimum():
    from drtran.mcp_server import _hereda_optimo, _hereda_restricciones, _ocurrencias
    from drtran.slots import FIXED, Slot, SlotTable

    t0 = SlotTable([Slot("a"), Slot("w[1]", FIXED, value=0.0), Slot("a")])
    t1 = SlotTable([Slot("a"), Slot("w[1]"), Slot("a"), Slot("w[2]")])
    _hereda_restricciones(t0, t1)
    assert t1.slots[1].kind == FIXED and t1.n_free == 3
    x = _hereda_optimo(t0, [1.0, 0.0, 2.0], t1, [9.0, 9.0, 9.0, 0.0])
    assert list(x) == [1.0, 0.0, 2.0, 0.0]
    assert _ocurrencias(["a", "b", "a"]) == [("a", 0), ("b", 0), ("a", 1)]
