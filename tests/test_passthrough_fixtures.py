"""The autonomous-lane WTI, pinned as a fixture.

`test_end_to_end_passthrough.py` walks the whole chain from the CSV, and so it
models WTI with `art.batch_build`, the one-call shortcut. The model an analyst
would actually bring is the one art's AUTONOMOUS LANE builds: node by node, with
the domain declared and every decision written down. That run is not a function
call and cannot be replayed in a test, so its `.pre` is kept here and checked
for what matters downstream:

- it is an optimum (fue does not move it);
- drtran's diagonal gate reproduces it exactly;
- mtram reaches the same transfer from it.

See `tests/data/passthrough/README.md` for where the files come from.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
pytest.importorskip("mcp", reason="needs the MCP extra: pip install 'drtran[mcp]'")
pytest.importorskip("fue")

from drtran import mcp_server as M  # noqa: E402
from drtran.pre import load_pre  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "passthrough")
WTI = os.path.join(DATA, "WTI_autonomo.pre")
IPC = os.path.join(DATA, "IPC_ES_cadena.pre")


def _num(text, label):
    m = re.search(re.escape(label) + r"\s*\**\s*([-+]?\d+\.\d+(?:e[-+]?\d+)?)",
                  text)
    assert m, f"{label!r} not in output"
    return float(m.group(1))


@pytest.fixture(scope="module")
def gate():
    return M.load_pre("FIXTURES", f"{IPC},{WTI}")


def test_the_autonomous_WTI_is_the_model_the_lane_chose():
    m = load_pre(WTI).model
    assert (m.boxlam, m.d, m.D) == (0.0, 1, 0)
    assert m.estimate_mu
    assert [len(f) for f in m.ar] == [1] and m.ar[0][0] > 0, \
        "AR(1) with φ>0: the domain's gradual adjustment"
    assert not any(len(f) for f in (m.ma or []))
    steps = [i for i in m.interventions if i.type == "step"]
    assert [len(i.omega) for i in steps] == [3, 3]


def test_it_is_a_fixed_point_of_fue(tmp_path):
    stem = str(tmp_path / "fp")
    shutil.copy(WTI, stem + ".inp")
    subprocess.run([sys.executable, "-m", "fue", "fp"], cwd=str(tmp_path),
                   capture_output=True, timeout=600)

    def vals(p):
        m = load_pre(p).model
        return np.array([o for it in m.interventions for o in it.omega]
                        + [c for f in m.ar for c in f], float)

    assert float(np.max(np.abs(vals(WTI) - vals(stem + ".pre")))) <= 5e-6


def test_the_diagonal_gate_reproduces_it(gate):
    assert "✅" in gate and "Coinciden" in gate, gate
    assert _num(gate, "| WTI |") == pytest.approx(-730.657424, abs=1e-3)
    assert _num(gate, "| IPC_ES |") == pytest.approx(-7.297333, abs=1e-3)
    assert abs(_num(gate, "Diferencia con la suma: **")) < 1e-5
    assert "WTI: lambda=0 d=1 D=0 refactor=100 deterministas=2" in gate


def test_mtram_lands_on_the_same_transfer(gate):
    """The same order and revision as the chain test, and the gain within its
    tolerance: 0.030555 here against 0.029370 there."""
    out = M.build_model("FIXTURES")
    assert "b=1 r=0 s=0  ->  b=0 r=0 s=1" in out, out
    assert _num(out, "ganancia nu(1) =") == pytest.approx(0.030555, abs=0.001)
