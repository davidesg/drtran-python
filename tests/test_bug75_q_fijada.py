"""BUG-75: un q[i,j] fijado que deja la Q de partida no definida positiva se
rechaza antes de estimar, con el rango admisible (como el C, rc = 12), en vez
de devolver loglik = -inf."""
import os
import warnings

import pytest

import drtran
from drtran.estimate import start_q_message
from drtran.slots import build_slots, read_cns

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "p07")
WTI = os.path.join(DATOS, "wti_ar1_clase.pre")
ES = os.path.join(DATOS, "es_cpi_m10_clase.pre")


def _cast():
    warnings.filterwarnings("ignore")
    specs = [drtran.load_pre(WTI), drtran.load_pre(ES)]
    return drtran.build_cast_spec(specs, [drtran.Link(out=0, inp=1, b=0, r=0, s=1)])


def _table(cs, tmp_path, q):
    t = build_slots(cs)
    p = tmp_path / "q.cns"
    p.write_text(f"q[2,1] = {q}\n")
    read_cns(str(p), t)
    return t


def test_q_imposible_se_rechaza(tmp_path):
    cs = _cast()
    t = _table(cs, tmp_path, 1)
    with pytest.raises(drtran.StartNotAdmissible) as e:
        drtran.fit(cs, x0=drtran.x0_full(cs, t), embed=True, slots=t)
    msg = str(e.value)
    assert "not positive definite" in msg
    assert "RELATIVE TO SERIES 1" in msg
    assert "q[2,1] = 1" in msg and "<- outside" in msg


def test_q_admisible_no_avisa(tmp_path):
    cs = _cast()
    t = _table(cs, tmp_path, 0.01)
    assert start_q_message(drtran.x0_full(cs, t), cs, t) is None


def test_sin_tabla_no_comprueba():
    cs = _cast()
    assert start_q_message(drtran.x0_full(cs, build_slots(cs)), cs, None) is None
