"""BUG-87: con mu fijada y d + D > 0, una media residual significativa se
avisa (como el C tras diagnose()); con mu libre, no."""
import os
import re
import warnings

import drtran
from drtran.school import fixed_mu_drift, report_fixed_mu_drift

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "p07")
WTI = os.path.join(DATOS, "wti_ar1_clase.pre")
ES = os.path.join(DATOS, "es_cpi_m10_clase.pre")


def _fit(es_path):
    warnings.filterwarnings("ignore")
    specs = [drtran.load_pre(es_path), drtran.load_pre(WTI)]
    cs = drtran.build_cast_spec(specs, [drtran.Link(out=0, inp=1, b=0, r=0, s=1)])
    return drtran.fit(cs, embed=True)


def test_mu_fijada_avisa(tmp_path):
    txt = open(ES).read()
    txt = re.sub(r"(\*\* Mean parameter \(mu\):\n)\S+ 1\n", r"\g<1>0.000000 0\n", txt)
    p = tmp_path / "ES_mu0.pre"
    p.write_text(txt)
    w = fixed_mu_drift(_fit(str(p)))
    assert len(w) == 1 and w[0]["series"] == 0 and w[0]["dd"] == 1
    assert abs(w[0]["mean"] - 0.0668886) < 1e-5 and abs(w[0]["t"] - 4.80) < 0.01
    assert "the residual mean is significant" in report_fixed_mu_drift(w)


def test_mu_libre_no_avisa():
    assert fixed_mu_drift(_fit(ES)) == []
