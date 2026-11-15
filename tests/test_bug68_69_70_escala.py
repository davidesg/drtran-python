"""BUG-68, 69 and 70: the scale check, the R2's Box-Cox and the units mtram
presents in. Each repro of scripts/ is a self-contained case on synthetic
series; it returns 0 when the defect is gone."""
import importlib.util
import pathlib

import pytest

pytest.importorskip("mcp")
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("nombre", [
    "repro_check_scale_mide_el_nivel",              # BUG-68
    "repro_stationary_series_boxcox",               # BUG-69
    "repro_presentacion_en_unidades_del_motor",     # BUG-70
])
def test_el_repro_sale_con_cero(nombre):
    spec = importlib.util.spec_from_file_location(nombre, SCRIPTS / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main() == 0


def test_the_norm_is_not_warned_of(tmp_path):
    """Two log series at refactor = 100 with a mean, d = 0: the suite's norm.
    The check called them "huge"; the same at refactor = 1 passed (BUG-68)."""
    import numpy as np
    import fue
    from drtran.pre import check_scale, load_pre, mixed_factors

    rng = np.random.default_rng(11)
    e = np.zeros(160)
    for t in range(1, 160):
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.01)
    y = 100 * np.exp(e)
    specs = {}
    for f in (100.0, 1.0):
        ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name="Y")
        m = fue.Model(ts, d=0, D=0, boxlam=0.0, ar=[[0.1]], ar_free=[[True]],
                      mu=f * float(np.log(y).mean()), estimate_mu=True, refactor=f)
        m.fit()
        p = tmp_path / f"y{f:g}.pre"
        m.write_pre(str(p))
        specs[f] = load_pre(str(p))
    assert check_scale(specs[100.0]) is None
    assert "refactor=100 " in check_scale(specs[1.0])
    assert mixed_factors([specs[100.0], specs[100.0]]) is None
    assert "do not share" in mixed_factors([specs[100.0], specs[1.0]])
