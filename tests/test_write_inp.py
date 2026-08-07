"""Writing the re-estimated univariate blocks back out.

fue leaves a `.pre`, drtran reads it as a seed and re-estimates everything
together — and the univariate blocks move while it does. This is how those
estimates leave the program.

The tests pin what it IS and, just as deliberately, what it is NOT: the written
`.pre` is a worse starting point for the diagonal than the original, and has to
be, because the diagonal's optimum is fue's separate estimates by definition.
That was the framing this feature was requested under, and it was wrong; the test
below is what keeps it from being believed again.
"""

import os
import shutil

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (Link, build_cast_spec, loglik_diagonal,  # noqa: E402
                         x0_from_pre)
from drtran.estimate import fit, loglik, standard_errors  # noqa: E402
from drtran.pre import next_inp_path, write_inp  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")
LINK = [Link(0, 1, b=0, r=0, s=1)]

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    d = tmp_path_factory.mktemp("rt")
    a, b = str(d / "ES_CPI_m10.pre"), str(d / "WTI_ar1.pre")
    shutil.copy(ES, a)
    shutil.copy(WTI, b)
    cs = build_cast_spec([drtran.load_pre(a), drtran.load_pre(b)], links=LINK)
    f = fit(cs, embed=True)
    se = standard_errors(f)
    out = [write_inp(f, series=i, path=next_inp_path(p), std_errors=se)
           for i, p in enumerate((a, b))]
    return (a, b), out, f


def test_the_written_pre_carries_the_JOINT_estimates(written):
    """Not the seeds it started from. On the canonical case the two `omega_d1`
    move to -0.040867 and -0.094588 when the transfer is fitted beside them."""
    (a, _b), out, f = written
    back = drtran.load_pre(out[0])
    assert back.name == "ES_CPI" and back.nobs == 216

    iv = back.model.interventions
    assert float(iv[0].omega[0]) == pytest.approx(-0.040867, abs=1e-5)
    assert float(iv[1].omega[0]) == pytest.approx(-0.094588, abs=1e-5)
    assert float(back.model.ar[0][0]) == pytest.approx(0.295207, abs=1e-5)
    assert float(back.model.mu0) == pytest.approx(0.140319, abs=1e-5)

    original = drtran.load_pre(a)
    assert float(original.model.interventions[0].omega[0]) != pytest.approx(
        float(iv[0].omega[0]), abs=1e-4), "it must differ from the seed"


def test_it_never_overwrites_its_input(written):
    (a, b), out, _f = written
    assert os.path.abspath(out[0]) != os.path.abspath(a)
    assert os.path.abspath(out[1]) != os.path.abspath(b)
    assert out[0].endswith("ES_CPI_m10.1.inp")
    assert next_inp_path("x/ES_CPI_m10.pre") == "x/ES_CPI_m10.1.inp"
    assert next_inp_path("ES_CPI_m10.pre", 3) == "./ES_CPI_m10.3.inp"


def test_the_cycle_reaches_the_same_optimum(written):
    """Re-estimating from the written files lands where the first run landed.
    That is the round trip closing: nothing was lost on the way out."""
    _paths, out, f = written
    cs2 = build_cast_spec([drtran.load_pre(out[0]), drtran.load_pre(out[1])],
                          links=LINK)
    f2 = fit(cs2, embed=True)
    assert f2.loglik == pytest.approx(f.loglik, abs=1e-5)
    assert f2.loglik == pytest.approx(-718.287406, abs=1e-4)


def test_the_written_pre_is_a_WORSE_diagonal_start_and_must_be(written):
    """The correction this feature was built on.

    The blocks written are optimal WITH the transfer. Evaluated on the diagonal
    they are not, and cannot be: the diagonal's optimum IS fue's separate
    estimates — that is the gate the whole port rests on. Measured: -772.840628
    against -767.424341.

    So "re-start the diagonal from the best point" is not what this does, and the
    assertion is here so the claim cannot creep back into the documentation.
    """
    _paths, out, _f = written
    cs_new = build_cast_spec([drtran.load_pre(out[0]), drtran.load_pre(out[1])])
    cs_old = build_cast_spec([drtran.load_pre(ES), drtran.load_pre(WTI)])

    ll_new, _ = loglik_diagonal(x0_from_pre(cs_new), cs_new)
    ll_old, _ = loglik_diagonal(x0_from_pre(cs_old), cs_old)

    assert ll_old == pytest.approx(-767.424341, abs=1e-4)
    assert ll_new == pytest.approx(-772.840628, abs=1e-3)
    assert ll_new < ll_old, "the written seeds are a WORSE diagonal start"


def test_it_refuses_without_standard_errors(written):
    """The format carries them, so there is nothing to write without them."""
    from drtran.estimate import StdErrors
    import numpy as np

    _paths, _out, f = written
    bad = StdErrors(cov=np.zeros((1, 1)), se=np.array([np.nan]),
                    se_of_slot=np.array([np.nan]), t=np.array([np.nan]),
                    p=np.array([np.nan]), ifault=2)
    with pytest.raises(RuntimeError, match="standard errors are not available"):
        write_inp(f, series=0, path="/tmp/should_not_appear.inp",
                  std_errors=bad)
    assert not os.path.exists("/tmp/should_not_appear.inp")


def test_what_is_written_is_a_specification_not_an_optimum(tmp_path):
    """Why the extension changed, stated as the invariant it turns on.

    A `.pre` claims to be the optimum of its own specification, and that claim
    is testable: run fue on it and the numbers do not move. What drtran writes
    back fails the test — necessarily, since these blocks are optimal WITH the
    transfer beside them and the univariate optimum is by definition fue's
    separate estimate.

    So the file is a perfectly good STARTING POINT and a false `.pre`. It
    carries no mark of who wrote it, so the wrong extension would make it
    indistinguishable downstream from an optimum fue had certified — and the
    ladder climbs by trusting exactly that.
    """
    import shutil
    import subprocess
    import sys

    import numpy as np

    import drtran
    from drtran.cast import Link, build_cast_spec
    from drtran.estimate import standard_errors
    from drtran.pre import load_pre, write_inp
    from drtran.slots import build_slots

    specs = [load_pre(ES), load_pre(WTI)]
    cs = build_cast_spec(specs, links=[Link(0, 1, 0, 0, 1)])
    table = build_slots(cs)
    f = drtran.fit(cs, x0=drtran.x0_full(cs, table), embed=True, slots=table)
    tgt = str(tmp_path / "block.inp")
    write_inp(f, series=0, path=tgt, std_errors=standard_errors(f))
    assert tgt.endswith(".inp")

    def vals(p):
        m = load_pre(p).model
        return np.array([o for it in m.interventions for o in it.omega]
                        + [c for fa in m.ma for c in fa], float)

    antes = vals(tgt)
    r = subprocess.run([sys.executable, "-m", "fue", "block"],
                       cwd=str(tmp_path), capture_output=True, timeout=600)
    pre = tmp_path / "block.pre"
    if not pre.exists():                       # fue no disponible en este entorno
        import pytest as _pt
        _pt.skip(f"fue no produjo el .pre: {r.stderr[-200:]!r}")
    movido = float(np.max(np.abs(antes - vals(str(pre)))))
    assert movido > 1e-6, (
        "si esto NO se moviera, el fichero SÍ sería un .pre y el cambio de "
        f"extensión estaría de más; se movió {movido:.6f}")
