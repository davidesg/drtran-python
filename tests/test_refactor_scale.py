"""`refactor=1`: the optimiser reaches the optimum but cannot certify it.

The C's recorded symptom was a hang: the same model that converges in 23
iterations with `refactor=100` (Delta-log ~0.2) ran >2 min without converging with
`refactor=1` (Delta-log ~0.002).

The cause is NOT the finite-difference gradient. It is that `qnewtopt` fixes the
typical parameter size to 1 (`qnewtopt.c:185,208,215`), so at `refactor=1`, where
the deterministic omegas are ~1e-4:

* the gradient test `|g|*(|x|+1)/(|f|+1)` becomes an ABSOLUTE tolerance it cannot
  meet, and
* the step test `|Dx|/(|x|+1)` an absolute one it meets at once.

The optimiser still finds the same optimum in the same number of iterations; what
it loses is the ability to RECOGNISE it. On an easy problem that only degrades
the termcode 1 -> 2; on a large one it iterates to `maxits`, which is the hang.

Uniform rescaling leaves the model identical — the log-likelihood differs by
exactly the Jacobian — so any difference in the fit is an artefact of the
optimiser, which is what makes this testable.
"""
import copy
import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec  # noqa: E402
from drtran.pre import PreSpec  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


def _at_scale(path, factor):
    """The same model with the series scaled by `factor`.

    mu and the deterministic omegas scale with the data; the AR/MA parameters and
    the transfer omegas are scale-invariant. So the ONLY thing that changes is the
    numerical scale of one class of parameter.
    """
    s = drtran.load_pre(path)
    m = copy.deepcopy(s.model)
    r = factor / float(m.refactor)
    m.refactor = factor
    for iv in m.interventions:
        iv.omega = [v * r for v in iv.omega]
    m.mu0 = float(m.mu0) * r
    return PreSpec(ts=m.series, model=m, path=s.path)


def _fit_at(factor, typx):
    cs = build_cast_spec([_at_scale(ES, factor), _at_scale(WTI, factor)],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return drtran.fit(cs, embed=True, typx=typx)


@pytest.fixture(scope="module")
def fits():
    return {"r100": _fit_at(100.0, None),
            "r1_c": _fit_at(1.0, None),
            "r1_fix": _fit_at(1.0, 1e-3)}


def test_rescaling_is_the_same_model(fits):
    """The log-likelihoods must differ by EXACTLY the Jacobian, n*m*log(100).

    If they do not, the two runs are not the same problem and nothing else here
    means anything.
    """
    jac = 215 * 2 * np.log(100.0)
    assert fits["r1_c"].loglik - jac == pytest.approx(fits["r100"].loglik, abs=1e-6)
    assert fits["r1_fix"].loglik - jac == pytest.approx(fits["r100"].loglik, abs=1e-6)


def test_the_optimum_is_reached_at_either_scale(fits):
    """The defect is in the certificate, not in the search: the transfer omegas
    agree to six decimals whatever the scale and whatever typx."""
    ref = np.asarray(fits["r100"].xfree[:2])
    np.testing.assert_allclose(fits["r1_c"].xfree[:2], ref, atol=1e-6)
    np.testing.assert_allclose(fits["r1_fix"].xfree[:2], ref, atol=1e-6)
    assert fits["r1_c"].nit == fits["r100"].nit


def test_typx_recovers_the_gradient_certificate(fits):
    """The defect and its fix, in one assertion each."""
    assert fits["r100"].termcode == 1, "well-scaled: certified by gradient"
    assert fits["r1_c"].termcode == 2, "the C's behaviour: falls to the step test"
    assert fits["r1_fix"].termcode == 1, "with typx: certified by gradient again"


def test_fit_defaults_to_the_fix():
    """`drtran.fit` must ship with typx on; the C is reachable via typx=None."""
    import inspect
    assert inspect.signature(drtran.fit).parameters["typx"].default == 1e-3
