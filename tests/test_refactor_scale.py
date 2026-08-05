"""A KNOWN LIMITATION, pinned so nobody rediscovers it: at `refactor=1` the
optimiser reaches the optimum but reports the wrong reason for stopping.

This is a documentation test. It asserts the CURRENT behaviour — including the
part that is unsatisfactory — because that behaviour was studied at length on
2026-08-04/05 and deliberately left alone. The full study, its measurements and
what was tried and rejected, is in `drtran/docs/OPTIMIZER_STOPPING_STUDY.md`.
Read it before changing anything here.

The short version. `qnewtopt` fixes Dennis & Schnabel's typical parameter size
to 1 (`qnewtopt.c:185,208,215`, the book's simplified A9.4.1; the full algorithm
takes it as an input). At `refactor=1` the deterministic omegas are ~1e-4, so
the gradient test `|g|*(|x|+1)/(|f|+1)` degenerates into an ABSOLUTE tolerance
it cannot meet and the step test `|dx|/(|x|+1)` into an absolute one it meets at
once. The optimiser still finds the same optimum in the same number of
iterations; what it loses is the ability to CERTIFY it, so termcode falls from
1 (gradient) to 2 (step).

Making the tests relative to each parameter's own size fixes the termcode and
measurably COSTS convergence depth, which on ill-conditioned data breaks the
scale-invariance of the point estimates. It was tried, measured and reverted.
The optimiser is Mauricio's published work; it does not get changed without a
proven better alternative.

Practical consequence for a user: rescale to 100, which is the standing
guidance anyway (see M0.2 in `drtran/TODO.md`) and what fue now emits.
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

    mu and the deterministic omegas scale with the data; the AR/MA parameters
    and the transfer omegas are scale-invariant. So the ONLY thing that changes
    is the numerical scale of one class of parameter — which is what makes this
    a controlled experiment rather than two different fits.
    """
    s = drtran.load_pre(path)
    m = copy.deepcopy(s.model)
    r = factor / float(m.refactor)
    m.refactor = factor
    for iv in m.interventions:
        iv.omega = [v * r for v in iv.omega]
    m.mu0 = float(m.mu0) * r
    return PreSpec(ts=m.series, model=m, path=s.path)


def _fit_at(factor):
    cs = build_cast_spec([_at_scale(ES, factor), _at_scale(WTI, factor)],
                         links=[Link(0, 1, b=0, r=0, s=1)])
    return drtran.fit(cs, embed=True)


@pytest.fixture(scope="module")
def fits():
    return {"r100": _fit_at(100.0), "r1": _fit_at(1.0)}


def test_rescaling_is_the_same_model(fits):
    """The log-likelihoods must differ by EXACTLY the Jacobian, n*m*log(100).

    This is the control. If it fails, the two runs are not the same problem and
    nothing else in this file means anything.
    """
    jac = 215 * 2 * np.log(100.0)
    assert fits["r1"].loglik - jac == pytest.approx(fits["r100"].loglik,
                                                    abs=1e-6)


def test_the_optimum_is_reached_at_either_scale(fits):
    """The search is NOT what is affected: same transfer omegas to six decimals
    and the same iteration count at both scales."""
    np.testing.assert_allclose(fits["r1"].xfree[:2], fits["r100"].xfree[:2],
                               atol=1e-6)
    assert fits["r1"].nit == fits["r100"].nit


def test_the_certificate_is_what_degrades(fits):
    """THE LIMITATION. Same optimum, different stated reason for stopping.

    termcode 2 normally means "suspect an ill-conditioned likelihood"; here it
    means nothing of the sort — it is an artefact of the parameter scale. That
    is the whole finding, and it is why the guidance is to rescale to 100.
    """
    assert fits["r100"].termcode == 1, "well scaled: certified by gradient"
    assert fits["r1"].termcode == 2, "raw scale: falls through to the step test"


def test_the_optimiser_takes_no_scale_argument():
    """`fit` must NOT grow a knob for this.

    A `typx` parameter was implemented, measured and reverted: it fixes the
    termcode and costs convergence depth, breaking the scale-invariance of the
    point estimates on ill-conditioned data. See the study document. If this
    test starts failing, someone is re-treading it — read the study first.
    """
    import inspect
    assert "typx" not in inspect.signature(drtran.fit).parameters
