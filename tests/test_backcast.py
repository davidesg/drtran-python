"""The pre-sample: backcasting instead of zeroing it (BUG-8, step 7).

The subtracting cast's own documentation said it computed "the exact likelihood
of the WRONG series", because the convolution needs input values before t=1 and
it set them to zero. These pin the two things that make the replacement
trustworthy: the backcast is exact on cases with a closed form, and the
pre-sample reproduces real data when real data is what it should reproduce.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import (Link, backcast, build_cast_spec,  # noqa: E402
                         compute_irf, x0_from_pre, _pre_sample)

PT8 = "/home/david/Dropbox/SRC/atws/Taste/oracle/data/passthrough8"


# ── the backcast, against closed forms ───────────────────────────────────────
@pytest.mark.parametrize("phi", [0.5, -0.7, 0.9])
def test_an_ar1s_backcast_is_exactly_phi_to_the_l_times_w0(phi):
    """`E[w_{-l} | data] = phi^l w_0` for an AR(1). Exact, not approximate.

    This is the check that says the reversal is right: the time-reversed AR(1)
    has the same phi, so the l-step backcast is the l-step forecast.
    """
    rng = np.random.default_rng(7)
    w = np.zeros(500)
    for t in range(1, 500):
        w[t] = phi * w[t - 1] + rng.standard_normal()
    pre = backcast(w, [phi], [], 0.0, 5)
    assert pre == pytest.approx([phi ** (l + 1) * w[0] for l in range(5)])


def test_the_mean_is_carried_through():
    rng = np.random.default_rng(11)
    mu, phi = 3.0, 0.6
    w = np.full(400, mu)
    for t in range(1, 400):
        w[t] = mu + phi * (w[t - 1] - mu) + rng.standard_normal()
    pre = backcast(w, [phi], [], mu, 3)
    assert pre == pytest.approx([mu + phi ** (l + 1) * (w[0] - mu)
                                 for l in range(3)])


def test_an_ma1_remembers_exactly_one_period():
    """Beyond its order an MA's backcast is the mean, to the last bit."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(400)
    pre = backcast(a[1:] - 0.6 * a[:-1], [], [0.6], 0.0, 4)
    assert pre[0] != 0.0
    assert pre[1:] == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)


def test_nothing_is_asked_for_nothing_is_returned():
    assert len(backcast(np.arange(10.0), [0.5], [], 0.0, 0)) == 0


# ── the pre-sample, against real data ────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(PT8), reason="the oracle's data is missing")
def test_the_pre_sample_reproduces_the_observations_the_trim_dropped():
    """Shorten the window on purpose: the pre-sample must return the real values.

    The strongest check available, because it has a known answer that owes
    nothing to the model. It also exercises the re-differenced (`alt`) path,
    where the values are reconstructed through Delta from the input's own
    series -- an indexing error there is invisible in the estimates and obvious
    here. It caught one.
    """
    from fue.cast_us import cast_us_py

    y = drtran.load_pre(os.path.join(PT8, "PT8_FR.pre"))
    x = drtran.load_pre(os.path.join(PT8, "PT8_WTI.pre"))
    cs = build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 1)])
    assert cs.needs_subtracting and 0 in cs.alt_delta

    xv = x0_from_pre(cs)
    idx = cs.npar_links
    phis, thetas, mus, ws = [], [], [], []
    for sc in cs.series:
        xi = xv[idx:idx + sc.npar]
        idx += sc.npar
        p, q, ph, th, mu, w, _ = cast_us_py(xi, sc.est_spec)
        phis.append(np.asarray(ph)[:int(p)])
        thetas.append(np.asarray(th)[:int(q)])
        mus.append(float(mu))
        ws.append(np.asarray(w))
    *_, alt, _ = cast_us_py(xv[idx - cs.series[1].npar:idx], cs.alt_est[0])
    alt = np.asarray(alt)

    nu = compute_irf([1.0, 0.5], [0.7], 0, 300)          # long reach on purpose
    for drop in (3, 8):
        pre = _pre_sample(cs, 0, cs.links[0], alt, len(alt) - drop, nu,
                          phis, thetas, mus, ws)
        assert pre[:drop] == pytest.approx(alt[drop - 1::-1], abs=1e-12)


@pytest.mark.skipif(not os.path.exists(PT8), reason="the oracle's data is missing")
def test_a_contemporaneous_transfer_needs_no_pre_sample_at_all():
    """`s = r = b = 0`: one nu weight, so the convolution is complete from t=1.

    Worth pinning because it is why DE and EMU did not move when this landed,
    and why the C's `(0,0,0)` homologation value did not move either.
    """
    y = drtran.load_pre(os.path.join(PT8, "PT8_DE.pre"))
    x = drtran.load_pre(os.path.join(PT8, "PT8_WTI.pre"))
    cs = build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 0)])
    nu = compute_irf([0.01], [], 0, 200)
    pre = _pre_sample(cs, 0, cs.links[0], np.zeros(203), 203, nu,
                      [np.zeros(0)] * 2, [np.zeros(0)] * 2, [0.0, 0.0],
                      [np.zeros(203), np.zeros(215)])
    assert len(pre) == 0
