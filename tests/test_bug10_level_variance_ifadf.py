"""BUG-10 — the level forecast's VARIANCE integrated with `d + D*s` and ignored
`ifadf`, and every series with the FIRST one's operator. integrated_weights now
takes one full operator per series (cast.differencing_poly, from fue), the same
source the mean and the common sample use."""
import numpy as np
from scipy.signal import lfilter

from drtran.forecast import error_variance, integrated_weights
from fue.forecast import _nonsop_coefs


def _op(d, D, s, ifadf=None):
    return np.r_[1.0, -np.asarray(_nonsop_coefs(d, D, s, ifadf=ifadf), float)]


def test_the_ifadf_factor_is_integrated():
    """psi* of white noise is the expansion of 1/delta(B): delta with the
    frequency factor (1 - 2cos(pi/6)B + B^2) of ifadf[1] = 1, s = 12."""
    delta = _op(1, 0, 12, [0, 1, 0, 0, 0, 0, 0])
    assert np.allclose(delta, [1, -2.732051, 2.732051, -1], atol=1e-6)
    L = 24
    psi = np.zeros((L + 1, 1, 1)); psi[0] = 1.0
    got = integrated_weights(psi, deltas=[delta])[:, 0, 0]
    want = lfilter([1.0], delta, np.r_[1.0, np.zeros(L)])
    assert np.allclose(got, want)
    old = error_variance(integrated_weights(psi, d=1, D=0, s=12), np.eye(1), L)[12, 0, 0]
    new = error_variance(integrated_weights(psi, deltas=[delta]), np.eye(1), L)[12, 0, 0]
    assert new > 20 * old          # the bands were ~4.6 times too narrow


def test_each_series_with_its_own_operator():
    """Output d = 1, input d = 0: the input's row is NOT integrated."""
    L = 6
    psi = np.zeros((L + 1, 2, 2)); psi[0] = np.eye(2)
    out = integrated_weights(psi, deltas=[_op(1, 0, 1), _op(0, 0, 1)])
    assert np.allclose(out[:, 0, 0], 1.0)           # random walk: all ones
    assert np.allclose(out[1:, 1, 1], 0.0)          # stationary: only psi_0


def test_the_old_signature_still_works():
    L = 5
    psi = np.zeros((L + 1, 1, 1)); psi[0] = 1.0
    assert np.allclose(integrated_weights(psi, d=1, D=0, s=1)[:, 0, 0], 1.0)
    assert np.allclose(integrated_weights(psi, deltas=_op(1, 0, 1))[:, 0, 0], 1.0)
