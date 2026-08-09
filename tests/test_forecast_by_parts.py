"""The subtracting route's own forecast: the parts apart, joined on LEVELS.

TASTE's procedure (`TFFO.PAS`), and it is a different one from the embedded
cast's rather than the same one with a correction — see
`docs/LEVEL_TRANSFER_PLAN.md`. With the transfer embedded, forecasting the VARMA
forecasts everything; with it subtracted, the engine only ever saw the noise, so
the transfer has to be put back and the inputs forecast too.
"""

import os

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import Link, build_cast_spec, cast_diagonal, x0_from_pre  # noqa: E402
from drtran.estimate import fit  # noqa: E402
from drtran.forecast import forecast, forecast_by_parts, to_level  # noqa: E402

PT8 = "/home/david/Dropbox/SRC/atws/Taste/oracle/data/passthrough8"

pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.skipif(not os.path.exists(PT8),
                       reason="the oracle's passthrough data is missing"),
]


def _pair(name, s):
    y = drtran.load_pre(os.path.join(PT8, "PT8_%s.pre" % name))
    x = drtran.load_pre(os.path.join(PT8, "PT8_WTI.pre"))
    return build_cast_spec([y, x], links=[Link(0, 1, 0, 0, s)])


# ── the bridge: with no transfer it must BE the univariate forecast ──────────
def test_with_omega_zero_it_is_exactly_the_univariate_forecast():
    """The structural check, and it is exact rather than close.

    With omega = 0 the noise IS the output, so joining the parts must return
    the output's own univariate forecast — every deterministic, every
    difference, the Box-Cox and its refactor included. Anything wrong anywhere
    in the recombination shows up here as a number that is nearly right.
    """
    y = drtran.load_pre(os.path.join(PT8, "PT8_FR.pre"))
    x = drtran.load_pre(os.path.join(PT8, "PT8_WTI.pre"))
    cs0 = build_cast_spec([y, x])                       # diagonal, no links
    cs1 = build_cast_spec([y, x], links=[Link(0, 1, 0, 0, 1)])

    lvl, _z, _v = forecast_by_parts(x0_from_pre(cs1), cs1, L=12)
    uni = to_level(forecast(x0_from_pre(cs0), cs0, L=12, embed=False), cs0, 0)
    assert lvl == pytest.approx(uni, abs=1e-12)


# ── it is the route a dispatched model actually takes ────────────────────────
def test_a_dispatched_model_takes_this_route():
    disp = _pair("FR", 1)
    assert disp.needs_subtracting
    r = fit(disp)
    fc = forecast(r.x, disp, L=6, embed=r.embed)
    assert to_level(fc, disp, 0) == pytest.approx(
        forecast_by_parts(r.x, disp, L=6)[0], abs=1e-10)


def test_on_a_MATCHED_model_the_two_routes_give_the_same_number():
    """And exactly, not approximately — which is the whole reason to trust this.

    It is the same commutation that made the two CASTS agree when the operators
    match: differencing is linear and commutes with nu(B), so joining the parts
    on the levels and integrating the embedded VARMA's forecast are the same
    arithmetic. The new route therefore agrees with the homologated one
    everywhere both apply, and can only differ where the old one was wrong.
    """
    match = _pair("ES", 1)
    assert not match.needs_subtracting
    r = fit(match)
    fc = forecast(r.x, match, L=6, embed=r.embed)
    assert to_level(fc, match, 0) == pytest.approx(
        forecast_by_parts(r.x, match, L=6)[0], abs=1e-9)


# ── the bands are calibrated, which is what says the variance is right ───────
@pytest.mark.parametrize("pais, s", [("FR", 1), ("DE", 0), ("EMU", 0)])
def test_the_reported_one_step_band_matches_the_one_it_achieves(pais, s):
    """`se(1)` against the RMSE of forty real one-step errors.

    A point forecast can be right with a badly scaled variance and nobody
    notices, so this is pinned separately. It also guards the specific mistake
    made while writing it: `cast_diagonal` returns Q, not Sigma — the
    likelihood is concentrated and sigma2 comes out apart — and reading one as
    the other gave a band six times too wide with the point forecast intact.
    """
    cs = _pair(pais, s)
    r = fit(cs)
    dat = np.asarray(cs.series[0].spec.model.series.data, float)
    nobs = len(dat)
    off = nobs - cast_diagonal(r.x, cs)[3].shape[0]
    err = [dat[t] - forecast_by_parts(r.x, cs, L=1, origin=t)[0][0]
           for t in range(nobs - 40, nobs)]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    se = float(np.sqrt(forecast(r.x, cs, L=1, embed=r.embed).var_level[1, 0, 0]))
    assert 0.75 < rmse / se < 1.35, f"{pais}: RMSE {rmse:.5f} vs se {se:.5f}"
    assert off > 0


# ── and it forecasts better than the route built for the other cast ──────────
@pytest.mark.parametrize("pais, s", [("FR", 1), ("DE", 0), ("EMU", 0)])
def test_it_beats_putting_the_transfer_back_with_the_wrong_vector(pais, s):
    """Measured: 6.6 % to 20.0 % lower one-step level RMSE.

    `_system_forecast` puts the subtracted transfer back using the input's OWN
    differenced column, which stopped being the vector nu is fitted against the
    moment the dispatch landed. The difference between the two contributions is
    0.47-0.69 residual standard deviations, so this is not a rounding question.
    """
    cs = _pair(pais, s)
    r = fit(cs)
    dat = np.asarray(cs.series[0].spec.model.series.data, float)
    nobs = len(dat)
    off = nobs - cast_diagonal(r.x, cs)[3].shape[0]

    old, new = [], []
    for t in range(nobs - 40, nobs):
        fc = forecast(r.x, cs, L=1, origin=t - off, embed=r.embed)
        # the old route, reached explicitly: `to_level` now dispatches
        saved = cs.alt_est
        try:
            cs.alt_est = {}                     # look matched to `to_level`
            old.append(dat[t] - to_level(fc, cs, 0, origin=t)[0])
        finally:
            cs.alt_est = saved
        new.append(dat[t] - forecast_by_parts(r.x, cs, L=1, origin=t)[0][0])

    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))   # noqa: E731
    assert rms(new) < rms(old), f"{pais}: {rms(new):.5f} vs {rms(old):.5f}"


def test_the_origin_is_in_data_indices_like_to_level():
    """Not in stationary ones, which is what `forecast_mean` wants.

    They differ by the operator's order — thirteen for a monthly `∇∇₁₂` — and
    passing one where the other belongs is silent. It was a real bug here.
    """
    cs = _pair("FR", 1)
    r = fit(cs)
    nobs = cs.series[0].spec.model.series.nobs
    nst = cast_diagonal(r.x, cs)[3].shape[0]
    assert nobs - nst == 13
    lvl, _z, _v = forecast_by_parts(r.x, cs, L=1, origin=nobs)   # the last one
    assert np.isfinite(lvl).all()
    with pytest.raises(ValueError):
        forecast_by_parts(r.x, cs, L=1, origin=nobs + 5)         # past the end
