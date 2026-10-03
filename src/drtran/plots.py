"""The three plots a transfer model is decided with.

Not decoration. In the guided protocol the analyst reads these and answers; the
numbers alone do not carry the shape of the evidence:

* **the prewhitened CCF** is the identification instrument. `(b, r, s)` is read
  off it — where the first significant bar sits, how many follow, whether the
  tail decays. And the LEFT half is the exogeneity check: bars there mean the
  output leads the input, which a single-input transfer model does not allow.
  This one is **drvarma's plot**, reused rather than rewritten, so a CCF looks
  the same across the suite — see `plot_ccf` for the sign-convention trap that
  reuse hides.
* **the impulse response** with its bands is the answer the analyst came for,
  and the bands are what says whether the answer is worth anything.
* **the forecast** in levels, with its band, which is asymmetric under a log
  model and therefore cannot be drawn as a symmetric ribbon.

Matplotlib is an optional dependency (`drtran[plots]`); each function imports it
so that importing `drtran` never requires it.
"""

from __future__ import annotations

import numpy as np


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:                                    # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install 'drtran[plots]'")


def plot_ccf(a_input, beta_output, freq=12, names=("X", "Y"), lags=None,
             ax=None):
    """The prewhitened CCF — **drvarma's canonical plot**, not a second one.

    Right of zero (k > 0): the input leads, which is the transfer, and `b` is the
    first significant bar. Left (k < 0): the OUTPUT leads — feedback, which a
    single-input transfer model assumes away. Bars on both sides mean the
    specification does not hold, and this plot is where that is seen first.

    The drawing is `drvarma.plots.plot_ccf`: GraphMaker's CCF, the one
    Treadway approved and drtran's C GUI draws (atsw-gui `lib/ccfplot`) — bars,
    dotted +/-2/sqrt(N) bands, a dashed vertical at lag 0, the title
    "input - output" above and Hosking's portmanteau below as GraphMaker
    labels it, P (not Ljung-Box's Q). Reusing it means a CCF looks the same in
    `mtram` and `sima`, and an analyst reads both the same way.

    **The arguments are swapped on purpose.** drvarma and drtran use OPPOSITE
    lag-sign conventions: drvarma's k=+1 is drtran's k=-1. Passing
    `(a_input, beta_output)` would draw the CCF MIRRORED, putting the transfer on
    the feedback side — a plot that looks perfectly normal and says the opposite
    of the truth. Verified on the canonical case: swapped, the two agree to six
    decimals on both sides (k=0..3 and k=-1..-3).
    """
    from drvarma.plots import plot_ccf as _dv_plot_ccf

    # drvarma's names go with (w1, w2) = (output, input) and its title puts w2,
    # the series leading at k > 0, first: "input - output", as GraphMaker.
    return _dv_plot_ccf(beta_output, a_input, lags=lags, freq=freq,
                        names=(names[1], names[0]), ax=ax)


def prewhitened_pair(cast_spec, link, x=None):
    """`(a_input, beta_output)`: the input prewhitened by its own ARMA, and the
    output filtered by THE SAME filter — the pair `plot_ccf` draws.

    Filtering only the input leaves the CCF contaminated by the output's own
    structure; this is the step that is usually forgotten.
    """
    import numpy as _np
    from fue.cast_us import cast_us_py

    from .cast import x0_from_pre
    from .identify import prewhiten

    x = _np.asarray(x0_from_pre(cast_spec) if x is None else x, float)
    idx = cast_spec.npar_links
    pieces = []
    for sc in cast_spec.series:
        pieces.append(x[idx:idx + sc.npar]); idx += sc.npar

    def prep(i):
        p, q, phi, theta, _mu, w, ifa = cast_us_py(pieces[i],
                                                   cast_spec.series[i].est_spec)
        return (_np.asarray(phi, float)[:p], _np.asarray(theta, float)[:q],
                _np.asarray(w, float), int(ifa))

    phi_x, theta_x, w_x, if_x = prep(link.inp)
    _p, _t, w_y, if_y = prep(link.out)
    if if_x or if_y:
        raise ValueError("the univariate cast failed; check the .pre files")
    n = min(len(w_x), len(w_y))
    w_x, w_y = w_x[len(w_x) - n:], w_y[len(w_y) - n:]
    return prewhiten(w_x, phi_x, theta_x), prewhiten(w_y, phi_x, theta_x)


def plot_irf(ir, title=None):
    """The impulse response and its cumulative sum, each with a 95 % band.

    Left: the response to a ONE-OFF unit shock. Right: to a PERMANENT one, which
    converges to the gain — drawn as a horizontal line, because that convergence
    is the thing to look at.
    """
    plt = _mpl()
    k = np.arange(len(ir.nu))
    has_se = ir.se is not None and np.all(np.isfinite(ir.se))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))

    a1.axhline(0, lw=0.8, color="black")
    if has_se:
        a1.vlines(k, ir.nu - 1.96 * ir.se, ir.nu + 1.96 * ir.se,
                  color="#93c5fd", lw=4)
    a1.plot(k, ir.nu, "o-", ms=4, color="#1d4ed8")
    a1.set_title("one-off shock:  nu(k)")
    a1.set_xlabel("k")

    a2.axhline(0, lw=0.8, color="black")
    if has_se:
        a2.fill_between(k, ir.cum - 1.96 * ir.se_cum, ir.cum + 1.96 * ir.se_cum,
                        color="#bbf7d0")
    a2.plot(k, ir.cum, "o-", ms=4, color="#15803d")
    if ir.gain == ir.gain:
        a2.axhline(ir.gain, ls="--", lw=1.0, color="#b91c1c")
        a2.annotate(f"gain {ir.gain:.6f}", (k[-1], ir.gain),
                    textcoords="offset points", xytext=(-6, 6),
                    ha="right", color="#b91c1c", fontsize=9)
    a2.set_title("permanent change:  cumulative")
    a2.set_xlabel("k")

    for ax in (a1, a2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(title or f"Impulse response — {ir.out_name} <- {ir.inp_name}"
                          f"   (b={ir.b}, r={ir.r}, s={ir.s})")
    fig.tight_layout()
    return fig


def plot_forecast(fit, cast_spec, series=0, horizon=12, origin=None):
    """fuf's forecast graph for one series of the joint model, drawn by pyfug.

    The same figure fuf, art and sima draw (`pyfug.plot_forecast`, fufplot.c):
    the last `horizon` observations and the forecasts of the series' annual
    change (a rate in % under a log model; the level with ±2σ bands when
    λ < 0), and below, ERR, the residuals of those observations. The data come
    from `build_forecast_result`, the one drtran's fuf report uses, through
    `fue.forecast.forecast_graph_data`.

    It used to be a figure of its own: the level and its band over the
    horizon. pyfug is the one graphics engine of the ladder.
    """
    _mpl()
    from fue.forecast import forecast_graph_data
    from pyfug.graphics import plot_forecast as _pyfug_forecast

    from .report import build_forecast_result

    fr, model = build_forecast_result(fit, cast_spec, series=series,
                                      horizon=horizon, origin=origin)
    return _pyfug_forecast(**forecast_graph_data(model, fr))


def plot_residuals(residuals, npar=0, freq=1, lags=None, title="",
                   start=None, nobs=None, refactor=1.0):
    """Residual series + ACF/PACF — the figure of fug -c, drawn by pyfug.

    The same figure art shows after a univariate fit (`pyfug.plot_combined`,
    with fug C's geometry), so the analyst reads one instrument, not two. It
    used to be assembled here from pieces of `fue.plots`; pyfug is the one
    graphics engine.

    `npar` is the number of estimated parameters, which is what the Q's degrees
    of freedom are corrected by. Passing 0 overstates the fit's adequacy.
    `start` is the (year, period) of the series' FIRST observation and `nobs`
    its length: the residuals are its last len(residuals) observations, and
    the rest go in `timeout`, so the year axis starts where fug C starts it.
    `refactor` turns the residuals into fractions, so the statistics read in %.
    """
    _mpl()
    from pyfug.core import Tseries
    from pyfug.graphics import plot_combined

    r = np.asarray(residuals, float) / float(refactor or 1.0)
    lost = max(0, int(nobs) - len(r)) if nobs else 0
    y0, p0 = (start if start else (1, 1))
    serie = Tseries(name=title or "residuals", freq=int(freq or 1), nobs=len(r),
                    begyear=int(y0), begtime=int(p0), data=r)
    return plot_combined(serie, npar=int(npar), timeout=lost,
                         tsnobs=len(r) + lost, nlags=int(lags or 0),
                         title=serie.name)


def save(fig, path, dpi=130):
    """Write the figure and close it. Returns the path."""
    plt = _mpl()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
