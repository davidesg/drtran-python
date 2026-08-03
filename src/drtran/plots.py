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

    The drawing is `drvarma.plots.plot_ccf`: impulse bars, dashed +/-2/sqrt(N)
    bands, seasonal dividers and the Hosking Q label, exactly as drvus drew it.
    Reusing it means a CCF looks the same in `art`, `mtram` and `sima`, and an
    analyst reads all three the same way.

    **The arguments are swapped on purpose.** drvarma and drtran use OPPOSITE
    lag-sign conventions: drvarma's k=+1 is drtran's k=-1. Passing
    `(a_input, beta_output)` would draw the CCF MIRRORED, putting the transfer on
    the feedback side — a plot that looks perfectly normal and says the opposite
    of the truth. Verified on the canonical case: swapped, the two agree to six
    decimals on both sides (k=0..3 and k=-1..-3).
    """
    from drvarma.plots import plot_ccf as _dv_plot_ccf

    return _dv_plot_ccf(beta_output, a_input, lags=lags, freq=freq,
                        names=(names[0], names[1]), ax=ax)


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


def plot_forecast(level, lower, upper, history=None, name="", n_hist=24,
                  title=None):
    """History and forecast in LEVELS, with the band drawn as it really is.

    The band comes from `level_band`, which forms it in the transformed scale and
    maps it back, so under a log model it is **asymmetric**. Drawing a symmetric
    ribbon around the point forecast would be a different, wider and wrong
    picture — on the canonical case ±0.47 where the truth is +0.39/−0.39 either
    side of a level that is not the centre.
    """
    plt = _mpl()
    level = np.asarray(level, float)
    L = len(level)
    fig, ax = plt.subplots(figsize=(9, 4))

    if history is not None and len(history):
        h = np.asarray(history, float)[-n_hist:]
        xh = np.arange(-len(h), 0)
        ax.plot(xh, h, "-", lw=1.2, color="#374151", label="observed")
        ax.plot([xh[-1], 0], [h[-1], level[0]], "-", lw=1.2, color="#374151")

    xf = np.arange(L)
    ax.fill_between(xf, lower, upper, color="#dbeafe", label="95 % band")
    ax.plot(xf, level, "o-", ms=4, lw=1.4, color="#1d4ed8", label="forecast")
    ax.axvline(-0.5, ls=":", lw=1.0, color="#6b7280")

    ax.set_xlabel("horizon (0 = first forecast)")
    ax.set_ylabel("level")
    ax.set_title(title or f"Forecast — {name}")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def plot_residuals(residuals, npar=0, freq=1, lags=None, title=""):
    """Residual series + ACF/PACF — **fue's panel**, not a second one.

    `fue.plots.plot_acf_pacf` in the Treadway-Jenkins design: impulse style,
    shared y-range, +/-2/sqrt(n) bands, seasonal grid lines and the Ljung-Box Q
    in the ACF's xlabel. Reused so that a residual panel looks the same after a
    univariate fit in `art` and after a joint one here — the analyst is reading
    the same instrument in both, and should not have to re-learn it.

    `npar` is the number of estimated parameters, which is what the Q's degrees
    of freedom are corrected by. Passing 0 overstates the fit's adequacy.
    """
    plt = _mpl()
    from fue.plots import plot_acf_pacf, plot_residuals_ts

    import matplotlib.gridspec as gridspec
    r = np.asarray(residuals, float)
    fig = plt.figure(figsize=(12, 5.5), layout="constrained")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.6, 1.0],
                           hspace=0.06, wspace=0.05)
    ax_ser = fig.add_subplot(gs[:, 0])
    plot_residuals_ts(r, title=title or "residuals", ax=ax_ser)
    plot_acf_pacf(r, npar=npar, freq=freq, lags=lags,
                  ax_acf=fig.add_subplot(gs[0, 1]),
                  ax_pacf=fig.add_subplot(gs[1, 1]))
    return fig


def save(fig, path, dpi=130):
    """Write the figure and close it. Returns the path."""
    plt = _mpl()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
