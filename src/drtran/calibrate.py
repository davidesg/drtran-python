"""Which observations are bending the instruments.

ART's `diagnose_interventions` does more than flag extreme residuals: it measures
**how much each one distorts the ACF/PACF, the JB and the Q**. That is the idea
worth carrying over, because a transfer model is identified and judged with
instruments too — the CCF and two portmanteaus — and they bend the same way.

The instruments differ, so the questions do:

===================  =========================================================
ART                  which observation distorts the **ACF/PACF** and the Q?
mtram                which observation distorts the **CCF** I read (b, r, s)
                     from, and the **adequacy portmanteau** I judge with?
===================  =========================================================

**Why this is not ART's scan run again.** ART calibrates on the *univariate*
rung, and its interventions arrive here already in the `.pre` as deterministics.
Re-running it would be worse than redundant: an anomaly in the output's
univariate residuals may be **explained by the input** once the transfer is in
the model, so carrying it in as an intervention attributes to a dummy what the
transfer explains — double counting, and it biases omega toward zero. What
survives the JOINT fit is the genuine intervention.

**The method is leave-one-out, not a contribution heuristic.** Recomputing the
CCF and the portmanteau without observation *t* answers the question the analyst
actually has — *would my conclusion change?* — instead of a proxy for it. On a
215-observation sample it costs nothing.

Which matters most at node **N6**. When adequacy fails there are two causes and
they need opposite responses:

* the **shape** is wrong  → re-identify (b, r, s);
* **one observation** is  → an intervention, and re-specifying the shape around
  it is how a model acquires a lag nobody can interpret.

Telling them apart is what this module is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .diagnose import _p_chi, chi_test
from .identify import ccf


@dataclass
class Anomaly:
    """One observation, and what it does to the instruments."""

    index: int                     # 0-based, into the residual array
    obs: int                       # 1-based observation number in the series
    date: str = ""
    z: float = 0.0                 # standardised structural residual
    variance_fraction: float = 0.0     # z^2 / sum z^2
    p_transfer_without: float = float("nan")   # adequacy p, leaving it out
    ccf_lags_affected: list = field(default_factory=list)
    # THE GLOBAL EFFECT, which is the primary one: by inflating the variance an
    # anomaly divides EVERY coefficient, not only the lags it touches.
    compression: float = 1.0       # mean|r(k)| without it / with it
    ccf_max: float = 0.0           # max|r(k)| with it
    ccf_max_without: float = 0.0   # and without
    ccf_without: object = None     # the whole CCF without it, to plot

    def __repr__(self):                                    # pragma: no cover
        return (f"Anomaly({self.date or self.obs}, z={self.z:+.2f}, "
                f"var={100*self.variance_fraction:.1f}%)")


@dataclass
class Calibration:
    """What the leave-one-out sweep found."""

    p_transfer: float = float("nan")       # with every observation in
    p_exog: float = float("nan")
    n: int = 0
    threshold: float = 3.5
    anomalies: list = field(default_factory=list)
    decisive: list = field(default_factory=list)   # those that flip the verdict
    names: tuple = ("X", "Y")
    ccf: object = None             # the CCF with every observation in

    @property
    def verdict(self):
        """`"shape"`, `"observation"` or `"adequate"` — the N6 branch to take."""
        if self.p_transfer >= 0.05:
            return "adequate"
        return "observation" if self.decisive else "shape"


def _date_of(model, obs):
    try:
        y, p = model.series._obs_to_date(obs)
    except Exception:                                      # pragma: no cover
        return str(obs)
    freq = int(getattr(model.series, "freq", 1) or 1)
    return f"{p:02d}/{y}" if freq > 1 else str(y)


def calibrate(fit, link_index=0, threshold=3.5, nlags=None,
              contrib_threshold=0.05):
    """Leave-one-out calibration of the CCF and the adequacy portmanteau.

    Returns a `Calibration`. An observation is **decisive** when dropping it
    moves the adequacy p-value across 0.05 — i.e. when the verdict on the
    transfer rests on that single point.
    """
    from .netid import residuals

    cs = fit.cast_spec
    link = cs.links[link_index]
    a, ifa = residuals(fit.x, cs, embed=fit.embed, structural=True)
    if ifa:
        raise RuntimeError(f"cannot obtain the residuals: ifault={ifa}")

    n = a.shape[0]
    if nlags is None:
        nlags = max(10, min(n // 4, 24))
    npar_tr = (link.s + 1) + link.r

    aX = np.asarray(a[:, link.inp], float)
    aN = np.asarray(a[:, link.out], float)

    def _p_adequacy(x_, n_):
        cpos = ccf(x_, n_, nlags)
        Qt, ncorr = chi_test(cpos, len(x_), first=0)
        return _p_chi(Qt, max(1, ncorr - npar_tr)), cpos

    p_full, ccf_full = _p_adequacy(aX, aN)
    cneg = ccf(aN, aX, nlags)
    Qe, dfe = chi_test(cneg, n, first=1)
    p_exog = _p_chi(Qe, dfe)

    # standardise the OUTPUT's innovation: that is where an intervention shows
    z = aN - aN.mean()
    sd = z.std()
    z = z / sd if sd > 1e-15 else z
    zz = float(np.sum(z * z)) or 1.0

    model = cs.series[link.out].spec.model
    lost = int(getattr(model.series, "nobs", n)) - n
    rng = float(np.max(np.abs(ccf_full))) or 1.0

    out, decisive = [], []
    for t in range(n):
        if abs(z[t]) < threshold:
            continue
        keep = np.ones(n, bool)
        keep[t] = False
        p_wo, _c = _p_adequacy(aX[keep], aN[keep])

        lags = []
        for k in range(nlags + 1):
            if t + k < n:
                c = abs(aX[t] * aN[t + k]) / (n * aX.std() * aN.std() + 1e-30)
                if c > contrib_threshold * rng:
                    lags.append(k)

        # The GLOBAL effect. An anomaly inflates the residual variance, which is
        # the DIVISOR of every correlation, so it shrinks all of them at once --
        # not just the lags it happens to touch. In the school's teaching this is
        # the thing to VERIFY, and it is easy to see: take the point out and the
        # coefficients come back.
        m_with = float(np.mean(np.abs(ccf_full)))
        m_wo = float(np.mean(np.abs(_c)))
        an = Anomaly(index=t, obs=lost + t + 1,
                     date=_date_of(model, lost + t + 1),
                     z=float(z[t]), variance_fraction=float(z[t] ** 2 / zz),
                     p_transfer_without=float(p_wo), ccf_lags_affected=lags,
                     compression=(m_wo / m_with if m_with > 1e-30 else 1.0),
                     ccf_max=float(np.max(np.abs(ccf_full))),
                     ccf_max_without=float(np.max(np.abs(_c))),
                     ccf_without=np.asarray(_c, float))
        out.append(an)
        if p_full < 0.05 <= p_wo:
            decisive.append(an)

    out.sort(key=lambda a_: -abs(a_.z))
    return Calibration(p_transfer=float(p_full), p_exog=float(p_exog), n=n,
                       threshold=threshold, anomalies=out, decisive=decisive,
                       names=(cs.names[link.inp], cs.names[link.out]),
                       ccf=np.asarray(ccf_full, float))


def report_calibration(cal):
    """The calibration table, and the N6 branch it implies."""
    xn, yn = cal.names
    L = ["=" * 66,
         f"  ANOMALY CALIBRATION — {xn} -> {yn}",
         "=" * 66,
         "  Which observations are bending the instruments this model is",
         "  identified and judged with: the CCF, and the adequacy portmanteau.",
         "",
         f"  adequacy p = {cal.p_transfer:.4f}    exogeneity p = {cal.p_exog:.4f}"
         f"    (|z| > {cal.threshold:g}, n = {cal.n})",
         ""]
    if not cal.anomalies:
        L += ["  No residual exceeds the threshold. Whatever the portmanteau says,",
              "  it is not one observation saying it.", "=" * 66]
        return "\n".join(L)

    L += ["    date         z     var%    CCF x    max|r|  ->  sin él    p sin él",
          "  " + "-" * 64]
    for a in cal.anomalies:
        L.append(f"  {a.date:>8s}  {a.z:+7.2f}  {100*a.variance_fraction:5.1f}%"
                 f"   {a.compression:5.2f}x   {a.ccf_max:6.3f} -> {a.ccf_max_without:6.3f}"
                 f"    {a.p_transfer_without:6.4f}")
    L += ["",
          "  «CCF x» es lo importante: en cuánto CRECEN todos los coeficientes al",
          "  quitar esa observación. Un anómalo infla la varianza residual, que es",
          "  el DIVISOR de toda correlación, así que aplasta TODOS los retardos a",
          "  la vez — no sólo el que toca. Es un hecho a VERIFICAR, y se verifica",
          "  mirando: quítalo y los coeficientes resucitan (plot_calibration).",
          ""]

    v = cal.verdict
    if v == "adequate":
        big = [a for a in cal.anomalies if a.variance_fraction > 0.15]
        if big:
            a = big[0]
            L += [f"  *** THE TRANSFER PASSES, AND {a.date} IS WHY IT PASSES SO WELL.",
                  f"      Taking it out multiplies every CCF coefficient by "
                  f"{a.compression:.2f} —",
                  f"      max|r| goes {a.ccf_max:.3f} -> {a.ccf_max_without:.3f}.",
                  f"      It carries {100*a.variance_fraction:.0f} % of the residual "
                  f"variance on its own.",
                  "",
                  "      A large anomaly INFLATES the residual variance, which",
                  "      shrinks every correlation, which makes the portmanteau",
                  f"      comfortable: p = {cal.p_transfer:.4f} here against "
                  f"{a.p_transfer_without:.4f} without it.",
                  "      An adequacy bought that way is not evidence of a good",
                  "      transfer — it is evidence of an uncalibrated",
                  "      intervention. It belongs in `art`, on the univariate",
                  "      rung, and travels here in the .pre."]
        else:
            L += ["  The transfer is adequate, and no single observation is",
                  "  carrying the verdict."]
    elif v == "observation":
        d = cal.decisive[0]
        L += ["  *** THE VERDICT RESTS ON ONE OBSERVATION.",
              f"      Adequacy is {cal.p_transfer:.4f} with {d.date} in and "
              f"{d.p_transfer_without:.4f} without it.",
              "",
              "      This is an INTERVENTION, not a misspecified shape.",
              "      Re-identifying (b, r, s) around it is how a model acquires",
              "      a lag nobody can interpret. Calibrate the anomaly instead —",
              "      in `art`, on the univariate rung, where interventions live",
              "      and travel here in the .pre."]
    else:
        L += ["  Adequacy fails and NO single observation explains it: dropping",
              "  any one of them leaves the verdict where it was. The SHAPE is",
              "  wrong — go back and re-identify (b, r, s)."]
    L.append("=" * 66)
    return "\n".join(L)


def plot_calibration(cal, anomaly=None, lags=None):
    """The verification: the CCF **with and without** the anomaly, overlaid.

    This is the "easy to see" part of the school's rule. The claim — that one
    observation is flattening every coefficient by inflating the variance — is
    not something to take on trust from a number. Remove the point and the
    correlations come back; the picture either shows that or it does not.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                                    # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install 'drtran[plots]'")

    a = anomaly or (cal.anomalies[0] if cal.anomalies else None)
    if a is None or a.ccf_without is None:
        raise ValueError("nothing to verify: no anomaly above the threshold")

    wo = np.asarray(a.ccf_without, float)
    wi = np.asarray(cal.ccf, float)
    k = np.arange(len(wo))
    band = 2.0 / math.sqrt(cal.n)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar(k - 0.19, wi, width=0.36, color="#9ca3af", label="con la anomalía")
    ax.bar(k + 0.19, wo, width=0.36, color="#1d4ed8", label=f"sin {a.date}")
    ax.axhline(band, ls="--", lw=0.9, color="#b91c1c")
    ax.axhline(-band, ls="--", lw=0.9, color="#b91c1c")
    ax.axhline(0, lw=0.8, color="black")
    ax.set_xlabel("lag k  (k >= 0: the transfer)")
    ax.set_ylabel("r(k)")
    ax.set_title(f"Verificación — quitando {a.date} cada coeficiente x"
                 f"{a.compression:.2f}   (media |r|)")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig
