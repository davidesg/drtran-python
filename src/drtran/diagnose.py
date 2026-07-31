"""Transfer adequacy: the Box–Jenkins check, on what was actually ESTIMATED.

Port of `transfer_adequacy` (`drtran.c`). `identify.py` reads the CCFs **before**
estimating, from the `.pre` seeds; this reads them **after**, from the residuals
of the fitted model. Two different questions: one proposes (b, r, s), the other
says whether the one that was estimated is enough.

Why the cast's residuals are already the right objects
------------------------------------------------------
In the cast, by construction, each series' residual is exactly what the check
needs:

* the **input**'s residual is the **prewhitened input** — its own ARMA filtered it;
* the **output**'s residual is the innovation of the **noise** N_t.

If (b, r, s) is right, the noise must carry no trace of the input, so the CCF
between the two residuals has to be white noise at every lag::

    significant at k >= 0   ->  structure missing in the TRANSFER
    significant at k <  0   ->  FEEDBACK: X is not exogenous, and a one-input
                                model is not valid for that pair

The verdict comes from the JOINT test
-------------------------------------
Not from the presence of an isolated peak: with 5 % bands, about 1 lag in 20 is
expected to cross by chance, so demanding zero significant lags would condemn even
a correct specification. The peaks are reported; the portmanteau decides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .identify import ccf


def chi_test(r, n, first=1):
    """The Q of `diagnose.c:ChiTestC`.

        Q = n(n+2)·Σ_i r_i² / (n − i + 1)

    `first=0` includes the contemporaneous lag, which is what the TRANSFER test
    wants; `first=1` skips it, which is what the exogeneity test wants.
    **The divisor is n−i+1, not n−i** — changing it moves Q just enough to stop
    matching the C.
    """
    r = np.asarray(r, float)
    idx = np.arange(first, len(r))
    if len(idx) == 0:
        return 0.0, 0
    div = np.array([n - i + 1 for i in range(1, len(idx) + 1)], float)
    return float(n * (n + 2) * np.sum(r[idx] ** 2 / div)), len(idx)


def _p_chi(Q, df):
    if df < 1:
        return float("nan")
    try:
        from scipy.stats import chi2
        return float(1.0 - chi2.cdf(Q, df))
    except Exception:                                     # pragma: no cover
        return float("nan")


@dataclass
class Adequacy:
    """The verdict on a link that has already been estimated."""

    inp: int
    out: int
    nlags: int
    threshold: float
    ccf_pos: np.ndarray = None          # k = 0..nlags
    ccf_neg: np.ndarray = None          # k = 0..nlags on the negative side
    Q_transfer: float = 0.0
    df_transfer: int = 0
    p_transfer: float = 1.0
    Q_exog: float = 0.0
    df_exog: int = 0
    p_exog: float = 1.0
    names: tuple = ()

    @property
    def adequate(self):
        return not (self.p_transfer < 0.05)

    @property
    def exogenous(self):
        return not (self.p_exog < 0.05)

    @property
    def significant_lags(self):
        """The k >= 0 crossing the individual band. Informative, not the verdict."""
        return [k for k in range(self.nlags + 1)
                if abs(self.ccf_pos[k]) > self.threshold]

    def __repr__(self):                                   # pragma: no cover
        a = "ADEQUATE" if self.adequate else "INADEQUATE"
        e = "exogenous" if self.exogenous else "FEEDBACK!"
        return (f"Adequacy({a} p={self.p_transfer:.4f}, {e} "
                f"p={self.p_exog:.4f})")


def transfer_adequacy(x, cast_spec=None, link_index=0, nlags=None, embed=True):
    """Check link `link_index` of the model ESTIMATED at `x`.

    `x` is the full parameter vector together with its `cast_spec`, or simply a
    `Fit`, which already carries both. Returns an `Adequacy`.
    """
    from .netid import residuals

    if hasattr(x, "x"):                       # a Fit is accepted directly
        cast_spec = x.cast_spec
        x = x.x
    if cast_spec is None:
        raise TypeError("cast_spec is required (or pass a Fit)")
    link = cast_spec.links[link_index]

    # STRUCTURAL residuals — see the note in `residuals`. With b=0 the
    # reduced-form ones are correlated by construction, and the portmanteau would
    # condemn a correct model: p goes to 0.0000 where the C reports 0.1966.
    a, ifault = residuals(x, cast_spec, embed=embed, structural=True)
    if ifault:
        raise RuntimeError(f"cannot obtain the residuals: ifault={ifault}")
    n = a.shape[0]

    if nlags is None:
        nlags = max(10, min(n // 4, 24))

    aX = a[:, link.inp]                       # the input, already prewhitened
    aN = a[:, link.out]                       # the noise innovation

    cpos = ccf(aX, aN, nlags)                # k >= 0
    cneg = ccf(aN, aX, nlags)                # the k <= 0 side

    # The transfer: portmanteau over k >= 0, CONTEMPORANEOUS INCLUDED. The
    # degrees of freedom discount the parameters of nu(B) — (s+1) + r — because
    # they were estimated from these same data.
    npar_tr = (link.s + 1) + link.r
    Qt, ncorr = chi_test(cpos, n, first=0)
    dft = max(1, ncorr - npar_tr)

    # Exogeneity: portmanteau over k < 0, skipping the contemporaneous lag, which
    # belongs to the other side.
    Qe, dfe = chi_test(cneg, n, first=1)

    return Adequacy(
        inp=link.inp, out=link.out, nlags=nlags,
        threshold=2.0 / math.sqrt(n), ccf_pos=cpos, ccf_neg=cneg,
        Q_transfer=Qt, df_transfer=dft, p_transfer=_p_chi(Qt, dft),
        Q_exog=Qe, df_exog=dfe, p_exog=_p_chi(Qe, dfe),
        names=(cast_spec.names[link.inp], cast_spec.names[link.out]))


def report_adequacy(ad):
    """The report, in the C's format."""
    xn, yn = ad.names if ad.names else ("X", "Y")
    L = ["=" * 61,
         f"  TRANSFER FUNCTION ADEQUACY — {yn} <- {xn}",
         "  CCF between the estimated noise and the prewhitened input",
         "=" * 61,
         "  If (b, r, s) is correct, this CCF must be white noise.",
         "    significant at k >= 0 -> structure missing in the TRANSFER",
         "    significant at k <  0 -> FEEDBACK (X is not exogenous)",
         ""]

    L.append("  Portmanteau of the CCF (k >= 0) — this is the test of the transfer:")
    L.append(f"    Q({ad.df_transfer}) = {ad.Q_transfer:.4f}   "
             f"[{ad.nlags + 1} lags − {(ad.nlags + 1) - ad.df_transfer} "
             f"parameters of nu(B)]")
    L.append(f"    p-value = {ad.p_transfer:.4f}")

    L += ["", "  VERDICT:"]
    sig = ad.significant_lags
    if not ad.adequate:
        L.append(f"    *** The transfer is NOT adequate (p = {ad.p_transfer:.4f}).")
        L.append("    The noise still carries a trace of the input at:")
        for k in sig:
            L.append(f"      k = {k:2d}   r = {ad.ccf_pos[k]:7.4f}")
        L.append("    Widen nu(B) to cover those lags: raise s up to the last one,")
        L.append("    or try r=1 if the weights decay. Then re-estimate.")
    else:
        L.append(f"    The transfer is ADEQUATE (p = {ad.p_transfer:.4f}): the CCF is")
        L.append("    consistent with white noise at k >= 0.")
        if sig:
            L.append(f"    ({len(sig)} lag(s) cross the individual bands; "
                     f"~{0.05 * (ad.nlags + 1):.1f} would be expected by chance,")
            L.append("     so they do not contradict the joint test:")
            for k in sig:
                L.append(f"       k = {k:2d}   r = {ad.ccf_pos[k]:7.4f}")

    L += ["", "  Exogeneity — portmanteau of the CCF at k < 0:",
          f"    Q({ad.df_exog}) = {ad.Q_exog:.4f}   p-value = {ad.p_exog:.4f}"]
    if ad.exogenous:
        L.append("    No sign of feedback: the input can be treated as exogenous.")
    else:
        L.append("    *** FEEDBACK: the output leads the input. A one-input model")
        L.append("    is not valid for this pair.")
    L.append("=" * 61)
    return "\n".join(L)
