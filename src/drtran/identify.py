"""Identifying (b, r, s) by prewhitening and the CCF — Box-Jenkins' step 1.

Port of `prewhiten_and_identify` (`drtran.c`). The classical procedure:

1. **Prewhiten the input** with its own ARMA: apply phi_X(B) and invert
   theta_X(B) until white noise `a_X` is left.
2. **Apply THE SAME filter to the output** -> `beta_Y`. This is the step usually
   forgotten: filtering only the input leaves the CCF contaminated by the
   output's own structure.
3. **The CCF between the two**. With the input already white, r(k) is
   proportional to the impulse response weights: nu(k) = r(k)*s_beta/s_a.
4. **Read (b, r, s)** off the pattern of nu.
5. **Check exogeneity** by looking at the k<0 side of the CCF: if the output
   leads the input there is feedback, and a transfer model with a single input
   does not hold.

Sources
-------
* **Box & Jenkins (1976, ch. 11)** — the SIMPLE prewhitening implemented here:
  filter the input with its ARMA and apply THE SAME filter to the output, so that
  r(k) estimates the scaled nu directly.
* **Haugh & Box (1977)**, JASA 72(357) 121-130 — the original paper. They propose
  DOUBLE prewhitening (each series with its own model) and give the asymptotic
  distribution of the cross-correlations.
* **Tsay (1985)**, JBES 3(3) 228-237 — the critique. It points out two problems
  with the approach; see below which one is covered and which is not.

Two of the C's decisions are replicated because they are what keeps the answers
sane:

* The structure is the **CONTIGUOUS block** starting at b. With 5% bands 1 lag in
  20 is expected to cross them by chance, so taking the last significant lag of
  the whole CCF sent s to absurd values (s=24).
* Exogeneity is judged with a **portmanteau** over k<0, not by counting how many
  cross: counting raises the warning almost always, for the same reason.

What the literature says, and what is done about it
---------------------------------------------------
**Haugh & Box (1977) §1.4**: `var{r_xy(k)} ~ (N-k)^-1`, so the band must widen
with the lag. The C uses a constant `2/sqrt(N)`. Measured on the canonical case,
that difference creates a false positive at k=24 — which is precisely the one
that forced the contiguous-block heuristic. Available as `band="haugh-box"`; the
default remains `"constant"` for homologation.

**Tsay (1985) §2** points out two problems with prewhitening:

1. *Unidirectionality is rarely true; there may be feedback.* COVERED: that is
   exactly the exogeneity portmanteau over k<0.
2. *In non-stationary series the transformation (differencing) may WEAKEN the
   relationship between the variables*, with contradictory conclusions between
   the analysis in levels and in transformed series (citing Feige & Pearce 1974).
   **NOT covered**: identification here runs on the already differenced series
   the `.pre` fixes. It is a real limitation of the method, not of the port.
   Tsay's alternative — a VAR approximation plus least squares plus the corner
   method, without prewhitening or differencing — is out of scope for now.

**Tsay (1985) eq. (2.2)**, as a bonus: the reduced-form innovation covariance is
diagonal **if and only if b != 0**. With a contemporaneous transfer (b=0) the
reduced form CANNOT have a diagonal Sigma — which is exactly why the embedded
cast needs `normalize_phi0` and why the C keeps the STRUCTURAL Q apart. A
theoretical confirmation of the design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Identification:
    """What the prewhitened CCF says about one link."""

    lags: np.ndarray                 # lags, from -nlags to +nlags
    ccf: np.ndarray                  # r(k)
    nu: np.ndarray                   # weights nu(k) = r(k)*s_beta/s_a
    threshold: float                 # the band at k=0
    b: int
    r: int
    s: int
    bands: np.ndarray = None         # band per lag (may vary with |k|)
    alternatives: list = field(default_factory=list)   # [(b, r, s, reason), ...]
    exogenous: bool = True
    Q_exogeneity: float = 0.0
    p_exogeneity: float = 1.0
    n_signif_negative: int = 0

    @property
    def has_relationship(self):
        return not (self.b == 0 and self.r == 0 and self.s == 0
                    and not self.significant_non_negative)

    @property
    def significant_non_negative(self):
        u = self.bands if self.bands is not None else self.threshold
        m = (self.lags >= 0) & (np.abs(self.ccf) > u)
        return self.lags[m].tolist()

    def __repr__(self):                                   # pragma: no cover
        ex = "exogenous" if self.exogenous else "FEEDBACK!"
        return (f"Identification(b={self.b}, r={self.r}, s={self.s}, "
                f"{ex} p={self.p_exogeneity:.3f})")


def prewhiten(w, phi, theta):
    """Filter `w` with the given ARMA: apply phi(B) and invert theta(B).

        u[t]  = w[t] - SUM_k phi_k*w[t-k]
        a[t]  = u[t] + SUM_k theta_k*a[t-k]

    The sums are truncated at the start of the sample (k < t), as in the C.
    """
    w = np.asarray(w, float)
    phi = np.asarray(phi, float).ravel()
    theta = np.asarray(theta, float).ravel()
    n = len(w)
    a = np.zeros(n)
    for t in range(n):
        u = w[t]
        for k in range(1, min(len(phi), t) + 1):
            u -= phi[k - 1] * w[t - k]
        a[t] = u
        for k in range(1, min(len(theta), t) + 1):
            a[t] += theta[k - 1] * a[t - k]
    return a


def ccf(d1, d2, nlags):
    """corr(d1_t, d2_{t+k}) for k = 0..nlags (`diagnose.c:Ccf`'s convention).

    It lives here, and the other modules IMPORT it. Having a copy in every place
    that looks at a CCF is how a fix gets applied in one and forgotten in the
    others.
    """
    d1 = np.asarray(d1, float)
    d2 = np.asarray(d2, float)
    n = len(d1)
    x1 = d1 - d1.mean()
    x2 = d2 - d2.mean()
    s1 = math.sqrt(float((x1 * x1).sum()) / n)
    s2 = math.sqrt(float((x2 * x2).sum()) / n)
    out = np.zeros(nlags + 1)
    if s1 < 1e-12 or s2 < 1e-12:
        return out
    for k in range(nlags + 1):
        out[k] = float((x1[:n - k] * x2[k:]).sum()) / (n * s1 * s2)
    return out


#: Internal alias of `ccf`. Inside `identify` the local variable `ccf` -- the
#: array of correlations -- shadows the module function, so the calls in there
#: go through this one.
_ccf = ccf


def identify(cast_spec, link, x=None, nlags=None, band="constant"):
    """Identify the `link`'s (b, r, s) from the prewhitened CCF.

    `x` defaults to the `.pre`'s seeds: the prewhitening uses the ARMA fue
    estimated for the input, which is exactly what the ladder puts there.

    `band` chooses the significance threshold:

    * `"constant"` (the default) — `2/sqrt(N)` for every lag, which is what the C
      does and what this port homologates against.
    * `"haugh-box"` — `2/sqrt(N-|k|)`, which is what the original paper says.
      Haugh & Box (1977, §1.4) give `var{r_xy(k)} ~ (N-k)^-1` and propose judging
      each estimate "against an approximate standard error of N^(-1/2) **or
      (N-k)^(-1/2)**". The band must WIDEN with the lag, because there is less
      overlap to estimate it from.

      This is not cosmetic: on the canonical case (N=215) the band at k=24 goes
      from 0.1364 to 0.1447, and the peak of -0.1423 stops being significant.
      That peak is precisely the false positive that forced the C to introduce
      the "contiguous block" heuristic. With the paper's band the false positive
      never arises: **the heuristic compensates for a badly calibrated band**.

      It is left optional so as not to break homologation with the binary.
    """
    from fue.cast_us import cast_us_py

    from .cast import x0_from_pre

    x = np.asarray(x0_from_pre(cast_spec) if x is None else x, float)

    # each series' univariate chunk
    idx = cast_spec.npar_links
    pieces = []
    for sc in cast_spec.series:
        pieces.append(x[idx:idx + sc.npar]); idx += sc.npar

    def prepare(i):
        p, q, phi, theta, mu, w, ifault = cast_us_py(pieces[i],
                                                     cast_spec.series[i].est_spec)
        return (np.asarray(phi, float)[:p], np.asarray(theta, float)[:q],
                np.asarray(w, float), int(ifault))

    phi_x, theta_x, w_x, if_x = prepare(link.inp)
    _phi_y, _theta_y, w_y, if_y = prepare(link.out)
    if if_x or if_y:
        raise ValueError("the univariate cast failed; check the .pre files")

    n = min(len(w_x), len(w_y))
    w_x, w_y = w_x[len(w_x) - n:], w_y[len(w_y) - n:]

    if nlags is None:                       # nlags = min(n/4, 24), at least 10
        nlags = max(10, min(n // 4, 24))

    # 1 and 2: prewhiten X and filter Y with THE SAME filter
    a_x = prewhiten(w_x, phi_x, theta_x)
    beta_y = prewhiten(w_y, phi_x, theta_x)

    s_a, s_b = a_x.std(), beta_y.std()
    lags = np.arange(-nlags, nlags + 1)
    ccf = np.zeros(2 * nlags + 1)
    if s_a < 1e-12 or s_b < 1e-12:
        return Identification(lags=lags, ccf=ccf, nu=ccf.copy(),
                             threshold=2.0 / math.sqrt(n), b=0, r=0, s=0)

    # 3: k >= 0 is "the input leads the output" (the transfer);
    #    k < 0 is "the output leads" (feedback: there should be none).
    cpos = _ccf(a_x, beta_y, nlags)
    cneg = _ccf(beta_y, a_x, nlags)
    for k in range(nlags + 1):
        ccf[nlags + k] = cpos[k]
        ccf[nlags - k] = cneg[k]

    nu = ccf * (s_b / s_a)                  # 4: impulse response weights
    if band == "haugh-box":
        # Haugh & Box (1977) §1.4: var{r_xy(k)} ~ (N-k)^-1
        bands = np.array([2.0 / math.sqrt(n - abs(int(k))) for k in lags])
    elif band == "constant":
        bands = np.full(len(lags), 2.0 / math.sqrt(n))
    else:
        raise ValueError(f"unknown band: {band!r}")
    threshold = float(bands[nlags])          # the one at k=0, for the report

    # 5: exogeneity by portmanteau over k<0 (not by counting).
    # `diagnose.c:ChiTestC`: Q = n(n+2)*SUM_{i=1..lags} r_i^2 / (n - i + 1),
    # skipping lag 0. The divisor is n-i+1, not n-i.
    cn = np.array([ccf[nlags - k] for k in range(1, nlags + 1)])
    divisors = np.array([n - i + 1 for i in range(1, nlags + 1)], float)
    Q = float(n * (n + 2) * np.sum(cn ** 2 / divisors))
    try:
        from scipy.stats import chi2
        pval = float(1.0 - chi2.cdf(Q, nlags))
    except Exception:                                    # pragma: no cover
        pval = float("nan")
    nsig_neg = int(np.sum([abs(cn[i - 1]) > bands[nlags - i]
                           for i in range(1, nlags + 1)]))

    # 6: b = first significant k>=0; the CONTIGUOUS block from there
    sig_pos = [k for k in range(nlags + 1)
               if abs(ccf[nlags + k]) > bands[nlags + k]]
    if not sig_pos:
        return Identification(lags=lags, ccf=ccf, nu=nu, threshold=threshold,
                             bands=bands, b=0, r=0, s=0,
                             exogenous=(pval >= 0.05),
                             Q_exogeneity=Q, p_exogeneity=pval,
                             n_signif_negative=nsig_neg)

    b_hat = sig_pos[0]
    last = b_hat
    while last + 1 <= nlags and abs(ccf[nlags + last + 1]) > bands[nlags + last + 1]:
        last += 1

    alternatives = []
    # Candidate A: every significant weight as a free omega
    s1 = last - b_hat
    alternatives.append((b_hat, 0, s1, "each significant weight as a free omega"))

    # Candidate B: if the tail decays approximately geometrically, a denominator
    # of order 1 summarises it with a single parameter.
    nblock = last - b_hat + 1
    if nblock >= 3:
        q1 = abs(nu[nlags + last]) / (abs(nu[nlags + last - 1]) + 1e-12)
        q2 = abs(nu[nlags + last - 1]) / (abs(nu[nlags + last - 2]) + 1e-12)
        if q1 < 0.95 and q2 < 0.95 and abs(q1 - q2) < 0.25:
            s2 = max(0, (last - 2) - b_hat)
            alternatives.append(
                (b_hat, 1, s2,
                 f"the tail decays ~geometrically (ratio ~{0.5 * (q1 + q2):.2f})"))

    b, r, s, _reason = alternatives[-1]      # the most parsimonious proposal
    return Identification(lags=lags, ccf=ccf, nu=nu, threshold=threshold,
                         bands=bands, b=b, r=r, s=s, alternatives=alternatives,
                         exogenous=(pval >= 0.05), Q_exogeneity=Q,
                         p_exogeneity=pval, n_signif_negative=nsig_neg)


def report_identification(ident, names=("X", "Y")):
    """Text report, in the style of the one the C prints.

    Named `report_identification`, not `report`: every other report in the
    package carries its subject (`report_network`, `report_adequacy`,
    `report_forecast`, `report_irf`, `report_rolling`, `report_aggregates`), and
    a bare `report` was shadowed at package level by the `report` MODULE the
    moment one was added. `drtran.report` silently stopped being this function
    and became `drtran/report.py` — the kind of collision that only shows up when
    something calls it.
    """
    xi, yi = names
    L = ["=" * 61,
         "  IDENTIFICATION — prewhitening and CCF (Box-Jenkins)",
         f"  {xi}  ->  {yi}",
         "=" * 61,
         "  The input is prewhitened with its ARMA and the SAME filter is",
         "  applied to the output.  r(k) = corr(a_t, beta_{t+k}):",
         f"    k > 0  ->  {yi} responds to {xi} with a lag of k periods",
         f"    k < 0  ->  {yi} leads {xi} (feedback: there should be none)",
         "",
         f"  Weights nu(k) = r(k)*s_beta/s_a   (band +/-{ident.threshold:.4f})",
         "     k      r(k)      nu(k)",
         "    ------------------------------"]
    u = ident.bands if ident.bands is not None else np.full(len(ident.lags),
                                                            ident.threshold)
    for k, c, v, uk in zip(ident.lags, ident.ccf, ident.nu, u):
        if abs(c) > uk:
            L.append(f"   {int(k):4d}  {c:8.4f}  {v:9.4f}  *")
    if not any(abs(ident.ccf) > u):
        L.append("     (none)")
    L += ["",
          "  Exogeneity — portmanteau of the CCF at k < 0:",
          f"    Q({len(ident.lags) // 2}) = {ident.Q_exogeneity:.4f}"
          f"   p = {ident.p_exogeneity:.4f}   "
          f"[{ident.n_signif_negative} significant]"]
    L.append("    " + ("the input behaves as exogenous. OK" if ident.exogenous
                       else "WARNING: the output leads the input. There may be "
                            "FEEDBACK (Y -> X); a single-input model assumes X "
                            "is EXOGENOUS."))
    L.append("")
    if ident.alternatives:
        L.append("  PROPOSALS:")
        for i, (b, r, s, reason) in enumerate(ident.alternatives):
            mark = " <-" if (b, r, s) == (ident.b, ident.r, ident.s) else "   "
            L.append(f"    [{chr(65 + i)}]  b={b}  r={r}  s={s}{mark}  {reason}")
    else:
        L.append("  No significant CCF at k >= 0: no relationship detected.")
        L.append("  Proposal: b=0, r=0, s=0 (no transfer).")
    return "\n".join(L)
