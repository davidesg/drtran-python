"""Identifying the NETWORK: the whole system's `-p`.

Port of `identify_network` (`drtran.c`). It is step 3 of the school's ladder:
once the **diagonal** model is estimated, the CCFs of its RESIDUALS are read and
from them the system's dynamic relationships are proposed — who moves whom, with
what delay — together with the contemporaneous covariances (Munoz Polo 2001,
§2.6).

The convention, the C's::

    ccf_ij(k) = corr(a_i(t), a_j(t+k)),  with i < j,  |r| > 2/sqrt(n) significant

    k > 0   a_i leads a_j        =>  link  i -> j
    k < 0   a_j leads a_i        =>  link  j -> i
    k = 0   contemporaneous      =>  free the covariance q[i,j]
    both sides significant       =>  FEEDBACK

It is a GUIDE, not the network
------------------------------
The C says so itself and it bears repeating: this proposes CANDIDATES. They must
be pruned by exogeneity (nothing enters an exogenous series), by acyclicity (a
DAG admits no cycles) and by how plausible the delay is. The identified network
is not the final one: the analyst's intervention between rungs is part of the
method.

Two inherited decisions, and why
--------------------------------
* **The search window is bounded** to 2 seasonal periods (or 8 without
  seasonality), and to n/4 and to 12: a transfer with a longer delay is
  implausible, and the more lags are looked at, the more false positives from
  multiple testing.
* **With feedback the dominant side is taken**, with a warning. It does not fit
  in a one-way DAG, and choosing silently would be worse than saying so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .identify import ccf

from .cast import Link


def residuals(x, cast_spec, embed=True, xitol=-1e-3, structural=False):
    """The model's residuals `a`, (n, m). This is `elf` with `atf=True`.

    They are not recomputed by hand: they come from the same `elf` that scores
    the likelihood, so they are the filter's EXACT residuals, with their
    pre-sample initialisation, and not a truncated approximation.

    `structural=True` undoes the Phi(0) normalisation: `a <- Phi(0)*a`. The
    transfer DIAGNOSTICS ask for it. With a contemporaneous link (b=0) the
    embedded cast puts omega_0 at lag zero, so Phi(0) != I and the reduced-form
    residuals come out correlated **by construction** (Sigma_12 =
    omega_0*sigma2_X). Measuring adequacy on those is measuring the correlation
    the transfer itself generates and calling it misspecification: the
    portmanteau blows up and condemns a correct model. For reading the CCFs of
    the network identification it makes no difference, because there are no links
    there yet.
    """
    from drvarma._engine import elf_c

    from .cast import cast_diagonal
    from .embed import cast_embedded

    phi0 = None
    if embed:
        out = cast_embedded(np.asarray(x, float), cast_spec, with_phi0=structural)
        if structural:
            phi, theta, mu, w, sigma, ifault, phi0 = out
        else:
            phi, theta, mu, w, sigma, ifault = out
    else:
        phi, theta, mu, w, sigma, ifault = cast_diagonal(np.asarray(x, float),
                                                         cast_spec)
    if ifault:
        return None, int(ifault)

    n, m = w.shape
    _ll, _f1, _f2, a, ifa = elf_c(m, n, phi.shape[0], theta.shape[0],
                                  mu, phi, theta, sigma, w, 1.0, xitol, True)
    if ifa:
        return None, int(ifa)
    res = np.asarray(a)[1:, 1:]
    if structural and phi0 is not None:
        res = res @ np.asarray(phi0, float).T
    return res, 0


def _bs_from_side(c, nlags, thr):
    """(b, s, peak) of one side's significant block, or None. Port of
    `net_bs_from_side`.

    The structure is the **CONTIGUOUS block** from b: an isolated distant peak is
    noise, and with 5% bands one lag in twenty is expected outside by chance.
    """
    b = last = -1
    peak = 0.0
    for k in range(1, nlags + 1):
        if abs(c[k]) > thr:
            if b < 0:
                b = k
            last = k
        elif b >= 0 and k > last + 1:
            break
        if abs(c[k]) > abs(peak):
            peak = c[k]
    if b < 0:
        return None
    last = b
    while last + 1 <= nlags and abs(c[last + 1]) > thr:
        last += 1
    return b, last - b, peak


@dataclass
class Candidate:
    """A proposed link, with the peak that supports it."""

    out: int
    inp: int
    b: int
    s: int
    peak: float

    @property
    def link(self):
        return Link(out=self.out, inp=self.inp, b=self.b, r=0, s=self.s)


@dataclass
class IdentifiedNetwork:
    """What reading the residual CCFs proposes."""

    candidates: list = field(default_factory=list)   # list[Candidate]
    covariances: list = field(default_factory=list)  # (i, j, r0), i < j
    feedback: list = field(default_factory=list)
    nlags: int = 0
    band: float = 0.0
    names: list = field(default_factory=list)

    @property
    def links(self):
        """The links as `Link`, ready for `build_cast_spec`."""
        return [c.link for c in self.candidates]

    @property
    def cycle(self):
        """The proposed graph's cycle, or `None`. One is expected now and then.

        Reading the CCFs pair by pair does not impose acyclicity, so the proposal
        may come out cyclic — and then it is NOT estimable as it stands. That is
        not a failure of the identification: it is the part of the job that falls
        to the analyst, pruning.
        """
        from .network import find_cycle

        return find_cycle(self.links, len(self.names))

    def __repr__(self):                                    # pragma: no cover
        return (f"IdentifiedNetwork({len(self.candidates)} links, "
                f"{len(self.covariances)} covariances, band {self.band:.3f})")


def identify_network(cast_spec, x=None, nlags=None, embed=True):
    """Read the CCFs of the DIAGONAL model's residuals and propose the network.

    `x` defaults to the `.pre`'s seeds. The natural thing is to pass it the
    diagonal rung's optimum (`fit(...).x`), which is what the C does: the network
    is read in the residuals of the ESTIMATED diagonal model.
    """
    from .cast import x0_from_pre

    x = x0_from_pre(cast_spec) if x is None else np.asarray(x, float)
    a, ifault = residuals(x, cast_spec, embed=embed)
    if ifault:
        raise RuntimeError(f"cannot obtain the residuals: ifault={ifault}")

    n, m = a.shape
    thr = 2.0 / math.sqrt(n)

    if nlags is None:
        nlags = min(max(n // 4, 6), 12)
        freq = getattr(cast_spec.series[0].spec.model.series, "freq", 1) or 1
        cap = 2 * freq if freq > 1 else 8
        nlags = min(nlags, cap)

    net = IdentifiedNetwork(nlags=nlags, band=thr, names=list(cast_spec.names))

    for i in range(m):
        for j in range(i + 1, m):
            cpos = ccf(a[:, i], a[:, j], nlags)      # k >= 0: the i -> j side
            cneg = ccf(a[:, j], a[:, i], nlags)      # k >= 0 of (j,i): j -> i
            if not np.any(cpos) and not np.any(cneg):
                continue                              # degenerate series

            if abs(cpos[0]) > thr:
                net.covariances.append((i, j, float(cpos[0])))

            pos = _bs_from_side(cpos, nlags, thr)
            neg = _bs_from_side(cneg, nlags, thr)

            if pos and neg:
                net.feedback.append((i, j, pos[0], pos[2], neg[0], neg[2]))
                # It does not fit a one-way DAG: take the dominant side and warn.
                if abs(pos[2]) >= abs(neg[2]):
                    neg = None
                else:
                    pos = None
            if pos:
                net.candidates.append(Candidate(out=j, inp=i, b=pos[0], s=pos[1],
                                                peak=pos[2]))
            if neg:
                net.candidates.append(Candidate(out=i, inp=j, b=neg[0], s=neg[1],
                                                peak=neg[2]))

    # The real links tend to be the strongest; spurious distant peaks fall to the
    # bottom. A stable sort, so that ties come out in series order and the report
    # is reproducible.
    net.candidates.sort(key=lambda c: -abs(c.peak))
    return net


def report_network(net):
    """The report, in the C's format, but with the covariances BY INDEX.

    The C's `-i` prints `q[EI,EU] = free` with NAMES under a heading that says
    "paste into a -c file" — and its `.cns` **does not read names** in the `q`,
    only numeric lower-triangle indices (`q[i,j]`, i > j), which is what its
    guided mode `-g` does write. Here what can be pasted is emitted directly,
    with the name alongside in a comment.
    """
    nb = net.names
    L = ["=" * 61,
         "  NETWORK IDENTIFICATION  (CCFs of the diagonal model's residuals)",
         "=" * 61,
         "  Munoz Polo (2001) §2.6: the system's dynamic relationships are read",
         "  in the CCFs of the DIAGONAL model's residuals. This is a GUIDE to",
         "  candidates, not the final network: prune by exogeneity (nothing",
         "  enters an exogenous series), acyclicity (a DAG admits no cycles) and",
         "  how plausible the delay is.",
         f"  Searched up to k={net.nlags}; |r| > {net.band:.3f} is significant.",
         ""]

    for i, j, bp, pp, bn, pn in net.feedback:
        L.append(f"  [feedback]  {nb[i]} <-> {nb[j]} : "
                 f"{nb[i]}->{nb[j]} k={bp}({pp:+.2f}), "
                 f"{nb[j]}->{nb[i]} k={bn}({pn:+.2f})  -> the dominant one is taken")

    L.append("  CONTEMPORANEOUS  (k=0; free the innovation covariance):")
    if not net.covariances:
        L.append("    (none above the band)")
    for i, j, r0 in net.covariances:
        L.append(f"    {nb[i]:<4s} - {nb[j]:<4s}   r(0) = {r0:+.3f}")

    L += ["", "  DIRECTED LINKS  (candidate transfers, strongest first):"]
    if not net.candidates:
        L.append("    (none above the band)")
    for c in net.candidates:
        L.append(f"    {nb[c.inp]:<4s} -> {nb[c.out]:<4s}   peak {c.peak:+.3f}"
                 f"   proposal  b={c.b} r=0 s={c.s}")

    if net.candidates:
        L += ["", "  PROPOSED NETWORK  (for a -n file / read_dag):"]
        for c in net.candidates:
            L.append(f"    {nb[c.out]} <- {nb[c.inp]}   {c.b} 0 {c.s}")

    if net.covariances:
        L += ["", "  PROPOSED COVARIANCES  (for a -c file / read_cns;",
              "  the indices are the position on the command line):"]
        for i, j, _r in net.covariances:
            L.append(f"    q[{j + 1},{i + 1}] = free      # {nb[i]} . {nb[j]}")

    if net.feedback:
        L += ["", f"  NOTE: {len(net.feedback)} pair(s) with feedback (both",
              "  directions). A one-way transfer DAG kept the dominant side;",
              "  they are worth looking at by hand."]
    L.append("=" * 61)
    return "\n".join(L)


def write_guided(net, name):
    """GUIDED mode (`-g`): write `NAME.dag` and `NAME.cns`, ready to estimate.

    They are written exactly as they come, **unpruned**: pruning is the analyst's.
    If the proposal carries a cycle it is written all the same, with the cycle
    noted in the header — hiding it, or pruning it unilaterally, would be worse:
    the file is the draft the decision is made on, and `read_dag` will reject it
    as long as the cycle is still there.

    Returns the two paths.
    """
    from .network import write_dag

    dag, cns = f"{name}.dag", f"{name}.cns"
    write_dag(dag, net.links, net.names)
    c = net.cycle
    if c is not None:
        route = " -> ".join(net.names[i] for i in c)
        with open(dag) as f:
            body = f.read()
        with open(dag, "w") as f:
            f.write(f"# WARNING: the proposal has a CYCLE and is not estimable\n"
                    f"# as it stands:\n"
                    f"#   {route}\n"
                    f"# A DAG admits no cycles. Prune one of those links -- the\n"
                    f"# one with the smaller peak, or the one that makes no\n"
                    f"# sense -- before estimating.\n"
                    + body)
    with open(cns, "w") as f:
        f.write("# Contemporaneous covariances proposed by reading the residual\n"
                "# CCFs of the diagonal model. Indices = position on the command\n"
                "# line; the lower triangle, i > j.\n")
        for i, j, r0 in net.covariances:
            f.write(f"q[{j + 1},{i + 1}] = free      "
                    f"# {net.names[i]} . {net.names[j]}, r(0) = {r0:+.3f}\n")
    return dag, cns
