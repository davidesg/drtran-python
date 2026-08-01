"""The impulse response of a transfer, with its standard errors.

`nu_k` is the response of the output, k periods later, to a ONE-OFF unit shock in
the input. The cumulative column is the response to a PERMANENT unit change, and
it converges to the gain `nu(1) = omega(1)/delta(1)`. For the canonical case that
is the answer to the question the whole model exists to ask: how much of an oil
shock reaches the Spanish CPI, and by when.

On identification
-----------------
In a VAR the impulse response is not identified without an ordering (a Cholesky
of Sigma), because Sigma is not diagonal. Here the identification **is the
model**: the input is exogenous and Sigma is diagonal. That is not magic — those
restrictions are DECLARED, they are TESTED (the exogeneity portmanteau, the
adequacy portmanteau) and they can be relaxed. Relax them next to a
contemporaneous transfer and you land back in the VAR's problem, which is what
`variance_decomposition` refuses to paper over.

On the standard errors
----------------------
The delta method: `Var(nu_k) = g' V g` with `g = d nu_k / d xfree` and V the
covariance of the FREE parameters. The C differentiates the nu recursion
analytically and maps each slot's derivative back through the constraints. This
port differentiates `nu` with respect to the **free vector** by central
differences instead, which is exact enough on a function that costs nothing to
evaluate — and, more to the point, makes shared slots, products and linear
combinations propagate on their own rather than needing their own chain rule.
Same reason `expand` goes inside the likelihood's objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cast import compute_irf


@dataclass
class ImpulseResponse:
    """`nu_k` and its cumulative sum, with standard errors."""

    link_index: int
    b: int
    r: int
    s: int
    out_name: str = ""
    inp_name: str = ""
    nu: np.ndarray = None          # (K+1,)
    se: np.ndarray = None          # (K+1,)
    cum: np.ndarray = None         # (K+1,)
    se_cum: np.ndarray = None      # (K+1,)
    gain: float = float("nan")     # nu(1) = omega(1)/delta(1)
    se_gain: float = float("nan")

    @property
    def t(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = np.isfinite(self.se) & (self.se > 1e-15)
            return np.where(ok, self.nu / np.where(ok, self.se, 1.0), 0.0)

    def __repr__(self):                                    # pragma: no cover
        return (f"ImpulseResponse({self.out_name} <- {self.inp_name}, "
                f"b={self.b} r={self.r} s={self.s}, gain={self.gain:.6f})")


def _link_offset(cast_spec, j):
    """Where link `j`'s omega/delta block starts in the full vector."""
    return sum((l.s + 1) + l.r for l in cast_spec.links[:j])


def impulse_response(fit, link_index=0, K=None, cov="auto", xitol=-1e-3):
    """`nu(k)` for link `link_index`, with delta-method standard errors.

    `K` defaults to the C's choice, `b + s + 12` capped at 36 — far enough to see
    the tail die.

    `cov` is the free-parameter covariance. `"auto"` computes it with
    `standard_errors`, which is the expensive part; passing one already computed
    avoids paying twice. Passing `None` asks for the weights **without**
    inference, and then the standard errors come back NaN rather than zero —
    a zero standard error reads as infinite precision, which is the opposite of
    "not computed".
    """
    from .estimate import standard_errors

    cs = fit.cast_spec
    if not cs.links:
        raise ValueError("the model has no transfer: there is no impulse response")
    link = cs.links[link_index]
    b, r, s = link.b, link.r, link.s
    if K is None:
        K = min(b + s + 12, 36)

    slots = fit.slots
    xfree = np.asarray(fit.xfree if slots is not None else fit.x, float)
    expand = slots.expand if slots is not None else (lambda v: v)
    off = _link_offset(cs, link_index)

    def weights(v):
        xf = np.asarray(expand(np.asarray(v, float)), float)
        om = xf[off:off + s + 1]
        de = xf[off + s + 1:off + s + 1 + r]
        return compute_irf(om, de, b, K + 1)

    nu = weights(xfree)
    cum = np.cumsum(nu)

    # the gain is a property of the WEIGHTS, not of the inference: it is
    # computed whether or not there is a covariance to attach an error to
    from .embed import nu_at_one

    xf = np.asarray(expand(xfree), float)
    gain = float(nu_at_one(xf[off:off + s + 1], xf[off + s + 1:off + s + 1 + r]))

    if isinstance(cov, str) and cov == "auto":
        se_all = standard_errors(fit, xitol=xitol)
        cov = None if se_all.ifault else se_all.cov
    if cov is None:
        nan = np.full(K + 1, float("nan"))
        return ImpulseResponse(link_index=link_index, b=b, r=r, s=s,
                               out_name=cs.series[link.out].name,
                               inp_name=cs.series[link.inp].name,
                               nu=nu, se=nan, cum=cum, se_cum=nan.copy(),
                               gain=gain)

    # Jacobian by central differences on the FREE vector. `nu` is an explicit
    # polynomial recursion, not a likelihood, so this costs nothing and the step
    # can be the usual cube root of macheps.
    k_free = len(xfree)
    eps = np.finfo(float).eps ** (1.0 / 3.0)
    J = np.zeros((K + 1, k_free))
    for i in range(k_free):
        h = eps * max(abs(xfree[i]), 1.0)
        up, dn = xfree.copy(), xfree.copy()
        up[i] += h
        dn[i] -= h
        J[:, i] = (weights(up) - weights(dn)) / (2.0 * h)

    var = np.einsum("ki,ij,kj->k", J, cov, J)
    se = np.sqrt(np.maximum(var, 0.0))

    Jc = np.cumsum(J, axis=0)
    var_c = np.einsum("ki,ij,kj->k", Jc, cov, Jc)
    se_cum = np.sqrt(np.maximum(var_c, 0.0))

    # the gain's s.e. comes from the same delta method, so it cannot drift from
    # the cumulative column
    g = np.zeros(k_free)
    for i in range(k_free):
        h = eps * max(abs(xfree[i]), 1.0)
        up, dn = xfree.copy(), xfree.copy()
        up[i] += h
        dn[i] -= h
        xu, xd = np.asarray(expand(up), float), np.asarray(expand(dn), float)
        gu = nu_at_one(xu[off:off + s + 1], xu[off + s + 1:off + s + 1 + r])
        gd = nu_at_one(xd[off:off + s + 1], xd[off + s + 1:off + s + 1 + r])
        g[i] = (gu - gd) / (2.0 * h)
    se_gain = math.sqrt(max(float(g @ cov @ g), 0.0))

    return ImpulseResponse(link_index=link_index, b=b, r=r, s=s,
                           out_name=cs.series[link.out].name,
                           inp_name=cs.series[link.inp].name,
                           nu=nu, se=se, cum=cum, se_cum=se_cum,
                           gain=gain, se_gain=se_gain)


def report_irf(irfs):
    """The impulse response tables, one per link, in the C's layout."""
    if not isinstance(irfs, (list, tuple)):
        irfs = [irfs]
    L = ["=" * 63,
         "  IMPULSE RESPONSE  nu(B) = omega(B)/delta(B) * B^b",
         "=" * 63,
         "  nu_k is the response of the output, k periods later, to a ONE-OFF",
         "  unit shock in the input. The CUMULATIVE column is the response to a",
         "  PERMANENT unit change; it converges to the gain.",
         "",
         "  In a VAR the impulse response is not identified without an ordering",
         "  (Cholesky), because Sigma is not diagonal. Here the identification IS",
         "  the model. That is not magic: the restrictions (exogenous input,",
         "  diagonal Sigma) are DECLARED, they are TESTED, and they can be",
         "  relaxed -- and if you relax them next to a contemporaneous transfer,",
         "  you land back in the VAR's problem, and the program says so."]
    for ir in irfs:
        L += ["",
              f"  {ir.out_name} <- {ir.inp_name}   "
              f"(b={ir.b}, r={ir.r}, s={ir.s})",
              "",
              "    k      nu_k     std.err       t   |   cumulative   std.err",
              "  " + "-" * 61]
        t = ir.t
        for k in range(len(ir.nu)):
            if ir.se[k] != ir.se[k]:                 # NaN: not computed
                L.append(f"  {k:3d}  {ir.nu[k]:9.6f}          -       -  "
                         f"|  {ir.cum[k]:9.6f}          -")
            else:
                L.append(f"  {k:3d}  {ir.nu[k]:9.6f}  {ir.se[k]:9.6f}  "
                         f"{t[k]:6.2f}  |  {ir.cum[k]:9.6f}  {ir.se_cum[k]:9.6f}")
        if ir.gain == ir.gain and ir.se_gain == ir.se_gain:
            tg = ir.gain / ir.se_gain if ir.se_gain > 1e-15 else 0.0
            L.append(f"    gain nu(1) = {ir.gain:.6f}  "
                     f"(s.e. {ir.se_gain:.6f}, t = {tg:.2f})")
        elif ir.gain == ir.gain:
            L.append(f"    gain nu(1) = {ir.gain:.6f}")
    L.append("=" * 63)
    return "\n".join(L)
