#!/usr/bin/env python3
"""BUG-7 — `chi_test(..., first=1)` shifts the Ljung-Box divisor by one lag,
so the EXOGENEITY portmanteau does not match the C.

`diagnose.py:chi_test` builds the divisor from the position in the sum rather
than from the lag:

    idx = np.arange(first, len(r))
    div = np.array([n - i + 1 for i in range(1, len(idx) + 1)], float)

`i` restarts at 1 whatever `first` is, so:

    first=0  (transfer test)     lag k  ->  divisor n - k        <- matches the C
    first=1  (exogeneity test)   lag k  ->  divisor n - k + 1    <- one lag late

The C (`drvarma_v.04/src/diagnose.c:278-285`) is 1-based and `corr[1]` is the
CONTEMPORANEOUS lag:

    for (i = 1; i <= lags; i++)
        chisqr += corr[i]*corr[i] / (nobs - i + 1);

so lag k always carries divisor `nobs - k`, on both sides. The C never
recomputes the k>0 sum from scratch — it SUBTRACTS the contemporaneous term
(`diagnose.c:430-431`):

    Q(k>0) = ChiTestC(corr, lags+1, n) - corr[1]*corr[1]*(n+2)

which is an identity the Python must satisfy and does not.

Consequence: `Q_exog` is understated by about `1/(n-k)` per lag — ~0.5 % at
n=215 — which biases the exogeneity test **towards declaring the input
exogenous**. Small, but it is a fidelity break in a port whose declared property
is reproducing the C, and it points the wrong way: the error hides feedback
rather than inventing it.

Run:  python3 scripts/repro_chitest_divisor_offbyone.py
"""
from __future__ import annotations

import numpy as np

from drtran.diagnose import chi_test

N = 215
NLAGS = 24
SEED = 7
TOL = 1e-9


def main() -> int:
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(NLAGS + 1) / np.sqrt(N)

    q_ge0, _ = chi_test(r, N, first=0)     # k >= 0, contemporaneous included
    q_gt0, _ = chi_test(r, N, first=1)     # k > 0

    # The C's identity, diagnose.c:430-431
    expected = q_ge0 - r[0] ** 2 * (N + 2)

    print(__doc__.split("Run:")[0].strip())
    print()
    print(f"  n = {N}   lags = {NLAGS}   seed = {SEED}")
    print()
    print(f"  chi_test(first=0)                    = {q_ge0:.9f}")
    print(f"  chi_test(first=1)  as implemented    = {q_gt0:.9f}")
    print(f"  chi_test(first=1)  by the C identity = {expected:.9f}")
    print(f"  discrepancy                          = {q_gt0 - expected:+.9f}"
          f"   ({100 * (q_gt0 - expected) / expected:+.4f} %)")
    print()
    print("  divisor applied to each lag:")
    print("    lag        the C / first=0        first=1 (as implemented)")
    for k in (1, 2, 3, NLAGS):
        print(f"    {k:3d}        {N - k:14d}        {N - k + 1:22d}")
    print()

    # An independent check that isolates the divisor from the lag-0 term:
    # zero out r[0] and the two branches MUST agree exactly.
    r0 = r.copy()
    r0[0] = 0.0
    a, _ = chi_test(r0, N, first=0)
    b, _ = chi_test(r0, N, first=1)
    print("  with r[0] set to 0 the two branches sum the SAME terms, so any")
    print("  difference is the divisor alone:")
    print(f"    first=0 = {a:.9f}")
    print(f"    first=1 = {b:.9f}")
    print(f"    gap     = {b - a:+.9f}   ({100 * (b - a) / a:+.4f} %)")
    print()

    if abs(b - a) > TOL:
        print("  BUG-7 REPRODUCED: the exogeneity branch uses the wrong divisor.")
        print()
        print("  Fix: key the divisor to the LAG, not to the position, e.g.")
        print("      div = np.array([n - k for k in idx], float)")
        print("  which reduces to the C for both values of `first`.")
        return 1

    print("  Branches agree — BUG-7 would be refuted by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
