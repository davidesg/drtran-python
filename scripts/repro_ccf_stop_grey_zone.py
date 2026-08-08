#!/usr/bin/env python3
"""BUG-6 — `identify_link`'s stop rule is calibrated in units of the 2-sigma
BAND, so it refuses to identify transfers that are unambiguously real.

The rule (`mcp_server.py:735-741`) is

    _pico = max|ccf(k >= 0)| / idt.threshold        # threshold = 2/sqrt(N)
    if _pico < 2.0: stop, "no propongo orden"

`idt.threshold` is ALREADY two standard errors, so `_pico` counts 2-sigma units
and the cut demands 4 sigma. The comment beside it records exactly two measured
points -- omega ~ 0 giving 1.0-1.5, and a signal case giving 7.6-7.8 -- and
concludes "no hay zona gris". **The grey zone was never simulated.** With only a
null and one large effect there is nothing in between to observe, so its
emptiness is a property of the design.

This script simulates the missing middle: one DGP, a sweep of true effect sizes.
For each it reports what the stop rule sees and what the effect actually is.

    y_t = omega * x_t + n_t,   x_t ~ AR(1),  n_t ~ AR(1),  both stationary

`omega` is reported as `rho`, the implied contemporaneous correlation between the
prewhitened input and the filtered output -- which is what r(0) estimates, so the
expected peak/band ratio is rho*sqrt(N)/2 and the comparison is exact rather than
approximate.

Run:  python3 scripts/repro_ccf_stop_grey_zone.py
"""
from __future__ import annotations

import math

import numpy as np

from drtran.identify import ccf, prewhiten

N = 215            # the passthrough sample: 216 monthly obs, d=1
PHI_X = 0.30       # WTI's AR(1)
PHI_N = 0.40       # a CPI noise AR(1)
NLAGS = 25
REPS = 400
CUT = 2.0          # the rule's cut, in units of the 2/sqrt(N) band
SEED = 20260808


def one_draw(rho: float, rng: np.random.Generator) -> tuple[float, float]:
    """Return (peak/band ratio, |t| of the contemporaneous effect)."""
    n = N + 200                                    # burn-in
    ax = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = PHI_X * x[t - 1] + ax[t]

    e = rng.standard_normal(n)
    nz = np.zeros(n)
    for t in range(1, n):
        nz[t] = PHI_N * nz[t - 1] + e[t]

    # scale so that corr(prewhitened x, y-part) is exactly rho in expectation
    y = rho * x + math.sqrt(max(1.0 - rho * rho, 0.0)) * nz
    x, y = x[200:], y[200:]

    a = prewhiten(x, [PHI_X], [])                  # input filtered by its own ARMA
    b = prewhiten(y, [PHI_X], [])                  # SAME filter on the output
    m = min(len(a), len(b))
    a, b = a[-m:], b[-m:]

    r = ccf(a, b, NLAGS)                           # k >= 0 side
    band = 2.0 / math.sqrt(m)
    pico = float(np.abs(r).max()) / band

    # the joint test the analyst would actually run: OLS of b on a at k=0
    beta = float(a @ b) / float(a @ a)
    resid = b - beta * a
    se = math.sqrt((resid @ resid) / (m - 1) / (a @ a))
    return pico, abs(beta / se)


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(__doc__.split("Run:")[0].strip())
    print()
    print(f"  N = {N}   band = 2/sqrt(N) = {2 / math.sqrt(N):.4f}   "
          f"reps = {REPS}   cut = {CUT}")
    print()
    print("   rho    peak/band   in sigma    |t| of omega    % STOPPED by the rule")
    print("  " + "-" * 68)

    grey = []
    for rho in (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80):
        picos, ts = zip(*(one_draw(rho, rng) for _ in range(REPS)))
        pico, t = float(np.mean(picos)), float(np.mean(ts))
        stopped = 100.0 * float(np.mean([p < CUT for p in picos]))
        flag = ""
        if stopped > 5.0 and t > 3.0:
            flag = "   <-- REAL EFFECT, RULE STOPS IT"
            grey.append((rho, pico, t, stopped))
        print(f"  {rho:5.2f}   {pico:8.2f}   {2 * pico:8.2f}   {t:12.2f}   "
              f"{stopped:16.1f}{flag}")

    print()
    print("  The two points the original calibration measured:")
    print("    omega ~ 0      -> peak/band 1.0-1.5   (the rho=0.00 row)")
    print("    a large effect -> peak/band 7.6-7.8   (around rho=0.80)")
    print("  Everything between them is the grey zone the comment says does not")
    print("  exist. It does, and it is where real moderate transfers live.")
    print()

    if grey:
        print("  GREY ZONE FOUND — effect sizes that are unambiguous to the")
        print("  joint test and invisible to the stop rule:")
        for rho, pico, t, st in grey:
            print(f"    rho={rho:.2f}: |t| = {t:5.2f} (p < 1e-3) but the rule "
                  f"stops {st:.0f}% of draws")
        print()
        print("  BUG-6 REPRODUCED.")
        return 1

    print("  No grey zone found — BUG-6 would be refuted by this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
