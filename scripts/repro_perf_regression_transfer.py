"""
BUG-9 reproduction — the transfer fit's per-iteration wall time blows up with the
SEASONAL-AR order, config-dependent, NOT a uniform slowdown.

Self-contained: uses repo-root `.pre` fixtures. Fits the SAME rational transfer
nu(B)=omega0/(1-delta1 B) (b=0, r=1, s=0) on two outputs that differ only in whether
the noise carries a seasonal AR:

  * Spain CPI (ES_CPI_m10, P=0, no seasonal AR)  <-  WTI  -> FAST
  * euro-area HICP (EA_HICP_sar, P=1, seasonal AR at B^12)  <-  Brent -> SLOW

Both converge in ~24 iterations (termcode=1). The seasonal-AR case takes ~15x the
per-iteration time — the cost is the exact-VARMA likelihood evaluation, whose state
grows with the max AR lag (12 for the seasonal factor). The C reference (its own
`check_scale` docstring) does this class of fit in "23 iterations and one second".

    python3 scripts/repro_perf_regression_transfer.py
"""
import os
import time

import drtran
from drtran.pre import load_pre
from drtran.cast import Link, build_cast_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def timed(out_pre, inp_pre, label):
    cs = build_cast_spec([load_pre(os.path.join(ROOT, out_pre)),
                          load_pre(os.path.join(ROOT, inp_pre))],
                         links=[Link(0, 1, b=0, r=1, s=0)])
    t = time.time(); f = drtran.fit(cs, maxits=300); el = time.time() - t
    print(f"{label:38s} wall {el:6.1f}s  nit={f.nit:>2}  termcode={f.termcode}  "
          f"{el / max(f.nit, 1):5.2f}s/iter")
    return el, f.nit


print("rational transfer  nu(B)=omega0/(1-delta1 B)  (b=0,r=1,s=0)")
a = timed("ES_CPI_m10.1.pre", "WTI_ar1.1.pre", "Spain CPI (P=0, no seasonal AR) <- WTI")
b = timed("EA_HICP_sar.pre", "BRENT_ar1.pre", "EA HICP (P=1, seasonal AR B^12) <- Brent")
print(f"\nper-iteration ratio (seasonal / non-seasonal): "
      f"{(b[0]/max(b[1],1)) / (a[0]/max(a[1],1)):.0f}x  "
      f"(C reference for either: ~0.04 s/iter)")
