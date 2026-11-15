#!/usr/bin/env python3
"""BUG-69: school.stationary_series transforms with y**lambda, and fue with
(y**lambda - 1)/lambda: for lambda not 0 or 1 Brajin's R2 (A.28) comes out
negative.

r2_brajin divides the residual variance -- in fue's scale, refactor *
(y**lambda - 1)/lambda -- by the variance of `stationary_series`, which uses
refactor * y**lambda. The two differ by a factor 1/lambda**2 in variance: with
lambda = +-0.5 the denominator is 4 times too small and R2 = 1 - 4(1 - R2).

Two synthetic series, Y_t = 0.5 Y_{t-1} + 0.4 X_{t-1} + a_t in logs, about 100;
each fitted by fue as an AR(1) with mu, with lambda = 1 (where both transforms
agree) and with lambda = 0.5; load_pre, a link X -> Y with b = 1, estimate, and
the R2 transition mtram reports. Exits 1 while the R2 with lambda = 0.5 is
negative; 0 when it is in [0, 1].
"""
import os
import re
import sys
import tempfile
import warnings

import numpy as np

import fue
from fue.forecast import _boxcox
import drtran.mcp_server as M

warnings.filterwarnings("ignore")


def pares(n=160, seed=11):
    rng = np.random.default_rng(seed)
    e, x = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + rng.normal(0, 0.01)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + 0.4 * x[t - 1] + rng.normal(0, 0.005)
    return 100.0 * np.exp(e), 100.0 * np.exp(x)


def r2(lam):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for nm, s in zip(("Y", "X"), pares()):
            ts = fue.TimeSeries(list(map(float, s)), freq=12, start=(2000, 1), name=nm)
            w = np.array([_boxcox(float(v), lam, 100.0) for v in s])
            m = fue.Model(ts, d=0, D=0, boxlam=lam, ar=[[0.1]], ar_free=[[True]],
                          mu=float(w.mean()), estimate_mu=True, refactor=100.0)
            m.fit()
            p = os.path.join(d, f"{nm}.pre")
            m.write_pre(p)
            paths.append(p)
        name = f"l{lam:g}"
        M.load_pre(name, ", ".join(paths))
        M.set_network(name, '[{"out":0,"inp":1,"b":1,"r":0,"s":0}]')
        out = M.estimate(name)
    g = re.search(r"R² \(Brajín A\.28.*?\*\*(-?[0-9.]+)\*\* a \*\*(-?[0-9.]+)\*\*", out)
    return float(g.group(1)), float(g.group(2))


def main():
    a = r2(1.0)
    b = r2(0.5)
    print(f"lambda = 1  : R2 from {a[0]:.4f} to {a[1]:.4f}")
    print(f"lambda = 0.5: R2 from {b[0]:.4f} to {b[1]:.4f}")
    abierto = min(b) < 0
    print("negative R2" if abierto else "R2 in [0, 1]")
    return 1 if abierto else 0


if __name__ == "__main__":
    sys.exit(main())
