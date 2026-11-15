#!/usr/bin/env python3
"""BUG-70: mtram presents in the engine's units: the forecast's "s.e.(%)" is a
percentage only at refactor = 100 with logs, and a gain between two series
carries the ratio of their rescaling factors.

Two synthetic series, Y_t = 0.5 Y_{t-1} + 0.4 X_{t-1} + a_t in logs, about
100. The same data in three sets of .pre files:

  A  both in logs at refactor = 100   (what art writes)
  B  both in logs at refactor = 1     (what atsw-gui's GUI writes)
  C  Y at 100 and X at 1

The fit is the same in the three (the likelihood in the units of y agrees).
mtram prints the forecast's "s.e.(%)" as 0.5457 in A and 0.0055 in B, and the
gain as 0.4496 in A and B and 44.96 in C, "en la escala transformada".

Exits 1 while the s.e.(%) of A and B differ, or the gain of C -- in the
series' units, which mtram gives next to the engine's figure when the files
do not share a factor -- is not A's; 0 when the three say the same.
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
N = 200


def datos(seed=11):
    rng = np.random.default_rng(seed)
    e, x = np.zeros(N), np.zeros(N)
    for t in range(1, N):
        x[t] = 0.6 * x[t - 1] + rng.normal(0, 0.01)
    for t in range(1, N):
        e[t] = 0.5 * e[t - 1] + 0.4 * x[t - 1] + rng.normal(0, 0.005)
    return 100 * np.exp(e), 100 * np.exp(x)


def corre(fy, fx, tag):
    d = tempfile.mkdtemp()
    paths = []
    for nm, s, f in zip(("Y", "X"), datos(), (fy, fx)):
        ts = fue.TimeSeries(list(map(float, s)), freq=12, start=(2000, 1), name=nm)
        w = np.array([_boxcox(float(v), 0.0, f) for v in s])
        m = fue.Model(ts, d=0, D=0, boxlam=0.0, ar=[[0.1]], ar_free=[[True]],
                      mu=float(w.mean()), estimate_mu=True, refactor=float(f))
        m.fit()
        p = os.path.join(d, f"{nm}.pre")
        m.write_pre(p)
        paths.append(p)
    M.load_pre(tag, ", ".join(paths))
    M.set_network(tag, '[{"out":0,"inp":1,"b":1,"r":0,"s":0}]')
    est = M.estimate(tag)
    g = float(re.search(r"omega\(1\) = ([-+0-9.]+)", est).group(1))
    u = re.search(r"EN LAS UNIDADES DE CADA SERIE: ([-+0-9.]+)", est)
    if u:                       # the engine's figure stays; the reading is this one
        g = float(u.group(1))
    fc = M.forecast(tag, 3)
    se = float(re.search(r"\n\s+1\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", fc).group(1))
    print(f"{tag}: refactor Y={fy:g}, X={fx:g}:  gain (series' units) = {g:+.6f}"
          f"   forecast s.e.(%) at h=1 = {se}")
    return g, se


def main():
    ga, sa = corre(100.0, 100.0, "A")
    gb, sb = corre(1.0, 1.0, "B")
    gc, sc = corre(100.0, 1.0, "C")
    abierto = abs(sb - sa) > 0.1 * sa or abs(gc - ga) > 0.1 * abs(ga)
    print("the same fit reads differently by the files' factors" if abierto else "ok")
    return 1 if abierto else 0


if __name__ == "__main__":
    sys.exit(main())
