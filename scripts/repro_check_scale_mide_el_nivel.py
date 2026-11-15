#!/usr/bin/env python3
"""BUG-68: check_scale measures the LEVEL of the stationary series, not its
spread, and with d = 0 it warns about a scale that is fine.

`pre.check_scale` takes the median of |w|, w the stationary series. With
d = D = 0, w is the transformed level itself, so the median is the level: a
series about 1000 in levels with refactor = 1 moves by ~1 per period -- the
comfortable scale the docstring aims at -- and is called "huge", with the
advice to regenerate with refactor = 0.01, which would make its moves ~0.01.

A synthetic monthly AR(1) about 1000 (innovations of sd 1), fitted by fue with
lambda = 1 and refactor = 1, written as a .pre. Exits 1 while check_scale warns
on it; 0 when it does not.
"""
import os
import sys
import tempfile
import warnings

import numpy as np

import fue
from drtran.pre import check_scale, load_pre

warnings.filterwarnings("ignore")


def main():
    rng = np.random.default_rng(7)
    n = 160
    y = np.zeros(n)
    y[0] = 1000.0
    for t in range(1, n):
        y[t] = 1000.0 + 0.6 * (y[t - 1] - 1000.0) + rng.normal(0, 1.0)
    ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name="NIVELES")
    m = fue.Model(ts, d=0, D=0, boxlam=1.0, ar=[[0.1]], ar_free=[[True]],
                  mu=float(y.mean()), estimate_mu=True, refactor=1.0)
    m.fit()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "NIVELES.pre")
        m.write_pre(p)
        aviso = check_scale(load_pre(p))
    print(f"sd of the series fue fits: {y.std():.3g}; of its changes: {np.diff(y).std():.3g}")
    print(f"check_scale: {aviso or 'no warning'}")
    return 1 if aviso else 0


if __name__ == "__main__":
    sys.exit(main())
