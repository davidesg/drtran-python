"""BUG-22: el binario en C aplica Box-Cox con lambda=1 como (y-1), y fue como y.

Con lambda=1 fue NO transforma: `fue.c:BoxCox` escribe `refactor*y`. El C de
drtran usa la formula general sin la excepcion,
`refactor*(pow(y,lam)-1)/lam`, que con lambda=1 es `refactor*(y-1)`
(`drtran.c:885-888`, y las mismas lineas en 1780 y 2300).

Dos consecuencias, y este script mide las dos:

  1. La serie estacionaria del C esta desplazada `refactor` unidades. Con la
     media LIBRE es una reparametrizacion (mu sale una unidad por debajo y la
     verosimilitud coincide), asi que el escalon diagonal homologa. Con la media
     FIJA --incluida mu=0, la linea `0` de siempre-- y SIN diferenciar
     (d=D=0) es OTRO modelo: la verosimilitud diagonal deja de ser la suma de
     las de fue. Con d>=1 o D>=1 la diferencia se come la constante.
  2. El C rechaza datos <= 0 aunque lambda=1 no los necesite positivos: una
     serie en niveles que cruce el cero (un tipo, un saldo, un crecimiento) no
     entra.

El puerto en Python (`drtran.load_pre`) usa la convencion de fue y homologa en
los tres casos: el defecto es solo del C.

Ejecutar:  python3 scripts/repro_boxcox_lambda1_c.py
"""
import os
import re
import subprocess
import tempfile
import warnings

import numpy as np

import fue

warnings.filterwarnings("ignore")

BIN = "/home/david/Dropbox/SRC/drtran/bin/drtran"
N = 120


def _ar(rng, phi):
    e = rng.standard_normal(N)
    y = np.zeros(N)
    for t in range(1, N):
        y[t] = phi * y[t - 1] + e[t]
    return y


def _drtran(d, y_pre, x_pre, tag):
    out = os.path.join(d, tag + ".txt")
    r = subprocess.run([BIN, y_pre, x_pre, "-0", "-o", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip().splitlines()[0]
    txt = open(out).read()
    ll = float(re.search(r"Log-likelihood = (\S+)", txt).group(1))
    mu = re.findall(r"^(mu\[\d\])\s+(-?\d+\.\d+)", txt, re.M)
    return ll, mu


def main():
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(5)
    X = _ar(rng, 0.5) + 50.0 / 7
    Y = _ar(rng, 0.6) + 10.0 / 3
    Y[60:] += 2.0

    tx = fue.TimeSeries(X, freq=12, start=(2000, 1), name="X")
    mx = fue.Model(tx, ar=[[0.3]], mu=7.0, estimate_mu=True).fit()
    x_pre = os.path.join(d, "X.pre")
    mx.write_pre(x_pre)
    print(f"X: lambda=1, mu libre; fue mu = {mx._result.params[-1]:.6f}\n")

    ty = fue.TimeSeries(Y, freq=12, start=(2000, 1), name="Y")
    casos = {
        "mu_Y libre": dict(mu=3.0, estimate_mu=True),
        "mu_Y fija en 0": dict(mu=0.0, estimate_mu=False),
        "mu_Y fija en 10/3": dict(mu=10.0 / 3, estimate_mu=False),
    }
    print(f"{'caso':20s} {'fue (suma)':>14s} {'drtran C':>14s} {'dif':>10s}  medias del C")
    for tag, kw in casos.items():
        my = fue.Model(ty, ar=[[0.5]], **kw).fit()
        y_pre = os.path.join(d, "Y.pre")
        my.write_pre(y_pre)
        s = mx._result.loglik + my._result.loglik
        ll, mu = _drtran(d, y_pre, x_pre, "o")
        print(f"{tag:20s} {s:14.6f} {ll:14.6f} {ll - s:+10.6f}  {mu}")

    # 2. datos que cruzan el cero, lambda=1
    rng = np.random.default_rng(1)
    z = _ar(rng, 0.5)
    tz = fue.TimeSeries(z, freq=12, start=(2000, 1), name="Z")
    mz = fue.Model(tz, ar=[[0.3]], mu=0.0, estimate_mu=True).fit()
    z_pre = os.path.join(d, "Z.pre")
    mz.write_pre(z_pre)
    _, err = _drtran(d, z_pre, x_pre, "z")
    print(f"\nserie que cruza el cero, lambda=1: fue estima (loglik "
          f"{mz._result.loglik:.6f}); drtran C: {err}")


if __name__ == "__main__":
    main()
