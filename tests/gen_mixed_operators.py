"""Tres brazos: mismo omega_0, y solo cambia DONDE esta el exceso de raices.

  M  output d=1 D=0  vs  input d=1     f0: 1 vs 1   Delta = 1        (control)
  S  output d=0 D=1  vs  input d=1     f0: 1 vs 1   Delta = S(B),  S(1)=12
  Z  output d=1 D=1  vs  input d=1     f0: 2 vs 1   Delta = 1-B^12,  Delta(1)=0

S es el caso de David: desajuste PURAMENTE estacional, mismo orden en f=0.
Z es el caso observado (FR/DE/EMU vs WTI).
"""
import numpy as np, os
FREQ, N, BURN, OMEGA = 12, 300, 400, 1.0
D = os.path.dirname(os.path.abspath(__file__))

def write_pre(path, name, data, lam, d, Dd):
    L = ["*"*48, "*        Input file for program DRVUS          *", "*"*48, "",
         "** Frequency of time series: either 1(A), 4(Q) or 12(M):", " %d" % FREQ,
         "** Number of observations and starting date of time series:",
         " %d  1 1990 %s" % (len(data), name),
         "** Number of deterministic variables (including seasonal components):", "0",
         "**Number and orders of regular AR operators:", "0",
         "** Number and orders of annual AR operators:", "0",
         "** Number and orders of regular MA operators:", "0",
         "** Number and orders of anual MA operators:", "0",
         "** Number and frequencies of regular AR(2) operators with fixed frequency:", "0",
         "** Number and frequencies of regular MA(2) operators with fixed frequency:", "0",
         "** Mean parameter (mu):", "0",
         "** Box-Cox lambda, regular differences and complete annual differences:",
         "%.2f %d %d" % (lam, d, Dd),
         "** Individual factors of the annual difference (from freq 0.0): ",
         " 0 0 0 0 0 0 0",
         "** ACF/PACF bands (0 Automatic) and reescaling factor: ", " 0.00 1.00",
         "** Time series (stochastic and non-standard deterministic variables): "]
    L += ["%.10f " % v for v in data]
    open(path, "w").write("\n".join(L) + "\n")

rng = np.random.default_rng(20260809)
n = N + BURN
e = rng.standard_normal(n)          # w_x, RUIDO BLANCO: input d=1
a = rng.standard_normal(n)          # innovacion del ruido N
x = np.cumsum(e)                    # input en NIVEL, d=1 D=0

def integrate_seasonal(v):          # invierte (1 - B^12)
    z = np.zeros(len(v))
    for t in range(len(v)):
        z[t] = v[t] + (z[t-FREQ] if t >= FREQ else 0.0)
    return z

arms = {}
arms["M"] = (np.cumsum(a),                              1, 0)  # N es grad-estacionario
arms["S"] = (integrate_seasonal(a),                     0, 1)  # N es grad_12-estacionario
arms["Z"] = (integrate_seasonal(np.cumsum(a)),          1, 1)  # N es grad grad_12-estacionario

for tag, (Nz, dy, Dy) in arms.items():
    y = OMEGA * x + Nz
    write_pre(os.path.join(D, "%s_Y.pre" % tag), "%s_Y" % tag, y[BURN:], 1.0, dy, Dy)
    write_pre(os.path.join(D, "%s_X.pre" % tag), "%s_X" % tag, x[BURN:], 1.0, 1, 0)
    print("%s: output d=%d D=%d  input d=1 D=0   omega_0 = %.1f" % (tag, dy, Dy, OMEGA))
