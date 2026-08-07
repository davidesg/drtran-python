"""The EMBEDDED cast: the transfer INSIDE the VARMA, with nothing subtracted.

Port of `build_embedded_varma` (`tran_shootx.c`). It is the **default** cast,
here as in the C, and the underlying reason is forecasting.

Why it is the right one for forecasting
---------------------------------------
`elf` receives the series **as they are** and **everything** comes out of it,
initial conditions included: the exact likelihood takes care of the pre-sample
initialisation. The subtracting cast, by contrast, has to truncate the
convolution at the start of the sample (it assumes w=0 before the first datum),
so it introduces an approximation the forecast carries along. With the embedded
cast no piece is computed outside `elf`.

**Beware of an easy and false reading:** the embedded cast does NOT give a higher
likelihood than the subtracting one. The two do not measure the same thing — the
subtracting one models the NOISE N = w_Y - transfer, and the embedded one models
the OBSERVED series w_Y with the transfer inside the VARMA. Measured, the
embedded one comes out lower (-721.8015 against -721.7202 on the canonical case
with r=1), and the C binary shows exactly the same pattern. The embedded cast's
advantage is the absence of truncation, not a better fit.

The mathematics
---------------
For row *i*, the transfer model is

    phi_i(B)*[ Y_i,t - SUM_k (omega_k(B)/delta_k(B))*B^b_k*X_in_k,t ] = theta_i(B)*a_i,t

Multiplying by D_i(B) = PROD_k delta_k(B) (the denominators of the links that
come INTO i) the fractions disappear and what is left is polynomial:

    diagonal:      phi_i(B)*D_i(B)
    off-diagonal:  -phi_i(B)*omega_k(B)*B^b_k*(D_i(B)/delta_k(B))
    MA:            D_i(B)*theta_i(B)

All in PLAIN form (a(B) = a_0 + a_1 B + ..., with a_0 = 1 in the ARMA). The ARMA
are stored as (1 - phi_1 B - ...), so they change sign on the way to plain form.

`p` and `q` depend ONLY on the orders (p_ord, q_ord, b, r, s), not on the
parameter values: they are constant throughout the optimisation.
"""

from __future__ import annotations

import math

import numpy as np

from .cast import build_sigma, compute_irf
from .cast import ar_is_stationary


def poly_mul(a, b):
    """Product of polynomials in plain form."""
    return np.convolve(np.asarray(a, float), np.asarray(b, float))


def _topological_order(m, links):
    """Series ordered with the inputs before their outputs.

    Needed for the means: an output's mean depends on its input's. If there is a
    cycle the natural order is returned (the C assumes acyclic; here at least it
    does not hang).
    """
    incoming = {i: set() for i in range(m)}
    for l in links:
        incoming[l.out].add(l.inp)
    order, pending = [], set(range(m))
    while pending:
        ready = [i for i in sorted(pending) if not (incoming[i] & pending)]
        if not ready:                        # a cycle: cannot be ordered
            return list(range(m))
        order.extend(ready)
        pending -= set(ready)
    return order


def normalize_phi0(phi, theta, sigma):
    """Premultiply by Phi(0)^-1 to leave Phi_0 = I, which is what `elf` expects.

    With Theta(0) = I:
        phi[k]   <- Phi_0^-1 * phi[k]
        theta[k] <- Phi_0^-1 * theta[k] * Phi_0
        Sigma    <- Phi_0^-1 * Sigma * Phi_0^-T

    It is needed as soon as there is a CONTEMPORANEOUS transfer (b=0), which puts
    omega_0 at lag zero. Returns `(phi, theta, sigma, phi0)`; `phi0` is kept
    because the diagnostics need the STRUCTURAL residuals, not the reduced-form
    ones: a_structural = Phi(0)*a_reduced. Without undoing it, the adequacy test
    measures the contemporaneous correlation the transfer itself generates
    (Sigma_12 = omega_0*sigma2_X) and calls it misspecification.
    """
    phi0 = phi[0].copy()
    m = phi0.shape[0]
    if np.allclose(phi0, np.eye(m), atol=1e-14):
        return phi[1:].copy(), theta[1:].copy(), sigma, phi0
    inv = np.linalg.inv(phi0)
    ph = np.array([inv @ phi[k] for k in range(1, len(phi))])
    th = np.array([inv @ theta[k] @ phi0 for k in range(1, len(theta))])
    sg = inv @ sigma @ inv.T
    return ph, th, sg, phi0


def cast_embedded(x, cast_spec, with_phi0=False):
    """Parameter vector -> VARMA structure with the transfer EMBEDDED.

    Same signature as `cast.cast_diagonal`: returns
    `(phi, theta, mu, w, sigma, ifault)` ready for `elf`, with `phi`/`theta`
    already normalised (Phi_0 = I) and `w` with **nothing subtracted**.

    `with_phi0=True` appends Phi(0). The DIAGNOSTICS ask for it: `elf` returns
    the REDUCED-FORM residuals, and with a contemporaneous transfer (b=0) those
    come out correlated **by construction** (Sigma_12 = omega_0*sigma2_X). The
    structural ones, which are what the model assumes orthogonal, come from
    undoing the normalisation: a_structural = Phi(0)*a.
    """
    from fue.cast_us import cast_us_py

    x = np.asarray(x, float)
    m = cast_spec.m
    links = cast_spec.links
    idx = 0

    om, de = [], []
    for l in links:
        om.append(np.asarray(x[idx:idx + l.s + 1], float)); idx += l.s + 1
        de.append(np.asarray(x[idx:idx + l.r], float)); idx += l.r

    ps, qs, phis, thetas, mus, ws = [], [], [], [], [], []
    for sc in cast_spec.series:
        xi = x[idx:idx + sc.npar]; idx += sc.npar
        p, q, phi, theta, mu, w, ifault = cast_us_py(xi, sc.est_spec)
        if ifault:
            return None, None, None, None, None, int(ifault)
        ps.append(int(p)); qs.append(int(q))
        phis.append(np.asarray(phi, float)); thetas.append(np.asarray(theta, float))
        mus.append(float(mu)); ws.append(np.asarray(w, float))

    sigma, idx, ifa_q = build_sigma(x, idx, m)
    if ifa_q:
        return None, None, None, None, None, int(ifa_q)

    # --- Row by row: polynomials in plain form -----------------------------
    P = [[np.zeros(1) for _ in range(m)] for _ in range(m)]
    M = [np.zeros(1) for _ in range(m)]

    for i in range(m):
        A = np.concatenate(([1.0], -phis[i][:ps[i]]))       # phi_i, plain form
        T = np.concatenate(([1.0], -thetas[i][:qs[i]]))     # theta_i, plain form

        incoming = [(k, l) for k, l in enumerate(links) if l.out == i]

        # D_i = product of the denominators of the links coming into i
        Di = np.ones(1)
        for k, l in incoming:
            if l.r > 0:
                Di = poly_mul(Di, np.concatenate(([1.0], -de[k])))

        P[i][i] = poly_mul(A, Di)                            # diagonal

        for k, l in incoming:
            # D_i without THIS link's delta
            acc = np.ones(1)
            for k2, l2 in incoming:
                if k2 != k and l2.r > 0:
                    acc = poly_mul(acc, np.concatenate(([1.0], -de[k2])))
            # omega_k(B)*B^b in plain form. BJR convention: the leading term
            # adds, the rest SUBTRACT — the same as compute_irf and fue's calcnu.
            wpoly = np.zeros(l.b + l.s + 1)
            for j in range(l.s + 1):
                wpoly[l.b + j] = om[k][j] if j == 0 else -om[k][j]
            acc = poly_mul(poly_mul(acc, wpoly), A)
            n1, n2 = len(P[i][l.inp]), len(acc)
            nn = max(n1, n2)
            buf = np.zeros(nn)
            buf[:n1] = P[i][l.inp]
            buf[:n2] -= acc
            P[i][l.inp] = buf

        M[i] = poly_mul(Di, T)                               # the row's MA

    p = max(1, max(len(P[i][j]) - 1 for i in range(m) for j in range(m)))
    q = max([0] + [len(M[i]) - 1 for i in range(m)])

    # --- elf's convention: Phi(B) = Phi_0 - SUM_{k>=1} Phi_k B^k -----------
    PHI = np.zeros((p + 1, m, m))
    THETA = np.zeros((q + 1, m, m))
    for i in range(m):
        for j in range(m):
            poly = P[i][j]
            PHI[0, i, j] = poly[0]
            for k in range(1, min(len(poly), p + 1)):
                PHI[k, i, j] = -poly[k]
        THETA[0, i, i] = M[i][0]
        for k in range(1, min(len(M[i]), q + 1)):
            THETA[k, i, i] = -M[i][k]

    # --- Means: NO adjustment. mu is the series' MEAN, not an intercept -----
    #
    # Box-Jenkins writes the model in DEVIATIONS from the mean,
    #     (w_Y - mu_Y) = nu(B)*(w_X - mu_X) + N_t,
    # and that is why the transfers come out clean: taking expectations,
    # E[w_Y] = mu_Y, inheriting nothing from the input. Multiplying by delta(B),
    #     phi_Y*delta*(w_Y - mu_Y) - phi_Y*omega*B^b*(w_X - mu_X) = delta*theta_Y*a_Y,
    # which is exactly row 1 of Phi(B)(w - mu) = Theta(B)a WITH mu = (mu_Y, mu_X).
    # There is no extra term to add.
    #
    # The C does `MU[i] += (SUM_k omega_k / delta(1))*MU[inp]`
    # (tran_shootx.c:288), which would correspond to the INTERCEPT
    # parametrisation, w_Y = c + nu(B)*w_X + N. The two are the same family
    # reparametrised AS LONG AS mu_Y is free — verified: with the output's mu
    # free they reach the same optimum to 1e-12. They diverge when mu_Y is FIXED,
    # and there they impose different things: in deviations, mu_Y = 0 means
    # E[w_Y] = 0; with an intercept it means E[w_Y] = nu(1)*mu_X != 0.
    #
    # The deviations specification is the one coherence with fue demands: the
    # `.pre`'s mu is the MEAN fue estimated for that series. If fue fixed it at
    # zero, that means the series has no drift, not that an intercept is zero.
    MU = np.asarray(mus, float)

    # --- The series, with NOTHING SUBTRACTED: that is what changes -----------
    n = min(len(w) for w in ws)
    W = np.column_stack([w[len(w) - n:] for w in ws])

    # The C's constraint (shootx [12]), CORRECTED — see `ar_is_stationary`.
    # The original tests |phi[0]| >= 0.999 for EVERY order, and phi[0] is only
    # a root when p = 1. This checks the roots, which is what the C does three
    # lines below for the MA, with `chekma`.
    for i in range(m):
        if ps[i] >= 1 and not ar_is_stationary(phis[i][:ps[i]]):
            return None, None, None, None, None, 1

    ph, th, sg, phi0 = normalize_phi0(PHI, THETA, sigma)
    if with_phi0:
        return ph, th, MU, W, sg, 0, phi0
    return ph, th, MU, W, sg, 0


def loglik_embedded(x, cast_spec, xitol=-1e-3):
    """Exact concentrated log-likelihood with the EMBEDDED cast."""
    from drvarma.estimate_py import _elf_f1f2

    phi, theta, mu, w, sigma, ifault = cast_embedded(x, cast_spec)
    if ifault:
        return float("-inf"), int(ifault)
    n, m = w.shape
    f1, f2, ifa = _elf_f1f2(w, mu, phi, theta, sigma, xitol)
    if ifa or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def nu_at_one(omega, delta):
    """nu(1) = omega(1)/delta(1), the link's steady-state gain.

    It is exposed because it is what multiplies the input's mean, and because the
    C computes it by summing ALL the omegas (`w1 += omega[k][kk]`,
    tran_shootx.c:288) instead of alternating the sign. With s=0 both agree; with
    s>0 they do not, and the program's own BJR convention says
    omega(1) = omega_0 - omega_1 - ... (that is how m6's `.cns` uses it:
    "nu_num(1)=0 => omega3[0] = omega3[1]+omega3[2]+omega3[3]").
    """
    omega = np.asarray(omega, float)
    delta = np.asarray(delta, float)
    w1 = omega[0] - float(np.sum(omega[1:]))
    d1 = 1.0 - float(np.sum(delta))
    return w1 / d1


def check_nu_consistency(omega, delta, b=0, length=400, tol=1e-6):
    """nu(1) must be the SUM of the nu weights. Checks the sign convention."""
    s = float(np.sum(compute_irf(omega, delta, b, length)))
    return abs(s - nu_at_one(omega, delta)) < tol, s, nu_at_one(omega, delta)
