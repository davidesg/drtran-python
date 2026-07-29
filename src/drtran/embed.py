"""El cast EMPOTRADO: la transferencia DENTRO del VARMA, sin restar nada.

Puerto de `build_embedded_varma` (`tran_shootx.c`). Es el cast por defecto del C
y es mejor que el de resta: `elf` recibe las series **tal cual** y su
verosimilitud exacta se ocupa de la inicialización pre-muestral, así que **no hay
truncamiento**. El cast por resta tiene que truncar la convolución al principio
de la muestra (asume w=0 antes del primer dato).

La matemática
-------------
Para la fila *i*, el modelo de transferencia es

    φᵢ(B)·[ Yᵢ,ₜ − Σₖ (ωₖ(B)/δₖ(B))·B^bₖ·X_inₖ,ₜ ] = θᵢ(B)·aᵢ,ₜ

Multiplicando por Dᵢ(B) = Πₖ δₖ(B) (los denominadores de los enlaces que ENTRAN
en i) desaparecen las fracciones y queda polinomial:

    diagonal:        φᵢ(B)·Dᵢ(B)
    fuera diagonal:  −φᵢ(B)·ωₖ(B)·B^bₖ·(Dᵢ(B)/δₖ(B))
    MA:              Dᵢ(B)·θᵢ(B)

Todo en forma LLANA (a(B) = a₀ + a₁B + …, con a₀ = 1 en los ARMA). Los ARMA se
guardan como (1 − φ₁B − …), así que al pasar a llana cambian de signo.

`p` y `q` dependen SÓLO de los órdenes (p_ord, q_ord, b, r, s), no de los valores
de los parámetros: son constantes a lo largo de la optimización.
"""

from __future__ import annotations

import math

import numpy as np

from .cast import compute_irf


def poly_mul(a, b):
    """Producto de polinomios en forma llana."""
    return np.convolve(np.asarray(a, float), np.asarray(b, float))


def _orden_topologico(m, links):
    """Series ordenadas con las entradas antes que sus salidas.

    Necesario para las medias: la media de una salida depende de la de su
    entrada. Si hay un ciclo se devuelve el orden natural (el C asume acíclico;
    aquí al menos no se cuelga).
    """
    entrantes = {i: set() for i in range(m)}
    for l in links:
        entrantes[l.out].add(l.inp)
    orden, pendientes = [], set(range(m))
    while pendientes:
        libres = [i for i in sorted(pendientes) if not (entrantes[i] & pendientes)]
        if not libres:                       # ciclo: no se puede ordenar
            return list(range(m))
        orden.extend(libres)
        pendientes -= set(libres)
    return orden


def normalize_phi0(phi, theta, sigma):
    """Premultiplica por Φ(0)⁻¹ para dejar Φ₀ = I, que es lo que espera `elf`.

    Con Θ(0) = I:
        phi[k]   ← Φ₀⁻¹·phi[k]
        theta[k] ← Φ₀⁻¹·theta[k]·Φ₀
        Σ        ← Φ₀⁻¹·Σ·Φ₀⁻ᵀ

    Hace falta en cuanto hay una transferencia CONTEMPORÁNEA (b=0), que mete ω₀
    en el retardo cero. Devuelve `(phi, theta, sigma, phi0)`; `phi0` se conserva
    porque los diagnósticos necesitan los residuos ESTRUCTURALES, no los de la
    forma reducida: a_estructural = Φ(0)·a_reducido. Sin deshacerlo, la prueba de
    adecuación mide la correlación contemporánea que la propia transferencia
    genera (Σ₁₂ = ω₀·σ²_X) y la llama mala especificación.
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


def cast_embedded(x, cast_spec):
    """Vector de parámetros → estructura VARMA con la transferencia EMPOTRADA.

    Misma firma que `cast.cast_diagonal`: devuelve
    `(phi, theta, mu, w, sigma, ifault)` listos para `elf`, con `phi`/`theta` ya
    normalizados (Φ₀ = I) y `w` **sin restar nada**.
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

    var = np.ones(m)
    for i in range(1, m):
        var[i] = math.exp(x[idx]); idx += 1
    sigma = np.diag(var)

    # --- Fila por fila: polinomios en forma llana --------------------------
    P = [[np.zeros(1) for _ in range(m)] for _ in range(m)]
    M = [np.zeros(1) for _ in range(m)]

    for i in range(m):
        A = np.concatenate(([1.0], -phis[i][:ps[i]]))       # φᵢ en forma llana
        T = np.concatenate(([1.0], -thetas[i][:qs[i]]))     # θᵢ en forma llana

        entrantes = [(k, l) for k, l in enumerate(links) if l.out == i]

        # Dᵢ = producto de los denominadores de los enlaces que entran en i
        Di = np.ones(1)
        for k, l in entrantes:
            if l.r > 0:
                Di = poly_mul(Di, np.concatenate(([1.0], -de[k])))

        P[i][i] = poly_mul(A, Di)                            # diagonal

        for k, l in entrantes:
            # Dᵢ sin el δ de ESTE enlace
            acc = np.ones(1)
            for k2, l2 in entrantes:
                if k2 != k and l2.r > 0:
                    acc = poly_mul(acc, np.concatenate(([1.0], -de[k2])))
            # ωₖ(B)·B^b en forma llana. Convención BJR: el líder suma, el resto
            # RESTA — la misma que compute_irf y que calcnu de fue.
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

        M[i] = poly_mul(Di, T)                               # MA de la fila

    p = max(1, max(len(P[i][j]) - 1 for i in range(m) for j in range(m)))
    q = max([0] + [len(M[i]) - 1 for i in range(m)])

    # --- Convención de elf: Φ(B) = Φ₀ − Σ_{k≥1} Φ_k B^k --------------------
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

    # --- Medias: aquí w_i es la serie OBSERVADA, no el ruido ---------------
    #     E[w_i] = mu_i + Σ_{k: out=i} g_k·E[w_inp(k)],  g_k = ν_k(1)
    # ν_k(1) = ω(1)/δ(1) con ω(1) = ω₀ − ω₁ − … (BJR) y δ(1) = 1 − δ₁ − …
    MU = np.zeros(m)
    for i in _orden_topologico(m, links):
        MU[i] = mus[i]
        for k, l in enumerate(links):
            if l.out != i:
                continue
            w1 = om[k][0] - float(np.sum(om[k][1:])) if l.s > 0 else om[k][0]
            d1 = 1.0 - float(np.sum(de[k])) if l.r > 0 else 1.0
            if abs(d1) < 1e-10:
                return None, None, None, None, None, 1
            MU[i] += (w1 / d1) * MU[l.inp]

    # --- Las series, SIN RESTAR NADA: es lo que cambia todo ----------------
    n = min(len(w) for w in ws)
    W = np.column_stack([w[len(w) - n:] for w in ws])

    for i in range(m):
        if ps[i] >= 1 and abs(phis[i][0]) >= 0.999:
            return None, None, None, None, None, 1

    ph, th, sg, _phi0 = normalize_phi0(PHI, THETA, sigma)
    return ph, th, MU, W, sg, 0


def loglik_embedded(x, cast_spec, xitol=-1e-3):
    """Log-verosimilitud exacta concentrada con el cast EMPOTRADO."""
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
    """ν(1) = ω(1)/δ(1), la ganancia en estado estacionario del enlace.

    Se expone porque es lo que multiplica la media de la entrada, y porque el C
    la calcula sumando TODOS los ω (`w1 += omega[k][kk]`, tran_shootx.c:288) en
    vez de alternar el signo. Con s=0 da lo mismo; con s>0 no, y la convención
    BJR del propio programa dice ω(1) = ω₀ − ω₁ − … (así lo usa el `.cns` de m6:
    «nu_num(1)=0 ⇒ omega3[0] = omega3[1]+omega3[2]+omega3[3]»).
    """
    omega = np.asarray(omega, float)
    delta = np.asarray(delta, float)
    w1 = omega[0] - float(np.sum(omega[1:]))
    d1 = 1.0 - float(np.sum(delta))
    return w1 / d1


def check_nu_consistency(omega, delta, b=0, length=400, tol=1e-6):
    """ν(1) debe ser la SUMA de los pesos ν. Comprueba la convención de signo."""
    s = float(np.sum(compute_irf(omega, delta, b, length)))
    return abs(s - nu_at_one(omega, delta)) < tol, s, nu_at_one(omega, delta)
