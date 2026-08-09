"""The cast: parameter vector -> VARMA structure.

Port of `tran_shootx.c` (`shootx`). It starts from the **diagonal case with no
transfer**, which is the design's gate: with a diagonal structure the exact
likelihood factorises, so the joint fit must reproduce the SUM of fue's
univariate ones. If that does not add up, the cast is wrong — never `elf`.

What is replicated from the C and what is not
---------------------------------------------
The SEMANTICS and the CONVENTIONS are replicated, not the engineering:

* `shootx`'s **parameter vector order**: transfers (omega, delta) per link ->
  ARMA per series -> deterministics -> means -> covariance. Keeping it is not an
  external contract (the `.cns` goes by name, not by position), but it makes a
  discrepancy with the C locatable by comparing position against position.
* The **covariance normalisation**: `Q[1][1] = 1` and `var_i = exp(x_i)` for
  i>1, with the scale concentrated into `sigma2`. That is a deliberate decision,
  not an accident: Mauricio's concentrated likelihood (1995, eq. 3.1) depends on
  Q only through a product invariant under Q -> cQ, so leaving all m variances
  free leaves a flat direction and a singular Hessian. The legacy code and
  drvarma leave them free; here they are not, on purpose.

Memory management, tensors, globals and 1-based indexing are NOT replicated.

What is NOT reimplemented
-------------------------
The stationary series. `fue.cast_us.cast_us_py()` already returns `w` with
Box-Cox, differencing and the deterministics subtracted — which is what
`build_stationary_series` does in the C. drtran uses fue's univariate cast per
series and only **assembles** the VARMA. Reimplementing it would create a second
source of truth for the most delicate part of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Link:
    """A transfer link: `out` receives from `inp` through omega(B)/delta(B)*B^b.

    A copy of the C's `struct Tlink`. Series indices are 0-based (the C uses
    1-based); `s` is the numerator's degree and `r` the denominator's, so the
    link contributes (s+1) + r free parameters.
    """

    out: int
    inp: int
    b: int = 0
    r: int = 0
    s: int = 0

    @property
    def npar(self):
        return (self.s + 1) + self.r


def compute_irf(omega, delta, b, length):
    """Weights nu of the rational filter omega(B)/delta(B) with delay b.

    Port of `compute_irf`::

        nu[t] = omega[t-1-b] + SUM_{j=1..r} delta[j]*nu[t-j]

    `nu[j]` weights the input at lag j-1 (1-based in the C; here `nu[0]` is the
    weight of lag 0).

    **Box-Jenkins sign convention**, the same as fue's `calcnu`: the numerator is
    omega(B) = omega_0 - omega_1 B - omega_2 B^2 - ..., that is, **the leading
    term adds and the rest subtract**. Porting this the other way round is the
    kind of one-character error that already cost a bug in the C itself (the
    Nyquist sign in `CalcNonsOp`).
    """
    omega = np.asarray(omega, float)
    delta = np.asarray(delta, float)
    s = len(omega) - 1
    r = len(delta)
    nu = np.zeros(length)
    for t in range(1, length + 1):
        lag = t - 1 - b
        acc = 0.0
        if 0 <= lag <= s:
            acc = omega[0] if lag == 0 else -omega[lag]
        for j in range(1, r + 1):
            if t > j:
                acc += delta[j - 1] * nu[t - j - 1]
        nu[t - 1] = acc
    return nu


@dataclass
class SeriesCast:
    """One series' univariate cast, precomputed (the fixed part)."""

    spec: object            # PreSpec
    est_spec: object        # fue.cast_us.EstSpec
    npar: int               # number of free parameters it takes from x
    name: str


@dataclass
class CastSpec:
    """The fixed part of the problem, precomputed once before optimising.

    The equivalent of the C's `populate_globals`, but without globals: all the
    state lives here and is passed explicitly.
    """

    series: list = field(default_factory=list)      # list[SeriesCast]
    links: list = field(default_factory=list)       # list[Link]
    m: int = 0
    n_stat: int = 0                                  # common length of the w's
    npar: int = 0
    # Links whose two series carry DIFFERENT differencing operators, so the
    # reported gain is nu(1)*Delta(1) and not nu(1). Empty for every matched
    # case, which is all of the legacy. See `check_operators` and BUG-8.
    delta_warnings: list = field(default_factory=list)

    @property
    def names(self):
        return [s.name for s in self.series]

    @property
    def npar_links(self):
        return sum(l.npar for l in self.links)


def _npar_univariate(model):
    """How many free parameters fue's univariate cast consumes.

    Taken from the length of fue's initial vector rather than recounted here:
    the count and the order are the same thing (`count_npar_build_par` in the C),
    and keeping a second copy of the count is exactly what ends up diverging.
    """
    from fue.cast_us import _build_initial_x

    return len(np.asarray(_build_initial_x(model), float))


def _end_date(ts):
    """(year, period) of the LAST observation, from `start`, `nobs` and `freq`."""
    y, p = ts.start
    f = int(ts.freq or 1)
    tot = y * f + (p - 1) + int(ts.nobs) - 1
    return tot // f, tot % f + 1


def check_alignment(specs):
    """The premise the alignment states and never checked (BUG-2).

    Below, the series are aligned AT THE END and trimmed to the shortest, on
    the assumption that the last observation is the same date. That is right
    for the case it was written for -- different d/D over the SAME window --
    and it is silent when the windows are different stretches of calendar. Two
    series that do not share a single period can be crossed and the fit goes
    through without a word.

    It was not theoretical. Loading a price series for 1700-1896 against
    rainfall for 1766-2024, the identification paired the 1700 price with the
    1766 rainfall -- 66 years apart -- and proposed b=18 in earnest. The only
    tell was that the printed band, 2/sqrt(n), did not match the real overlap.
    And the reproduction is starker: declare the input 50 years later and the
    output is IDENTICAL to the last decimal, because the date never enters the
    computation.

    This is a refusal, not a trim. Trimming to the common calendar window is
    probably what an analyst wants, but it changes which observations the model
    is fitted on, and that decision belongs upstream -- rebuild the `.pre` in
    `art` over the window you mean. Guessing it here would replace a silent
    wrong answer with a quiet different one.
    """
    ref = specs[0].ts
    fin = _end_date(ref)
    for sp in specs[1:]:
        ts = sp.ts
        if int(ts.freq or 1) != int(ref.freq or 1):
            raise ValueError(
                f"{sp.name!r} is {ts.freq}-per-year and {specs[0].name!r} is "
                f"{ref.freq}: they cannot be modelled jointly. Rebuild them at "
                f"the same frequency in `art`.")
        f2 = _end_date(ts)
        if f2 != fin:
            raise ValueError(
                f"the series do NOT end on the same date: {specs[0].name} ends "
                f"{fin[1]:02d}/{fin[0]} and {sp.name} ends {f2[1]:02d}/{f2[0]}. "
                f"The joint cast aligns at the END and trims to the shortest, "
                f"which assumes a common last observation; with different "
                f"windows it would pair observations that are years apart and "
                f"say nothing. Rebuild both `.pre` in `art` over the window you "
                f"mean to model.")


def differencing_poly(model):
    """The FULL non-stationary operator as a polynomial in B, `[1, -r1, -r2...]`.

    Regular, seasonal and the individual annual factors, all of it. The
    coefficients come from `fue._nonsop_coefs` rather than being rebuilt here,
    for the reason `forecast.py` gives: the `ifadf` factors are exactly the kind
    of delicate detail that must have a single source of truth. Getting them
    from there means `∇∇₄` written the school's way -- `d=2, D=0,
    ifadf=[0,1,1]`, as m6's EA carries it -- yields the same polynomial as
    `∇∇₄` written any other way, which is the whole point of comparing
    polynomials instead of comparing `(d, D)` tuples.
    """
    from fue.forecast import _nonsop_coefs
    freq = int(model.series.freq or 1)
    r = np.asarray(_nonsop_coefs(model.d, model.D, freq,
                                 ifadf=(model.ifadf or None)), float)
    poly = np.empty(len(r) + 1)
    poly[0] = 1.0
    poly[1:] = -r
    return poly


def delta_operator(out_model, in_model, tol=1e-9):
    """`Δ(B) = op_out / op_in` — the operator the transfer term silently applies.

    BUG-8, stated as arithmetic. The embedded cast relates each series
    differenced by ITS OWN operator, but the model says

        ∇^d ∇ₛ^D N  =  ∇^d ∇ₛ^D y  −  ν(B) · (∇^d ∇ₛ^D x)

    so the input must enter differenced by the OUTPUT's operator. When they
    differ, what gets fitted is not ν but **ν·Δ**, and the reported gain is
    `ν(1)·Δ(1)` -- measured, see `docs/LEVEL_TRANSFER_PLAN.md` §2d.

    Returns `(delta, nested, resto)`:

    * `delta`  -- the quotient polynomial, `[1.0]` when the operators agree;
    * `nested` -- whether the division is EXACT, i.e. the output's operator
      contains the input's. When it is not, the two are not merely mismatched:
      neither differencing implies the other, and no single vector can serve
      both roles. That case needs refusing, not dispatching;
    * `resto`  -- the remainder's max abs coefficient, so the caller can say
      how far from nested it was.

    Δ(1) is `float(delta.sum())`: **0** means the gain is annihilated (an excess
    root at frequency zero), **s** means it is multiplied by the period (an
    excess purely at the seasonal frequencies), **1** means the operators agree
    and there is nothing to do.
    """
    a = differencing_poly(out_model)
    b = differencing_poly(in_model)
    if len(b) > len(a):
        # The INPUT is differenced harder. Not nested by construction, and the
        # quotient is not a polynomial: say so rather than return nonsense.
        return np.array([1.0]), False, float('inf')
    q, r = np.polydiv(a[::-1], b[::-1])           # numpy wants highest degree first
    resto = float(np.max(np.abs(r))) if r.size else 0.0
    return q[::-1], resto <= tol, resto


def operators_agree(out_model, in_model, tol=1e-9):
    """True when the transfer needs no correction: `Δ(B) = 1`."""
    delta, nested, _ = delta_operator(out_model, in_model, tol)
    return bool(nested and len(delta) == 1 and abs(delta[0] - 1.0) <= tol)


def check_operators(cast_spec):
    """The interim guard for BUG-8: say when the reported gain is not ν(1).

    Refuses what cannot be modelled at all, warns about what is modelled wrong,
    and stays silent when the operators agree -- which is every legacy case,
    every m6 series and the whole network, so this costs them nothing.

    It warns rather than refuses on the mismatched-but-nested case for a reason
    the plan argues at length: `IPC_FR <- WTI` is a legitimate model -- the
    French CPI has stochastic seasonality and WTI does not, and that is the data
    rather than an analyst's mistake. The oracle fits it happily, because in its
    formulation the input enters in levels and there is no Δ to mismatch.
    Refusing it would be trading a wrong answer for no answer. What is NOT
    acceptable is returning the wrong gain in silence, which is what happened
    until now -- and the diagonal gate cannot catch it (measured: −8.31e−08,
    unchanged, because the factorisation identity holds under any differencing
    as long as both sides use the same one).

    Returns the list of warnings; also emits each through `warnings.warn`.
    """
    import warnings as _w

    out = []
    for l in cast_spec.links:
        my = cast_spec.series[l.out].spec.model
        mx = cast_spec.series[l.inp].spec.model
        nom_y = cast_spec.series[l.out].name
        nom_x = cast_spec.series[l.inp].name
        delta, nested, resto = delta_operator(my, mx)
        if not nested:
            raise ValueError(
                f"{nom_y} <- {nom_x}: neither series' differencing implies the "
                f"other, so no single vector can both carry {nom_x}'s own "
                f"univariate model and feed the transfer. The transfer relates "
                f"the LEVELS, and the noise carries the differencing, so the "
                f"input must enter differenced by {nom_y}'s operator -- which "
                f"here would leave {nom_x} under-differenced. Respecify one of "
                f"the two in `art`.")
        if len(delta) == 1 and abs(delta[0] - 1.0) <= 1e-9:
            continue
        d1 = float(delta.sum())
        if abs(d1) <= 1e-9:
            efecto = ("ANNIHILATED: Δ(1) = 0, an excess root at frequency zero "
                      "(∇∇ₛ is order TWO there, not one)")
        else:
            efecto = f"MULTIPLIED by {d1:.6g}: Δ(1) = {d1:.6g}"
        msg = (f"{nom_y} <- {nom_x}: the two series are differenced by "
               f"DIFFERENT operators, so what is fitted is not ν(B) but "
               f"ν(B)·Δ(B), with Δ = op({nom_y})/op({nom_x}) of degree "
               f"{len(delta) - 1}. The reported GAIN is therefore wrong -- "
               f"{efecto} -- by up to that factor, depending on how much of Δ "
               f"the fitted (b, r, s) has the reach to absorb. ν₀, the "
               f"contemporaneous impact, is unaffected. Do NOT divide the gain "
               f"by Δ(1) to correct it: the error is partial and its size is "
               f"not knowable from the output. See BUG-8 in "
               f"docs/LEVEL_TRANSFER_PLAN.md.")
        out.append(msg)
        _w.warn(msg, RuntimeWarning, stacklevel=2)
    return out


def build_cast_spec(specs, links=None):
    """Precompute the cast from the `.pre` files read (one per series).

    `specs[0]` is the OUTPUT (series 1 of the VARMA, the one that receives the
    transfers by default); the rest are the inputs. `links` is the list of links;
    with no links the model is the diagonal one, which is the validation gate.
    """
    from fue.cast_us import build_est_spec

    if len(specs) < 2:
        raise ValueError("the joint fit needs at least 2 series")

    check_alignment(specs)

    cs = CastSpec(links=list(links or []))
    for s in specs:
        m = s.model
        cs.series.append(SeriesCast(spec=s, est_spec=build_est_spec(m),
                                    npar=_npar_univariate(m), name=s.name))
    cs.m = len(cs.series)
    for l in cs.links:
        if not (0 <= l.out < cs.m and 0 <= l.inp < cs.m):
            raise ValueError(f"link out of range: {l}")
        if l.out == l.inp:
            raise ValueError(f"a link cannot go from a series to itself: {l}")
    cs.delta_warnings = check_operators(cs)
    # Vector order, following shootx: transfers -> univariate -> covariance.
    # Covariance: var[0] is fixed at 1 (the scale is concentrated into sigma2),
    # then log(var_i/var_1) for i>0. The off-diagonal covariances start out FIXED
    # at zero: they are only freed if the .cns asks for it.
    cs.npar = cs.npar_links + sum(s.npar for s in cs.series) + (cs.m - 1)
    return cs


def build_sigma(x, idx, m):
    """The vector's covariance block: `(Q, idx, ifault)`.

    Q carries the variance RATIOS, not the variances: Q[0][0] = 1 and
    Q[i][i] = exp(x), with the scale concentrated into sigma2 (see the header
    note). The off-diagonal covariances go in **raw**, in lower-triangle order by
    rows — `q[2,1]`, `q[3,1]`, `q[3,2]`, … — which is the slot table's.

    They are only read if the vector carries them: without a slot table the
    vector ends at the ratios and the model has a diagonal covariance, which is
    the default case. Freeing a covariance is the analyst's decision
    (`q[5,2] = free` in the `.cns`), not something switched on in bulk.

    **Rejected if Q is not positive definite.** The raw covariances are not
    reparametrised, so the optimizer may step into the region where Q stops being
    so; there the likelihood does not exist. `ifault` is returned and the
    objective answers 1.0, which is Mauricio's own strategy (1995 §3): the point
    does not improve on the start and the search moves away from it. It is what
    the C does, and in real cases that boundary has never bitten — m6's strongest
    correlation (-0.41) converges without going near it.
    """
    var = np.ones(m)
    for i in range(1, m):
        var[i] = math.exp(x[idx]); idx += 1
    Q = np.diag(var)

    n_off = m * (m - 1) // 2
    if len(x) - idx >= n_off:
        for i in range(1, m):
            for j in range(i):
                Q[i, j] = Q[j, i] = x[idx]; idx += 1
        if n_off and np.any(Q[np.triu_indices(m, 1)] != 0.0):
            try:
                np.linalg.cholesky(Q)
            except np.linalg.LinAlgError:
                return None, idx, 1
    return Q, idx, 0


def cast_diagonal(x, cast_spec):
    """Parameter vector -> diagonal VARMA structure (transfer by SUBTRACTION).

    Order of `x`, following `shootx`:

        1. transfers: omega[0..s] and delta[1..r] of each link
        2. ARMA + deterministics + mean of each series (fue's univariate cast, in
           `count_npar_build_par` order)
        3. covariance: log(var_i / var_1) for i = 2..m

    Returns `(phi, theta, mu, w, sigma, ifault)` ready for `elf_varma`: `phi`
    (p,m,m) and `theta` (q,m,m) block-diagonal, `w` (n,m) with the stationary
    series aligned at the end, and `sigma` diagonal.
    """
    from fue.cast_us import cast_us_py

    x = np.asarray(x, float)
    m = cast_spec.m
    idx = 0

    # --- 1. Transfers: omega[0..s] and delta[1..r] of each link -------------
    om, de = [], []
    for l in cast_spec.links:
        om.append(x[idx:idx + l.s + 1]); idx += l.s + 1
        de.append(x[idx:idx + l.r]); idx += l.r

    ps, qs, phis, thetas, mus, ws = [], [], [], [], [], []

    for sc in cast_spec.series:
        xi = x[idx:idx + sc.npar]
        idx += sc.npar
        p, q, phi, theta, mu, w, ifault = cast_us_py(xi, sc.est_spec)
        if ifault:
            return None, None, None, None, None, int(ifault)
        ps.append(int(p)); qs.append(int(q))
        phis.append(np.asarray(phi, float)); thetas.append(np.asarray(theta, float))
        mus.append(float(mu)); ws.append(np.asarray(w, float))

    sigma, idx, ifa_q = build_sigma(x, idx, m)
    if ifa_q:
        return None, None, None, None, None, int(ifa_q)

    # Alignment: if the series have different d/D their w's have different
    # lengths. They are aligned at the END (the last observation is the same
    # date) and trimmed to the shortest, which is what build_stationary_series
    # does.
    n = min(len(w) for w in ws)
    W = np.column_stack([w[len(w) - n:] for w in ws])

    # --- Transfers: each link SUBTRACTS from its output ---------------------
    # tr[o][t] = SUM_k nu_j[k]*w_in[t-k+1]; series 1 of the VARMA becomes the
    # NOISE N_t = w_Y - SUM_j transfer_j. With omega = 0 nothing is subtracted
    # and the model splits into independent univariate ones: the bridge's test.
    if cast_spec.links:
        tr = np.zeros_like(W)
        for l, o_j, d_j in zip(cast_spec.links, om, de):
            nu = compute_irf(o_j, d_j, l.b, n)
            xin = W[:, l.inp]
            acc = np.zeros(n)
            for t in range(n):                      # SUM_{k=0..t} nu[k]*x[t-k]
                acc[t] = float(np.dot(nu[:t + 1], xin[t::-1]))
            tr[:, l.out] += acc
        W = W - tr

    p = max(ps) if ps else 0
    q = max(qs) if qs else 0
    PHI = np.zeros((p, m, m))
    THETA = np.zeros((q, m, m))
    for i in range(m):
        for k in range(ps[i]):
            PHI[k, i, i] = phis[i][k]
        for k in range(qs[i]):
            THETA[k, i, i] = thetas[i][k]

    # The C's constraint (shootx [12]), CORRECTED — see `ar_is_stationary`.
    # The original tests |phi[0]| >= 0.999 for EVERY order, and phi[0] is only
    # a root when p = 1. This checks the roots, which is what the C does three
    # lines below for the MA, with `chekma`.
    for i in range(m):
        if ps[i] >= 1 and not ar_is_stationary(phis[i][:ps[i]]):
            return None, None, None, None, None, 1

    return PHI, THETA, np.asarray(mus, float), W, sigma, 0


def loglik_diagonal(x, cast_spec, xitol=-1e-3):
    """EXACT CONCENTRATED log-likelihood of the diagonal joint model.

    The C's `est()` does not estimate the scale: it decomposes Sigma = sigma2*Q
    with Q[1][1]=1 and **concentrates** sigma2, which comes out analytically from
    f1. That is why `Q` only carries the variance RATIOS. Evaluating `elf_varma`
    with an absolute Sigma instead gives a different likelihood — it is the error
    of passing the identity as Sigma when the real variances differ by a factor
    of 1000.

    drvarma's `_elf_f1f2` is used (the same one its estimator uses) together with
    its concentrated formula, without touching `elf`: any discrepancy with fue is
    a bug of this cast.

    `xitol = -1e-3` selects the **exact** likelihood, not the approximate one.
    """
    from drvarma.estimate_py import _elf_f1f2

    phi, theta, mu, w, sigma, ifault = cast_diagonal(x, cast_spec)
    if ifault:
        return float("-inf"), int(ifault)
    n, m = w.shape
    f1, f2, ifa = _elf_f1f2(w, mu, phi, theta, sigma, xitol)
    if ifa or not (f1 > 0.0 and f2 > 0.0):
        return float("-inf"), int(ifa or 5)
    # drvmlest.c:est [4] — the concentrated likelihood
    ll = (-0.5 * m * n * (math.log(2.0 * math.pi) - math.log(m) - math.log(n) + 1.0)
          - 0.5 * n * (m * math.log(f1) + math.log(f2)))
    return float(ll), int(ifa)


def _sigma2_univariate(sc, x_i, xitol=-1e-3):
    """One series' sigma2 at its seeds, through the same `elf` with m=1.

    Nothing is re-estimated: the univariate likelihood is evaluated at the
    `.pre`'s seeds and the concentrated variance is taken, sigma2 = f1/(n*m) with
    m=1.
    """
    from drvarma.estimate_py import _elf_f1f2
    from fue.cast_us import cast_us_py

    p, q, phi, theta, mu, w, ifault = cast_us_py(x_i, sc.est_spec)
    if ifault:
        return None
    w = np.asarray(w, float).reshape(-1, 1)
    phi = np.asarray(phi, float).reshape(-1, 1, 1) if p else np.zeros((0, 1, 1))
    theta = np.asarray(theta, float).reshape(-1, 1, 1) if q else np.zeros((0, 1, 1))
    f1, _f2, ifa = _elf_f1f2(w, np.array([float(mu)]), phi, theta,
                             np.ones((1, 1)), xitol)
    if ifa or not f1 > 0.0:
        return None
    return float(f1) / len(w)


def x0_from_pre(cast_spec):
    """Initial vector: the `.pre`'s seeds, which are fue's estimates.

    It is the natural starting point — and it explains the C's `termcode 3`: on
    the diagonal rung these seeds ALREADY are the optimum, so the line search
    cannot improve and stops. That is not a failure.

    The variance RATIOS log(var_i/var_1) are NOT seeded at zero: the series'
    scales can differ by orders of magnitude (in the canonical case, sigma2 =
    0.0627 against 68.84, a ratio of 1098) and starting at 1 leaves the initial
    point very far off — logL -1371 instead of -767. They are computed with the
    same `elf`, m=1, on each `.pre`'s seeds.
    """
    from fue.cast_us import _build_initial_x

    # Transfers at ZERO: that is the diagonal model, which we already know
    # homologates with fue. Starting the network there is what the ladder does —
    # first the diagonal, then whatever dynamics its residual CCF suggests.
    parts, s2 = [np.zeros(cast_spec.npar_links)], []
    for sc in cast_spec.series:
        xi = np.asarray(_build_initial_x(sc.spec.model), float)
        parts.append(xi)
        s2.append(_sigma2_univariate(sc, xi))

    ratios = np.zeros(cast_spec.m - 1)
    if s2[0]:
        for i in range(1, cast_spec.m):
            if s2[i]:
                ratios[i - 1] = math.log(s2[i] / s2[0])
    parts.append(ratios)
    return np.concatenate(parts)


def ar_is_stationary(phi, tol=0.0):
    """Is the AR operator `1 - phi_1 B - ... - phi_p B^p` stationary?

    Stationary iff every root of that polynomial lies OUTSIDE the unit circle.

    This replaces the C's guard at `tran_shootx.c:629`, which tests
    `|phi[0]| >= 0.999` for **every** order `p >= 1`. For p = 1 that is at
    least the right quantity — there `phi[0]` is the reciprocal of the root —
    but for p >= 2 it is not a root at all. In an AR(2) the stationary region
    is the triangle |phi2| < 1, phi2 + phi1 < 1, phi2 - phi1 < 1, in which
    **phi1 reaches 2**, so every AR(2) with complex roots and phi1 > 1 is
    stationary and was rejected. That is not an exotic corner: it is exactly
    where the persistent cycles live.

    Measured, with the guard disabled, on the canonical pair:

        phi1    phi2     |root|   stationary   ifault
        0.9500  -0.5200   1.387       yes         0
        1.0354  -0.5184   1.389       yes         0     <- was rejected
        1.6000  -0.8000   1.118       yes         0     <- was rejected
        2.1000  -1.0500   0.782       NO          3     <- elf catches it

    And the guard turns out to be over-broad even at p = 1, its own stated
    case. Unguarded, every stationary AR(1) evaluates cleanly up to
    phi = 0.9999 (|root| = 1.0001, ifault 0); phi = 1 exactly gives ifault 2
    and phi > 1 gives ifault 3. So `elf` already rejects precisely the right
    set, and the 0.999 threshold was discarding a live strip of stationary
    parameter space below it.

    The guard is KEPT rather than deleted, for the reason it exists: rejecting
    before calling `elf` is cheaper than a full likelihood evaluation, and it
    hands the line search a clean refusal. What changes is only that it now
    rejects the right region — and it mirrors what `tran_shootx.c` already does
    three lines below for the MA, where `chekma` builds the companion matrix
    and looks at eigenvalue moduli, generic in the operator.

    `tol` widens the rejection if a numerical margin is ever wanted; the
    default is the mathematical boundary, which is where `elf` also sits.
    """
    import numpy as _np

    phi = _np.asarray(phi, float)
    if not len(phi):
        return True
    if not _np.all(_np.isfinite(phi)):
        return False
    # 1 - phi_1 z - ... - phi_p z^p, highest power first for np.roots
    coef = _np.r_[-phi[::-1], 1.0]
    nz = _np.flatnonzero(coef)
    if not len(nz):
        return True
    coef = coef[nz[0]:]                     # descarta ceros de cabecera
    if len(coef) < 2:
        return True
    try:
        r = _np.abs(_np.roots(coef))
    except Exception:                       # noqa: BLE001
        return False
    return bool(len(r) == 0 or _np.min(r) > 1.0 + tol)
