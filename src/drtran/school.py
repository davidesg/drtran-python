"""Readings the school takes from an estimated model, which drtran computed and
never read.

Every quantity here is derived from numbers drtran already produces. What was
missing is that a table of parameters with standard errors is the *input* to the
Box-Jenkins-Treadway method, not its output: the analyst then reads specific
things off it, and reacts. The sources are the theses in `drtran/literature/`
(Muñoz Polo 2000 §2.6 and §6.4; Brajín Rodríguez 2004 §2), and each function
below cites the case that motivates it.

This module is MUTE, like the rest of the library: it returns findings, never
advice. Turning a finding into "and therefore you should" is the assistant's
job, and lives in `mcp_server`.
"""
from __future__ import annotations

import math

import numpy as np


def variance_reduction(fit_transfer, fit_diagonal, series_index=0,
                       embed=None):
    """Fraction by which the transfer cuts the output's residual variance.

    The school closes every case with this number and states it in percent:
    "una reducción del 44 % de la varianza residual de lnQ en relación a su
    modelo univariante" (Muñoz 6.4.1; also 53 %, 23 % and 37 % in 6.4.2-6.4.4).
    It answers "was the transfer worth it" in the units an analyst cares about,
    where the likelihood ratio answers it in units only a statistician does.

    Note the school's own qualification, which is what makes the number honest:
    in 6.4.1 and 6.4.4 the reduction is achieved "empleando un parámetro MENOS
    de intervención" — the transfer explained an anomaly the univariate model
    had to absorb with a dummy. Compare parameter counts alongside.

    Returns the fraction in [0, 1) — 0.44 for a 44 % cut — or NaN if either
    residual set is unavailable.
    """
    from .netid import residuals

    def _var(f):
        if f is None:
            return float("nan")
        emb = f.embed if embed is None else embed
        a, ifa = residuals(f.x, f.cast_spec, embed=emb, structural=True)
        if ifa:
            return float("nan")
        a = np.asarray(a, float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if series_index >= a.shape[1]:
            return float("nan")
        return float(np.var(a[:, series_index]))

    v1, v0 = _var(fit_transfer), _var(fit_diagonal)
    if not (np.isfinite(v1) and np.isfinite(v0)) or v0 <= 0:
        return float("nan")
    return 1.0 - v1 / v0


def dead_time_suspect(fit, table, se=None, link_index=0, tol=1.96):
    """Is the leading numerator weight indistinguishable from -1?

    THE reading of the Muñoz cases, and the least obvious one. Twice, an
    estimated omega_0 of exactly -1.00 is read as a statement about the DEAD
    TIME rather than about the weight:

      6.4.4: "Se observa que el parámetro MA estimado en el retardo cero es
              -1.00. Este valor indica que es altamente probable que el tiempo
              muerto especificado sea ERRÓNEO y que es al menos superior en un
              trimestre."
      6.4.5: "el parámetro MA de orden 0 resulta no significativamente distinto
              de -1.00, lo que parece indicar que el tiempo muerto es de dos
              periodos."

    In 6.4.4 the coefficient stays pinned at -1.00 through several
    reformulations until b is corrected and the restriction imposed. An analyst
    who does not know this reflex can spend a long while adding MA terms that
    are all non-significant.

    **BUT THE RULE IS CONDITIONAL, AND THE CONDITION MATTERS.** Tested against
    known truth (generated with b=2, fitted at b=0,1,2,3) omega_0 came out
    -0.169, +0.036, +0.794, +0.471 — nowhere near -1, and the check never
    fires. It does not reproduce in the plain parametrisation, and here is why:
    those cases run on a TRANSFORMED output, with the long-run restriction
    imposed by subtracting the input from it. Muñoz's own algebra (§2.6, p. 37)
    gives the numerator of that parametrisation as

        omega*(B) = omega_s(B) - delta_r(B)

    and delta's leading coefficient is 1. So when the true numerator has no
    term at that lag — exactly the case of an understated dead time — one is
    left with omega*_0 = 0 - 1 = -1. The -1 is the DENOMINATOR showing through
    the subtraction, not a property of transfer models.

    So this fires only where the output carries that transformation, which is
    also where it is genuinely useful. It stays because it costs nothing and
    helps in that parametrisation; it is NOT a general dead-time test, and
    presenting it as one would be reading a 25-year-old reflex out of its
    setting.

    Returns (suspect, omega0, se_omega0, t_vs_minus_one) — `suspect` True when
    omega_0 is within `tol` standard errors of -1.
    """
    val, err = _slot(fit, table, se, f"omega{link_index + 1}[0]")
    if val is None:
        return False, float("nan"), float("nan"), float("nan")
    if err is None or not np.isfinite(err) or err <= 0:
        return (abs(val + 1.0) < 0.05, val, float("nan"), float("nan"))
    t = (val + 1.0) / err
    return bool(abs(t) < tol), val, err, float(t)


def denominator_near_unit(fit, table, se=None, link_index=0, thresh=0.95):
    """Is a denominator root close to the unit circle?

    Muñoz 6.4.5 reads delta = .99 (.01) as implausible, and reports the
    consequence rather than the parameter: "Los valores estimados tanto de la
    ganancia a largo plazo como del retardo medio también son excesivamente
    altos (en valor absoluto), además de NO significativamente distintos de
    cero." A denominator approaching 1 sends nu(1) to infinity, so the gain and
    the mean lag stop being estimable long before the fit stops converging.

    Returns a list of (name, value, se) for denominator coefficients above
    `thresh` in absolute value.
    """
    out = []
    for nm in (table.names if table is not None else []):
        if not nm.startswith(f"delta{link_index + 1}["):
            continue
        val, err = _slot(fit, table, se, nm)
        if val is not None and abs(val) > thresh:
            out.append((nm, val, err if err is not None else float("nan")))
    return out


def worst_correlations(fit, se, table=None, top=5, flag=0.9):
    """The largest correlations between estimated parameters.

    Brajín §2.3.1 detects overparametrisation "con errores estándar altos en
    relación con el valor estimado del parámetro y/o CORRELACIONES ALTAS entre
    parámetros estimados", and Muñoz 6.4.3 uses exactly this to abandon an
    overfitting experiment: "en todos los casos se observa que la situación de
    estimación está mal definida (altas correlaciones entre muchos de los
    parámetros de relación), por lo que este experimento puede considerarse
    FALLIDO."

    But a high correlation is a QUESTION, not a verdict, and Muñoz 6.4.4 is the
    case that proves it: mu and an intervention parameter correlate at -.93 and
    both are kept, because "esta sobreparametrización es necesaria" — dropping
    the intervention moves many other parameters, and dropping mu leaves the
    residual mean away from zero. So this returns the pairs and says nothing
    about what to do with them.

    Returns (pairs, n_flagged) with pairs = [(name_i, name_j, rho), ...] sorted
    by |rho| descending.
    """
    if se is None or getattr(se, "cov", None) is None or getattr(se, "ifault", 1):
        return [], 0
    cov = np.asarray(se.cov, float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ok = d > 0
    if ok.sum() < 2:
        return [], 0
    with np.errstate(divide="ignore", invalid="ignore"):
        R = cov / np.outer(d, d)
    names = _free_names(table, cov.shape[0])
    pairs = []
    for i in range(cov.shape[0]):
        for j in range(i + 1, cov.shape[0]):
            if ok[i] and ok[j] and np.isfinite(R[i, j]):
                pairs.append((names[i], names[j], float(R[i, j])))
    pairs.sort(key=lambda p: -abs(p[2]))
    return pairs[:top], sum(1 for p in pairs if abs(p[2]) >= flag)


# ── helpers ────────────────────────────────────────────────────────────────
def _free_names(table, n):
    """Names of the FREE parameters, in the order the covariance uses."""
    if table is None:
        return [f"par[{i}]" for i in range(n)]
    try:
        return [table.names[table.slot_of_free[i]] for i in range(n)]
    except Exception:                                      # noqa: BLE001
        return [f"par[{i}]" for i in range(n)]


def _slot(fit, table, se, name):
    """(value, standard error) of a named slot, or (None, None)."""
    if table is None or name not in table.names:
        return None, None
    idx = table.names.index(name)
    x = np.asarray(fit.x, float)
    if idx >= len(x):
        return None, None
    val = float(x[idx])
    err = None
    if se is not None and getattr(se, "se", None) is not None:
        try:
            for k in range(table.n_free):
                if table.slot_of_free[k] == idx:
                    err = float(np.asarray(se.se, float)[k])
                    break
        except Exception:                                  # noqa: BLE001
            err = None
    return val, err


def noise_adequacy(fit, series_index=0, nlags=None, npar=None, embed=None):
    """Ljung-Box on the OUTPUT residual's own ACF — is the NOISE adequate?

    `transfer_adequacy` reads the CCF and answers a question about the
    RELATION. This reads the ACF and answers a question about the NOISE. The
    two are separate verdicts, and telling them apart is what makes Muñoz §2.6
    p.42's reformulation order usable:

      "La especificación inadecuada de la relación v(B) puede generar la
       apariencia (en acf/pacf residuales) de especificación inadecuada del
       ruido theta(B) a la vez que una ccf que requiere reformulación de la
       relación. SIN EMBARGO, la especificación inadecuada del ruido NO puede
       dar la impresión en ccf de especificación inadecuada de la relación."

    The contamination runs one way, so with both instruments in hand the repair
    order is forced rather than a matter of taste.

    Deliberately `fue.diagnostics.ljung_box`, not `diagnose.chi_test`: this is
    the same statistic printed under the ACF of the residual panel in `art`,
    and an analyst comparing the two must not find two numbers.

    `npar` defaults to `npar_for_series`, NOT to the joint parameter count —
    see that function for why the difference is not cosmetic.

    Returns (Q, p, lags, df) — p is NaN when it cannot be computed.
    """
    from .netid import residuals

    if npar is None:
        npar = npar_for_series(fit, series_index)

    emb = fit.embed if embed is None else embed
    a, ifa = residuals(fit.x, fit.cast_spec, embed=emb, structural=True)
    if ifa:
        return float("nan"), float("nan"), 0, 0
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if series_index >= a.shape[1]:
        return float("nan"), float("nan"), 0, 0
    r = a[:, series_index]
    k = int(nlags or min(24, max(4, len(r) // 5)))
    try:
        from fue.diagnostics import ljung_box
        res = ljung_box(r, lags=k, df_correction=int(npar))
        return (float(res["statistic"][0]), float(res["pvalue"][0]), k,
                max(1, k - int(npar)))
    except Exception:                                      # noqa: BLE001
        return float("nan"), float("nan"), k, max(1, k - int(npar))


def decay_pattern(weights, ses=None, tol=1.96):
    """Read a denominator off the pattern of freely estimated weights.

    The step every Muñoz case takes and drtran has no tool for. A generous pure
    MA is estimated first — "de hecho, esto equivale a una estimación de los
    primeros términos de la ccf" (6.4.1) — and then the SHAPE of the estimates
    decides the parametrisation:

      "Se observa que el valor absoluto de los mismos DECRECE conforme aumenta
       el retardo, lo que parece indicar que la relación requiere un factor
       AR(1) con parámetro positivo."

    That is strictly more informative than reading the CCF once, because the
    weights are estimated jointly with the noise while the CCF is not.

    The sign pattern carries the sign of the denominator parameter: weights of
    constant sign mean delta > 0, alternating means delta < 0. The ratio of
    consecutive weights estimates it, and the median is used rather than the
    mean because one near-zero weight would otherwise dominate.

    Returns a dict: `significant` (the lags whose weight clears `tol` standard
    errors), `decaying`, `alternating`, `ratio`, `suggests_denominator`.
    """
    w = np.asarray(weights, float)
    n = len(w)
    sig = list(range(n))
    if ses is not None:
        e = np.asarray(ses, float)
        sig = [k for k in range(n)
               if k < len(e) and np.isfinite(e[k]) and e[k] > 0
               and abs(w[k] / e[k]) > tol]
    if not sig:
        return dict(significant=[], decaying=False, alternating=False,
                    ratio=float("nan"), suggests_denominator=False)

    # Read the shape over the SIGNIFICANT span only. A tail of noise beyond the
    # last real weight would flatten any pattern there is.
    span = w[sig[0]:sig[-1] + 1]
    mag = np.abs(span)
    decaying = bool(len(mag) >= 3 and np.all(np.diff(mag) < 1e-12)
                    and mag[0] > 0)
    ratios = [span[i + 1] / span[i] for i in range(len(span) - 1)
              if abs(span[i]) > 1e-12]
    ratio = float(np.median(ratios)) if ratios else float("nan")
    alternating = bool(len(span) >= 3
                       and np.all(np.diff(np.sign(span[span != 0])) != 0))
    # A denominator is worth trying when the tail dies away GRADUALLY: three or
    # more weights in a row, shrinking, with a ratio inside the unit circle.
    suggests = bool(decaying and len(span) >= 3
                    and np.isfinite(ratio) and 0.05 < abs(ratio) < 0.95)
    return dict(significant=sig, decaying=decaying, alternating=alternating,
                ratio=ratio, suggests_denominator=suggests)


def npar_for_series(fit, series_index=0):
    """Parameters to correct a SINGLE series' residual ACF by.

    Not the length of the joint parameter vector, which is what a naive reading
    of "number of estimated parameters" gives. The Ljung-Box on series i's ACF
    is a statement about series i's own model, and in `art` the correction is
    that model's own `npar`. Correcting by the joint vector inflates chi-square
    significance on every multivariate fit — on the canonical case it cut the
    degrees of freedom from 21 to 7 and turned p = 0.29 into p = 0.0017.

    So: the series' own ARMA orders, plus the parameters of the links that FEED
    it, because those are estimated from the same residual.
    """
    from fue.cast_us import cast_us_py

    cs = fit.cast_spec
    x = np.asarray(fit.x, float)
    idx = cs.npar_links
    n = 0
    for i, sc in enumerate(cs.series):
        piece = x[idx:idx + sc.npar]
        idx += sc.npar
        if i != series_index:
            continue
        try:
            p, q, *_rest, ifa = cast_us_py(piece, sc.est_spec)
            if not ifa:
                n = int(p) + int(q)
        except Exception:                                  # noqa: BLE001
            n = 0
    for lk in (cs.links or []):
        if lk.out == series_index:
            n += (lk.s + 1) + lk.r
    return n


def ccf_pairs(fit, link_index=0, lag=None, top=5, embed=None):
    """Which PAIRS of observations build a residual-CCF coefficient.

    `calibrate` asks "which observation bends the instruments" and answers with
    leave-one-out over one series. Muñoz's cases ask something one step
    narrower, and the answer is a PAIR:

      "se justifica por distorsión negativa entre el ruido preblanqueado y los
       residuos del input en II/94 y II/92, y III/86 y III/84" (6.4.2, lag 8)

    Two dates, one in each series, separated by the lag. That is not a
    rephrasing of a single outlier: neither observation need be extreme on its
    own. The CCF coefficient at lag k is

        r_k = SUM_t  aX_t · aN_{t+k}  /  (n · sX · sN)

    a sum of PRODUCTS, and a product is large when the two factors are
    moderately large TOGETHER and of the same sign. A pair of ordinary
    observations that happen to line up can carry a coefficient, and no
    single-series scan will ever show it.

    That distinction is what makes it worth computing. An analyst who finds a
    spike supported by two dates has a different problem from one supported by
    an outlier: an outlier is an intervention, while a coincidence of two
    moderate values at a fixed distance is either a real dynamic the model
    misses, or nothing at all.

    `lag` defaults to the coefficient with the largest absolute value over
    k >= 0. Returns (lag, r_k, pairs) with pairs = [(date_input, date_noise,
    contribution as a fraction of r_k), ...], largest |contribution| first.
    """
    from .calibrate import _date_of
    from .netid import residuals

    cs = fit.cast_spec
    link = cs.links[link_index]
    emb = fit.embed if embed is None else embed
    a, ifa = residuals(fit.x, cs, embed=emb, structural=True)
    if ifa:
        return 0, float("nan"), []
    a = np.asarray(a, float)
    n = a.shape[0]
    aX = a[:, link.inp] - a[:, link.inp].mean()
    aN = a[:, link.out] - a[:, link.out].mean()
    sX = math.sqrt(float((aX * aX).sum()) / n)
    sN = math.sqrt(float((aN * aN).sum()) / n)
    if sX < 1e-12 or sN < 1e-12:
        return 0, float("nan"), []

    nlags = max(10, min(n // 4, 24))
    if lag is None:
        rs = [float((aX[:n - k] * aN[k:]).sum()) / (n * sX * sN)
              for k in range(nlags + 1)]
        lag = int(np.argmax(np.abs(rs)))
    lag = int(lag)
    prod = aX[:n - lag] * aN[lag:]
    r_k = float(prod.sum()) / (n * sX * sN)
    if abs(r_k) < 1e-14:
        return lag, r_k, []

    mX = cs.series[link.inp].spec.model
    mN = cs.series[link.out].spec.model
    lostX = int(getattr(mX.series, "nobs", n)) - n
    lostN = int(getattr(mN.series, "nobs", n)) - n

    contrib = prod / (n * sX * sN * r_k)
    order = np.argsort(-np.abs(contrib))[:top]
    pairs = [(_date_of(mX, lostX + int(t) + 1),
              _date_of(mN, lostN + int(t) + lag + 1),
              float(contrib[int(t)])) for t in order]
    return lag, r_k, pairs


def noise_ma_roots(fit, series_index=0, near=1.10):
    """Near-unit FACTORS of the noise's MA polynomial, grouped by modulus.

    The operational signature of OVER-DIFFERENCING. If a series was
    differenced more than it needed, the extra factor (1 - B) has to be undone
    by the moving average, and it can only do that with a root ON the unit
    circle — so in a sample it comes out just outside, and the closer it sits
    the more the data are saying the difference was not needed.

    The convention is `theta(B) = 1 - theta_1 B - ... - theta_q B^q`, the one
    `cast_us_py` returns, so a modulus of 1 is the boundary of invertibility.
    Verified against a known airline model: the regular factor with
    theta = -0.4212 gives |z| = 2.374 = 1/0.4212, and the seasonal one with
    Theta = 0.8147 gives twelve roots at 1.0172 = (1/0.8147)^(1/12).

    **Roots are grouped, and the grouping is the point.** A SEASONAL factor
    (1 - Theta B^s) contributes s roots of equal modulus spread evenly round
    the circle — INCLUDING a real positive one at frequency zero. Reading that
    one root on its own says "regular differencing", which is wrong, and sends
    the analyst to undo a difference that is not the one in question. What
    identifies the factor is the multiplicity: one isolated root is the regular
    difference, a ring of `freq` of them is the seasonal one.

    Returns [(modulus, multiplicity, kind), ...] closest to the circle first,
    with `kind` one of "regular", "estacional" or "" when it is neither.
    """
    from fue.cast_us import cast_us_py

    cs = fit.cast_spec
    x = np.asarray(fit.x, float)
    idx = cs.npar_links
    theta, freq = None, 1
    for i, sc in enumerate(cs.series):
        piece = x[idx:idx + sc.npar]
        idx += sc.npar
        if i != series_index:
            continue
        freq = int(getattr(sc.spec.model.series, "freq", 1) or 1)
        try:
            p, q, _phi, th, *_r, ifa = cast_us_py(piece, sc.est_spec)
            if not ifa and q:
                theta = np.asarray(th, float)[:q]
        except Exception:                                  # noqa: BLE001
            return []
    if theta is None or not len(theta):
        return []

    co = np.r_[1.0, -theta]                    # 1 - th1 z - th2 z^2 - ...
    try:
        rts = np.roots(co[::-1])               # np.roots wants descending
    except Exception:                          # noqa: BLE001
        return []
    mods = sorted(float(abs(z)) for z in rts
                  if np.isfinite(abs(z)) and abs(z) < near)
    out, k = [], 0
    while k < len(mods):
        j = k
        while j + 1 < len(mods) and mods[j + 1] - mods[k] < 1e-3:
            j += 1
        m, mult = float(np.mean(mods[k:j + 1])), j - k + 1
        kind = ("estacional" if freq > 1 and mult in (freq, freq - 1)
                else "regular" if mult == 1 else "")
        out.append((m, mult, kind))
        k = j + 1
    out.sort(key=lambda t: t[0])
    return out


def integration_order_moved(fit_transfer, fit_diagonal, series_index=0,
                            near=1.10):
    """Did the noise become OVER-differenced once the input was accounted for?

    The most interesting conceptual result in Muñoz's thesis, and the one that
    sounds impossible until you see the mechanism: lnE is I(2) on its own and
    I(1) after the effects of DlnM1 are removed. She flags it as paradoxical
    and possible only in finite samples.

    It is not really paradoxical. Univariate identification has one instrument
    and must explain everything with it, so influence arriving from an input —
    a smooth, persistent, wandering influence — reads as a trend of the
    series' own, and gets differenced away. Put the input in the model and that
    part of the wandering has an owner, so the extra difference is now removing
    something that is no longer there.

    The comparison is FREE and it is the right one: the diagonal rung is
    already estimated (it is `load_pre`'s gate) and it carries the same noise
    model fitted WITHOUT the transfer. So we ask whether a root that was
    comfortably inside moved out toward the unit circle once the transfer went
    in. Nothing else changes between the two fits, which is what makes the
    movement attributable.

    **This does not compare likelihoods across differencing orders, and must
    not.** Models with different d are models of different data. What is
    compared is where the MA roots sit within ONE differencing order.

    Returns (moved, root_joint, root_diagonal, kind) — `moved` True when the
    factor sits CLOSER to the unit circle in the joint fit than in the
    diagonal one. False whenever there is no diagonal fit to compare against:
    a position is not a movement.
    """
    if fit_diagonal is None:
        # Sin el escalón diagonal no hay MOVIMIENTO que medir, sólo una
        # posición. Decir "se movió" comparando contra nada sería inventarse la
        # mitad del hallazgo, y es justo la mitad interesante.
        return False, float("nan"), float("nan"), ""
    rj = noise_ma_roots(fit_transfer, series_index, near=near)
    if not rj:
        return False, float("nan"), float("nan"), ""
    mj, mult, kind = rj[0]
    # el MISMO factor, identificado por su multiplicidad: comparar la raíz
    # regular de un ajuste con la estacional del otro no mide nada
    md = float("nan")
    for m, mu, _k in noise_ma_roots(fit_diagonal, series_index, near=near * 3):
        if mu == mult:
            md = m
            break
    moved = bool(np.isfinite(md) and mj < md - 1e-6 and mj < near)
    return moved, mj, md, kind


def seasonality_mismatch(specs, out=0, inp=1):
    """Can the input's filter remove the output's seasonality? Often it cannot.

    Muñoz 6.4.1 hits this and drtran would hit it identically. The output's
    model has STOCHASTIC seasonality; the input's has none. Prewhitening
    filters the output by the INPUT's ARMA, so the output's seasonality
    survives the filter untouched and the filtered series is still
    non-stationary at that frequency:

      "genera una serie todavía no estacionaria en la frecuencia uno y una ccf
       muy poco informativa."

    An uninformative CCF is the dangerous outcome, not a loud one: it does not
    announce itself. It looks like a CCF with structure everywhere, and the
    contiguous-block heuristic will happily read an order off it.

    The remedy in the thesis is a deliberate split between the model used to
    IDENTIFY and the model used to ESTIMATE:

      "Los parámetros estimados de estacionalidad determinista obtenidos en
       M3.Q se emplean para calcular la desviación del output de sus
       componentes deterministas, que posteriormente se filtra por el modelo
       univariante del input. … No obstante, el modelo M2.Q, con estacionalidad
       mixta, se emplea como modelo univariante del ruido en cada una de las
       estimaciones eficientes."

    An alternative output model, with the seasonality made deterministic,
    exists only to make the CCF readable. The real model does the fitting.

    **The class of the output's seasonality matters, not just its presence.**
    A multiplicative SARIMA makes EVERY seasonal frequency stochastic at once,
    which is the worst case for a filter that carries none. A hybrid (MEG,
    "Modelos de Estacionalidad Generalizada", Abraham & Box 1978) leaves
    stochastic only the frequencies that were found to be so, so the surviving
    non-stationarity is narrower and the CCF degrades less. Reporting them as
    the same thing would overstate the first.

    The model class is long established; what is recent is the TESTING used to
    resolve each frequency, whose critical values are under active research.
    So `mcp_server` is careful to put the caution on the decision procedure
    rather than on the specification — a distinction worth keeping, because
    "experimental model" and "test whose critical values are being pinned
    down" call for different amounts of hesitation.

    Returns (mismatch, detail) where detail["out"] and detail["inp"] are one of
    "sarima", "meg" or "determinista".
    """
    def _seasonal(sp):
        m = sp.model
        freq = int(getattr(m.series, "freq", 1) or 1)
        if freq <= 1:
            return "", freq
        # SARIMA multiplicativo: (1-B^s) o factores estacionales completos.
        # TODAS las frecuencias estacionales son estocásticas a la vez.
        if (int(getattr(m, "D", 0) or 0) > 0 or getattr(m, "ar_s", None)
                or getattr(m, "ma_s", None)):
            return "sarima", freq
        # MEG / híbrido: factores de frecuencia FIJA, una frecuencia cada uno.
        if getattr(m, "ar_f", None) or getattr(m, "ma_f", None):
            return "meg", freq
        return "determinista", freq

    ko, fo = _seasonal(specs[out])
    ki, _fi = _seasonal(specs[inp])
    mismatch = bool(ko in ("sarima", "meg") and ki not in ("sarima", "meg")
                    and fo > 1)
    n_f = len((getattr(specs[out].model, "ar_f", None) or [])) + \
        len((getattr(specs[out].model, "ma_f", None) or []))
    detail = {"out": ko, "inp": ki, "freq": fo, "out_n_fixed_freq": n_f}
    return mismatch, detail
