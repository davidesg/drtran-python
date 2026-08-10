# `mtram` — MCP tool reference

*Generated from the docstrings by `tools/gen_tools_md.py`. Do not edit by hand — edit the docstring.*

**21 tools.** In an MCP server the docstring is what the model reads, so this page and the instruction the model receives are the same text by construction.

---

| tool | what it answers |
|---|---|
| [`build_model`](#build-model) | AUTONOMOUS run: walk the ladder taking the documented default at every |
| [`calibrate`](#calibrate) | Which observations are BENDING the instruments — and what to do about it. |
| [`check_operators`](#check-operators) | Do the two series of each link carry the SAME differencing operator? |
| [`diagnose`](#diagnose) | Is the transfer adequate, and is the input really exogenous? |
| [`estimate`](#estimate) | Estimate every parameter JOINTLY by exact maximum likelihood. |
| [`evaluate`](#evaluate) | Out-of-sample evaluation: estimate ONCE on 1..window, then roll the origin. |
| [`forecast`](#forecast) | Forecast the LEVEL with its band, plus the period and annual variations. |
| [`identify_link`](#identify-link) | Identify (b, r, s) of ONE link by prewhitening and the CCF. |
| [`identify_network`](#identify-network) | Propose the whole NETWORK from the residual CCFs of the DIAGONAL fit. |
| [`impulse_response`](#impulse-response) | nu(k), its cumulative sum and the GAIN, with standard errors. |
| [`load_pre`](#load-pre) | Load a case's `.pre` **or `.inp`** files, CONFIRM the roles, prove the bridge. |
| [`overfit`](#overfit) | Overfit ON PURPOSE: enlarge the transfer and see whether it protests. |
| [`plot_calibration`](#plot-calibration) | PLOT the CCF **with and without** the dominant anomaly — the verification. |
| [`plot_ccf`](#plot-ccf) | PLOT the prewhitened CCF of one link — the identification instrument. |
| [`plot_forecast`](#plot-forecast) | PLOT the level forecast with its band, over the recent history. |
| [`plot_impulse_response`](#plot-impulse-response) | PLOT nu(k) and its cumulative sum, each with a 95 % band. |
| [`plot_residuals`](#plot-residuals) | PLOT the residual series with its ACF and PACF — fue's own panel. |
| [`refine_link`](#refine-link) | Estimate a GENEROUS pure MA and read the denominator off its weights. |
| [`set_network`](#set-network) | Fix the network to estimate: a JSON list of {out, inp, b, r, s}. |
| [`variance_decomposition`](#variance-decomposition) | How much of the forecast error comes from EACH source of innovation. |
| [`write_inp`](#write-inp) | Write the JOINTLY re-estimated univariate blocks back out as `.inp`. |

---

## `build_model`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |

AUTONOMOUS run: walk the ladder taking the documented default at every
    decision node, and REPORT WHICH ONE IT TOOK at each.

    The nodes are in `docs/DECISION_NODES.md`. The two rules this obeys:

    * **it never makes a claim the data cannot make for it** — it does not free a
      covariance, does not invent a constraint, and does not prune a cycle;
    * **it differs from guided only in who decides**, never in what is computed.
      A guided run that made the same choices reaches the same model.

    It STOPS, rather than choosing, at two points: a cyclic network (the system
    is simultaneous — `sima`) and a failed exogeneity test (a single-input
    transfer model does not hold). Both are findings, not obstacles.

---

## `calibrate`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |
| `threshold` | number | no | `3.5` |

Which observations are BENDING the instruments — and what to do about it.

    Leave-one-out over the CCF and the adequacy portmanteau. Its verdict is the
    branch to take at node N6 when adequacy fails, and the two branches need
    OPPOSITE responses:

    * **shape** — no single observation explains the failure → re-identify
      (b, r, s);
    * **observation** — the verdict rests on one point → that is an
      INTERVENTION. Re-specifying the shape around it is how a model acquires a
      lag nobody can interpret. Interventions are calibrated in `art`, on the
      univariate rung, and travel here in the `.pre`.

    This is NOT ART's scan run again: an anomaly in the output's univariate
    residuals may be explained by the INPUT once the transfer is in the model.
    What survives the joint fit is the genuine one.

---

## `check_operators`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |

Do the two series of each link carry the SAME differencing operator?

    The question BUG-8 turned on, and it is decided before estimating anything.
    The transfer relates the LEVELS with the noise carrying the differencing, so
    the input must enter differenced by the OUTPUT's operator. When the two
    operators agree that is the input's own column and nothing has to happen;
    when they differ, what a naive cast would fit is not nu(B) but nu(B)·Delta(B)
    and the reported GAIN comes out wrong by Delta(1).

    Reports, per link, the quotient's degree and Delta(1): **1** means the
    operators agree, **0** means an excess root at frequency zero and the gain
    would be annihilated, **s** means an excess purely at the seasonal
    frequencies and it would be multiplied. It also reports the COMMON WINDOW
    and any series carrying spare history, which is a different problem that
    passes every other check.

    Nothing here estimates. It is the first gate, and it is cheap.

---

## `diagnose`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |

Is the transfer adequate, and is the input really exogenous?

    Two portmanteaus on the CCF between the estimated noise and the prewhitened
    input: k >= 0 tests the TRANSFER (lag zero belongs to it), k < 0 tests
    EXOGENEITY — significance there is feedback, and a single-input transfer
    model does not hold.

    Measured on the STRUCTURAL residuals: with a contemporaneous transfer the
    reduced-form ones are correlated by construction and would condemn a correct
    model.

---

## `estimate`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `embed` | boolean | no | `True` |
| `cns_path` | string | no | `` |

Estimate every parameter JOINTLY by exact maximum likelihood.

    `embed=True` (default, as in the C) puts the transfer INSIDE the VARMA, so
    there is no pre-sample truncation. `embed=False` subtracts it instead — the
    old cast, and what TASTE does.

    **`embed=True` is a request, not a guarantee.** When a link's two series
    carry DIFFERENT differencing operators the transfer needs the input
    re-differenced by the OUTPUT's, which is a second vector for the same series
    and the embedded cast has one column per series. The fit is then dispatched
    to the subtracting cast, and the result SAYS SO — silently honouring the
    flag while doing something else is what BUG-8 was.

    `cns_path` is an optional constraints file (free / fixed / shared / product /
    linear combination).

    PRESENT THE EQUATION BLOCK VERBATIM. Do not rebuild the parameter table.

---

## `evaluate`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `window` | integer | yes | — |
| `horizon` | integer | no | `6` |

Out-of-sample evaluation: estimate ONCE on 1..window, then roll the origin.

    The only thing here that compares the model against WHAT HAPPENED instead of
    against itself — every other figure the suite prints is theoretical. Run it
    on two specifications to decide empirically which forecasts better.

---

## `forecast`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |
| `series_index` | integer | no | `0` |

Forecast the LEVEL with its band, plus the period and annual variations.

    The band is formed in the TRANSFORMED scale and mapped back, so with a log
    model it is ASYMMETRIC. Never build it by adding 1.96 standard errors to a
    level: the STD columns are relative (percentages), not index points.

    With a DISPATCHED model the forecast follows TASTE's route -- each input
    forecast by its own model, the noise by its own ARMA, joined on the LEVELS
    -- rather than the embedded cast's recursion. It is a different procedure,
    not the same one corrected, and the result says which one ran.

---

## `identify_link`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `input_index` | integer | no | `1` |
| `band` | string | no | `constant` |
| `ident_pre` | string | no | `` |

Identify (b, r, s) of ONE link by prewhitening and the CCF.

    Filters the input with ITS OWN ARMA and applies the SAME filter to the
    output, so r(k) estimates the impulse response weights directly. Proposes
    (b, r, s) from the CONTIGUOUS block, and judges exogeneity by a portmanteau
    over k < 0 — feedback there means a single-input transfer model does not
    hold, and you must say so.

    `band`: "constant" (2/sqrt(N), what the C does) or "haugh-box"
    (2/sqrt(N-|k|), what the original paper says).

    `ident_pre` is an ALTERNATIVE `.pre` for the output, used ONLY here, to
    compute the deviation that gets prewhitened. The estimation keeps the real
    model. This is Muñoz §2.4's artifice, and it exists for one situation: the
    output carries STOCHASTIC seasonality and the input does not, so the
    input's filter cannot remove it and the filtered output is still
    non-stationary at that frequency — "una ccf muy poco informativa", which
    the contiguous-block heuristic will read an order off anyway. Build the
    alternative in `art` with the seasonality made DETERMINISTIC, and pass it
    here.

---

## `identify_network`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `nlags` | integer | no | `0` |

Propose the whole NETWORK from the residual CCFs of the DIAGONAL fit.

    Estimates the diagonal model (no transfers) and reads the cross-correlations
    of its residuals: who leads whom, at what lag, and which innovations move
    together contemporaneously.

    ⚠ IF THE PROPOSAL CONTAINS A CYCLE the system is SIMULTANEOUS: it has no
    topological order and cannot be cast as a triangular VARMA. Tell the analyst
    and route them to `sima`. Do not prune the cycle yourself to make it fit.

---

## `impulse_response`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |

nu(k), its cumulative sum and the GAIN, with standard errors.

    nu_k is the response of the output k periods later to a ONE-OFF unit shock;
    the cumulative column is the response to a PERMANENT change and converges to
    the gain. This is usually the answer the analyst came for.

    Unlike a VAR's, this impulse response IS identified without an ordering —
    because the restrictions (exogenous input, diagonal Q) are declared and
    tested, not assumed.

---

## `load_pre`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `paths` | string | yes | — |
| `check` | boolean | no | `True` |

Load a case's `.pre` **or `.inp`** files, CONFIRM the roles, prove the bridge.

    `paths` is a comma-separated list; **the FIRST one is the OUTPUT**. mtram
    does not build univariate models — if one is missing, send the analyst to
    ART.

    **Both file kinds are accepted, and they are not the same claim.** A `.pre`
    is an OPTIMUM in re-runnable form; an `.inp` is a SPECIFICATION whose
    values are seeds. Either works here because this tool RE-ESTIMATES each
    series with fue on the way in, so the stored values are only a starting
    point: fed art's `.inp` with every parameter at zero it reaches the same
    likelihoods as with fue's `.pre` (−1744.135582 both ways on the passthrough
    case).

    That matters more than it looks, because by the convention **a `.pre` that
    an analyst edited has become an `.inp` again** — so a specification is the
    normal input after any reformulation, not an exception. The tool's name is
    historical; the contract is "a specification, optionally already optimal".

    Three things happen here, and the last two are the point.

    **It states the roles and asks you to confirm them.** Which series is the
    output is not something the data can decide (decision node N0); it is the
    question the analyst arrives with. Loading in the wrong order silently
    produces a model of the wrong thing, so the roles are printed back for
    confirmation before anything else happens.

    **It then estimates the DIAGONAL model — no transfer — and checks it
    reproduces the univariate fits.** With a diagonal structure the exact
    likelihood factorises, so the joint fit MUST equal the sum of the separate
    ones. That identity is drtran's validation gate: it is what says the cast,
    the transform, the deterministics and the seeds all survived the crossing
    from fue intact. Until it holds, no transfer result from this case means
    anything, because the thing the transfer is added to is already wrong.

    **And it says WHICH of the two you actually brought.** The gate re-estimates
    anyway, so comparing the stored values against the re-estimated ones costs
    nothing — and it answers a question nothing in the suite could answer
    before: were these optima, or something that still needed estimating? The
    likelihood gap it reports is the rigorous form (it is >= 0 always, and zero
    exactly when the files were optima). Neither answer is a problem; being
    unable to tell them apart was.

    It also reads out the estimation situation before any transfer complicates
    it: how the optimiser terminated on a problem whose answer is known.

    `check=False` skips the estimation and only loads — for a case big enough
    that the wait is not worth it, at the price of flying blind.

---

## `overfit`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |

Overfit ON PURPOSE: enlarge the transfer and see whether it protests.

    Every one of Muñoz's six cases does this, and none of them skips it even
    when nothing looks wrong — an adequate portmanteau says the model is not
    contradicted, and that is not the same as saying nothing better was
    available. Brajín §2.3.1 states the doctrine: a model is confirmed by
    enlarging it and finding the extra parameters non-significant.

    Two enlargements, one at a time: one more numerator weight (s+1) and one
    denominator (r+1). Each is refitted jointly and compared by likelihood
    ratio with 1 degree of freedom.

    **A failed experiment is not a confirmation.** Muñoz 6.4.3 abandons an
    overfit because "la situación de estimación está mal definida (altas
    correlaciones entre muchos de los parámetros de relación), por lo que este
    experimento puede considerarse FALLIDO" — the enlarged model says nothing
    about the small one, in either direction. That distinction is reported.

    The session's estimated model is left untouched.

---

## `plot_calibration`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |
| `path` | string | no | `` |

PLOT the CCF **with and without** the dominant anomaly — the verification.

    In the school's teaching this is a fact to VERIFY, not to infer, and it is
    easy to see: an anomaly inflates the residual variance, which is the divisor
    of every correlation, so it flattens ALL the lags at once — not only the ones
    it touches. Take the point out and the coefficients come back.

    Show this beside `calibrate`'s table. The number (`CCF x`) states the claim;
    the picture is what lets the analyst check it.

---

## `plot_ccf`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `input_index` | integer | no | `1` |
| `lags` | integer | no | `0` |
| `path` | string | no | `` |

PLOT the prewhitened CCF of one link — the identification instrument.

    Show this to the analyst and read it WITH them; the numbers alone do not
    carry the shape. Right of zero the input leads, and the first significant bar
    is `b`; left of zero the OUTPUT leads, which is feedback and which a
    single-input transfer model assumes away. Bars on both sides mean the
    specification does not hold.

    It is drvarma's canonical CCF drawing, so it looks the same as in `sima` and
    as drvus drew it. Writes a PNG and returns its path.

---

## `plot_forecast`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |
| `series_index` | integer | no | `0` |
| `path` | string | no | `` |

PLOT the level forecast with its band, over the recent history.

    The band is ASYMMETRIC under a log model — it is formed in the transformed
    scale and mapped back — so it is drawn as it really is, not as a symmetric
    ribbon. Writes a PNG and returns its path.

---

## `plot_impulse_response`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `link_index` | integer | no | `0` |
| `path` | string | no | `` |

PLOT nu(k) and its cumulative sum, each with a 95 % band.

    Left panel: the response to a ONE-OFF unit shock. Right: to a PERMANENT
    change, converging to the gain, which is drawn as a line because that
    convergence is the thing to look at. Writes a PNG and returns its path.

---

## `plot_residuals`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `series_index` | integer | no | `0` |
| `lags` | integer | no | `0` |
| `path` | string | no | `` |

PLOT the residual series with its ACF and PACF — fue's own panel.

    The same drawing `art` shows after a univariate fit, so the analyst reads one
    instrument, not two. These are the STRUCTURAL residuals: with a
    contemporaneous transfer the reduced-form ones are correlated by
    construction. Writes a PNG and returns its path.

---

## `refine_link`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `input_index` | integer | no | `1` |
| `b` | integer | no | `-1` |
| `smax` | integer | no | `6` |

Estimate a GENEROUS pure MA and read the denominator off its weights.

    What Muñoz's cases actually do, and what `identify_link` does not: the CCF
    is read once to get a delay, and then a free-form numerator is ESTIMATED —

      "v(B) = .35 + .21B + .40B² + .16B³ + .64B⁴ + .29B⁵ + .34B⁶ … De hecho,
       esto equivale a una estimación de los primeros términos de la ccf."

    and the SHAPE of the estimates decides the parametrisation:

      "Se observa que el valor absoluto de los mismos decrece conforme aumenta
       el retardo, lo que parece indicar que la relación requiere un factor
       AR(1) con parámetro positivo."

    Strictly more informative than reading the CCF once, and for one concrete
    reason: these weights are estimated JOINTLY with the noise model, while the
    prewhitened CCF is not. It is also how a denominator gets proposed at all —
    r is the one order the CCF cannot show you, because an infinite geometric
    tail and a long finite one look alike in a sample.

    `b` defaults to the delay `identify_link` reads. The session's network and
    estimated model are left untouched.

---

## `set_network`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `links_json` | string | yes | — |

Fix the network to estimate: a JSON list of {out, inp, b, r, s}.

    `out` and `inp` are indices into the loaded `.pre` list (0 is the output).
    A cycle is refused here too — `read_dag`'s rule, in memory.

---

## `variance_decomposition`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `series_index` | integer | no | `0` |
| `horizon` | integer | no | `12` |

How much of the forecast error comes from EACH source of innovation.

    REFUSES when the structural Q is not diagonal: with correlated innovations
    the decomposition is not unique, it needs an ordering, and that is the VAR's
    problem. Transmit the refusal — do not go looking for a number anyway.

---

## `write_inp`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `outdir` | string | no | `.` |

Write the JOINTLY re-estimated univariate blocks back out as `.inp`.

    The blocks MOVE when the transfer is fitted beside them, and this is how
    those estimates leave mtram: into a modified specification, or back to ART
    and fuf, which read this format.

    **`.inp` and NOT `.pre`, and the extension is the claim.** A `.pre` is an
    `.inp` with the estimates as new initial values — an OPTIMUM, in re-runnable
    form — and the invariant is testable: run fue on a `.pre` and the numbers do
    not move. What is written here fails that test by 13.11 on the passthrough
    case, and cannot pass it: these blocks are optimal WITH the transfer, so on
    the diagonal they evaluate worse, necessarily.

    That makes it a perfectly good starting point and a false `.pre`. The file
    carries no mark of who wrote it, so a wrong extension here would be
    indistinguishable downstream from an optimum fue had certified — and the
    ladder climbs by trusting exactly that.

    The transfer is not written: this is a univariate file.

---
