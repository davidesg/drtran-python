# The ladder as an optimisation algorithm

*A study of the stage-to-stage flow in building transfer models, and of what
the `.inp` / `.out` / `.pre` conventions actually guarantee. Written for
researchers and developers: the first group needs to know what the files
assert, the second needs to know what they may not break.*

Every claim below is measured on the passthrough case (WTI → IPC_ES, monthly,
215 observations, `tests/test_end_to_end_passthrough.py`) rather than argued.
Where the analogy with an optimisation algorithm breaks, §6 says so — that
section is the reason this document is worth reading twice.

---

## 1. Two levels, and the files are the interface

Model building in this suite is a **bilevel optimisation**, and almost every
design decision follows from keeping the two levels apart.

```
   OUTER (discrete)          structure: lambda, d, D, p, q, deterministics,
                             and for mtram the network (b, r, s)
        |  proposes
        v
   INNER (continuous)        exact maximum likelihood over the parameters
                             of a FIXED structure
        |  returns an optimum + a diagnosis
        v
   OUTER reads the diagnosis and proposes again
```

The inner problem is a well-posed maximisation with a unique answer. The outer
problem is a search over a combinatorial space with no gradient, driven by
diagnostics. **The file conventions are how the two levels talk**, and each
extension marks which level produced it:

| file | produced by | asserts |
|---|---|---|
| `.inp` | the OUTER level (art, an analyst, drtran writing back) | a **specification**. Parameter values are seeds — a starting point, nothing more |
| `.out` | the INNER level | the full record of one estimation *and its diagnosis* — the outer level's only feedback |
| `.pre` | the INNER level | that same `.inp` with the estimates as new initial values: **an optimum, in re-runnable form** |

The cycle is then literally a fixed-point iteration with a structural search
wrapped round it:

```
.inp --(fue estimates)--> .pre --(analyst reformulates)--> .inp --> ...
```

**A `.pre` that is touched becomes an `.inp` again.** Editing the specification
unmakes the claim that these values are its optimum. This is not bookkeeping
etiquette: it is what keeps the outer level from mistaking a proposal for a
result.

---

## 2. The invariant

The `.pre`'s claim is testable, and this is the load-bearing fact of the whole
convention:

> **Run fue on a `.pre` and the numbers do not move.**

Measured:

| file | max abs change after re-running fue |
|---|---|
| `.pre` written by fue | **0.000000** |
| the univariate block written back by drtran after a JOINT fit | **13.109261** |

The second is not a defect in the arithmetic. Those blocks are optimal *with
the transfer in the model*, and the univariate optimum is by definition fue's
separate estimate; they cannot be a fixed point of the univariate operator.
That is why drtran writes `.inp` and not `.pre` (`write_inp`, CLI `-W`), and
why it was a real defect when it did otherwise (`BUGS.md`, BUG-3).

The general rule for developers: **only the program that performed an
estimation may write a `.pre`.** The file carries no mark of its author, so a
fabricated one is indistinguishable downstream from a certified one — and the
ladder climbs by trusting exactly that.

### 2.1 The gap is one-sided, and its sign is a certificate

Something stronger holds. Because the diagonal fit *maximises* the same
likelihood the stored values merely *evaluate*, we always have

```
    logL(diagonal fit)  >=  logL(at the stored values)
```

with equality **iff** the stored values are the univariate optima. So the
difference is a non-negative optimality gap. Measured:

| starting file | logL at stored values | logL of the diagonal fit | gap |
|---|---|---|---|
| genuine `.pre` | −1744.135583 | −1744.135582 | **+0.000000** |
| non-optimal block | −1748.794396 | −1744.135582 | **+4.658814** |

**This certificate is free, and the gate now claims it** (§7.1). It had not
before: `mtram.load_pre` re-estimates each series with fue on the way in, so
its factorisation check compares the joint diagonal fit against *re-estimated*
univariate likelihoods, and that difference is ≈0 whatever it was handed. One
extra likelihood evaluation at the stored values — no optimisation — is enough
to tell an analyst whether the files they were given are optima or something
that still needed estimating.

---

## 3. Crossing to the multivariate stage

The transition from *m* univariate optima to a joint model rests on one
identity and one inequality:

```
    SUM_i logL(series i)   =   logL(joint DIAGONAL fit)   <=   logL(joint model)
         \_______________________________/                      \___________/
              the factorisation                                  the transfer
              -> mtram's GATE                                     -> the LR test
```

The **equality** holds because with diagonal structure the exact likelihood
factorises. It is therefore a complete check on the crossing: the
transformation, the differencing, the deterministics and the seeds either all
arrived intact or the identity fails. Measured on the passthrough pipeline:
**−1.50e−07**.

The **inequality** holds because the diagonal model is the joint model with the
transfer restricted to zero. That is what makes the diagonal rung the right
null for the likelihood-ratio test, and it is why `estimate` reports it:
"converges" and "was worth fitting" are different claims. Measured: diagonal
−1744.135582, joint −1704.423918, LR = 79.42 on 2 df.

Note that these are two independent claims. The identity proves the CROSSING;
it says nothing about whether the `.pre` files were optima, because the
re-estimation on the way in makes it hold either way. That second question is
what §2.1's gap answers, and both are now reported side by side.

---

## 4. Do the files transport useful information? Yes, and it is measurable

The claim implicit in the design is that a `.pre` carries a *point* closer to
the optimum than a cold start. Measured on the joint fit, same structure, same
convergence criterion:

| starting point | logL at start | iterations | wall | reaches |
|---|---|---|---|---|
| the `.pre` seeds | −1744.1356 | **139** | 20.4 s | −1704.423918 |
| cold (univariate blocks zeroed) | −1977.5070 | **200** | 29.2 s | −1704.423918 |

Two things, and the second matters more than the first.

**It saves 61 iterations, about 30 %.** The files are worth what they cost.

**Both reach the same optimum, to 1e−4.** So what is transported changes the
PATH and not the DESTINATION. That is the property that makes the ladder
trustworthy rather than merely convenient: a better starting point is an
efficiency, never a thumb on the scale. If seeding changed the answer, every
result in the suite would depend on the order in which its models were built.

---

## 5. The stages, end to end

| # | from | by | to | level | what it guarantees |
|---|---|---|---|---|---|
| 1 | raw levels | `art.load_data` | `.inp` (data only) | — | the series enter UNTRANSFORMED. λ and d are decisions, not preprocessing |
| 2 | `.inp` | `art` identification | `.inp` (structure, seeds at 0) | outer | a specification. art does not estimate, and does not claim to |
| 3 | `.inp` | `fue` | `.out` + `.pre` | inner | an optimum, plus the diagnosis the outer level needs |
| 4 | `.pre` | analyst + `.out` | `.inp'` | outer | reformulation. The edit demotes the file, by the rule |
| 5 | *loop 3–4* | | | | until the diagnosis is clean |
| 6 | `.pre` × m | `mtram.load_pre` | the diagonal rung | — | the crossing, by the factorisation identity |
| 7 | the rung | `identify_link` | (b, r, s) | outer | a PROPOSAL, read off the prewhitened CCF |
| 8 | (b, r, s) | `estimate` | the joint model | inner | exact ML, with the LR against the rung |
| 9 | the model | `diagnose` | a verdict + a branch | outer | and the branch is where the school's rules live |
| 10 | the model | `write_inp` | `.inp` × m | outer | a starting point. NOT a `.pre` — see §2 |

Stage 7 deserves one note for researchers. **Prewhitening uses the univariate
models as fue estimated them** — `identify()` defaults `x` to
`x0_from_pre(cast_spec)`, i.e. the values carried in the specification, not the
diagonal fit's. That is doctrinally correct: Muñoz §2.6 states that the input's
univariate model "permanece inalterado desde el inicio hasta el fin del
proceso". It is also numerically indistinguishable in the normal case, because
the diagonal fit reproduces those same values exactly (measured: 0.00000000) —
they differ only when the file was not an optimum, which is precisely the
situation §2 exists to prevent.

---

## 6. Where the analogy breaks

An optimisation algorithm is expected to ascend. **The outer loop does not, and
cannot, and this is the most important thing in this document.**

**Likelihoods are not comparable across integration orders.** A model with
d = 1 and one with d = 2 are models of *different data* — different variables,
different sample lengths. Their log-likelihoods are numbers on incommensurable
scales, and comparing them is meaningless rather than merely imprecise. So the
outer level has no single objective to ascend, and any implementation that
picked a specification by comparing likelihoods across d would be producing
confident nonsense.

Three consequences that developers should treat as constraints:

1. **Order of operations is not convention, it is necessity.** The integration
   order must be settled *before* the ARMA structure is searched, because only
   within a fixed d is the likelihood a usable objective. This is why the unit
   root tests come first, and why they are tests rather than a search.

2. **Within a fixed d, the outer search IS an ascent**, and AIC/BIC comparison
   is legitimate — `art` uses it. Outside it, only the diagnosis arbitrates.

3. **No global guarantee is available, and the convention never claims one.**
   What the ladder guarantees is that every *inner* answer is a true optimum of
   its own specification, that every crossing preserved the state, and that
   every step is re-runnable from the file it left behind. Reproducibility and
   local optimality, not global optimality.

A fourth, specific to mtram: the same non-comparability recurs when the
transfer changes the noise's apparent integration order (`SCHOOL_PRACTICE_STUDY`
§ Tier 3, item 11). `estimate` reports the near-unit MA root and explicitly
refuses to compare the two likelihoods, for exactly the reason above.

---

## 7. What the analysis said should change — both done

Two gaps, both small, both following from §2.1: places where the
implementation knew less than the convention did. Both are now closed, in
`load_pre`'s gate.

### 7.1 The gate now claims the certificate

One likelihood evaluation at the stored values, taken **before** `m.fit()`
overwrites them, yields the optimality gap of §2.1. It is reported alongside a
per-file reading of how far each set of coefficients moved when re-estimated.
Measured on the passthrough files:

| loaded | logL at the values brought in | gap vs the diagonal fit | per-file movement | verdict |
|---|---|---|---|---|
| fue's `.pre` | −1744.135583 | **+0.000000** | 3.5e−05, 3.2e−05 | an OPTIMUM |
| art's `.inp` | −1914.407710 | **+170.272128** | 47.36, 14.51 | a SPECIFICATION |

Two signals, and they agree. The gap is the rigorous one — non-negative by
construction, zero *iff* the files were optima. The per-file movement is the
readable one, and its threshold is **measured rather than chosen**: a genuine
`.pre` does not return exactly to its own values (the file stores six decimals
and the optimiser stops inside its own tolerance, leaving ~3e−5), while a
specification moves by 14–47 on the same case. Six orders of magnitude apart,
no grey zone, and 1e−3 sits comfortably between.

Note this is reported, never refused. Both inputs are legitimate.

### 7.2 `.inp` input is first-class, and says so

`load_pre` accepts either kind and its docstring now states the contract as it
really is: *a specification, optionally already optimal*. It re-estimates on
the way in, so the seeds are only seeds — fed art's `.inp` with every parameter
at zero it reaches the same likelihoods as with fue's `.pre`.

This is not an edge case being tolerated. By the rule in §1, **a `.pre` an
analyst edited has become an `.inp`**, so a specification is the normal input
after any reformulation. The tool's name is historical; the contract is not.

The two together answer a question the suite could not answer before: *were
these optima, or something that still needed estimating?* Neither answer is a
problem. Being unable to tell them apart was — an analyst who believes they
started from the best univariate model of a series, and started from a
half-estimated specification, is misreading their own work, and nothing was
going to tell them.

---

## Sources for the measurements

`tests/test_end_to_end_passthrough.py` walks stages 1–8 and pins the fixed
point, the crossing, the `.inp` acceptance and the diagonal identity.
`tests/test_write_inp.py` pins §2 in the direction that matters: it asserts the
written file *does* move when re-estimated, so that if it ever stops moving,
someone has to come back and ask why. `tests/test_walkthrough_mtram.py` pins
§7 in both directions — an optimum reported as one, and a specification
reported as one, on the same gate.
