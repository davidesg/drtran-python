# The pass-through × MEG bank: the experiment that would settle the mixed case

*Design only. Nothing here is run yet. It is David's proposal: repeat SF_MEG's
empirical study with WTI as an input, so that the deterministic / stochastic /
mixed seasonality comparison becomes a comparison of TRANSFER models.*

---

## 1. Why this is the right bank, and not just a bigger one

The synthetic three-arm bank (`tests/gen_mixed_operators.py`) established the
law `ν̂(1) = ν(1)·Δ(1)` on data I generated. Its weakness is exactly that: I
chose the truth. SF_MEG's study has the same three arms **on real series, with
the univariate models already identified, published and defended** — and, which
is what makes it decisive, **it already runs the out-of-sample forecast
comparison**. The transfer version inherits all of that instead of rebuilding
it.

The mapping is exact, and it is not a coincidence — it is the same distinction
in both studies:

| SF_MEG variant | operator | order at f=0 | vs WTI's `∇` | synthetic arm |
|---|---|---|---|---|
| **D** — harmonics at every frequency | `∇` | 1 | **Δ = 1**, matched | M |
| **S** — `ifadf[f]=1` at the stochastic frequencies | `∇ · Π factors` | 1 | Δ = Π factors | **S** |
| **∇∇₁₂** — the full seasonal difference | `(1−B)²S(B)` | **2** | Δ = `1−B¹²` | **Z** |
| **MEG / mixed** — some frequencies each way | `∇ · Π (stochastic only)` | 1 | Δ = that product | **S**, partial |

So the study's own model set already spans the three cases, and the MEG models
— the point of the paper — are precisely the ones with no oracle.

---

## 2. The material

**Outputs.** `SF_MEG/empirical/cases/`, nine cases with variants already built:

```
DE_CPI 9   ES_CPI 8   FR_CPI 15   IE_CPI 7   NL_CPI 9
DE_CORE 12 ES_CORE 14 IE_CORE 6                        (FR_CORE has none yet)
```

Monthly, `data/*.csv` running 2002-01 to 2026-05 (293 observations).

**Input.** WTI. Two sources, and they disagree on the window:
`Taste/oracle/data/passthrough8/PT8_WTI.pre` has 216 from 2002-01, and
`Nivel de Precios y Energia/IPC.xlsx` has 286 from 2002-01. **The common window
has to be fixed first and fixed once**, because `check_alignment` refuses series
that do not end on the same date, and rightly — that refusal exists because a
silent misalignment once produced `b=18` in earnest.

**Deterministics.** Each output keeps the interventions its published univariate
model carries. They are not re-identified: the whole design rests on changing
ONLY the seasonal representation, exactly as `FORECAST_COMPARISON.md` insists
for the univariate exercise. Its warning transfers verbatim — in FR the D model
was once built with a different noise than the pre-MEG baseline and contaminated
the comparison.

---

## 3. What has an oracle and what does not

| variant | TASTE can fit it | why |
|---|---|---|
| **D** (harmonics, `d=1`) | **yes** | deterministic regressors, `d=1`: the `pt8_*` cases already do this |
| **∇∇₁₂** (`d=1, D=1`) | **yes** | plain differencing |
| **S** and **MEG/mixed** | **no** | `ifadf` frequency-by-frequency has no counterpart in TASTE's model record |

That is the whole difficulty and it is David's framing: **the models the paper is
about are the ones the oracle cannot check.** Two arms can be homologated
against an independent implementation; the interesting arm cannot.

---

## 4. The certificate for the mixed models

David's criterion, and it is the contribution that makes the experiment
decidable. With no oracle, a mixed model earns its place by being **bracketed**
and **anchored**:

**(a) Bracketed.** The mixed model's forecast must lie BETWEEN the pure-D and
the pure-S forecasts, horizon by horizon. It has to: its seasonal representation
is literally intermediate — some frequencies deterministic, some stochastic —
and Abraham & Box's argument says the two extremes bound the forecast function.
A mixed model outside the bracket is not a compromise between them and something
is wrong.

```
    min(ŷ_D(h), ŷ_S(h)) - tol  <=  ŷ_MEG(h)  <=  max(ŷ_D(h), ŷ_S(h)) + tol
```

**(b) Anchored.** The transfer forecast must not be far from the UNIVARIATE
forecast of the same output. A transfer refines a forecast; it does not
relocate it. Measured against the model's own band:

```
    |ŷ_TF(h) - ŷ_univariate(h)|  <<  se(h)
```

**(c) And it must earn its keep.** `drtran/docs/FORECAST_DIAGNOSIS.md` already
states the rule and the test: a transfer model is kept over its univariate
**iff** it forecasts better out of sample, by Diebold-Mariano with the HAC
variance and the HLN correction. (a) and (b) say the model is sane; (c) says it
is worth having. All three are needed: (b) alone would be satisfied by a
transfer that does nothing.

**The sandwich is the point.** Better than the univariate, but close to it, and
between the two extremes it interpolates. A model that fails (a) or (b) has a
defect; one that passes them and fails (c) is merely useless.

---

## 5. What the experiment measures, beyond passing

Three things the bank can settle that the synthetic one could not:

1. **The gain, on real data, across the three arms.** `ν(1)` estimates the same
   economic pass-through in all three variants of a country, so the three should
   agree within their standard errors. Before the dispatch they could not: the
   `∇∇₁₂` variants had `Δ(1) = 0` and their gains were annihilated. This is the
   real-data replication of §2d, with nine countries instead of one synthetic
   series.

2. **Whether the seasonal-only mismatch bites in practice.** Arm S is the case
   nobody had tried, and synthetically its gain was wrong by a factor of twelve.
   The `S` and `MEG` variants are arm S on real data, and the dispatch should
   remove it. If it does not, the synthetic result does not transfer and that is
   a finding.

3. **Whether the by-parts forecast survives contact.** It is measured on three
   countries and one input so far (6.6-20.0 % better one-step). Nine cases, four
   variants each, twelve horizons is a different order of evidence.

---

## 5b. First results — run 2026-08-09

`SF_MEG/empirical/passthrough/` (`run_tf.py`, `RESULTADOS.md`). Window
2005-01..2019-12, 180 observations, WTI re-estimated on it (`WTI_2005.pre`,
AR(1) φ=0.3263). Link fixed at `b=0, r=0, s=1` across every variant.

**The three arms classify themselves** — `Δ(1) = 1.000` for the harmonic
variants, `0.268` and `0.804` for the `ifadf` ones, `0.000` for `∇∇₁₂` — from
comparing the two operators, with nothing declared.

**Check §5.1 passes, and cleanly.** ES_CPI's eight variants give a gain of
**0.0264-0.0274**, across arms M and Z alike. The same economic pass-through,
estimated by specifications that resolve the seasonality in opposite ways. All
eight converged on the gradient (termcode 1).

**And what the program gave before**, with the dispatch disabled — which is
literally the old path, `xin = W[:, l.inp]`, not an approximation of it:

| | arm | before | now |
|---|---|---|---|
| ES_CPI_airline | Z | 0.006745 | **0.026545** |
| ES_CPI_airMA_mu | Z | 0.005197 | **0.026402** |
| DE_CPI_airline | Z | 0.007298 | **0.013023** |
| **DE_CPI_sto1** | **S** | **−0.000505** | **+0.015359** |

Eight variants of ES_CPI: before, the five `air*` said the pass-through was
~0.006 and the three harmonic ones ~0.027. An analyst would have concluded that
the seasonal specification changes the economics by a factor of five. It does
not. And `DE_CPI_sto1` came out with the **wrong sign**.

**Not yet certified.** Several DE and DE_CORE variants stop at **termcode 3** —
steptol, not the gradient, `ifault = 0` — which is a long-standing open item.
Their figures are indicative. ES_CPI's are not affected.

### Two corrections the run forced

**The link order is identified, not imposed.** Fixing `s=1` across every variant
was my decision and it was wrong. Where `omega1` does not exist — DE_CPI, at
−0.000332 — the surface is flat in that direction and the optimiser stops at
termcode 3, which is CORRECT behaviour: it is saying the parameter is not
identified. The same cases converge at `s=0`. Where it does exist — ES_CPI, at
−0.010408 — `s=1` converges. The order belongs to the CASE, identified once and
held across that case's variants; what is held fixed is the link, not the
seasonal representation.

**And the diagonal already fails, which is David's check.** All eighty DIAGONAL
fits, with no link at all: **57 converge on gradtol, 10 on steptol, and 12 stop
at termcode 3** — plus one broken `.pre` (`FR_CPI_S_f5`). So in those twelve the
problem is UPSTREAM of the transfer, in the univariate models or the joint
diagonal, and not in the link at all. They cluster in DE_CORE (6 of 12), ES_CORE
(2) and DE_CPI (2). Any transfer figure from those cases inherits the defect,
and the check is cheap enough to be the first gate of the bank rather than an
afterthought.

### The diagonal gap — established facts, and an unexplained cause

David's follow-up: the diagonal fits START practically at the optimum, so
converging in twenty iterations by anything other than gradtol is odd. It is,
and chasing it produced facts but not yet a cause. What is measured:

| case | nit | termcode | logL at the stored values | fitted | **gap** |
|---|---|---|---|---|---|
| DE_CORE_m00 | 13 | 2 | −629.168775 | −629.168775 | **0.000000** |
| ES_CPI_airline | 8 | 1 | −664.448426 | −664.288274 | 0.160152 |
| ES_CPI_m00 | 19 | 1 | −681.034215 | −680.586954 | 0.447261 |
| DE_CPI_m00 | 21 | 3 | −648.925875 | −647.566048 | 1.359826 |
| FR_CPI_D | 21 | 1 | −586.127714 | −584.002893 | 2.124821 |

Three hypotheses tested, **all three refuted**:

1. **The `.pre` are not optima.** They are. Re-run through `fue` they move by
   at most 1e-6 — the ladder's own invariant, and it holds.
2. **The variance ratio is badly seeded**, which is the natural suspect in a
   pass-through where the input's variance is a thousand times the output's.
   It is not: `log(var2/var1)` starts at 6.9157 and ends at 6.9236, moving
   0.008, and re-seeding it from the univariate sigmas changes nothing.
3. **Some coefficients are fixed in one case and free in another.**
   `DE_CPI_m00` and `DE_CORE_m00` each carry exactly one fixed coefficient and
   their gaps are 1.36 and 0.00; `FR_CPI_D` carries none and has the largest.

What moves is the **deterministic coefficients** — up to 0.054 in `DE_CPI_m00`
(`omega_d1[2,0]`: −0.074221 → −0.127816) — while `DE_CORE_m00` moves nothing at
all beyond WTI's phi at 5e-5. So a zero gap is achievable and something makes it
non-zero in most cases.

**The cause is not established and I am not going to guess it.** This is a
finding about the ladder's diagonal gate rather than about BUG-8, it is
reproducible in one command, and it deserves its own investigation rather than a
paragraph at the end of someone else's. Until it is settled the bank's figures
are indicative: the RANKING across variants of one case is what the experiment
rests on, and every variant of a case carries the same gap, but the absolute
likelihoods are not clean.

## 6. Order of work

1. **Fix the window and build the `.pre` files.** WTI over the common span; each
   output's variants re-cut to the same end date. This is where
   `check_alignment` earns its keep — let it refuse rather than trimming.
2. **The two arms with an oracle first** (D and ∇∇₁₂), as `pt8_*` cases in
   `Taste/oracle/cases/`. They are the control: if they do not homologate, the
   third arm's numbers mean nothing.
3. **The mixed arm, with the §4 certificate** as the acceptance criterion,
   wired as a bank rather than a report.
4. **The forecast exercise**, reusing `FORECAST_COMPARISON.md`'s design (the D
   model is the pre-MEG baseline; only the seasonal representation changes) and
   `FORECAST_DIAGNOSIS.md`'s recursive fixed-parameter scheme with balanced
   origins.
5. **Compare against the univariate results already in the study.** The
   univariate side is done and defended; every transfer model has a published
   counterpart to be anchored to.

---

## 7. On "this would make drtran a unique program"

David's assessment, and it is worth stating precisely rather than either
repeating or dismissing.

What the combination would be is: **joint exact-ML estimation of transfer models
whose output carries hybrid seasonality resolved frequency by frequency, with
correct gains and with forecasts.** Within this tree, nothing else does it —
TASTE has no `ifadf`, `art`/`fue` do MEG but univariately, `drvarma` does VARMA
but not transfers with per-frequency seasonality. Whether anything OUTSIDE this
tree does, I have not surveyed and should not claim.

What is certainly true is narrower and still worth having: **before BUG-8 this
program could not have run the experiment at all.** Every mixed-operator gain it
produced was wrong by `Δ(1)`, and every forecast put the transfer back with the
wrong vector. The experiment is possible now and was not three days ago; whether
it makes the program unique is for the results to argue.
