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
