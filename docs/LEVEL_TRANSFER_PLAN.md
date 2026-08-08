# Moving the transfer to LEVELS: the finding, the precedent, and the plan

*BUG-8's cause is established and the fix is major surgery in both `drtran` C
and Python. This is the plan, the evidence it rests on, and the test banks it
needs. Nothing here is implemented.*

---

## 1. What is wrong

The two implementations specify the cast differently:

| | what it relates |
|---|---|
| **TASTE** (and `fue`) | the **levels**: `y_t = ν(B)x_t + N_t`, with the differencing carried by the **noise** |
| **drtran**, C and Python | the series **already differenced**, each by **its own** (d, D) |

`drtran.c:365-372` — `apply_univariate_model(&Tm[i], …)` per series, then
`trim_to_common`. `cast.py` does the same. **Inherited from the original, not a
port regression.**

### Why they agree when (d, D) match

Differencing is linear and commutes with ν(B), both being polynomials in B:

```
    ∇(y − ν(B)x) = ∇y − ν(B)∇x
```

So relating the differenced series gives the same ν as relating the levels with
differenced noise. Measured: the four matched cases agree with the oracle to
between 1e-6 and 1.2e-4, and — more tellingly — **both estimators contract the
raw OLS coefficient by the same factor to two decimals**:

| caso | (d, D) | drtran/OLS | TASTE/OLS |
|---|---|---|---|
| ES | d=1 D=0 | 0.81 | 0.81 |
| USA | d=1 D=0 | 0.66 | 0.66 |
| CA | d=1 D=0 | 0.68 | 0.67 |
| UK | d=1 D=0 | 0.86 | 0.86 |

That agreement is not luck: it is the same estimator doing the same thing.

### Why they diverge when (d, D) differ

They stop being the same model. drtran relates `∇∇₁₂y` to `∇x`: the output's
extra `∇₁₂` removes the seasonal-frequency variation from the LEFT side and
leaves it on the right, so part of the input's variance has no counterpart to
explain and **ν shrinks**.

| caso | (d, D) | drtran/OLS | TASTE/OLS |
|---|---|---|---|
| FR | d=1 **D=1** | **0.36** | 0.77 |
| DE | d=1 **D=1** | **0.56** | 0.95 |
| EMU | d=1 **D=1** | **0.50** | 0.96 |

The contraction factor is the single dimension along which they separate.

### The confirmation

Give the input the OUTPUT's differencing and drtran lands on the oracle:

| caso | drtran as it is | input differenced by the output's (d,D) | TASTE |
|---|---|---|---|
| FR | 0.004282 | **0.009072** | **0.009070** |
| DE | 0.006441 | 0.011131 | 0.011660 |
| EMU | 0.005849 | 0.010750 | 0.011880 |

France to 2e-6.

---

## 2. The precedent: `fue` already does this, in C and in Python

`fue`'s `custom` intervention applies ω(B)/δ(B) to an ARBITRARY series **in
levels**, with the differencing carried by the noise. That is design (c),
already implemented and shipped in both languages. Driving it with WTI as the
custom input:

| caso | `fue` custom, LEVELS | drtran, differenced | TASTE |
|---|---|---|---|
| FR | 0.011414 | 0.004282 | 0.009070 |
| DE | **0.011607** | 0.006906 | **0.011660** |
| EMU | **0.011772** | 0.006194 | **0.011880** |

Germany and the euro area land on the oracle to 0.5 %. France is 26 % away and
that gap is NOT explained — it is smaller than drtran's and on the right side,
but it is not agreement, and the plan below treats it as an open question
rather than rounding it away.

**So the algebra is not speculative.** It exists, it is estimated by exact ML in
`fue`, and it reproduces the oracle. What drtran needs is not a new method but
the same specification.

---

## 2b. The oracle's specification, read from its source

Five lines of `MRQEST.PAS:108-116` are the whole thing:

```pascal
wobs := M - TFM.NOISE.d - TFM.NOISE.ds * TFM.NOISE.sp;
CalcNoise(nts, DATA, M, TFM, PointMoSe);                    { n = y - nu(B)x, LEVELS }
TransDiff(nts, 1.0, TFM.NOISE.d, TFM.NOISE.ds, sp, 1, M);   { the NOISE is differenced }
BackForeCast(nts, TFM.NOISE.parms, wobs, Back, ...);
CalcRes1st(nts, Resi^, TFM.NOISE.parms, wobs, Back, ...);
```

Read it in order:

1. **`CalcNoise` forms `n_t = y_t − Σ_j Σ_k ν_j[k]·x_j[t−k]` on the transformed
   LEVELS.** No differencing appears anywhere in that loop
   (`BACKTF.PAS`). `DATAR[...]^.Data` holds what `TFEST.PAS:592,609` left there
   — `TransDiff(…, lambda, 0, 0, 1, 1, nn)`, i.e. Box-Cox and nothing else.
2. **Then `TransDiff` differences the NOISE**, and the orders it uses are
   `TFM.NOISE.d` and `TFM.NOISE.ds`. That is the decisive detail and it is in
   the names: **the differencing belongs to the NOISE MODEL**, not to the output
   series and not to the inputs.
3. `wobs = M − d − ds·sp` is the effective sample, lost once, on the noise.
4. Backforecasting and the residual recursion then run on the differenced noise
   with the noise's own ARMA.

Two more details worth carrying into any implementation:

* **The inputs are extended BACKWARDS, not truncated.** `CalcNoise` indexes
  `Data[t−k]` from `t=1`, so the convolution reaches before the sample. That is
  why `TFEST.PAS:655-670` backforecasts each input's differenced series and then
  calls `BackLevel` to rebuild the extended LEVEL. The input's own univariate
  model is used for exactly two things — prewhitening at identification, and
  extending the level backwards — and **never to difference it for the fit**.
* **`MaxLag = 20` when `r > 0`** (`CalcNoise`): the infinite tail of `1/δ(B)` is
  truncated at twenty lags. A documented approximation of the oracle, and
  something to match or improve on deliberately rather than by accident.

### What this settles

There is no such thing as "the input's differencing" in a transfer model. The
input enters in levels. And the output's differencing is not the output's
either — it is the **noise's**. drtran's framing, in which each series is
differenced by its own univariate `(d, D)` and the differenced series are then
related, is a DIFFERENT MODEL, not a different implementation of the same one.

### And where drtran is genuinely more complex

TASTE does not estimate the input's model at all: it takes the input's
univariate model as GIVEN and frozen, which is also the school's doctrine
(Muñoz §2.6 — "el modelo U del input permanece inalterado desde el inicio hasta
el fin del proceso"). drtran estimates the whole system JOINTLY, so the input
does have an equation, and that equation needs the input differenced by its own
orders.

So the two requirements are:

* the OUTPUT's equation is on `∇^d ∇ₛ^D (y − ν(B)x)` — the noise, differenced
  once, with the input entering in levels;
* the INPUT's equation is on `∇^{d_x} ∇ₛ^{D_x} x` — its own.

They are both satisfiable, and they are not both satisfiable **by a single
VARMA on one `W` matrix with one column per series**. That is the whole
difficulty, stated exactly.

## 3. What actually has to change

The key simplification, and it is worth stating before anyone starts writing
level-based code: **drtran does not need to work in levels.** Because ∇ commutes
with ν(B),

```
    ∇^d ∇ₛ^D N_t  =  ∇^d ∇ₛ^D y_t  −  ν(B) · (∇^d ∇ₛ^D x_t)
```

so the correct stationary noise is the output's `w_y` minus the transfer applied
to **the input differenced by the OUTPUT's operator**. The whole defect is which
operator differences the input in the transfer term.

### And here is the difficulty

The input plays TWO roles and they want different vectors:

1. as the transfer's input feeding the output's equation → the **output's**
   operator;
2. as a series with its own univariate model inside the joint VARMA → **its
   own** operator.

The cast has one column per series in `W`. A series cannot be differenced two
ways at once. That is why the code does what it does, and it is the reason this
is surgery rather than a patch.

### Three ways out, and they are not equally good

**(A) Subtraction cast (`-S`).** It computes the noise explicitly
(`N = w_y − Σ tr_j`), so the transfer term can be built from a separate,
output-differenced vector while `W` keeps each series' own. Small change,
confined. But the subtraction cast is not the default and truncates the sample,
which is exactly what the embedded cast was introduced to avoid.

**(B) Embedded cast, input carried twice.** Add the input a second time as an
extra series differenced by the output's operator, with its own equation
constrained out. Keeps the embedded cast and its exactness, at the price of an
`m+1` system and a constraint scheme that has to be got right.

**(C) Embedded cast, single differencing for the whole system.** Difference
every series by the output's operator. Simplest and it is what the confirmation
above did — but the input's univariate block then no longer means what its
`.pre` says, and its ARMA would need re-specification.

None is free. (A) is the cheapest and least ambitious; (B) preserves what the
embedded cast is for; (C) is simplest to write and changes the most.

### What the diagonal gate will NOT tell you

Measured: with the input re-differenced by the output's operator the gate still
reports **−8.31e−08**, identical to before. The factorisation identity holds
under any differencing as long as both sides use the same one, so **the gate is
blind to this class of defect**. It is not a safeguard here and must not be
treated as one.

---

## 4. Test banks the change needs

The existing bank is blind to this: every canonical case is matched-(d, D), and
`battery.py`'s tolerance is ABSOLUTE (5e-3) on coefficients of order 5e-3, so
FR and DE reported OK at 112 % and 69 % relative error.

1. **Relative tolerance in `battery.py`.** Mixed or relative, so small
   coefficients are not compared against a tolerance that is 100 % of their
   value. Without this the oracle bank cannot see the thing it exists to see.
2. **Mixed-(d, D) cases in the drtran bank.** `gen_synthetic.py` builds both
   series with the same structure; it needs cases where the output carries an
   extra `∇ₛ` and the truth is known, so the fix can be checked against a
   generator rather than against another program.
3. **The oracle cases already exist** — `Taste/oracle/cases/pt8_*.json` and
   `scripts/repro_bug8_mixed_dD_vs_oracle.py`, which exits 1 while the bug is
   live, 0 when fixed, and 2 if the MATCHED cases start disagreeing. That last
   branch is the one that matters: losing the matched agreement would
   invalidate the comparison rather than resolve it.
4. **A `fue`-custom cross-check**, since `fue` reaches the oracle on the same
   data by the level route. Any drtran fix should land on `fue`'s number, and
   where it does not — France — that is a finding, not a rounding error.
5. **Homologation against the C must be re-run in full** (48 runs, both casts),
   and it WILL move for mixed-(d, D) cases. That is the point; what must not
   move is the matched ones.

---

## 5. The order of work

1. Relative tolerance in `battery.py` — otherwise nothing below is measurable.
2. Mixed-(d, D) cases with known truth in `gen_synthetic.py`.
3. Understand France: `fue` and TASTE agree on DE and EMU and differ by 26 % on
   FR. Until that is explained, "landing on the oracle" is not a criterion that
   can be applied uniformly.
4. Choose among (A), (B), (C) — a decision, not a derivation.
5. Implement in Python, verify against the three banks.
6. Port to the C and re-homologate.

Steps 1-3 are study and cost little. Step 4 is where the design decision lives
and should not be taken until 3 is answered.
