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

## 2c. Why the legacy never hit this — and it is not luck

The m6 models and the network are the embedded specification's home ground, and
they work. Checked, all six series of `tests/data/m6/`:

```
M6_EA.pre  d=2 D=0 freq=4 ifadf=[0, 1, 1]
M6_EC.pre  d=2 D=0 freq=4 ifadf=[0, 0, 0]
M6_EI.pre  d=2 D=0 freq=4 ifadf=[0, 0, 0]
M6_EP.pre  d=2 D=0 freq=4 ifadf=[0, 0, 0]
M6_EU.pre  d=2 D=0 freq=4 ifadf=[0, 0, 0]
M6_P.pre   d=2 D=0 freq=4 ifadf=[0, 0, 0]
```

**All six carry the same `(d, D) = (2, 0)`.** EA is the seasonal one — Relloso's
Table 4 gives it `∇∇₄` — and it still carries `D=0`, because
`∇∇₄ = (1−B)²(1+B)(1+B²)`: the `(1−B)` inside `∇₄` ADDS to the regular
difference, so the encoding is `d=2` plus fixed-frequency factors
(`ifadf=[0,1,1]` at π/2 and π), not `D=1`. `build_m6.py` says so in as many
words — *"d=2, D=0, ifadf=[0,1,1] (¡NO d=1!)"*.

So the school's own encoding of a seasonal series inside a multivariate system
**avoids mixed `(d, D)` by construction**, and with matched operators the
embedded cast is right.

### And the network, which is where the transfers actually are

The diagonal has no transfers, so it cannot exercise the defect at all. The
network can, and it is the repository's only `.dag`:

```
EP <- EI   b=1 r=0 s=1
EP <- EC   b=1 r=0 s=2
EI <- EU   b=1 r=0 s=3
EU <- EC   b=2 r=0 s=1
```

Four transfers over EP, EI, EC, EU — **all four at `(d, D) = (2, 0)`, and every
one of them identical on both sides of the arrow**. EA, the one seasonal series,
is a node of the diagonal but **appears in no transfer at all**. So the network
does not merely avoid the mixed case by a happy encoding; the seasonal series is
not related to anything.

That is why m6 homologates, why the network works, and why the defect survived:
**no transfer in the legacy ever relates two series with different
differencing.**

This does not make BUG-8 less of a defect — `art` writes `D=1` for FR, DE and
EMU, and those models are legitimate — but it does three things for the plan:

1. **It scopes the surgery.** The embedded specification is correct wherever
   the system shares its differencing, which is every legacy case and every
   canonical test. What is broken is a region the design never entered.
2. **It warns that the legacy banks cannot validate the fix.** m6, the network
   and the canonical cases will pass before and after, because they never
   exercise the path. The mixed-(d, D) cases of §4 are not a nicety; without
   them there is no test.
3. **It suggests a fourth route, and its limit.** One could re-encode `∇∇₁₂` the
   school's way — `d=2` plus fixed-frequency factors — instead of `D=1`. That
   removes the D-mismatch. It does NOT remove the mismatch: FR would then be
   `d=2` against WTI's `d=1`, and the same algebra bites. In m6 the `d`s
   coincide because the six series genuinely all want `d=2`; in the passthrough
   the output wants `∇∇₁₂` and the input wants `∇`, and they genuinely differ.
   Harmonising there would be forcing the data, not encoding it better.

## 2d. The controlled experiment, and the law it establishes

The measured cases could not separate two explanations, because
`∇∇₁₂ / ∇ = (1−B¹²)` carries an excess root at frequency zero AND eleven at the
seasonal frequencies, both at once. A synthetic bank separates them
(`tests/gen_mixed_operators.py`). Three arms, the same true `ω₀ = 1`, an input
that is a random walk, and only the location of the excess roots changes:

| arm | output | input | order at f=0 | Δ(B) = op_y/op_x | Δ(1) |
|---|---|---|---|---|---|
| M | `∇` | `∇` | 1 vs 1 | 1 | 1 |
| S | `∇₁₂` | `∇` | **1 vs 1** | `S(B) = 1+B+⋯+B¹¹` | **12** |
| Z | `∇∇₁₂` | `∇` | **2 vs 1** | `1−B¹²` | **0** |

Arm Z is the observed case (FR/DE/EMU against WTI). Arm S is the one nobody had
tried: a mismatch **purely at the seasonal frequencies**, with the regular
differencing identical on both sides.

### The law

What the cast fits is not ν but **ν̃ = ν · Δ**, and every number confirms it:

| arm | Δ(1) | ν̂(1), s=1 | ν̂(1), r=1 | ν̂(1), s=12 | ν̂₁₂ | ν̂₀ |
|---|---|---|---|---|---|---|
| M | 1 | 0.95 | 0.95 | **0.98** | −0.01 | 0.95 |
| S | 12 | 1.93 | 14.11 | **12.04** | 0.00 | 0.94 |
| Z | 0 | 1.06 | 1.03 | **0.07** | **−1.01** | 0.93 |

Read the S row across: with `s=1` the fit reports 1.93 — the first **two** terms
of `S(B)`, which is all a two-lag numerator can hold. Give it a rational tail
and it reaches 14.1; give it twelve lags and it lands on **12.04 = S(1)**. Read
the Z row: with a short numerator it reports ≈1, because its error sits at
**lag 12** and neither `s=1` nor a positive geometric tail can reach there. Give
it twelve lags and it finds `ν̂₁₂ = −1.008` — exactly the `−ω₀` that `(1−B¹²)`
predicts — and the gain collapses to **0.07 ≈ Δ(1) = 0**.

So:

> **ν̂(1) = ν(1) · Δ(1)**, as soon as the fitted transfer has the reach to see
> where Δ puts its weight. Until it does, the damage is partial and the reported
> gain is somewhere between the truth and the truth times Δ(1).

That also explains why the real cases contract by *varying* amounts (0.36, 0.56,
0.50 of OLS) rather than all collapsing: each fitted `(b, r, s)` has a different
reach.

**And `ν̂₀ ≈ 0.94` in all three arms.** The contemporaneous impact survives
whatever the operators do. It is the GAIN — the long-run multiplier, the one
quantity a transfer model is usually built to report — that breaks.

### What this settles about seasonal frequencies

The natural hope is that only the regular differencing has to agree, and that
differing seasonal integration is harmless. **It is not.** Arm S has identical
order at frequency zero and its gain is wrong by a factor of **twelve**. In
fairness it is wrong in a recoverable way — `Δ(1) = s` is a known constant, so
`ν(1) = ν̂(1)/s` — whereas arm Z's `Δ(1) = 0` destroys the quantity outright.
But "recoverable by a correction nobody applies" is not "valid".

The requirement is therefore the whole operator, not its regular part:

> **`∇^{d_y}∇ₛ^{D_y} = ∇^{d_x}∇ₛ^{D_x}`, frequency by frequency**, including
> whatever `ifadf` carries — or the reported gain is wrong by `Δ(1)`.

And it confirms the reading of the observed case exactly. `∇∇₁₂ = (1−B)²S(B)`
has **order 2 at frequency zero** against WTI's 1; that excess `(1−B)` puts
`Δ(1) = 0` in the law, and the gain is annihilated. Arm Z reproduces it from
scratch.

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

**(D) Frozen input, the oracle's own way out.** (A)–(C) all fight the same
thing: the input needs two vectors at once. TASTE does not have that problem
**because it does not estimate the input's model at all** — it takes it as given
and frozen (`TFEST.PAS`; and Muñoz §2.6 as doctrine). Offer that as a mode and
the conflict evaporates: the input appears only through the transfer,
differenced by the OUTPUT's operator, and there is no second equation asking for
a different vector. It is the simplest of the four and the only one with
doctrinal backing. The price is real and specific — no joint estimation of the
input, and therefore **no LR test against the diagonal rung**, which is what
drtran adds over TASTE.

None is free. (A) is the cheapest and least ambitious; (B) preserves what the
embedded cast is for; (C) is simplest to write and changes the most; (D) is
simplest of all and gives up the joint fit.

### (E) Dispatch on Δ — embedded when the operators match, subtracting when they do not

**This is the route to take**, and it is better than any of (A)–(D) taken alone.
Δ is known before estimation: compare the two operators frequency by frequency.
Then

* **Δ = 1 → embedded cast, exactly as today.** Every legacy case, m6, the
  network and every canonical test are matched, so they are *untouched*. The 48
  homologation runs do not move, and the regression risk is nil.
* **Δ ≠ 1 → subtracting cast**, with the transfer fed an input differenced by
  the OUTPUT's operator.

It keeps the joint fit and the LR test — which (D) gave up — and needs no `m+1`
system with constraints — which (B) does. Three things must be true, and the
third is the one to watch.

**1. Dispatching alone fixes nothing.** `cast.py:323` reads
`xin = W[:, l.inp]`: today the subtracting cast takes the input's OWN column,
the same defect. What the route buys is *room*: `tr` is built explicitly
outside the VARMA, so `xin` can be a separately-differenced vector without
touching `W`. In the embedded cast the transfer IS off-diagonal VARMA
coefficients acting on `W`'s columns, and there is no physical place to put a
second vector. That is the structural difference, and it is the whole reason
this works.

**2. The subtracting cast is not the oracle.** By its own documentation it
builds the noise outside the engine and, needing input values before `t=1`,
*sets them to zero* — "the likelihood it then computes is exact, for the WRONG
series". TASTE performs the same subtraction but **backforecasts the input to
levels first** (`BackLevel`, `TFEST.PAS:655-670`) so the convolution has
support. So the dispatch buys the right *specification* with a truncated
pre-sample. The measured cost of that truncation is small — ω's bias
+0.0017 → −0.0002 on 69 observations, RMSE under 1 % — and trading it against a
gain that is wrong by a factor of twelve is not a close call. Adding
backforecasting is the natural follow-on that reaches the oracle exactly.

**3. It changes the estimator without saying so, and that must be announced.**
Across different `(d, D)` this is harmless, because §6 already establishes those
likelihoods are not comparable. The live risk is *within* one output model:
choosing between candidate inputs, one matched and one not, would compare a
number from the embedded cast against one from the subtracting cast. Those are
comparable in principle and computed differently in fact. The dispatch must
therefore be reported in the output, not performed quietly.

### What must NOT be the fix: requiring the operators to match

Tempting, and wrong as a resolution. It would refuse FR ← WTI, which is a
legitimate model: the French CPI has stochastic seasonality and WTI does not,
and that is the data rather than an analyst's mistake. **The oracle does not
require it** — not by being laxer, but because in its formulation the question
does not arise: the input enters in levels, has no operator, and there is no Δ
to mismatch.

It is, however, the right **interim guard**, and it should ship before the
surgery does: today drtran returns a wrong gain silently, and §3's last
paragraph shows the diagonal gate cannot see it.

**And post-hoc correction must not be attempted.** `ν̂(1) = ν(1)·Δ(1)` invites
dividing by `Δ(1)` and moving on. It is not safe: the law holds only once the
fit has full reach. Arm S with `s=1` reported **1.93**, not 12 — dividing by 12
would give 0.16, worse than leaving it alone. The reported gain sits somewhere
between `ν(1)` and `ν(1)Δ(1)` depending on each `(b, r, s)`'s reach, and that
point is not knowable from the output.

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

1. ~~Relative tolerance in `battery.py`~~ — **DONE**, `taste-port 16161c5`.
   `tol = min(absolute, max(1e-4, 0.05·|expected|))`: only ever tightens, so
   nothing that failed can start passing. Verified on all fourteen cases —
   eleven pass, including the four matched passthrough cases and all five
   synthetic and forecast ones; the three that fail are FR, DE and EMU, which
   is the point. **The bank can now see BUG-8**; before, FR missed the
   threshold by twenty microns and reported OK at 112 % relative error.
2. ~~Mixed-operator cases with known truth~~ — **DONE**,
   `tests/gen_mixed_operators.py`, and §2d is what they established.
3. Understand France: `fue` and TASTE agree on DE and EMU and differ by 26 % on
   FR. Until that is explained, "landing on the oracle" is not a criterion that
   can be applied uniformly.
4. ~~Choose the route~~ — **DECIDED: (E)**, dispatch on Δ.
5. Implement in Python. **Δ and the guard are DONE** (`75ea12e`, `ef40867`,
   `8156910`): `delta_operator` divides the two operators' polynomials -- so the
   school's `∇∇₄` as `d=2, ifadf=[0,1,1]` is recognised as the SAME operator as
   `d=1, D=1`, which comparing `(d, D)` tuples would not -- and
   `check_operators` warns inside `build_cast_spec`, silently for every matched
   case. Full battery 371 passed.

   **The dispatch itself is still to do**: feed the subtracting cast's transfer
   term an input differenced by the OUTPUT's operator (`cast.py`,
   `xin = W[:, l.inp]` today) and route Δ ≠ 1 to it.

   One correction the battery forced, and it matters for the design: the guard
   first REFUSED the non-nested case, and `EP <- EA` in the m6 network showed
   that up. **Route (E) never needs Δ.** It needs `∇^{d_y}∇ₛ^{D_y} x`, which is
   computable whatever the two operators are; Δ exists only to say how wrong the
   gain is. Requiring nestedness was an artefact of reasoning through the
   quotient.
6. ~~Port to the C and re-homologate~~ — **DONE**, `drtran 655e255`.
   `operators_differ` compares the two full `rnsop` polynomials (so the school's
   encoding is recognised, as in Python); `build_stationary_series` also builds
   `w_alt[j]` by swapping ONLY `ornsop`/`rnsop` and calling the same
   `apply_univariate_model`; `tran_shootx.c` feeds the transfer `w_alt[j]` when
   it exists; and `main` switches to the subtracting cast, announcing it on
   stderr with both orders and the reason.

   **FR: `ω₀ = 0.009162`, `ω₁ = −0.005973`, logL `−638.941000` — identical to
   the Python to the last digit.** C battery 303 PASS 0 FAIL, m6's canonical
   likelihoods untouched (`−1709.511575` diagonal, `−1697.613401` network),
   Python battery 378 passed, homologation 12 passed.

   The matched control was verified against the PREVIOUS BINARY rather than
   argued: `ES_CPI <- WTI` gives `0.016400` and logL `−718.287406` before and
   after, identical. (`0.016402` is the same case under the SUBTRACTING cast —
   the two casts differ slightly, and confusing them is exactly the mistake that
   control exists to avoid. It caught it once already.)
8. **THE FORECAST — open, and the next piece of work.** Raised by David the
   moment the estimation was homologated, and he is right that it does not
   follow for free.

   `forecast.py:_system_forecast` puts the subtracted transfer back with

   ```python
   acc += nus[k][j] * we[tt - j, lk.inp]      # the input's OWN column
   ```

   and `we[:, lk.inp]` is the input differenced by ITS OWN operator. That was
   the right vector before the dispatch, because it was the vector nu had been
   fitted against. **It is no longer**: for a dispatched model nu multiplies the
   input differenced by the OUTPUT's operator, so putting the transfer back with
   the other one is the same category error the estimation just stopped making.
   The future values the recursion needs have the same problem — they are the
   input's own-differenced forecasts, and the convolution wants
   output-differenced ones.

   Two more things to settle in the same pass:

   * the reintegration to levels uses `m0`'s `d`, `D` and `s` — the OUTPUT's,
     which is correct — but the transfer contribution now arrives already in the
     output's differences, so the two must be checked to line up rather than
     assumed to;
   * `kmax` is `max(l.b + l.s + 1)`, capped at 200 when any `r > 0`. With a
     mismatched pair the effective nu is `nu*Delta`, which reaches
     `deg(Delta)` further — twelve extra lags for a monthly seasonal
     difference. The cap has to know that.

   **Measured.** Two probes, and the first was worthless: comparing `forecast`'s
   output against `w - a` failed on the MATCHED case too, so the identity was
   wrong (forecast returns levels), not the forecast. Nor does the one-step RMSE
   separate them — 40 origins give ratios of 0.70, 1.01, 1.18, 1.09 against the
   reported sigma, all near one.

   What does measure it is the defect itself: the transfer contribution computed
   with the input's own column against the same thing computed with the vector
   nu was fitted against.

   | | sigma_a | sd(contribution) | sd(DIFFERENCE) | as sigma |
   |---|---|---|---|---|
   | FR | 0.15854 | 0.15173 | 0.10982 | **0.69** |
   | DE | 0.21796 | 0.14162 | 0.10229 | **0.47** |
   | EMU | 0.17649 | 0.14460 | 0.10444 | **0.59** |

   Half to two-thirds of a residual standard deviation, systematically, in the
   transfer's contribution to every forecast. Not a rounding difference. It does
   not show up in a one-step RMSE because part of it is absorbed and forty
   origins are few — which is exactly why the direct measurement was needed.

   ### The design: copy TASTE, because this is not the embedded logic

   David's call, and `TFFO.PAS` bears it out. TASTE's transfer forecast is not
   the embedded cast's recursion with a correction; it is a different procedure,
   and the parallel with its ESTIMATION is exact:

   ```pascal
   FOR t := 1 TO L DO
      FOR j := 1 TO T DO                     { each input }
         FOR i := 0 TO lags1[j] DO
            IF t > i THEN sum2 += NU[j][i] * fc[j][t-i]              { its FORECAST }
            ELSE          sum2 += NU[j][i] * Data[Punt[j]][N-B+t-i]  { its LEVEL }
      fc[0][t] := sum1 + (zt[N-B+t] if in-sample else fcN[t])        { + the NOISE }
   ```

   Three separate problems, recombined on LEVELS:

   1. **Each input is forecast univariately**, by its own model — `ForeCastUS`
      after `BackForeCast`, exactly as a standalone series (line 150).
   2. **The noise is forecast by its own ARMA**, after `CalcNoise` builds it on
      levels and `BackForeCast` extends it (lines 242, 258, 292).
   3. **They are recombined as `y = nu(B)x + N` on the LEVELS**, with each input
      observed where the index is in the past and forecast where it is not.

   The error variance follows the same split (lines 336-360):

   ```
   vr[t] = sigma2_noise * SUM_j PSI_noise[j]^2
         + SUM_i sigma2_i * SUM_j (NU_i * PSI_i)[j]^2
   ```

   the noise's own psi weights, plus each input's propagated through `nu(B)`.
   `CalcPsiB` is called with `d` and `ds`, so these are LEVEL psi weights and
   `vr` is directly the level forecast error variance.

   **And the question that started all this never arises.** No differencing
   appears anywhere in the recombination, so "which operator does the input
   carry" has no meaning here — the input enters as a level, exactly as it does
   in `CalcNoise`. The dispatched forecast inherits the estimation's own
   coherence instead of needing a patch on top of a recursion built for a
   different cast.

   ### Why two paths and not one with a branch

   The same split as in estimation, and for the same reason.

   **Embedded: the transfer IS the VARMA.** Forecasting the VARMA forecasts
   everything — one recursion and it is done. That is the same property that
   removes the pre-sample truncation: the engine sees the whole system, so the
   exact initialisation and the forecast both fall out of a single object.

   **Subtracting: the engine only ever sees the NOISE.** The transfer was
   removed before it looked at anything, so it has to be put back — and putting
   it back requires forecasting the inputs too, each by its own model. That is
   not a shortcoming of the subtracting cast; it is what "subtracting" means.

   So the embedded cast is unaffected throughout, and it is what every matched
   case and all of the legacy use. This is a SECOND forecast path for the
   subtracting route, not a replacement.

   **And the two will not agree on the same fitted model.** That is expected,
   not a defect: two correct procedures for two different casts, exactly as
   their likelihoods already stopped being comparable. Worth saying in the
   output before it alarms someone who compares them.

   Related material already in the tree: `drtran/docs/FORECAST_DIAGNOSIS.md`
   (the out-of-sample criterion and its Diebold-Mariano test) and the `predice`
   sources under `atws/fuf/predice/`, whose `usfo.c` is the univariate forecast
   the input step needs.

7. ~~Backforecast the input instead of zeroing the pre-sample~~ — **DONE**,
   `bade3a6` (Python) and `drtran 14e948c` (C). FR's `omega[1]` went from 2.3e-4
   to **6.0e-6** of the oracle. Detail below.

   Backforecast the input instead
   of zeroing the pre-sample, which is what takes the subtracting cast from "the
   right specification, truncated" to the oracle exactly. `TFEST.PAS:655-670`
   shows how — difference each input by its own model, backforecast, then
   `BackLevel` to rebuild the extended level. Worth doing, not urgent: the
   remaining disagreement with the oracle is 0.24-1.01 %, inside the band the
   four matched cases sit in (0.00-0.93 %).

Step 3 answered itself. The 26 % France gap was a property of the `fue`-custom
LEVELS route (0.011414 against the oracle's 0.009070), not of the specification:
drtran's dispatch lands at 0.009162, **1.01 %** away. So "agrees with the
oracle" IS a usable criterion, and the three mixed cases now meet it as well as
the four matched ones ever did.

### How the bank stays green without forgetting — `taste-port 270868c`

The three failing cases would make `battery.py` exit 1 permanently, and a bank
that is always red cannot signal a NEW regression. Marking them expected-to-fail
has the opposite problem: the marker fossilises and the defect is forgotten,
which is approximately how this survived four years.

Both properties are kept by making the marker two-sided. A case carrying
`falla_conocido`:

* **fails →** reported as `CONOCIDO`, does not count towards the exit code, and
  is printed in full — note and offending values — on every run. A known defect
  is not a hidden one.
* **passes →** reported as `YA NO FALLA` and **counted as a failure**, asking
  for the marker to be removed. Either the fix landed or something moved the
  reference; neither is worth staying quiet about. That branch is what stops the
  marker from rotting.

Verified by making both branches fire rather than by reading the code: the three
marked cases exit 0 as `CONOCIDO`, and marking `pt8_es` — which passes — gives
exit 1 with *"YA NO FALLA … quita falla_conocido"*. So **when step 5 lands, the
bank turns red on success**, and the way to make it green is to come back to
this document.
