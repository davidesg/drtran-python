# Porting drtran to Python — the record of the process

This document tells **how** the port was done and **on what evidence** each rung
was accepted. `../TODO.md` says what is done and what is missing; the `README.md`
says what the program is. This is the logbook: the decisions, the figures that
support them and the traps not to fall into again.

Written on 2026-07-29, with steps 0, 1 and 2 closed; updated as the later rungs
went in.

---

## 1. The method: a ladder of gates, not a line-by-line port

The port does **not** translate the C function by function. It translates the
*cast* — the map from parameters to VARMA structure — and reuses everything else,
which already exists in Python. Every rung has a **gate**: a numerical equality
that must hold before going on. If it does not close, nothing advances; and the
fault always lies with the cast.

> **Non-negotiable principle.** drvarma's `elf` is used **as it is**: not
> modified, not patched, not special-cased. Any discrepancy with fue is a bug of
> drtran's cast, **never** of `elf`.

That principle is what turns the port into a measuring instrument: when something
does not add up, the place to look is bounded in advance.

The second rule, inherited from the work on drvarma: **before declaring a defect,
build a case with a known answer.** The three discrepancies in §5 were closed
that way, not by reading code.

## 2. What is reused and what is ported

Of the C's **12,615** lines, most is not ported: it is reused.

| C | destination | lines |
|---|---|---|
| `elfvarma.c` + `drvmlest.c` + `qnewtopt.c` + `nlatools.c` | already in **drvarma** Python | 3,350 |
| `fue_pre_reader.c` | `fue.load()` — **not ported** (see §3.1) | 601 |
| `gnuplot_i.c` + `fuf_graphic.c` | matplotlib | 1,860 |
| `tran_shootx.c` | `cast.py` + `embed.py` | 668 |
| `diagnose.c` | `identify.py` + `diagnose.py` + `netid.py` | 1,687 |
| `forecast.c` | `forecast.py` | — |
| `drtran.c` | CLI and orchestration -> `cli.py` | 4,201 |

The port now occupies **2,750 lines** of Python in `src/drtran/` and about
**1,600** of tests, with **111 tests**.

## 3. The rungs, with their evidence

### Step 0 — the input arrives intact

**Gate:** that the `.pre` fue writes reaches the cast **field by field**, and that
fue Python reproduces the reference univariate models.

Verified: lambda/d/D, `refactor`, mu **with its fixed flag**, orders and
coefficients of the regular and seasonal AR/MA with their free-flags, the
fixed-frequency AR(2)/MA(2), the deterministics with their omegas and flags, and
the series. Canonical case: phi 0.402839 / 0.299193, logL −7.3917 / −760.0326,
**sum −767.424341** (differing by 7.6e-09 from fue).

Tests: `test_pre_roundtrip.py` (8), `test_baseline_univariate.py` (3).

#### 3.1. Decision: `fue_pre_reader.c` is not ported

601 lines of parser. Duplicating it would create a **second source of truth** that
falls out of sync the moment fue changes one line of its format. `fue.load()`
already reads `.pre` and `.inp`. The `.pre` is the contract between the two
programs, and a contract with two different readers is not a contract.

Known cost: `model.write_pre()` requires the fitted model, so the cycle
`estimate -> .pre -> reread` is still untested. That is what closes continuity
towards the next rung of the methodological ladder; it does not block the cast.

### Step 1 — the diagonal cast

`cast.py` reuses fue's **univariate** cast per series (`build_est_spec` +
`cast_us_py`), which already returns `w` with Box–Cox applied, differences taken
and deterministics subtracted — it is the C's `build_stationary_series`. drtran
only **ASSEMBLES**: block-diagonal Phi and Theta, mu per series, the `w`'s aligned
at the end and a normalised Q.

Two things that are not obvious and cost the first afternoon:

- **The likelihood is CONCENTRATED.** `est()` does not estimate the scale: it
  decomposes Sigma = sigma2*Q with Q[1][1] = 1 and concentrates sigma2.
  Evaluating `elf_varma` with an absolute Sigma gives something else — passing the
  identity gave **−7802** instead of −767. drvarma's `_elf_f1f2` is used, plus the
  formula from `drvmlest.c:est [4]`.
- **The variance ratios are not left at zero.** The scales differ by a factor of
  1098 in the canonical case; starting at 1 leaves the initial point at **−1371**.
  They are computed with the same `elf`, m = 1, on the `.pre`'s seeds.

### Step 2 — THE GATE

**Diagonal joint fit == fue run separately.** With a diagonal structure the exact
likelihood factorises, so the joint fit must reproduce the **sum** of the
univariate ones.

| | logL |
|---|---|
| ES_CPI_m10 (fue) | −7.3917 |
| WTI_ar1 (fue) | −760.0326 |
| **sum = target** | **−767.424341** |
| **diagonal joint fit (port)** | **−767.424341** (diff. 3.9e-07) |

And it is reached **already at the `.pre`'s seeds**, which confirms empirically
why the C reports `termcode 3` on this rung: the seeds *are* the optimum.

Tests: `test_cast_diagonal.py` (4).

### Step 3 — the transfers

Two casts, as in the C, and they **do not measure the same thing**:

- **By subtraction.** `Link(out, inp, b, r, s)`, `compute_irf` and the
  subtraction from the output: series 1 of the VARMA becomes the noise
  `N_t = w_Y - SUM_j transfer_j`. It models the NOISE and **truncates** at the
  start of the sample.
- **Embedded** (`embed.py`), the **default**, as in the C. Polynomial algebra:
  row i = diagonal phi_i*D_i, off-diagonal −phi_i*omega_k*B^b_k*(D_i/delta_k), MA
  D_i*theta_i, with D_i = PROD_k delta_k over the incoming links; the series with
  **nothing subtracted**. It models the OBSERVED series, without truncating.

That the embedded cast does **not** give a higher likelihood than the subtracting
one is not a defect: they are different objectives on different data. The C shows
the same pattern.

A consistency check on both: **with omega = 0 the likelihood is exactly the
diagonal one, difference 0.0.** It is the gate's own test, one rung higher.

The joint estimation (`estimate.py`) uses Mauricio's scaled objective (1995, eq.
3.5) normalised to 1 at x0 — in the multivariate case (f1/f1_0)^m*(f2/f2_0),
because ll = C − 0.5n(m*log f1 + log f2) — minimised with drvarma's `raxopt`. A
rejected point returns 1.0 and the optimizer moves away.

Measured on the canonical case Y <- X, (b,r,s) = (0,0,0): **omega_0 = 0.016002**,
logL −736.774 against the diagonal's −767.424, **LR = 61.3** (1 df, p = 4.9e-15),
by gradient in 21 iterations.

### Step 4 — identifying (b, r, s)

`identify.py` ports `prewhiten_and_identify`: it prewhitens the input with ITS
ARMA, applies **the same filter** to the output, computes the CCF and reads
nu(k) = r(k)*s_beta/s_a.

It homologates with the binary: band 0.13640, r(0) = 0.492, r(1) = 0.310,
r(2) = 0.025, r(−1) = −0.077, r(−6) = −0.128, and the same proposal **b = 0,
r = 0, s = 1**. Exogeneity by portmanteau over k < 0: **Q(24) = 18.2969,
p = 0.7884**, identical to the C — `ChiTestC`'s divisor is n−i+1, not n−i.

The C's two decisions that keep the answers sane are replicated **on purpose**:

1. the structure is the **CONTIGUOUS block** from b (there is a significant peak
   at lag 24 that does NOT enter the proposal: with 5% bands one lag in 20 is
   expected outside);
2. exogeneity is judged by **portmanteau**, not by counting how many cross.

Reviewed against Haugh–Box (1977) and Tsay (1985). Tests:
`test_identification.py` (12), including a synthetic transfer with a known delay
(b = 3).

### Step 5 — the network, the DAG and `expand_params`

Up to here the model was one output and its inputs. The **network** is the general
case: a series may receive transfers and be an input to another at the same time,
which is what the school's systems really are (m6-1: EC -> EU -> EI -> EP, plus
EC -> EP). And with the network comes what the legacy `shootx` did by hand:
parameters shared between the transfer and the input's ARMA, factorized
numerators, fixed factors.

Three pieces, deliberately separate:

| piece | file | what it says |
|---|---|---|
| **the graph** | `.dag` -> `network.py` | who moves whom, with which (b, r, s) |
| **the parameters** | `.cns` -> `slots.py` | what is free, fixed, shared or an expression |
| **contemporaneity** | `.cns`, `q[i,j]` | what moves together within the same instant |

Separating the graph from the parameters is not tidiness for its own sake: the
DAG says **dynamics with a delay** and Sigma says **simultaneity**. Mixing them is
exactly the error the C's near-collinearity warning chases — a contemporaneous
transfer (b=0) and the covariance of those two innovations explain the same thing
at lag zero.

**The `.dag`** is lines of `OUTPUT <- INPUT b r s`, with the series by **name**,
not by position: a `.dag` must not depend on the order of the command line. A
cycle is rejected, and the message **says which one it is** — without a
topological order the system stops being a recursive DAG and becomes a
simultaneous-equations model, which is not what this cast represents.

**The slot table** is the DSL. Every position of the full vector has a stable name
and one of five natures: `free`, `fixed`, `alias` (SHARED), `product`
(`x = -y * z`) and `lincomb` (`x = t1 + t2 - t3`, each term a slot or a product of
two). The optimizer sees only the free ones:

```
xfree --expand--> xfull --cast--> Phi, Theta, mu, w, Sigma --elf--> l
```

`expand` goes **inside** the objective, so the gradient comes out by finite
differences with no chain rule: adding expressions to the DSL does not touch the
optimizer. It is the C's decision, and it is why the product and the linear
combination fitted without touching `_qnewt`.

Two design points worth not losing:

- **The slot order is not the C's, the names are.** The C groups by class (all the
  ARMA, then all the deterministics); here it groups by series, because the
  univariate block is produced by `fue._build_initial_x` and fue decides the
  order. It does not matter, because **the `.cns` goes by name** — the C repo's
  own `.cns` files are read here. What IS checked is that the total matches
  `cast_spec.npar`: if the two enumerations drifted apart, the names would stop
  matching the positions and the `.cns` would constrain the wrong parameter,
  **silently**.
- **The covariances are born fixed at zero.** They always enter the map, but a
  diagonal covariance is the default case and freeing one is the analyst's
  decision (`q[5,2] = free`), not something switched on in bulk: the legacy m6-1
  frees **three** of its fifteen. Outside the region where Q is positive definite
  the point is rejected (the objective returns 1.0 and the search moves away),
  which is Mauricio's strategy (1995 §3); that boundary has never bitten in the
  real cases.

Homologation, on five m6 series with the chain EC -> EU -> EP, one input with two
outputs, a denominator with r=1 and two free covariances:

| | drtran C | port | diff |
|---|---|---|---|
| free network (40 free of 48 slots) | −1434.696068 | −1434.696068 | 1.9e-10 |
| + product + linear combination (38 free) | −1439.505804 | −1439.505804 | 9.4e-08 |

and `expand` reconstructs the derived slots starting from **only the free ones**
of the C's optimum (diff 3e-07, which is the 6-decimal rounding of its report).
That tests the cast; whether the **search** arrives is another matter, and it is
tested apart: a 3-series network with 24 free, where the port converges to
**−912.244333 in 180 iterations** against the C's 181.

### Step 6 — identifying the network

With the diagonal rung closed, the ladder says: **read the CCFs of its residuals**
to discover the system's dynamic relationships (Munoz Polo 2001, §2.6).
`netid.py` ports it:

| k | reading | proposal |
|---|---|---|
| k > 0 | a_i leads a_j | link i -> j, with b and s from the contiguous block |
| k < 0 | a_j leads a_i | link j -> i |
| k = 0 | contemporaneous | free q[i,j] |
| both | feedback | does not fit a DAG: the dominant side is taken, with a warning |

The residuals are **not recomputed**: they come from the same `elf` that scores
the likelihood (`atf=True`), so they are the exact ones, with their pre-sample
initialisation. Rebuilding them with a hand-written filter would have created a
second source of truth for something that already exists — the same criterion as
with `fue.load()` (§3.1) and with `cast_us_py`.

It homologates with the binary **line by line** on m6: the three covariances
(EI.EU +0.358, EI.EA −0.314, EC.EA −0.408), the eight links with their peaks and
their (b, s), and the same order.

**A trap that cost a while:** a bare `-i` does **not** identify from the diagonal
model. It builds its own — in m6, 61 slots, 46 free, logL −1716.36, with a
diagonal Sigma — and reads the CCFs of *those* residuals. Compared against it the
port's figures did not add up and it looked like a bug; asking it for `-0 -i` with
the same constraints, they agree exactly. Comparing two different fits is the
easiest way to invent a bug.

**The proposal may come out cyclic, and in m6 it does.** Reading the CCFs pair by
pair does not impose acyclicity: the draft brings EP -> EC -> EA -> EP, and
**two** prunings are needed to make it estimable. The guided mode writes the
`.dag` all the same, with the cycle noted in the header, and `read_dag` rejects it
as long as it is still there. The library warns and does not prune: which link
falls is judgement, not arithmetic, and pruning silently invites estimating the
draft — which is exactly what the school's doctrine says not to do.

### Step 7 — diagnostics, forecasting and the CLI

`diagnose.py` ports the transfer's portmanteau (k >= 0, which includes the
contemporaneous lag: that is where omega_0 acts) and the exogeneity one (k < 0).
The subtlety is **which residuals it is fed**: with a contemporaneous transfer the
reduced-form ones are correlated by construction (Sigma_12 = omega_0*sigma2_X), so
the test must run on the **structural** ones, a = Phi(0)*a_reduced. Otherwise the
portmanteau condemns a correct model — it gave p = 0.0000 where the C reports
0.1966.

`forecast.py` ports the MA(infinity) weights, the three error variances (level,
variation, annual variation) and the level layer: `to_level` composes the future
deterministic component, the integration against delta(B) and the inverse Box–Cox,
reusing fue's `_build_xi`, `_nonsop_coefs` and `_inv_boxcox`.

That level layer carried **the port's last real defect**, and it is worth
recording because of its failure mode. `to_level` built xi from the deterministic
omegas **in the `.pre`** — the univariate seeds — instead of from the ones
**re-estimated jointly**: on the canonical case the two `omega_d1` move to
−0.040867 and −0.094588 once the transfer is fitted alongside them. With the seeds
the level forecast is silently *the univariate one*, which even matches the C's
own `-0` run, so it only shows up against a fit that actually has a transfer in
it. What located it was **instrumenting the binary** (a temporary `fprintf` in
`transfer_forecast`): the C's `f1` turned out to be identical to the port's, which
left the level layer as the only suspect.

`cli.py` keeps the C's own option letters, `getopt` string included, and rewrites
`-estwin` to `-R` before parsing exactly as the C does — a command line written for
the binary runs here unchanged. What is not ported (`-a`, `-estwin`/`-R`, `-C`,
`-L`) is refused with exit code 2, not ignored. The executable is `drtran-py`, not
`drtran`: that name belongs to the C binary on a machine with both, and shadowing
it silently is how a battery starts comparing a program against itself.

## 4. Porting decisions that are not translation

| decision | why |
|---|---|
| Not porting the `.pre` reader | a single source of truth for the contract (§3.1) |
| **Concentrated** likelihood | it is what `drvmlest.c:est` does; with an absolute Sigma it gives −7802 |
| Seeding the variance ratios | scales differ x1098; starting at 1 leaves the start at −1371 |
| **BJR** sign convention: omega(B) = omega_0 − omega_1 B − … | the leading term adds, the rest **subtract**; it is the C cast's |
| `normalize_phi0` (Phi_0^-1 on the left) | a contemporaneous transfer puts omega_0 at lag zero and `elf` requires Phi(0) = I |
| **mu is the mean, not an intercept** | coherence with fue; see §5.2 |
| Embedded by default | as in the C; it does not truncate the sample |
| Structural residuals for the diagnostics | the reduced-form ones are correlated by construction with b=0 |

## 5. What the port found in the C

The port turned out to be a test bench: three discrepancies, and in two of them
the one at fault was the original.

### 5.1. The impulse response inverted the sign (a C bug)

The report at `drtran.c:1371` added the non-leading numerator terms where the cast
subtracts. It published a gain of 0.005610 (= omega_0+omega_1) where the real one
is omega_0−omega_1 = **0.027195**: a factor of almost 5, for every s > 0. **The
documentation had the same error**, so code and document confirmed each other.
Both fixed in the C repo, plus a new battery section that pins the sign.

### 5.2. The embedded cast's mean (a C bug)

The C did, in topological order, `mu_i += (SUM_k omega_k/delta(1))*mu_inp`. The
suspicion noted in its TODO was a **sign**; the defect was the **whole
parametrisation**.

mu is THE MEAN of the series, not an intercept. Box–Jenkins writes the model in
**deviations**,

```
(w_Y - mu_Y) = nu(B)*(w_X - mu_X) + N_t     =>     E[w_Y] = mu_Y
```

so the output's mean **inherits nothing** from the input's. Multiplying by
delta(B),

```
phi_Y*delta*(w_Y - mu_Y) - phi_Y*omega*B^b*(w_X - mu_X) = delta*theta_Y*a_Y
```

which is exactly row 1 of Phi(B)(w − mu) = Theta(B)a with mu = (mu_Y, mu_X).
**There is no term to add.**

The alternative (`w_Y = c + nu(B)*w_X + N`) is the **intercept** parametrisation.
They are the same family reparametrised **as long as mu_Y is free** — verified,
the same optimum to 1e-12. They diverge when mu_Y is **fixed**: in deviations
mu_Y = 0 means E[w_Y] = 0; with an intercept, E[w_Y] = nu(1)*mu_X != 0. Coherence
with fue rules: if fue fixed the mean at zero it is because the series has no
drift.

**Why it survived so long:** the canonical case has WTI as its input with mu = 0,
and there the term is zero under either convention. **An input with a free mean**
is needed to see it. Fixed on both sides (`embed.py:cast_embedded` and
`tran_shootx.c:build_embedded_varma`) and documented in the C's technical note
(the observation *The means are means, not intercepts*).

### 5.3. The modified Cholesky (a drvarma porting bug)

The embedded cast was blocked: `elf` rejected it with `ifault = 3`. The cause was
not in drtran: drvarma's `_chol_lower` used `np.linalg.cholesky` (**the strict
one**) where the C uses the **MODIFIED** Cholesky (`nlatools.c:choldcp`), which
accepts semidefinite matrices. Since the embedded cast produces a singular Phi_p
**by construction**, the strict one knocked it down.

Fixed in drvarma (`fix(as311): faithfully port the C's MODIFIED Cholesky`), and it
closed along the way the three C-parity tests its suite had been carrying.

### 5.4. `compimp` degraded to `pulse` (a fue Python bug) — FIXED

While trying to reproduce m6's targets a discrepancy of 1.9 appeared on the
diagonal rung, and it was not drtran's. Bisection put it where it was:

1. m6's diagonal joint fit did not match the C even when evaluated at its optimum;
2. with a strictly diagonal Sigma it still did not => it was neither the
   covariance nor the network;
3. series by series, evaluating **at fue C's optimum**, five of the six matched to
   5e-8 and only **EI** differed: −292.495 against −290.613.

EI is the only one of the six with a **`compimp`** deterministic, the
*compensated* impulse: +1 at the date and **−1 at the next**
(`fue_pre_reader.c:194`, `fue.c:317`). fue Python's reader mapped it to a plain
`pulse` (`fue/inp.py:276`), and swallowed the −1.

Confirmed with a known answer: rebuilding the compensated regressor (+1, −1) by
hand on the same `.pre` with the same coefficients, fue Python gives
**−290.613205**, exactly fue C.

**Fixed in fue 0.1.9** (BUG-0006), together with two other gaps the review of the
nine deterministics uncovered: `easter` and `trend` did not exist in the port,
and — this one in fue C itself, BUG-0007 — its `.pre` writer lost them, so that
**fue C could not reread its own `.pre`**. The vocabulary was unified along the
way: `impulse` is the canonical name, because fue C **does not reject** a word it
does not know, it takes it for a non-standard variable and silently estimates
something else.

With that, m6's canonical targets are reproduced: **diagonal −1709.511575**
(diff 5.0e-07) and **free network −1697.613401** (diff 5.9e-07). The validation on
the five clean series is kept: it exercises the same machinery without depending
on which version of fue is installed.

## 6. Homologation with the binary

`test_homologation_c.py` and `test_network.py` **relaunch the binary live**
instead of comparing against stored references, so as not to carry stale figures
when the C changes.

Canonical case `ES_CPI_m10` <- `WTI_ar1` (input with mu = 0):

| (b,r,s) | subtracting cast | embedded cast |
|---|---|---|
| (0,0,0) | −736.774158 | −736.774158 |
| (0,1,0) | −721.720197 | −721.801539 |
| (0,0,1) | −718.183933 | −718.287406 |
| (1,1,1) | −756.528944 | −756.602851 |
| diagonal | −767.424341 | −767.424341 |

The case that discriminates the means convention, `ES_CPI_m10` <-
`DE_CPI_mar3sar` (**both with a free mean**), embedded cast:

| | drtran C | drtran Python |
|---|---|---|
| (0,0,0) | 24.408974 | 24.408974 |
| (0,0,1) | 35.487981 | 35.487981 |
| (0,1,1) | 35.555382 | 35.555382 |

And the whole chain on that same case's diagonal: fue C (−7.3917271 + 11.2056885)
= **3.8139613** = fue Python = drtran C (3.813961) = drtran Python (3.8139611).

**THE NETWORK**, on five m6 series (EP, EU, EC, EA, P) with the DAG EP <- EU,
EP <- EC, EU <- EC (the chain EC -> EU -> EP, one input with two outputs and a
denominator with r=1) and two free covariances:

| | drtran C | port | diff |
|---|---|---|---|
| free network (40 free / 48 slots) | −1434.696068 | −1434.696068 | 1.9e-10 |
| + product + linear comb. (38 free) | −1439.505804 | −1439.505804 | 9.4e-08 |
| 3-series network, **optimised** (24 free) | −912.244333 (181 it) | −912.244333 (180 it) | 9.0e-08 |

The first two rows evaluate **at the C's optimum** and test the *cast*; the third
starts on the diagonal rung and tests the *search*. They are different things and
worth not confusing: a correct cast with an optimizer that does not arrive, and an
optimizer that arrives on a crooked cast, fail in very different ways.

**THE FORECAST**, canonical case with `-b 0 -r 0 -s 1 -V -f 6`, both series, in
original units and against the C's own table:

| | drtran C | port |
|---|---|---|
| ES_CPI 1–6/2020 | 82.01 82.02 82.38 83.17 83.33 83.44 | the same |
| s.e. | 0.24 0.44 0.60 0.74 0.85 0.95 | the same |
| WTI 1–6/2020 | 60.76 … 61.14, s.e. 8.29 … 27.10 | the same |

## 7. How to reproduce it

```sh
# the port
cd ~/Dropbox/SRC/drtran-python && python -m pytest -q          # 111 passed, ~3 min

# the original
cd ~/Dropbox/SRC/drtran && make && ./test_battery.sh           # 296 PASS, 0 FAIL

# a one-off comparison
./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/DE_CPI_mar3sar.pre \
             -b 0 -r 0 -s 1 -V -o /tmp/t.out | grep '^Log-likelihood'
```

### Traps when comparing with the binary

- **Do not extract the likelihood with `grep -oE '[0-9]+\.[0-9]+' | tail -1`.**
  That captures the **exogeneity p-value** from the diagnostics block, not the
  logL. A false positive of a C <-> Python divergence came from exactly there: it
  looked like 0.1630 against 35.487981 when both gave 35.487981. Filter by
  `^Log-likelihood`.
- `-V` is the **embedded** cast and `-S` the **subtracting** one; comparing one
  with the other measures the truncation, not an error.
- Series with mu = 0 **do not discriminate** the means convention (§5.2).
- The C's forecast table publishes two decimals; comparing strings makes 83.44
  fail against 83.4398 for no reason but the rounding. Compare numbers.

## 8. What is missing

- **The write round trip**: `estimate -> .pre -> reread` (§3.1).
- **Standard errors.** The Hessian is not computed, so the parameter table has no
  `s.e.`/`t` column. The C does give it. It is what is left for the report to be
  comparable line by line.
- What the C prints and the CLI does not yet: the forecast-error variance
  decomposition, the monthly and annual variation columns, and the estimated
  nu(k) weights with their standard errors.
- The C-only options: aggregates (`-a`), fixed-window estimation
  (`-estwin`/`-R`), the rolling out-of-sample errors (`-C`) and the LaTeX report
  (`-L`).

### Inherited from the C — to watch

- **The optimizer degrades with `refactor = 1`.** In the C it hangs for over 2
  minutes without converging at Delta-log ~0.002, and converges in 23 iterations
  with `refactor = 100`. `cdgrad`'s finite-difference step (~6e-6 absolute) has a
  terrible signal-to-step ratio at raw scale. The port inherits `_qnewt` from
  drvarma, so it probably inherits the fragility. `check_scale()` already warns;
  whether to also condition internally is still to be decided.
- **`termcode 3` is NOT a failure here.** Starting from the `.pre`'s seeds (which
  on the diagonal already ARE the optimum) the line search cannot improve and
  stops. The correct classification: 1–2 convergence, **3 stopped without
  improvement**, 4–5 a real failure. The right test is to perturb the
  pre-estimates and check that it converges by gradient to the same point. NB:
  multiart (drvarma) **does** reject termcode 3 in its order search, where the
  seeds are OLS. The two criteria coexist; do not confuse them.
