# Defects found in use

Found while building climate → wheat-price transfer functions on annual data
(project *Joseph's Cycles*, 2026-08-06/07): five markets, samples of 45–81
observations, output = ARIMA(2,1,1) on log prices, input = a rainfall series in
levels. That workload is not the one the port was homologated on — the canonical
case is monthly, `refactor=100`, AR(1) — and it is what surfaced these.

Each entry says what it is, how it was found, and what it costs. The ones with a
number are defects; the rest are things to watch.

**Reviewed 2026-08-07**, with each claim re-verified rather than taken on
trust, and each defect assigned to a package — `drtran` (the library and the C
it ports) or `mtram` (the MCP layer). Verdicts are recorded in each entry
below. Two were confirmed as real, one is real but minor, one was closed by
policy, one has since been fixed a different and better way, and the
"unconfirmed" one turned out not to be a computation defect at all.

Reproductions are self-contained (they use the `.pre` files in the repo root):

```
python3 scripts/repro_ar2_phi1_bound.py
python3 scripts/repro_alineacion_por_indice.py
```

---

## BUG-1. The unit-circle guard is applied to every AR order — FIXED in Python, OPEN in the C

> **Verdict 2026-08-07: CONFIRMED, and it is `drtran`'s (plus the C).** The
> decisive test is not that stationary AR(2)s are rejected — the report already
> showed that — but whether the guard is protecting against something. It is
> not. With the guard disabled, every stationary AR(2) with phi1 > 1 evaluates
> normally (`ifault=0`, finite logL, in BOTH casts), and a genuinely
> non-stationary point (phi1=2.10, phi2=-0.5.., |root|=0.78) is caught by `elf`
> **on its own**, returning `ifault=3`. So the engine already has the check it
> needs downstream; the 0.999 guard adds nothing for p=1 that is not already
> there, and for p>=2 it removes a legitimate region of the parameter space.
> Not a corner case: the excluded region is exactly the persistent cycles.
>
> **Fixed 2026-08-07 in the Python** (`cast.ar_is_stationary`, used by both
> casts). The guard is KEPT — rejecting before calling `elf` is cheaper than a
> likelihood evaluation and gives the line search a clean refusal — and only
> the region it rejects changed. It now does for the AR what `chekma` does for
> the MA three lines below it in the C.
>
> **It is wider than this report said: the guard was over-broad at p = 1 too,
> its own stated case.** With it disabled, every stationary AR(1) evaluates
> cleanly up to phi = 0.9999 (|root| = 1.0001, ifault 0); phi = 1 exactly
> returns ifault 2 and phi > 1 returns ifault 3. `elf` already rejects
> precisely the right set by itself, so 0.999 was discarding a live strip of
> stationary space *below* the true boundary. The corrected guard sits at the
> mathematical boundary, which is where `elf` sits.
>
> **Validated three ways, none of them numpy's word for it:**
>
> * the ANALYTIC modulus — for an AR(2) with complex roots the product of roots
>   is −1/phi2, so |z| = 1/sqrt(|phi2|). Agreement to 4e−16.
> * **Schur's stationarity triangle** (|phi2| < 1, phi2 + phi1 < 1,
>   phi2 − phi1 < 1) over ~3000 grid points, boundary excluded: identical
>   verdict at every one.
> * the ORACLE's convention. TASTE — no shared ancestor with this code — has a
>   roots library in `ROOT.PAS`: `zroots`/`laguer` (Laguerre) and `FACPOL`,
>   which factorises an operator and reports each root's real part, imaginary
>   part and MODULUS. It builds the polynomial as `a[1] = 1`, `a[i+1] = -c[i]`,
>   i.e. `1 - c_1 z - ... - c_p z^p` — exactly the convention used here, and the
>   same one `art._shrink_stationary` builds. Three independent
>   implementations, one convention.
>
> The full battery passes unchanged (339): nothing homologated against the C
> moved, because the canonical cases never sat near the barrier.
>
> **What remains is a decision, not a task.** The same one-line correction
> belongs in `tran_shootx.c:629`, and the machinery is already in that block —
> `chekma` builds the companion matrix and reads eigenvalue moduli, generic in
> the operator, and the C already reuses it for the delta(B) stability guard.
> Calling it on `phi` with `p` is the whole fix. It is Mauricio's source, so it
> is not ours to change unilaterally.

**Symptom.** `estimate` and `identify_network` return `ifault=1` ("the likelihood
could not be evaluated") with no further diagnosis, for a model fue estimates
without complaint.

**Cause.** `cast.py:286` and `embed.py:224` carry, identically:

```python
if ps[i] >= 1 and abs(phis[i][0]) >= 0.999:
    return None, None, None, None, None, 1
```

and so does the original, `tran_shootx.c:629`:

```c
if (p_ord[i] >= 1 && fabs(phi[i][1]) >= 0.999) { *ifaultx = 1; goto cleanup; }
```

The comment states the intent — *an AR(1) pinned to the unit circle* — and for
p = 1 the test is right: there `phi[0]` **is** the root. But the condition is
`ps[i] >= 1`, i.e. every order. In an AR(2), `phi[0]` is not a root: the
stationary region is the triangle |phi2| < 1, phi2 + phi1 < 1, phi2 − phi1 < 1,
in which **phi1 reaches 2**. Every AR(2) with complex roots and phi1 > 1 is
stationary and is rejected.

**Evidence** (`scripts/repro_ar2_phi1_bound.py`, both casts):

| phi1 | phi2 | \|roots in B\| | stationary | ifault diag | ifault emb |
|---|---|---|---|---|---|
| 0.3000 | 0.0000 | 3.333 | yes | 0 | 0 |
| 0.9500 | −0.5200 | 1.387 | yes | 0 | 0 |
| 0.9900 | −0.5200 | 1.387 | yes | 0 | 0 |
| 1.0020 | −0.5425 | 1.358 | yes | **1** | **1** |
| 1.0354 | −0.5184 | 1.389 | yes | **1** | **1** |
| 1.6000 | −0.8000 | 1.118 | yes | **1** | **1** |

**Why it is worse than an `ifault`.** When the starting point sits *below* 0.999
nothing fails. `estimate.py` returns 1.0 ("does not improve") at every rejected
point, so the line search never crosses the barrier and the optimiser **pins**
against it. On a real case — London wheat prices, 1816–1896, n=81 — seeding
phi1 = 0.90 gives

```
phi_1[B^1]   0.998998   s.e. 1e-06   t = 1.04e+06
```

against a true maximum of 1.0354 (fue, univariate). A publishable-looking
estimate, pressed against an invisible wall, whose only tell is an absurd
t-statistic. The distortion is not random: it excludes exactly the **persistent
cycles**, which is what this class of study is about. Two of five markets in that
study (London era B, period 8.2 y; Strasbourg era B with the 1847 intervention,
7.6 y) could not be tested at all.

**Note the asymmetry.** Three lines below, in the same block, the MA *is* checked
properly, by roots, with `chekma` — which builds the companion matrix and looks
at eigenvalue moduli, and is generic in the operator. The right machinery is
already next door.

**Fix.** Check AR stationarity by the **roots** of the polynomial (the same
companion-matrix route as `chekma`, or a reflection-coefficient parametrisation,
Monahan 1984), and keep the `phi[0]` guard only when `ps[i] == 1`. It should be
fixed in the C as well.

---

## BUG-2. The series are paired by INDEX, not by date, silently

> **Verdict 2026-08-07: CONFIRMED, and it belongs to BOTH.** The dates are
> available — `spec.ts.start`, `.freq`, `.nobs` are all read from the `.pre`
> and sit unused. `cast.py:252` states the premise in a comment ("the last
> observation is the same date") and never checks it: that half is `drtran`'s,
> and a library may be mute about advice without being mute about accepting
> incompatible input. `mcp_server.load_pre` is the declared GATE and prints a
> summary with no dates in it: that half is `mtram`'s. The two fixes differ —
> the library should refuse, the gate should show the window.
>
> `tests/test_end_to_end_passthrough.py` now checks the premise holds on the
> pair that the real art -> fue -> mtram pipeline produces.

**Symptom.** None. That is the problem: two series that do not share a single
period can be crossed and the fit goes through without a word.

**Cause.** Each `.pre` declares its start date and fue reads it (`spec.ts.start`),
but nothing compares it across series:

* `pre.load_pre` reads each file on its own and never looks at another's `start`;
* `mcp_server.load_pre` walks the list checking only that the file exists and
  calling `check_scale`. It compares neither `start`, nor `freq`, nor even
  `nobs`, and its summary prints `(N obs, freq F)` — **the date appears nowhere**;
* `cast.py:252` aligns "at the END (the last observation is the same date)" and
  trims to the shortest. That comment describes the case it was written for —
  different d/D over the *same* window — but nothing checks the premise when the
  windows are different stretches of calendar.

**Evidence** (`scripts/repro_alineacion_por_indice.py`). The same pair is
identified twice, changing only the declared start date of the input:

```
--- input declared at (2002, 1)   (output at (2002, 1); years in common: yes)
    band  : 0.136399  => n = (2/band)^2 = 215
    r(0..4): [0.492301  0.309836  0.024815  0.027188 -0.107454]
    (b,r,s): (0, 0, 1)      Q exog: 18.296880 (p=0.788374)

--- input declared at (2052, 1)   (output at (2002, 1); years in common: NONE)
    band  : 0.136399  => n = (2/band)^2 = 215
    r(0..4): [0.492301  0.309836  0.024815  0.027188 -0.107454]
    (b,r,s): (0, 0, 1)      Q exog: 18.296880 (p=0.788374)
```

Identical to the last decimal, with **no overlapping period at all** in the
second: the date takes no part in the computation.

**How it showed up in real use.** Loading a price series for 1700–1896 (197 obs)
against rainfall for 1766–2024 (259 obs), the identification reported a band of
0.1429 = 2/sqrt(196) — so it had used 196 pairs, when the real overlap
(1766–1896) is 131 observations. It was pairing the 1700 price with the 1766
rainfall, 66 years out, and proposed `b=18` in earnest. The only thing that gave
it away was that 2/sqrt(n) did not match the overlap.

**Fix.** Intersect by date when the case is built — or, at a minimum, refuse the
load when `start`/`freq`/`nobs` are not compatible. And print the date range in
`mcp_server.load_pre`'s summary, which today gives only a count.

---

## BUG-3. `write_pre` wrote files that are not `.pre` files (mtram's) — FIXED

**Found 2026-08-07; fixed the same day.** `write_pre` is now `write_inp`, at
all three levels — `drtran.pre.write_inp`, the CLI's `-W`, and mtram's tool —
and `next_pre_path` is `next_inp_path`, emitting `NAME.1.inp`. Nothing about
the CONTENT changed: the file was always a legitimate starting point, and it is
the extension that was making a claim the content could not support.
`tests/test_write_inp.py` pins the invariant in the direction that matters —
it asserts the written file DOES move when fue re-estimates it, so that if it
ever stops moving someone has to come back and ask why.

Found while checking a claim of mine that turned out to be
backwards. I had recorded the `.inp -> .pre` step as a "gap in the ladder"
because no MCP tool performs it. It is not a gap. The extensions carry the
division of labour, and each is a different claim:

| file | claim |
|---|---|
| `.inp` | a SPECIFICATION. art writes one, every parameter at `0.000000` |
| `.out` | the full record of an estimation and its diagnosis |
| `.pre` | the same `.inp` with the estimates as new initial values — **an optimum, in re-runnable form** |

That last one is what makes the ladder climbable: each rung's output is the
next rung's input *in the same format*, so a model can be edited and re-run
toward a better one. An MCP must not write a `.pre`, because a `.pre` asserts
that an estimation ran and converged here, and the file carries no mark of who
wrote it — a fabricated one is indistinguishable downstream from a genuine one.

**The invariant is testable.** Run fue on a `.pre` and the numbers must not
move.

**Symptom (before the fix).** `write_pre` emitted `<name>.1.pre`, and those
files fail the invariant:

| file | max abs change after re-running fue |
|---|---|
| `.pre` written by fue | **0.000000** |
| the same block written by drtran | **13.109261** |

**Cause.** What it writes is the univariate block as re-estimated BESIDE a
transfer. Those values are optimal for the joint model and, necessarily,
not for the univariate one. The docstring says so plainly — *"NOT a better
starting point — they are optimal WITH the transfer, so on the diagonal they
evaluate worse"* — but that honesty lives in the tool's reply, not in the
file. Once the file exists the warning is gone and the extension speaks for it.

**Why nothing catches it.** The diagonal gate prints ✅ on these files, with
the same likelihoods as the genuine ones (−1744.135582 in both). That is not a
flaw in the gate: `load_pre` re-estimates each series with fue on the way in,
so the non-optimal values are silently replaced by the univariate optimum
before anything is compared. The gate tests the CROSSING, exactly as
documented.

Two consequences follow. Nothing in the suite distinguishes a real `.pre` from
a file that merely has the extension. And the round trip loses the very thing
`write_pre` existed to carry: reload its output into mtram and the jointly
re-estimated values are discarded. (Still true of `write_inp`, and now
harmless — re-estimating a specification is the right thing to do with one.)

**Fix, applied.** Write `.inp`. What is held is a STARTING POINT, not an
optimum, and the format for a starting point is `.inp`. Still open as design:
if mtram later needs to persist a MULTIVARIATE case — the network, the links,
the constraints — that too should be its own `.inp`-analogue. The `.pre` is the
contract between univariate optima and must not carry multivariate content.

### The rule this follows from, and what it settles

**A `.pre` is immutable: touch it and it becomes an `.inp` again.** Editing a
specification unmakes the claim that these values are its optimum, so the file
drops back to being a proposal. The ladder is then a fixed-point iteration,
and each extension marks which half of the cycle you are in:

```
.inp  ──(fue estimates)──▶  .pre  ──(analyst edits)──▶  .inp  ──▶ …
```

Two consequences, both measured on the passthrough pipeline:

**drtran should accept `.inp`, and already does.** Same format, and
`mcp_server.load_pre` re-estimates each series with fue on the way in, so the
stored values are seeds and nothing more. Fed art's `.inp` with every parameter
at zero, the gate reaches the same likelihoods as with fue's `.pre`
(−1744.135582 both ways) and closes the same way. This also sharpens the
ladder's contract: mtram needs a SPECIFICATION and estimates the univariate
optima itself — the `.pre` is welcome for its seeds, not required. An edited
`.pre`, which by the rule is an `.inp`, can be handed straight back.

**drtran should NOT produce univariate `.pre` files.** Not because it cannot:
off a DIAGONAL fit it legitimately could, since the likelihood factorises and
the joint blocks are the univariate optima exactly — measured, max|difference|
against fue's own file is 0.00000000. But that file already exists and is
identical, so writing it adds no information and a second place for the two to
drift. Off a fit WITH a transfer the blocks are optimal for the joint model and
therefore not for the univariate one, which is BUG-3 itself. Between redundant
and false there is no third case where it would earn its keep.

---

## To watch

### `check_scale`'s advice is right for lambda=0 and wrong for lambda=1

> **Verdict 2026-08-07: real, minor, `drtran`'s** (`pre.py:91`). Confirmed:
> the rule is `refactor < 10` and the message is unconditional, with no
> reference to `boxlam` anywhere in the function.

The rule is `refactor < 10` and the message always says *regenerate the .pre with
refactor=100*. For a log model with d=1 that is correct and it does solve the
problem: it puts the series in percent. For an **untransformed** series in levels
it is the wrong direction — rainfall at ~900 mm becomes ~90000. What such a
series needs is to be **divided**. The message should be conditioned on
`boxlam` and on the magnitude of the series.

Also: the consequence is worse than "the optimizer degrades" (the TODO's wording,
inherited from the C). At `refactor=1` the two casts can reach **contradictory
scientific conclusions** on the same data. On one of the study's links:

| | omega | t | theta |
|---|---|---|---|
| `embed=True`, refactor=1 | 0.001423 | 1.26 | 0.500 |
| `embed=False`, refactor=1 | 0.000070 | **3.87** | **1.000** (boundary) |
| either one, refactor=100 | — | **1.26** | 0.500 |

Both of the first two reported `CONVERGED (step)`. Fixing the scale makes both
converge by gradient and agree. Anyone who reads only the second line publishes a
significant effect that does not exist.

See `scripts/repro_refactor1_relgrad.py` and the TODO entry *The optimizer
degrades with `refactor=1`*.

### `CONVERGED (step)` is a generous label for termcode 2

> **Verdict 2026-08-07: will not be changed, and this is policy rather than
> disagreement.** The optimiser and its announcements are Mauricio's published,
> refereed work; the criteria and the wording stay as he wrote them unless
> something demonstrably better has been tested. See
> `OPTIMIZER_STOPPING_STUDY.md`, whose whole point was that the alternatives
> tried were not better.

`termcode 2` is the step-tolerance stop: the step collapsed while the gradient may
still be appreciable. The accompanying warning says exactly that and says it well,
but the status line leads with the word CONVERGED, which is an invitation to move
on. `STOPPED (steptol)` would read closer to the truth, and returning the gradient
norm would let the analyst judge instead of guess.

### `identify_link` hands over economically impossible lags with no joint test

> **Verdict 2026-08-07: addressed, differently and better, and it is
> `mtram`'s.** The entry asked for the k>=0 portmanteau to be reported at
> identification time. What shipped instead is the STOPPING RULE
> (`_before_you_choose`): the peak-to-band ratio, which was measured to
> separate cleanly — 1.0-1.5 on pure noise against 7.6-7.8 on a real transfer,
> no grey zone — where the portmanteau does not. On noise mtram now refuses to
> propose an order at all. Verified in `tests/test_bank_mtram.py`.

It takes "each significant weight as a free omega" from the contiguous block, so
an isolated noise spike at k=7, k=14 or k=18 becomes a formal proposal. With
n≈46 and ~40 lags scanned, two crossings are exactly what chance delivers. The
tool already computes the k>=0 portmanteau in `diagnose`; reporting it in
`identify_link` too would let the analyst see *there is no joint evidence of a
transfer* **before** being handed a `b=7` that looks like a finding.

### Unconfirmed: `plot_ccf`'s Q does not match `identify_link`'s

> **Verdict 2026-08-07: NOT a computation defect. The suspicion pointed at
> something real and diagnosed it wrongly.** They are two different statistics
> that share a letter. The figure's label is drvarma's `qccf` — **Hosking's
> bivariate portmanteau on the stacked 2-variate series**, which aggregates all
> four entries of the cross-correlation matrix, the two autocorrelations
> included, at every lag. `identify_link`'s is `chi_test` over ONE side of ONE
> cross-correlation. 102.8 against 24.5 is the expected ratio, not a
> discrepancy.
>
> What IS a defect, and a small `mtram` one: both are printed as `Q(20) = ...`
> with nothing saying which is which. Same class of problem as the residual
> panel's degrees-of-freedom correction — one label, two meanings.

The figure was labelled `Q(20) = 102.8` while `identify_link` reported
`Q(20) = 24.5` for k < 0 on the same link; from the plotted r(k) neither side
comes near 102.8 (roughly 28 for k >= 0). Either it is a different statistic or
the label is wrong. Not chased down — recorded as a suspicion, not a diagnosis.
