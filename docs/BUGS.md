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
python3 scripts/repro_identify_link_output_schema.py
python3 scripts/repro_refactor1_relgrad.py
python3 scripts/repro_ccf_stop_grey_zone.py       # BUG-6; synthetic, needs no .pre
```

**BUG-4 was added 2026-08-08** from a different workload — monthly IPC_ES → WTI
passthrough, the canonical shape — and is an `mtram` defect, not a numerical
one: it blocked node N1 whenever the CCF was successfully drawn. Fixed the same
day, with the guard the suite was missing.

---

## BUG-1. The unit-circle guard is applied to every AR order — FIXED in both

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
>
> ### Applied to the C (2026-08-07, `drtran` a4e5051) — and verifying it
> ### corrected two things this entry got wrong
>
> The fix itself went in as expected: `chekma(m, p, armax->phi, ...)` in place
> of the scalar test, with the work vectors made dynamic (`chekma` indexes
> 1..m*p, and `wr[4*MAX_SER]` = `wr[32]` is short as soon as m·max(p,q) > 32 —
> reachable with a seasonal model and few series. That bound already applied to
> the MA call; leaving it and adding a second consumer would have been worse).
>
> **1. In the DEFAULT path that guard never ran.** The embedded cast
> (`embed_varma`, the default since `ad8eac3`) returns at line 543, before
> block 12. Its own check, lines 521-528, calls `chekma` on the MA only: in the
> default path the AR was not checked at all. The broken guard lived only in
> the `-S` (subtraction) path.
>
> **2. So the damage described above was NOT inherited from the C — it is the
> PORT's.** `drtran-python` copied the guard into BOTH of its casts, the
> embedded one included, where the C never had it. The pinning at 0.998998 with
> t = 1.04e+06 was measured in the Python, and that is where it lived.
>
> This entry's original heading — "a C bug, inherited" — was therefore wrong in
> both halves: the bug was reachable in the C only under `-S`, and the port
> added its own.
>
> **Verification.** That the C fix does something, on the path that runs it: an
> AR(2) pinned at phi1 = 1.6, phi2 = −0.8 (|roots| = 1.118, stationary), with
> `-S` — before, `ifault = 1 (estimates not reliable)`, logL = 0.000000; after,
> logL = −832.259422. And re-homologation over 8 outputs × 3 (b, r, s) × both
> casts: **48 runs, byte-identical output**. The canonical cases never sat near
> the barrier, which is why nothing moves.

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

## BUG-4. `identify_link` declared `-> str` while returning a list — FIXED

Found 2026-08-08 on a different workload from the rest of this file: monthly
IPC_ES (INE, 2002-01…2019-12, n=216) as output, WTI as input — the canonical
passthrough shape, `refactor=100`, AR(1) both sides. It is not a numerical
defect; it is `mtram`'s, and it blocks node N1 outright.

```python
def identify_link(name: str, input_index: int = 1, band: str = "constant",
                  ident_pre: str = "") -> str:
    ...
    return _con_figura("\n".join(txt), grafico)
```

`_con_figura` returns the plain string only when there is **no** figure; with
one it returns `[TextContent(...), ImageContent(...)]` — which is the whole
point of the call, as the comment three lines above it says: *"EL GRÁFICO VA CON
LOS NÚMEROS, no en otra llamada"*. FastMCP builds the structured-output schema
from the return annotation, so the declared contract is

```
identify_link -> {'properties': {'result': {'type': 'string'}}, 'required': ['result'], ...}
```

and the list is rejected before it reaches the caller:

```
Error executing tool identify_link: 1 validation error for identify_linkOutput
result
  Input should be a valid string [type=string_type, input_value=[TextContent(...)]]
```

**The failure is conditional in the worst possible direction: the tool works
only when `plot_ccf` raises**, because that is the branch that returns a string.
When everything goes right, the call fails. And N1 — the (b, r, s) decision — is
precisely the node the analyst is supposed to take by *looking* at the CCF, so
in practice no transfer model can be identified through mtram at all.

The two sibling tools already show the shape of the answer: `plot_ccf` and
`plot_impulse_response` are annotated `-> list` and get `outputSchema: None`,
i.e. no structured validation, which is why the same content passes through them
untouched. `identify_link` is the only tool in the module that calls
`_con_figura`, and the only one of the three that declares a schema.

> **FIXED 2026-08-08, same day, and it was mine.** The annotation was missed
> when `identify_link` was given its figure: the five `plot_*` tools were
> retyped `-> list` and this one was not, because it was not in that list.
>
> **The lesson is the guard, not the one-word fix.** The full battery — 349
> tests — passed while the tool was unusable, because every test calls the
> FUNCTION and the function was correct. Nothing exercised the registered tool,
> which is the only place the schema exists. Two tests now do:
> `test_every_tool_that_can_return_content_blocks_is_annotated_list` reads the
> annotations, and `test_the_tools_survive_a_round_trip_through_the_mcp_layer`
> goes through `mcp.call_tool`. Both were checked by reverting the annotation
> and confirming they FAIL — a guard that does not fail when the defect returns
> is decoration.

**Fix: annotate `-> list`.** One word, at `mcp_server.py:843`. Verified locally:
with the annotation changed, `outputSchema` becomes `None` and the call returns
normally, both with the figure and (PART 3) without it. The change was reverted
after checking — it is not applied in this tree.

Worth doing alongside: a guard so that a return-annotation of `str` on any tool
that can reach `_con_figura` is caught by the test suite rather than by an
analyst mid-identification. The defect is invisible to unit tests that call the
function directly, since the function itself behaves correctly — only the
registered tool's schema is wrong.

```
python3 scripts/repro_identify_link_output_schema.py
```

Self-contained: it uses the repo-root `ES_CPI_m10.1.pre` and `WTI_ar1.1.pre`,
prints the declared schemas of the three tools, the actual return type, the
failure through `mcp.call_tool`, and the inverted-branch check.

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
>
> **Superseded 2026-08-08 — the cure overshot. See BUG-6.** The "no grey zone"
> claim came from measuring only two points, a null and one large effect; the
> middle was never simulated. It is populated, and the rule discards transfers
> that exact ML finds at |t| > 4. The original complaint here was real and the
> fix was the right shape — a gate before the order is read — but its threshold
> is set in units of the 2-sigma band, so it demands 4 sigma. Both entries
> should move together: report the portmanteau AND lower the cut.

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

---

## BUG-5. The gate compared DIFFERENT SAMPLES when the series differ in D — FIXED

Found 2026-08-08 on the monthly IPC → WTI passthrough, extending the ES case to
the other seven series of `IPC.xlsx` (2002-01…2019-12, n=216, freq=12,
`refactor=100`, λ=0). Three of them identify as SARIMA with `D=1`; the other
four stay at `D=0`. **`load_pre`'s diagonal gate passes for every D=0 series
and fails for every D=1 series, by the same amount.**

```
                sum of univariates   drtran diagonal      gap
IPC_FR             -723.131413        -681.788527     +41.342886
IPC_DE             -766.909798        -725.566912     +41.342886
EMU                -733.373133        -692.030246     +41.342887

CPI_USA            -779.700096        -779.700096      -7.44e-08
IPC_UK             -668.822105        -668.822106      -8.39e-07
IPC_CA             -801.308205        -801.308205      -2.56e-07
IPC_JP             -724.268981        -724.268981      -1.94e-07
IPC_ES             -767.435263        -767.435263      -1.75e-07
```

With a diagonal structure the exact likelihood FACTORISES, so the joint fit must
equal the sum of the separate ones. It does, to 1e-7, in all five D=0 cases —
and misses by 41.34 in all three D=1 cases. `mtram` correctly refuses to
continue ("no sigas"), so the defect **blocks** those three transfer models.

> **FIXED 2026-08-08, and the engine was never wrong.** The missing term is not
> a constant in the likelihood: **the two sides were scoring different data.**
>
> `cast.py:252` aligns the series at the END and trims to the shortest, which is
> correct for a joint fit — you cannot use observations of an input that have no
> counterpart in the output, and the oracle does the same (`TFEST.PAS:71` takes
> a single `nob` from the output and applies each model's differencing to it).
> But `fue` estimates each univariate on its own MAXIMAL sample. So with the
> output at D=1 the joint fit scored 203 observations of BOTH series while the
> sum scored 203 of the output and 215 of the input.
>
> **The gap is exactly the likelihood of the 12 input observations the joint fit
> discards.** WTI scores −760.032614 on 215 observations and −718.689727 on 203:
> a difference of **41.342887** against the observed **41.342886**. And the
> identity closes to **0.000000000** when both sides use the common sample —
> the output scores −26.234052, WTI −718.689727, and the joint fit
> −744.923779 = their exact sum.
>
> That is also why the gap was CONSTANT across three series sharing nothing: it
> never depended on the output. It depended on the INPUT, which was WTI in every
> case, and on the same 12 observations. Everything the report ruled out —
> sigma, the parameters, the data range, the Box-Cox Jacobian — was ruled out
> correctly, and for the right reason.
>
> **Fix: in the gate, not the engine.** `_diagonal_gate` now estimates each
> univariate on the COMMON stationary sample (`_muestra_comun`,
> `_en_muestra_comun`) before summing. Measured: the D=1 case goes from +41.34
> to −8.31e−08, and the D=0 control is unchanged at −1.74e−07. No closed form
> was needed and none was guessed — which the report was right to warn against.

### Reproduction

Build any of the three `.pre` files with `art` and load it against the WTI
input:

```python
confirm_and_estimate(inp_path=<series>.inp, output_path=<series>_b2.inp,
                     lam=0, d=1, D=1, p=1, q=0, Q=1, estimate_mu=False)
load_pre(name=..., paths="<series>_b2.pre,WTI_m10.pre")   # gap +41.342886
```

The D=0 controls are the same series with `D=0, n_harmonics=5, p=1, q=0,
estimate_mu=True` — they cross at 0.

### Impact

Blocking for any output with stochastic seasonality, which on this evidence is
the majority of European CPIs (3 of the 4 euro-area series tested). The gate
itself is working exactly as designed — it caught this before a single transfer
coefficient was reported, which is what it is for.

### Regression to keep

The identity to assert is not "the gap is small" but **"both sides score the same
sample"**. On the common stationary sample the sum must reproduce the diagonal
fit exactly: output −26.234052 + WTI −718.689727 = −744.923779, and the joint
diagonal returns −744.923779. Measured after the fix: D=1 case −8.31e−08, D=0
control unchanged at −1.74e−07.

The trigger is **any difference in `(d, D)` between series**, not `D=1`
specifically. It did not fire on the five D=0 pairs here only because both series
carried `d=1, D=0` and their `w` lengths matched exactly — so a regression that
only exercises equal-differencing pairs will not see it. Cover a mixed pair.

---

## BUG-6. `identify_link`'s "the CCF is indistinguishable from noise" stop is calibrated in units of the 2-sigma BAND, so it discards genuine transfers at 3-4 sigma — mtram's

Found 2026-08-08 extending the IPC → WTI passthrough to eight countries. Two of
the five estimable links were STOPPED by this rule and both turned out to carry
a transfer that exact ML finds at better than 1e-4.

```
                peak/band   =>  sigma   identify_link      joint ML
IPC_UK <- WTI     1.85         3.70     "no propongo orden"  omega_0 = 0.005223
                                                             t = 4.02, LR p = 8.0e-05
IPC_JP <- WTI     1.44         2.88     "no propongo orden"  gain   = 0.009722
                                                             t = 4.36, LR p = 1.4e-04
```

Both models then PASSED `diagnose`'s adequacy test (UK p=0.092, JP p=0.745), so
the transfers the rule refused to identify are well specified.

### The defect

`mcp_server.py:735-741`:

```python
_pico = float(np.abs(_ccf[_lags >= 0]).max()) / float(idt.threshold)
...
if _pico < 2.0:
    parar = True
```

`idt.threshold` is the plotted band, **2/sqrt(N)** — already two standard
errors. So `_pico` is measured in units of 2 sigma, and the cut at 2.0 demands
the peak exceed **4 sigma** before an order may be read. The docstring's own
reference range makes the mis-scaling explicit:

| `_pico` | in sigma | two-sided p at a pre-specified lag | the comment calls it |
|---|---|---|---|
| 1.0 | 2.0 | 0.046 | noise |
| 1.5 | 3.0 | 0.0027 | noise |
| **1.85** | **3.70** | **0.00022** | noise (UK) |
| 2.0 | 4.0 | 6.3e-05 | the cut |
| 7.6 | 15.2 | ~0 | signal |

A peak at 3 sigma is not noise. The band exists to protect a SEARCH over ~25
lags, where a 2-sigma crossing is unremarkable — but the peak that fires this
rule sits at **k=0** in both cases, which is not a searched lag: it is where the
economics says to look, and where every other country in the batch put its
largest weight.

### Why the calibration missed it — MEASURED

The comment records exactly two measured points — `omega ~ 0` giving 1.0-1.5,
and a signal case giving 7.6-7.8 — and concludes "no hay zona gris". **The grey
zone was never simulated.** With only a null and one large effect there is
nothing between 1.5 and 7.6 to observe, so its emptiness is a property of the
design, not of the statistic.

`scripts/repro_ccf_stop_grey_zone.py` simulates the missing middle: one DGP
(`y = rho*x + noise`, both AR(1), N=215, 400 reps) swept across effect sizes.

```
   rho    peak/band   in sigma    |t| of omega    % STOPPED by the rule
   0.00       1.10       2.19           0.81              100.0
   0.05       1.11       2.21           1.02              100.0
   0.10       1.19       2.37           1.57               99.2
   0.15       1.30       2.60           2.18               97.2
   0.20       1.54       3.08           3.00               86.8   <-- real, stopped
   0.25       1.83       3.66           3.76               64.8   <-- real, stopped
   0.30       2.21       4.41           4.64               31.0   <-- real, stopped
   0.40       2.90       5.80           6.33                1.5
   0.60       4.37       8.74          10.93                0.0
   0.80       5.85      11.70          19.50                0.0
```

Two things to note. **The null row reproduces the original calibration** (1.10
against their 1.0-1.5), so the setups are comparable and the disagreement is
about the middle, not the method. And **the real cases land exactly where the
simulation puts them**: the UK's r(0)=0.2521 is the `rho=0.25` row — simulated
|t| = 3.76 against the 4.02 that exact ML actually returned, with the rule
stopping 65% of such draws.

The grey zone is not narrow. Between `rho` 0.20 and 0.30 the effect is
unambiguous to any test (|t| from 3.0 to 4.6, p < 1e-3) and the rule discards
between a third and seven eighths of the samples. The batch's own gains span a
factor of 6 (0.0052 to 0.0320), and the two smallest fell inside.

### Impact

False negatives on small-but-real transfers, delivered as a hard stop
(`parar = True`, "No propongo orden") rather than a warning. The analyst is told
the relationship cannot be seen; nothing suggests estimating anyway. In this
batch that would have discarded the two most interesting cases — the UK and
Japan are precisely where fuel taxation and the exchange rate predict a small
elasticity, which is the finding.

It also inverts the tool hierarchy. The CCF is an IDENTIFICATION device
computed from a prewhitening filter; the joint exact-ML estimate with its
Hessian standard error is the TEST. Letting the weaker instrument veto the
stronger one is backwards, and it is the opposite of the policy applied
elsewhere in this suite (`formal_tests` is explicitly told not to overrule
Shin-Fuller or the acf/pacf).

### Suggested fix

- **Re-express the criterion in sigma, not in bands**, and set the cut where it
  is meant to be. If the intent is "a 2-sigma peak among 25 lags is expected by
  chance", then apply a multiplicity correction to a SEARCHED maximum, and treat
  k=0 separately since it is pre-specified.
- **Downgrade the stop to a warning.** Say the CCF cannot pin down `b`, propose
  the contemporaneous form `b=0, r=0, s=0` as the falsifiable default, and let
  the joint estimate decide. That is what was done by hand here and it produced
  two adequate models.
- **Extend the calibration** across effect sizes, not just null vs large, before
  claiming there is no grey zone.

### Reproduction

`IPC.xlsx`, 2002-01…2019-12, n=216. Output `.pre` = harmonics + AR(1) + mu
(UK) or + step 04/2014 (JP); input = `WTI_m10.pre` (AR(1), no mean).

```python
identify_link(name=..., input_index=1)     # -> 1.85 / 1.44, "No propongo orden"
set_network(..., '[{"out":0,"inp":1,"b":0,"r":0,"s":0}]')   # UK
estimate(...)                               # -> t = 4.02 / 4.36
diagnose(...)                               # -> adequate, exogenous (UK)
```

Self-contained (no `.pre` files needed), and it is the one that settles the
calibration question rather than the anecdote:

```
python3 scripts/repro_ccf_stop_grey_zone.py     # exits 1 when the grey zone is found
```

### Related, and possibly the same root

For Japan `identify_link` printed "the input behaves as exogenous. OK" at
p=0.0596 with 3 significant lags at k<0, and after estimation `diagnose`
returned "*** FEEDBACK ... a one-input model is not valid" at p=0.0188 on the
same pair. The two are different statistics (pre-fit prewhitened vs post-fit
structural residuals), so they need not agree — but the user-facing VERDICT
flipped from OK to blocking, and the borderline pre-fit value was reported
without any hedge. Worth deciding whether the identification-stage exogeneity
check should warn in a band around its own critical value.

---

## BUG-7. `chi_test(first=1)` keys the Ljung-Box divisor to the POSITION in the sum, not to the lag, so the exogeneity portmanteau does not match the C — drtran's

Found 2026-08-08 while auditing `diagnose.py` after the eight-country
passthrough batch. Small in magnitude, but it is a fidelity break in the one
place the docstring makes a point of, and it errs in the direction that hides
feedback.

### The defect

`diagnose.py:40-55`:

```python
idx = np.arange(first, len(r))
div = np.array([n - i + 1 for i in range(1, len(idx) + 1)], float)
return float(n * (n + 2) * np.sum(r[idx] ** 2 / div)), len(idx)
```

`i` restarts at 1 regardless of `first`, so the divisor tracks the position in
the sum rather than the lag:

| branch | used by | lag k gets divisor |
|---|---|---|
| `first=0` | the TRANSFER test | `n - k`     ← matches the C |
| `first=1` | the EXOGENEITY test | `n - k + 1` ← one lag late |

The C (`drvarma_v.04/src/diagnose.c:278-285`) is 1-based with `corr[1]` the
CONTEMPORANEOUS lag, so `nobs - i + 1` is `nobs - k` for lag k, on **both**
sides:

```c
for (i = 1; i <= lags; i++)
    chisqr += corr[i] * corr[i] / (nobs - i + 1);
```

And the C never recomputes the k>0 sum — it SUBTRACTS the contemporaneous term
(`diagnose.c:430-431`):

```c
Q(k>0) = ChiTestC(corr, lags+1, n) - corr[1]*corr[1]*(n+2)
```

That is an identity the Python must satisfy and does not. The docstring right
above the code is emphatic — *"**The divisor is n−i+1, not n−i** — changing it
moves Q just enough to stop matching the C"* — and the implementation then
gets it right for `first=0` and wrong for `first=1`.

### Impact

`Q_exog` is understated by about `1/(n-k)` per lag: **−0.4988 %** at n=215,
nlags=24. No verdict in the eight-country batch flips (Japan fails exogeneity at
p=0.0188 either way), so nothing published here is affected.

What makes it worth fixing anyway is the **direction**: a smaller Q means a
larger p, so the error makes the input look MORE exogenous than it is. The
exogeneity test is the one that decides whether a one-input transfer model is
admissible at all — an error that hides feedback is the expensive sign. And
`diagnose`'s verdict is a hard stop that redirects the analyst to `sima`, so it
should not be biased towards not stopping.

### Reproduction

```
python3 scripts/repro_chitest_divisor_offbyone.py     # exits 1 while the bug is live
```

Self-contained, no `.pre` needed. It checks the C's own identity and then
isolates the divisor from the lag-0 term by zeroing `r[0]`, after which the two
branches sum literally the same quantities and must agree exactly:

```
first=0 = 20.378027493
first=1 = 20.276384689
gap     = -0.101642804   (-0.4988 %)
```

### Fix

Key the divisor to the lag rather than to the position:

```python
div = np.array([n - k for k in idx], float)
```

which reduces to the C for both values of `first`. The regression to keep is the
identity, not a tolerance: with `r[0] = 0`, `chi_test(first=0)` and
`chi_test(first=1)` must return the same number bit for bit.
