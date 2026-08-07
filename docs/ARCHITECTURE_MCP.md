# ATSW and its assistants — an architecture

> **The file conventions** — what `.inp`, `.out` and `.pre` each assert, why
> only the estimating program may write a `.pre`, and what the stage-to-stage
> flow guarantees — are studied with measurements in
> [`LADDER_AS_OPTIMISATION.md`](LADDER_AS_OPTIMISATION.md). Read it before
> changing anything that writes a file.

What each MCP server is for, how they relate, and which decisions this settles.
Written 2026-08-02, when drtran reached its first beta and the question became
unavoidable: **one multivariate assistant or two?**

Studied first: `art/mcp_server.py` (3658 lines, 33 tools, shipped),
`drvarma/mcp_server.py` (929 lines, 14 tools, shipped as `multiart`),
`drvarma/docs/DESIGN_MCP.md` (the design it was built from), and the `atsw`
umbrella (1.1.0: fue + pyfug + art-tseries + drvarma).

---

## 1. The answer: three assistants, not two

**`art`**, **`mtram`** and **`sima`** — separate servers, one ladder, and a testable rule for moving between them.

The reason is not that one is univariate and two are multivariate. It is that
**drvarma and drtran do not answer the same question, do not start from the same
material, and do not make the same claim about the world.**

### 1.1 They start from different material

| | arrives with | first question |
|---|---|---|
| `art` | one raw series | guided or autonomous? |
| `mtram` | **`.pre` files** — the univariate rung already climbed and written down | which series is the output? |
| `sima` | *m* raw series, nothing decided | guided or autonomous? |

drtran's entry material is *the output of ART*. That is not an implementation
detail: the `.pre` is the **continuity contract** of the ladder, and it means the
transfer assistant opens the conversation in a completely different place —
there is nothing to characterise, it has already been done and committed to a
file the analyst can read.

### 1.2 They ask the analyst for different decisions

drvarma: the order (p, q); whether Σ is diagonal; which parameters to constrain.
Evidence: cross-correlation matrices, Tiao–Box partial autoregression, IC grid.

drtran: **the DAG** — who feeds whom; (b, r, s) per link; which covariances to
free; the constraint table (shared / product / linear combination).
Evidence: prewhitening and the CCF, and the residual CCFs of the diagonal fit.

Different decisions, different evidence, different plots. A single server would
have to switch its whole dialogue on a mode flag.

### 1.3 They make different claims — and this is the deep one

**drvarma's system is simultaneous.** Everything is endogenous, Σ is generally
not diagonal, and the impulse response is **not identified without an ordering**
(a Cholesky). That is the VAR's well-known problem and drvarma inherits it
honestly.

**drtran's system is recursive.** Exogeneity is **declared** and **tested** — the
exogeneity portmanteau over k < 0 — and that is what makes ν(k) identified
without ordering anything. The port enforces this: `variance_decomposition`
**refuses** when Q is not diagonal rather than picking an order quietly, and the
impulse response report says so in as many words.

Merging the two would put the analyst one prompt away from two *epistemically
different* answers about the same data, with no signpost between them. That is
the strongest argument in this document.

### 1.4 The handoff rule is already in the code

drtran's `-i`/`-g` reads the residual CCFs of the diagonal fit and proposes a
DAG. If the proposal contains a **cycle**, `find_cycle` rejects it: a cycle has
no topological order, so it cannot be cast as a triangular VARMA — the system is
**simultaneous**.

> **That is exactly when the analyst should move to the VARMA assistant.**

It is a contrast, not a matter of taste, and it fires on real data: in m6 the raw
proposal *is* cyclic and needs two prunings before it is estimable.

```
   art  ──.pre──▶      mtram        ──cycle in the proposed DAG?──▶      sima
 (one series)    (declared exogeneity,                            (everything
                  recursive, testable)                             endogenous)
```

---

## 2. The name

`multiart` reads as "the multivariate ART", which suggests it is ART's natural
continuation. **It is not.** ART's natural continuation is drtran: it consumes
ART's `.pre` files directly, it inherits Box–Jenkins' prewhitening, and its
lineage runs ART → fue → drtran. drvarma is a *classical symmetric VARMA* with a
different ancestry (Mauricio's exact likelihood, AS 311).

Keeping the name would teach every new user the wrong map on their first day.

Three schemes, with a recommendation:

| assistant | models | engine | status |
|---|---|---|---|
| **`art`** | one series: ARIMA + interventions | fue | shipped, 33 tools |
| **`mtram`** | transfer functions and networks (DAG) | drtran | **18 tools** |
| **`sima`** | simultaneous VARMA | drvarma | 14 tools (was `multiart`) |

`mtram` — **M**ultivariate **TRA**nsfer **M**odels. `sima` — **SI**multaneous
**M**ultivariate **A**nalysis. Short, pronounceable, a word rather than a path,
which is the register the suite already uses. Neither claims descent from ART,
and `sima` says the one thing an analyst most needs to know about drvarma
before touching it: everything in there is simultaneous.

Rejected: a family prefix (`art-tf`, `art-varma`) re-asserts exactly the false
lineage this is removing; engine names (`drtran-mcp`, `drvarma-mcp`) leak the
engine into the analyst's vocabulary, which the suite has avoided — nobody says
"I ran fue", they say "I built the model in ART".

`multiart` has been released, but the suite is at 1.1.0 and drtran is not in it
yet: renaming is as cheap now as it will ever be.

---

## 3. What is shared, and at which layer

The rule: **share implementations and artifacts, never the conversational
surface.** Building `mtram` tested it, and it held — three times, each time
catching something that writing a second copy would have hidden:

| borrowed | from | what reuse caught |
|---|---|---|
| the CCF plot | `drvarma.plots.plot_ccf` | **opposite lag-sign conventions.** drvarma's k=+1 is drtran's k=−1, so the arguments go swapped. Unswapped it draws the CCF mirrored — the transfer on the feedback side — and the picture looks perfectly normal. |
| the residual panel | `fue.plots.plot_acf_pacf` | nothing, which is the point: a residual panel now looks the same after a univariate fit in `art` and a joint one here. |
| anomaly calibration | `art.interventions`, the **idea** | that it must NOT be ART's scan re-run. An anomaly in the output's univariate residuals may be explained by the INPUT once the transfer is in the model. What survives the joint fit is the genuine intervention. |

**And how the borrowing is done: by importing the library, never by calling the
other server.** An MCP is a conversational surface for a model, not a calling
convention between programs; server-to-server would make `mtram` depend on `art`
*running* rather than on `art` being installed. `sima` set the precedent
(`DESIGN_MCP.md` §3: "for the art-seeding step multiart *imports* `art` as a
library"). The dependency graph stays acyclic — drtran → fue, drvarma,
art-tseries; art depends on none of them — which is the same discipline the DAG
demands of the models.

**Shared implementation** (already true, keep it true):
- `elf` — drtran scores its cast with drvarma's exact likelihood, unmodified.
- `fue.load()` — one reader for `.pre`/`.inp`, in all three.
- `fue.cast_us` — one univariate cast.
- `_qnewt` — one optimizer, one `fdhess`.
- fuf's forecast report — drtran's `-L` adapts it rather than inventing a second
  one, so a univariate report and a transfer report are the same page.

**Shared artifacts — these are the real interface between the assistants:**

| artifact | written by | read by | carries |
|---|---|---|---|
| `.inp` | ART | fue, ART | the series and the model to estimate |
| `.pre` | ART, drtran (`-W`) | ART, drtran, fuf | **estimated** parameters as new seeds |
| `.dag` | drtran (`-g`) | drtran | the network: who feeds whom, with (b,r,s) |
| `.cns` | drtran (`-g`) | drtran | free / fixed / shared / product / lincomb |

They are inspectable text **on purpose**. The analyst's intervention between
rungs is part of the method — the identified network is a guide, not a verdict —
so the handoff must be something a person can open, read and edit. An in-memory
object hand-off would quietly remove that.

**Shared discipline** (the part that is easy to lose and expensive to regain):

1. **Declare, do not choose.** When something is not identified — a
   decomposition with correlated innovations, a Hessian that is not positive
   definite, a cyclic DAG — say so and stop. Never pick an ordering silently.
2. **Never hardcode the rescaling factor.** Read `model.refactor` / `model.scale`.
   The suite has three logged bugs from getting this wrong.
3. **Never rebuild a table the engine already prints.** ART's instructions
   already forbid the assistant from composing its own parameter table; the same
   rule must hold in both multivariate servers, for the same reason (signs,
   standard errors, conventions).
4. **The refusal list is a feature.** drtran's CLI refuses what it has not
   ported with exit code 2 rather than ignoring the option. The MCP layer should
   behave the same way: a silently dropped instruction is how an analysis starts
   answering a different question.

---

## 4. `mtram` — the transfer assistant

`mtram` has first functions written; the protocol below is what they should
grow into. Its precursor is already in the C: `-g`, the **guided driver of the
ladder**,
which estimates the diagonal, reads the residual CCFs, writes `NAME.dag` and
`NAME.cns` and *prints the next command*. The MCP is that, made conversational.

Proposed protocol, mirroring ART's four stages:

```
0. opening question       guided or autonomous
1. load_pre               one .pre OR .inp per series; the first is the OUTPUT
                          → validate (check_scale), report what each model is
                          → the DIAGONAL GATE: joint == sum of the univariate
                          → and WHICH each file was: an optimum, or a
                            specification still to be estimated (both valid)
2. identify_link          prewhitening + CCF for one input   → propose (b, r, s)
                          `ident_pre=` an alternative output model, for the CCF
                          only, when seasonality is stochastic on one side
   refine_link            a generous free MA → read the DENOMINATOR off the
                          shape of nu(k). The CCF cannot show `r`.
   identify_network       residual CCFs of the diagonal fit  → propose the DAG
                          ⚠ CYCLE → say so and route the analyst to `sima`
                          → WAIT for the analyst to prune. Never prune alone.
3. set_network            fix the network in memory (a cycle is refused)
   (drtran.write_guided)  library-only: the draft .dag + .cns, unpruned
4. estimate               with -n/-c; report the equation VERBATIM
                          → the output's model whole, the inputs' NAMED
5. diagnose               transfer adequacy (k ≥ 0) and exogeneity (k < 0)
                          → and the REFORMULATION ORDER: relation before noise
   calibrate              which observation — and which PAIR — bends the CCF
   overfit                enlarge on purpose (s+1, r+1) and see if it protests
6. impulse_response       nu(k), cumulative, gain, mean lag, with std. errors
   variance_decomposition ⚠ refuses if Q is not diagonal, and says why
7. forecast               level + period + annual with bands; aggregates
   evaluate               -estwin/-C: MAE/RMSE/MAPE by horizon
8. write_inp              -W: the re-estimated univariate blocks back out,
                          as `.inp` — they are a starting point, not an optimum
```

Two tools with no counterpart in ART or drvarma, and they are the reason this
server has to exist:

- **`identify_network`** — the only tool in the suite that proposes a *causal
  structure*, and the only one that can conclude "this is simultaneous, you
  want `sima`".
- **`evaluate_out_of_sample`** — the only tool that compares a model against
  *what happened* rather than against itself. Every other figure in the suite is
  theoretical.

---

## 4a. What `mtram` turned out to be

Eighteen tools. The protocol in §4 survived contact, with one addition that came
out of building it and one refinement that came out of a domain correction.

**The addition — the guided/autonomous split has decision nodes**, and they are
listed in `docs/DECISION_NODES.md`. A node is a point where *the evidence does
not determine the answer*, and they were found by reading where the code returns
`alternatives`, returns `candidates`, or **refuses**. Two rules fell out:
*autonomous never makes a claim the data cannot make for it* (it does not free a
covariance, invent a constraint, or prune a cycle), and *the modes differ in who
decides, never in what is computed*.

The autonomous run self-corrects on the canonical case: the network scan proposes
the wrong shape (b=1 s=0, adequacy p = 0.0000), node N6 catches it, it returns to
N1 with the finer instrument, and lands on logL −718.287406. Every default it
took is in its report — which is what makes an autonomous answer auditable rather
than merely fast.

**The refinement — N6 had two causes and treated them as one.** When adequacy
fails it can be the *shape* or it can be *one observation*, and they need
opposite responses: re-identify, or calibrate an intervention. Re-specifying
around an anomaly is how a model acquires a lag nobody can interpret.
`calibrate` tells them apart, leave-one-out.

And the measurement that matters there is **global, not per-lag**: an anomaly
inflates the residual variance, which is the divisor of every correlation, so it
flattens all the lags at once. Measured on an injected 6 % jump — one point
carrying 47 % of the residual variance, and removing it multiplies every CCF
coefficient by 1.46 while the peak barely moves. In the school's teaching this is
a fact to **verify**, so `plot_calibration` draws both CCFs and the analyst
checks it rather than trusting the number.

## 4b. The umbrella: `polytropos` — and whether it should speak

**It should install. It should not speak.** With one exception, below.

### Why it should not speak

In MCP the client sees **every connected server at once**. An analyst with `art`,
`mtram` and `sima` configured already has all ~60 tools and all three instruction
blocks in front of the model. Routing is not a missing capability — it is
happening. A fourth voice would add something to learn without adding anything to
do.

Worse, it would work against the thing this architecture exists to protect. The
value of three assistants is that **the analyst always knows which bench they are
at**, because the three make different claims (§1.3). An umbrella that answers
questions blurs exactly that: it is a single interlocutor that will happily
discuss a recursive transfer model and a simultaneous VARMA in the same breath.

And a re-exporting router (one server proxying the other three) buys nothing and
costs a lot: every tool added anywhere must be re-exported, versions couple, and
the three instruction blocks have to be merged into one that contradicts itself
at the first opening question — `art` and `sima` ask "guided or autonomous?",
`mtram` asks "which series is the output?".

### The one exception

A **resource**, not a tool: *what is installed, and is it consistent?*

No individual server can answer it — each knows only itself. And the failure it
prevents is real and silent: `drtran` built against a `drvarma` whose `elf`
changed, or an `art` older than the `.pre` fields `mtram` expects. That is the
kind of mismatch that produces plausible numbers, which this project has spent
its whole history learning to distrust.

So: `polytropos` exposes no analysis tools, and at most one read-only resource
reporting the suite's versions and their compatibility.

### The map belongs in the servers, not in the umbrella

Each assistant knows its neighbours and says so at the right moment. `mtram` is
the one that detects a cyclic DAG, so `mtram` is the one that says "this system
is simultaneous, you want `sima`" — it does not need `sima` installed to say it,
any more than `find_cycle` needs a VARMA to reject a cycle. Centralising the map
would move that sentence away from the only place that can know it is time to say
it.

### On the name

The umbrella already exists and is called **`atsw`** (1.1.0: fue + pyfug +
art-tseries + drvarma). Creating a second umbrella name for the same idea gives
one concept two names, which is the problem §2 is fixing, in a new place.

Recommendation: keep `atsw` as the distribution and add an extra —
`pip install atsw[mcp]` pulls the three servers. If the codename is wanted, give
it to the **launcher**: `polytropos`, the single process that mounts the three so
the client config has one entry instead of three. Odysseus *polytropos* is "of
many turns" — which is what a launcher with three faces is, and not what a
distribution is.

## 5. What this locks in

- `atsw` gains `drtran` and the two multivariate extras. The umbrella stays what
  it is: a compatible set, no code of its own.
- ART does not depend on drvarma or drtran. drtran depends on fue and drvarma.
  drvarma depends on neither. **No cycles** — the same discipline the DAG demands
  of the models.
- Three servers can run at once, and should: an analyst climbing the ladder has
  ART's `.pre` on disk while the transfer assistant reads it.
- The `.pre` written by drtran `-W` is a valid fue file, so ART and fuf can read
  a jointly estimated model. That closes the loop the other way, and it is why
  `-W` exists.

## 6. Open questions

- Whether `polytropos` names the launcher or nothing at all (§4b).
- Whether the transfer assistant should be able to *invoke* ART for a series
  whose `.pre` does not exist yet, or refuse and tell the analyst to build it
  there. Leaning: refuse. The ladder's rungs are separate on purpose, and a
  transfer assistant that quietly builds univariate models is a transfer
  assistant that has opinions about them.
- Where this document should live once there is a real home for suite-wide
  documentation. It is in drtran-python for now because drtran is the new piece.
