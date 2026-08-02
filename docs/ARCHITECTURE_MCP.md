# ATSW and its assistants — an architecture

What each MCP server is for, how they relate, and which decisions this settles.
Written 2026-08-02, when drtran reached its first beta and the question became
unavoidable: **one multivariate assistant or two?**

Studied first: `art/mcp_server.py` (3658 lines, 33 tools, shipped),
`drvarma/mcp_server.py` (929 lines, 14 tools, shipped as `multiart`),
`drvarma/docs/DESIGN_MCP.md` (the design it was built from), and the `atsw`
umbrella (1.1.0: fue + pyfug + art-tseries + drvarma).

---

## 1. The answer: three assistants, not two

**ART**, **the transfer assistant** and **the VARMA assistant** — separate
servers, one ladder, and a testable rule for moving between them.

The reason is not that one is univariate and two are multivariate. It is that
**drvarma and drtran do not answer the same question, do not start from the same
material, and do not make the same claim about the world.**

### 1.1 They start from different material

| | arrives with | first question |
|---|---|---|
| ART | one raw series | guided or autonomous? |
| drtran | **`.pre` files** — the univariate rung already climbed and written down | which series is the output? |
| drvarma | *m* raw series, nothing decided | guided or autonomous? |

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
   ART  ──.pre──▶  transfer assistant  ──cycle in the proposed DAG?──▶  VARMA assistant
  (one series)     (declared exogeneity,                              (everything
                    recursive, testable)                               endogenous)
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

| | univariate | transfer / network | simultaneous VARMA |
|---|---|---|---|
| **A (recommended)** | `art` | **`tram`** | **`sima`** |
| B — family prefix | `art` | `art-tf` | `art-varma` |
| C — engine names | `art` | `drtran-mcp` | `drvarma-mcp` |

**A** keeps the register the suite already uses: short, pronounceable, a word
rather than a path. `tram` — TRAnsfer Models — and `sima` — SImultaneous
Multivariate Analysis. Neither claims descent from ART, and `sima` in particular
says the one thing the analyst most needs to know about drvarma: everything in
it is simultaneous.

**B** is the safest for discoverability and the least informative: `art-varma`
re-asserts exactly the false lineage we are trying to remove.

**C** is unambiguous and charmless; it also leaks the engine name into the
analyst's vocabulary, which the suite has so far avoided (nobody says "I ran
fue", they say "I built the model in ART").

Whatever is chosen, the entry point should change **now**: `multiart` has been
released but the suite is at 1.1.0 and drtran is not in it yet, so this is the
cheapest it will ever be.

---

## 3. What is shared, and at which layer

The rule: **share implementations and artifacts, never the conversational
surface.**

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

## 4. The transfer assistant — the one that does not exist yet

drtran already has its precursor: `-g`, the **guided driver of the ladder**,
which estimates the diagonal, reads the residual CCFs, writes `NAME.dag` and
`NAME.cns` and *prints the next command*. The MCP is that, made conversational.

Proposed protocol, mirroring ART's four stages:

```
0. opening question       guided or autonomous
1. load_pre               one .pre per series; the first is the OUTPUT
                          → validate (check_scale), report what each model is
2. identify_link          prewhitening + CCF for one input   → propose (b, r, s)
   identify_network       residual CCFs of the diagonal fit  → propose the DAG
                          ⚠ CYCLE → say so and route to the VARMA assistant
                          → WAIT for the analyst to prune. Never prune alone.
3. write_guided           .dag + .cns, unpruned, cycle annotated
4. estimate               with -n/-c; report the equation VERBATIM
5. diagnose               transfer adequacy (k ≥ 0) and exogeneity (k < 0)
6. impulse_response       nu(k), cumulative, gain, with standard errors
   variance_decomposition ⚠ refuses if Q is not diagonal, and says why
7. forecast               level + period + annual with bands; aggregates
   evaluate_out_of_sample -estwin/-C: MAE/RMSE/MAPE by horizon
8. write_pre              -W: the re-estimated univariate blocks back out
```

Two tools with no counterpart in ART or drvarma, and they are the reason this
server has to exist:

- **`identify_network`** — the only tool in the suite that proposes a *causal
  structure*, and the only one that can conclude "this is simultaneous, you are
  in the wrong assistant".
- **`evaluate_out_of_sample`** — the only tool that compares a model against
  *what happened* rather than against itself. Every other figure in the suite is
  theoretical.

---

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

- The names (§2) — the only decision here that is purely taste, and the only one
  that gets more expensive with every release.
- Whether the transfer assistant should be able to *invoke* ART for a series
  whose `.pre` does not exist yet, or refuse and tell the analyst to build it
  there. Leaning: refuse. The ladder's rungs are separate on purpose, and a
  transfer assistant that quietly builds univariate models is a transfer
  assistant that has opinions about them.
- Where this document should live once there is a real home for suite-wide
  documentation. It is in drtran-python for now because drtran is the new piece.
