# The decision nodes of a multivariate transfer model

What the guided/autonomous split is actually splitting. A node is a point where
**the evidence does not determine the answer** — where two analysts looking at
the same plot can reasonably differ. Everything else is procedure, and procedure
should never stop to ask.

The list comes from the code: from where `identify` returns *alternatives*, where
`identify_network` returns *candidates*, and where a function **refuses**.

---

## The gate, before any node

**Is there a `.pre` for every series?** If not, the analyst goes to `art`. This
is not a node — there is nothing to weigh. A transfer assistant that builds
univariate models is one that has opinions about them, and those opinions belong
one rung down.

---

## N0 — Which series is the output

**Underdetermined by:** everything. No amount of data answers it.

This is the question the analyst arrives with, not one the program can help with.
drtran takes the first `.pre` as the output. In a star that settles it; in a
**network** it does not, because a series can be an output of one link and an
input of another — in m6, EU is an output of EC and an input of EI.

**Autonomous:** the first `.pre`, as the CLI does. It cannot do better.
**Guided:** ask first, before loading anything else.

---

## N1 — (b, r, s) for each link

**Underdetermined by:** the shape of the tail. `identify` returns *alternatives*,
which is the honest signal that this is a node:

- **A** — every significant weight as a free `omega` (r = 0);
- **B** — if the tail decays roughly geometrically, a denominator of order 1
  summarises it with a single parameter (r = 1).

Both fit the same CCF. B is more parsimonious and imposes a shape; A is agnostic
and costs parameters.

**Evidence:** `plot_ccf`. Where the first significant bar sits (that is `b`), how
many follow contiguously, and whether what follows decays or stops.

**Autonomous:** the most parsimonious proposed — `alternatives[-1]`, which is B
when it exists. Defensible, and it is what the C does.
**Guided:** show the CCF, state both alternatives with the reason attached to
each, and wait. The reason is the content: "the tail decays ~geometrically
(ratio ≈ 0.6)" is an argument the analyst can reject.

---

## N2 — Which links belong in the DAG  ← **the node**

**Underdetermined by:** the residual CCFs, which speak pair by pair and know
nothing about the system. `identify_network` returns **candidates** sorted by
peak, and the report says so in as many words: this is a *guide*, prune by
exogeneity, acyclicity and lag plausibility.

Three separate judgements hide here:

1. **Exogeneity.** Nothing enters a genuinely exogenous series. That is a claim
   about the world, testable afterwards but not derivable from a CCF.
2. **Plausibility of the delay.** A link at k = 7 with a peak of 0.26 may be a
   real seven-period lag or a coincidence; the analyst knows which variables can
   plausibly take seven periods to act.
3. **Acyclicity.** Not a matter of taste — see below.

**Autonomous:** take the candidates as proposed, and *if the result is acyclic*,
estimate it. Report which links it kept and their peaks, so the choice is
visible.
**Guided:** present the candidates with their peaks and proposals, and wait. Do
not pre-prune: the draft is what the decision is made on.

### N2′ — The cycle: not a node, a **stop**

If the proposal contains a cycle there is no topological order, so it cannot be
cast as a triangular VARMA: the system is **simultaneous**. No pruning makes that
false — pruning only hides it.

**Autonomous must refuse here**, not choose. Pruning the weakest link to make the
model estimable would be inventing a recursive structure the data did not
support, silently. The answer is `sima`. On m6 the raw proposal *is* cyclic and
needs two prunings, so this fires on real data.

---

## N3 — Which contemporaneous covariances to free

**Underdetermined by:** the CCF at k = 0, which shows correlation and cannot say
whether it is structure or coincidence.

The covariances `q[i,j]` are **born fixed at zero** precisely because this is a
node: "a diagonal covariance is the default and freeing one is a modelling
decision, not a switch". The legacy m6-1 frees **three** of its fifteen.

And freeing one has a price the analyst must be told: **it destroys the variance
decomposition.** With a non-diagonal Q the decomposition is not unique, needs an
ordering, and `variance_decomposition` refuses. That is the trade — a better fit
against an answer you can no longer give.

**Autonomous: never free one.** It is a claim about the world, and an autonomous
run should not make claims the data cannot make for it. Report the contemporaneous
correlations it found and say they were left fixed.
**Guided:** offer them one at a time, with the peak and with the cost named.

---

## N4 — The constraint table (`.cns`)

**Underdetermined by:** the data entirely. `delta1[1] = phi_2[B^1]` — the
transfer's denominator IS the input's own AR — is *theory*. So is the fixed
`(1−B)` factor that forces `nu_num(1) = 0`.

**Autonomous: never invent one.** These encode beliefs; an autonomous run has
none.
**Guided:** recognise the standard patterns and offer them by name. The analyst
who wants a rational transfer knows to ask for it.

---

## N5 — The cast: embedded or subtraction

**Underdetermined by:** almost nothing, in practice.

They model different things — the subtraction cast models the NOISE and
truncates; the embedded one models the observed series and does not — but the
embedded cast is better on every axis that matters and is the default in both
the C and the port.

**Both modes: embedded.** Offer `-S` only for comparison with TASTE or with the
older literature. This is on the list because it *looks* like a node and is not;
treating it as one wastes the analyst's attention.

---

## N6 — After diagnosis: accept, revise, or stop

Two portmanteaus, and they mean different things:

- **Adequacy** (k ≥ 0) fails → the transfer's *shape* is wrong. Revise (b, r, s):
  back to N1. This is a **revision**.
- **Exogeneity** (k < 0) fails → the input is not exogenous, so a single-input
  transfer model does not hold. This is not a tuning problem. It is the same
  finding as a cyclic DAG arriving by another route: the system is simultaneous.

**Autonomous:** at most one revision loop on adequacy, then report whatever it
reached, adequate or not. On exogeneity failure, **stop and say so** — do not
re-specify around it.
**Guided:** present both p-values and the CCF, and let the analyst decide whether
the failure is structural or a lag they can add.

---

## N7 — Everything after: not nodes

Horizon, forecast origin, which aggregates, the evaluation window. These come
from the question the analyst is asking, not from the data. Both modes take them
as arguments and neither should stop to deliberate.

---

## The shape of the split

| node | autonomous | guided |
|---|---|---|
| N0 output | first `.pre` | ask first |
| N1 (b,r,s) | most parsimonious | CCF + both alternatives, wait |
| N2 the DAG | candidates as proposed, if acyclic | candidates + peaks, wait |
| **N2′ cycle** | **refuse → `sima`** | **refuse → `sima`** |
| N3 covariances | **never free** | offer one at a time, with the cost |
| N4 constraints | **never invent** | offer the known patterns |
| N5 cast | embedded | embedded |
| N6 diagnosis | one revision, then report | present and wait |

Two rules fall out of the table, and they are the ones worth keeping:

**Autonomous never makes a claim the data cannot make for it.** Freeing a
covariance, imposing a shared parameter, pruning a cycle — all three are
assertions about the world. An autonomous run that made them would be producing
a model nobody chose.

**The two modes differ in who decides, never in what is computed.** The same
functions, the same defaults, the same refusals. Guided stops at the nodes;
autonomous takes the documented default and *says which one it took*. A guided
run and an autonomous run that made the same choices must reach the same model —
otherwise one of them is doing something it has not declared.
