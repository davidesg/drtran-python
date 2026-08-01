# drtran (Python)

**Box–Jenkins transfer function models by exact maximum likelihood.**

A Python port of [`drtran`](../drtran) (C). It is the **bridge** between two
programs that already work:

- **[fue]** — identifies and estimates **univariate** models (ARIMA + Box–Cox +
  deterministics). Produces one `.pre` per series.
- **[drvarma]** — evaluates Mauricio's **exact VARMA likelihood** (`elf`) and
  maximises it with factored BFGS.

drtran reads the `.pre` files already specified in fue — one output and one or
more inputs — and estimates them **jointly**, every parameter at once:

```
Y_t  =  SUM_j  omega_j(B)/delta_j(B) * B^b_j * X_j,t  +  N_t
```

## The `.pre` is the contract, not an input format

fue leaves each series' best estimated univariate model in the `.pre`; drtran
takes it as a seed. `.pre` and `.inp` share a format — the difference is that a
`.pre`'s seeds are the estimates of the last iteration. That is what makes the
chain **iterative and continuous**: one rung's output feeds the next.

```
fue -> .pre -> drtran (diagonal) -> residual CCFs -> .dag/.cns -> drtran (network) -> ...
```

The artifacts are inspectable text files **on purpose**: the analyst's
intervention between rungs is part of the method (the identified network is a
guide, not the final one), not a limitation to be abstracted away.

## Design principle, non-negotiable

> drvarma's `elf` is used **as it is**. Not modified, not patched, not
> special-cased. It is the reference implementation of the exact likelihood. Any
> discrepancy with fue is a bug of drtran's cast, **never** of `elf`.

## Validation criterion (the gate to everything else)

**Diagonal joint estimation == fue run separately.** With a diagonal structure
the exact likelihood factorises, so the joint fit must reproduce the sum of the
univariate ones. Canonical case `ES_CPI_m10` <- `WTI_ar1`:

| | phi | logL |
|---|---|---|
| ES_CPI | 0.402839 | −7.3917 |
| WTI | 0.299193 | −760.0326 |
| **sum = joint target** | | **−767.424341** |

## An example

```python
import drtran
from drtran.cast import build_cast_spec, Link
from drtran.estimate import fit, unpack

Y = drtran.load_pre("ES_CPI_m10.pre")       # the output
X = drtran.load_pre("WTI_ar1.pre")          # the input

cs = build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=1)])
f  = fit(cs)                                # embedded by default, as in the C
print(f.loglik, unpack(f)["links"])
```

`identify(cs, link)` proposes (b, r, s) by prewhitening and the CCF before
estimating. `forecast(f, L=12)` and `to_level(fc, cs, serie=0)` give the forecast
back in the original units.

And a **network** of transfers, with its constraints:

```python
from drtran import build_slots, read_cns, read_dag

cs = build_cast_spec(specs)                       # the m series
cs = build_cast_spec(specs, links=read_dag("m6.dag", cs.names))
slots = build_slots(cs)                           # the q[i,j] are born fixed at 0
read_cns("m6.cns", slots)                         # free / fix / share / x=y*z
f = fit(cs, slots=slots)
```

```
# m6.dag                      # m6.cns
EP <- EI   1 0 1              q[5,2] = free
EI <- EU   1 0 3              omega1[1] = omega1[0] * theta_2[B^1]
EU <- EC   2 0 1              omega3[0] = omega3[1] + omega3[2] + omega3[3]
```

## The command line

The C's own options, verbatim — a command line written for the binary runs here
unchanged. The executable is `drtran-py` (not `drtran`: that name belongs to the
C binary), and `python -m drtran` works too.

```
drtran-py ES_CPI_m10.pre WTI_ar1.pre -b 0 -r 0 -s 1 -V -f 6
```

`-estwin E` estimates once on the first E observations, holds the parameters
fixed and rolls the forecast origin forward, reporting MAE / RMSE / MAPE by
horizon — the only way to decide **empirically** whether one specification
forecasts better than another, since every other figure the program prints is
theoretical. `-C FILE` writes the per-origin errors as CSV.

What is not ported yet (`-a`, `-L`) is **refused with exit code 2, not
ignored**: a silently dropped option is how a script starts publishing numbers
that answer a different question.

## Status

**Steps 0 to 7 — closed.** The input is validated field by field, the diagonal
cast passes the gate (−767.424341, differing by 3.9e-07 from fue's sum), both
transfer casts are in — by subtraction and **embedded**, the latter the default —
with joint estimation, the identification of (b, r, s) by prewhitening + CCF, and
the **network**: the `.dag`, the `.cns` slot table (fixed, shared, products and
linear combinations) and the non-diagonal covariance. Then the diagnostics, the
forecast — core and level layer — and the CLI. All homologated against the C
binary to ~1e-7, with tests that **relaunch it live**. **111 tests**, green.

Relloso's **m6** system (1997) is reproduced in full: diagonal −1709.511575 and
free network −1697.613401, the C's own targets.

The **network identification** (`-i`/`-g`) is there too: having read the CCFs of
the diagonal model's residuals, `identify_network(cs, x=f.x)` proposes the links
with their (b, s), the contemporaneous covariances and the pairs with feedback;
`write_guided` writes the draft `.dag` and `.cns`. It is a **guide**: prune by
exogeneity, acyclicity and how plausible the delay is before estimating.

Standard errors come from the Hessian recomputed **at the optimum** by finite
differences, not from the optimiser's accumulated BFGS matrix — the latter is
path-dependent and is never even built when the search starts at the optimum,
which is drtran's normal case. All 17 of the canonical case match the binary.
`docs/PORTE.md` §9 records why Mauricio left that call commented out, and what
measuring it settled.

**Missing:** aggregates (`-a`), the LaTeX report (`-L`), and the ERR column of
the forecast table. Details in [`TODO.md`](TODO.md).

> **[`docs/PORTE.md`](docs/PORTE.md) — the record of the process.** How it was
> done, which decisions are not translation and why, the homologation figures,
> the three defects the port found in the original (among them that **mu is the
> mean, not an intercept**) and the traps in comparing against the binary.

## Scope of the port

Most of the C is **not ported**, it is reused: `elfvarma` + `drvmlest` +
`qnewtopt` + `nlatools` are already in drvarma Python; `gnuplot_i` -> matplotlib;
the `.pre` reader -> `fue.load()`. That is ~5,800 of the C's 12,615 lines.

Ported: `tran_shootx.c` (the cast, 668) -> `cast.py` + `embed.py`, `diagnose.c`
-> `identify.py` + `diagnose.py` + `netid.py`, `forecast.c` -> `forecast.py`, and
`drtran.c`'s CLI/orchestration -> `cli.py` (4201 lines of C become 490 of
Python).
