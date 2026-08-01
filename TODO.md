# drtran (Python) — TODO

Port of `drtran` (C) reusing `fue` and `drvarma`. See the README for the design,
the non-negotiable principle and the validation criterion, and
[`docs/PORTE.md`](docs/PORTE.md) for the **record of the process**: the decisions
that are not translation, the homologation figures and the defects the port found
in the original.

## Step 0 — the input — DONE

- [x] **The `.pre` arrives intact.** `fue.load()` reads `.pre`/`.inp`; verified
      field by field (lambda/d/D, refactor, mu0 + estimate_mu, regular and
      seasonal ar/ma with their free-flags, ar_f/ma_f, deterministics with
      omega/omega_free, the series).
      Decision: **`fue_pre_reader.c` is not ported** (601 lines). Duplicating the
      parser would create a second source of truth that falls out of sync.
      Tests: `tests/test_pre_roundtrip.py` (8).
- [x] **Univariate baseline.** fue Python reproduces the canonical case:
      phi 0.402839 / 0.299193, logL −7.3917 / −760.0326, and the SUM gives
      **−767.424341** (diff 7.6e-09), which is the joint fit's target.
      Tests: `tests/test_baseline_univariate.py` (3).
- [ ] **The write round trip.** `model.write_pre()` requires the fitted model, so
      the cycle `estimate -> .pre -> reread` is still untested. It is what closes
      continuity towards the next rung. It does not block the cast.

## Step 1 — the cast — DONE (diagonal case)

- [x] **Diagonal cast** (`cast.py`). fue's univariate cast is reused per series
      (`build_est_spec` + `cast_us_py`), which already returns `w` with Box-Cox,
      differencing and the deterministics subtracted — it is the C's
      `build_stationary_series`. drtran only ASSEMBLES: block-diagonal Phi and
      Theta, mu per series, the w's aligned at the end, Q normalised.
- [x] **CONCENTRATED likelihood.** `est()` does not estimate the scale: it
      decomposes Sigma = sigma2*Q with Q[1][1]=1 and concentrates sigma2.
      Evaluating `elf_varma` with an absolute Sigma gives something else (passing
      the identity gave −7802 instead of −767). drvarma's `_elf_f1f2` is used plus
      the formula from `drvmlest.c:est [4]`.
- [x] **Seeding the variance ratios.** They are not left at zero: the scales
      differ x1098 in the canonical case and starting at 1 leaves the initial
      point at −1371. They are computed with the same `elf`, m=1, on the `.pre`'s
      seeds.
- [x] **Transfers by SUBTRACTION.** `Link(out, inp, b, r, s)`, `compute_irf` (BJR
      convention: omega(B)=omega_0−omega_1 B−…, the leading term adds and the rest
      subtract) and the subtraction from the output: series 1 of the VARMA becomes
      the noise N_t = w_Y − SUM_j transfer_j. Verified: with omega=0 the
      likelihood is EXACTLY the diagonal one (difference 0.0).
- [x] **Joint estimation** (`estimate.py`): Mauricio's scaled objective (1995 eq.
      3.5) normalised to 1 at x0 — in the multivariate case (f1/f1_0)^m*(f2/f2_0),
      because ll = C − 0.5n(m*log f1 + log f2) — minimised with drvarma's
      `raxopt`. A rejected point returns 1.0 and the optimizer moves away.
      Measured on the canonical case Y<-X (b=0,r=0,s=0): **omega_0 = 0.016002**,
      logL −736.774 against the diagonal's −767.424, **LR = 61.3 (1 df,
      p=4.9e-15)**. Converges by gradient in 21 iterations.
- [x] **EMBEDDED cast (`embed.py`) — UNBLOCKED and homologated.**
      Polynomial algebra: row i = diagonal phi_i*D_i, off-diagonal
      −phi_i*omega_k*B^b_k*(D_i/delta_k), MA D_i*theta_i, with D_i = PROD_k
      delta_k over the incoming links; means in topological order; the series with
      NOTHING subtracted; and `normalize_phi0` (Phi_0^-1 on the left) because a
      contemporaneous transfer puts omega_0 at lag zero. Verified: with no links
      and with omega=0 it matches the diagonal EXACTLY.
      It was blocked by a porting bug in drvarma: `_chol_lower` used
      `np.linalg.cholesky` (the strict one) where the C uses the MODIFIED Cholesky
      (`nlatools.c:choldcp`), which accepts semidefinite matrices. Since the
      embedded cast produces a singular Phi_p by construction, `elf` rejected it
      with ifault=3. **Fixed in drvarma** (commit `fix(as311)`), and it closed
      along the way the three C-parity tests its suite had been carrying.
      It homologates with the binary (`-V`) to ~1e-7: (0,0,0) −736.774158,
      (0,1,0) −721.801539, (0,0,1) −718.287406, (1,1,1) −756.602851.
      `fit(..., embed=True)` is now the DEFAULT, as in the C.
      **Note:** the embedded cast does NOT give a higher likelihood than the
      subtracting one — they do not measure the same thing (the subtracting one
      models the NOISE and truncates; the embedded one models the OBSERVED
      series). The C shows the same pattern.
- [x] **Homologation with the C binary.** The subtracting cast reproduces the
      compiled `drtran` to ~1e-7 on four b/r/s combinations:
      (0,0,0) −736.774158, (0,1,0) −721.720197, (0,0,1) −718.183933,
      (1,1,1) −756.528944; and the diagonal −767.424341. One test relaunches the
      binary live so as not to carry stale references.
      Tests: `tests/test_homologation_c.py` (12).
- [x] **Identifying (b, r, s) by prewhitening + CCF** (`identify.py`).
      Port of `prewhiten_and_identify`. It prewhitens the input with ITS ARMA,
      applies THE SAME filter to the output, computes the CCF and reads
      nu(k)=r(k)*s_beta/s_a.
      It homologates with the binary: band 0.13640, r(0)=0.492, r(1)=0.310,
      r(2)=0.025, r(−1)=−0.077, r(−6)=−0.128, and the same proposal **b=0 r=0
      s=1**. Exogeneity by portmanteau over k<0: **Q(24)=18.2969, p=0.7884**,
      identical to the C (`ChiTestC`'s divisor is n−i+1, not n−i).
      The C's two decisions that keep the answers sane are replicated: the
      structure is the **CONTIGUOUS block** from b (there is a significant peak at
      lag 24 that does NOT enter the proposal — with 5% bands one lag in 20 is
      expected outside), and exogeneity is judged by portmanteau, not by counting
      how many cross.
      Tests: `tests/test_identification.py` (12), including a synthetic transfer
      with a known delay (b=3).
- [x] **The MEAN is the mean, not an intercept** (`embed.py`). The model is
      written in DEVIATIONS, (w_Y − mu_Y) = nu(B)(w_X − mu_X) + N_t =>
      E[w_Y] = mu_Y: the output **inherits nothing** from the input, and
      multiplying by delta(B) gives row 1 of Phi(B)(w − mu) = Theta(B)a with
      mu = (mu_Y, mu_X), with no extra term.
      Before, `MU[i] += (omega(1)/delta(1))*MU[inp]` was done in topological
      order, which is the INTERCEPT parametrisation. The same family as long as
      mu_Y is free (the same optimum to 1e-12); they diverge with mu_Y FIXED.
      Coherence with fue rules: the `.pre`'s mu is the mean fue estimated, and a
      zero means the series has no drift. **The same defect was in the C** and was
      fixed there (`tran_shootx.c:build_embedded_varma`, commit `fix(cast)`),
      where it stood as a suspected sign error in its TODO.
      It does not show on the canonical case: its input (WTI) has mu = 0 and the
      term is zero under either convention. An input with a **free mean** is
      needed. Verified with `ES_CPI_m10` <- `DE_CPI_mar3sar`: fue C = fue Python =
      drtran C = drtran Python = 3.8139613 on the diagonal, and with a transfer
      C == Python at (0,0,0) 24.408974, (0,0,1) 35.487981, (0,1,1) 35.555382.
      Tests: `tests/test_transfer.py`, two new ones.
- [x] **Transfer NETWORK, DAG and `expand_params` — DONE and homologated.**
      Three new pieces:
      * `network.py` — the `.dag` (`OUTPUT <- INPUT b r s`, series by NAME) and the
        rejection of cycles, which **says which cycle it is**. With a cycle there
        is no topological order: it would stop being a recursive DAG and become a
        simultaneous-equations system, which is not what the cast represents.
      * `slots.py` — the slot table with the C's NAMES (`omega1[1]`,
        `theta_2[B^1]`, `q[5,2]`, `log(var3/var1)`), the `.cns` and
        `expand_params` with the five natures: free, fixed, **shared**,
        **product** (`x = -y * z`) and **linear combination**
        (`x = t1 + t2 - t3`, each term a slot or slot*slot). The gradient needs no
        chain rule: `expand` goes INSIDE the objective and `cdgrad` differentiates
        it by finite differences, the same decision as the C's.
      * a **non-diagonal** covariance in both casts (`build_sigma`): the `q[i,j]`
        always enter the map but are **born fixed at zero**, and freeing them is
        the analyst's decision (`q[5,2] = free`), not a global switch — the legacy
        m6-1 frees THREE of its 15. Outside the PSD region the point is rejected
        (objective 1.0) instead of evaluating the impossible.

      The slot order is NOT the C's (it groups by class, here by series, because
      fue produces the univariate block), but the NAMES are — and the `.cns` goes
      by name. `build_slots` checks the total against `cast_spec.npar`: if the
      enumeration drifted from fue's, the names would stop matching the positions
      **silently**.

      Homologated with the binary on a 5-series network with the chain
      EC -> EU -> EP, one input with two outputs, a denominator with r=1 and two
      free covariances: **−1434.696068** (diff 1.9e-10); with a product and a
      linear combination, **−1439.505804** (diff 9.4e-08), and `expand`
      reconstructs the derived slots from only the free ones (diff 3e-07, which is
      the 6-decimal rounding of the C's report). The **optimizer** arrives too: a
      3-series network, 24 free, **−912.244333 in 180 iterations** against the C's
      181. Tests: `tests/test_network.py` (19).
- [x] **The canonical m6 — VALIDATED** (after fixing fue, 2026-07-30). It was
      blocked because **fue Python degraded the `compimp` deterministic to a
      `pulse`**: the compensated impulse is +1 at the date and **−1 at the next**,
      and the reader swallowed the −1. Only `M6_EI.pre` uses it, and only because
      of that: evaluating at fue C's optimum, five of the six series matched to
      5e-8 and EI gave −292.495 instead of −290.613. Fixed in **fue 0.1.9**
      (BUG-0006/0007); the port now reproduces the C's two targets:
      **diagonal −1709.511575** (diff 5.0e-07) and **free network −1697.613401**
      (diff 5.9e-07), plus the variants with products and with the full structure.
      Tests: `tests/test_network.py`. See `docs/PORTE.md` §5.4.

## Step 2 — the gate — PASSED

- [x] **Diagonal joint fit == fue run separately.** logL = **−767.424341**,
      difference 3.9e-07. And it is reached ALREADY at the `.pre`'s seeds, which
      confirms empirically why the C reports `termcode 3` on this rung: the seeds
      are the optimum. Tests: `tests/test_cast_diagonal.py` (4).

## Step 3 — the rest

- [x] **NETWORK identification (`-i`/`-g`) — DONE** (`netid.py`). Having read the
      CCFs of the **diagonal** model's residuals, it proposes the directed links
      (with their b and s), the contemporaneous covariances and the pair with
      feedback. The residuals come from `elf` with `atf=True`: they are the EXACT
      ones, with their pre-sample initialisation, not a hand-written filter.
      It homologates with the binary **line by line** on m6: the same covariances
      (EI.EU +0.358, EI.EA −0.314, EC.EA −0.408), the same eight links with the
      same peaks and the same (b, s) proposals, and in the same order.
      **Careful when comparing:** a bare `-i` does NOT identify from the diagonal —
      it builds its own model (61 slots, 46 free, logL −1716.36) and reads the CCFs
      of THOSE residuals. It must be asked for `-0 -i` with the same constraints;
      confusing them makes the figures disagree and look like a bug in the port.
      The guided mode writes the `.dag` and the `.cns` **unpruned**, with the
      covariances by numeric index (the `.cns` does not read names in the `q`, and
      the C's `-i` prints them with names under a heading that says "paste into a
      -c file"). If the proposal carries a cycle it is written all the same,
      **noted**: in m6 it does, and two prunings are needed to make it acyclic.
      Deciding which one falls is judgement, not arithmetic, so the library warns
      and does not prune.
      Tests: `tests/test_network_identification.py` (6).
- [x] **`diagnose.c`'s diagnostics — DONE** (`diagnose.py`). The transfer's
      portmanteau (k >= 0, contemporaneous lag included) and the exogeneity one
      (k < 0). It matches the binary exactly: adequacy p = 0.1966, exogeneity
      p = 0.9136.
      **They are measured on the STRUCTURAL residuals**, not the reduced-form
      ones: with b = 0 the embedded cast puts omega_0 at lag zero, so the
      reduced-form ones come out correlated by construction
      (Sigma_12 = omega_0*sigma2_X) and the portmanteau condemns a correct model —
      it gave p = 0.0000.
      Tests: `tests/test_diagnose.py` (5).
- [x] **Forecasting — the CORE and the LEVEL layer** (`forecast.py`). The
      MA(infinity) psi weights, the error variances (level, variation, annual
      variation), the point forecasts, and `to_level`, which composes xi (the
      future deterministics), the integration against delta(B) and the inverse
      Box-Cox, reusing fue's `_build_xi`, `_nonsop_coefs` and `_inv_boxcox`.
      **sigma2 is applied**: the cast returns Q, not Sigma.
      **Verified against the binary on both series**: WTI (60.76 61.02 61.10
      61.13 61.14 61.14) and ES_CPI, the one that receives the transfer (82.01
      82.02 82.38 83.17 83.33 83.44). It agrees to the two decimals the C
      publishes. Tests: `tests/test_forecast.py` (13).
- [x] **CLOSED — the output's level.** It was a defect of the port, in the level
      layer and not in the model. `to_level` built xi from the **deterministic
      omegas in the `.pre`** (the univariate seeds) instead of from the ones
      **re-estimated jointly**: in the canonical case the two `omega_d1` move to
      −0.040867 and −0.094588 once the transfer is fitted alongside them.
      Fixed with `_fitted_deterministics`, which recovers them from the estimated
      vector — they head the series' univariate block, in `build_slots` order:
      every free omega first, then every delta.
      Silent by construction: with the seeds the forecast is **the univariate
      one**, which even matches the C's own `-0`. It only shows up against a fit
      that actually has a transfer in it.
      What located it was **instrumenting the binary** (a temporary `fprintf` in
      `transfer_forecast`, since reverted): the C's `f1` turned out to be identical
      to the port's (0.1133308248 0.1326379020 0.1381367427 …), which left the
      level layer as the only suspect. The same technique that resolved the `elf`
      hang and the two `nlatools` defects.
- [x] **CLI — DONE** (`cli.py`, `__main__.py`). The **same option letters** as the
      C, `getopt` string included and with the single-dash alias `-estwin`: a
      command line written for the C runs here untouched, or it is not a port.
      `gnu_getopt` is used (it permutes) because the `.pre` files come first.
      Done: `-b/-r/-s` (lists), `-0`, `-V/-S`, `-n`, `-c`, `-f`, `-O`, `-i`, `-g`,
      `-p`, `-m`, `-o` (`-` for stdout), `-N/-X/-D/-E/-M`, `-v`, `-h`.
      The bulk switches fix **by index, not by name**: names repeat (two AR factors
      give two `phi_1[B^1]`) and `set_fixed` only reaches the first.
      **What is not ported is refused with exit code 2, not ignored**: `-a`,
      `-estwin/-R`, `-C`, `-L`. A silently dropped option is how a script starts
      publishing numbers that answer a different question.
      The executable is called **`drtran-py`**, NOT `drtran`: that name is the C
      binary's on this machine and shadowing it is exactly the trap that bit `fue`.
      Verified against the binary on the canonical case: logL −718.287406, omega
      0.016400/−0.010747, and the forecast table with dates and standard errors
      equal to the C's (82.01/0.24 … 83.44/0.95; WTI 60.76/8.29 … 61.14/27.10).
      `-0` gives −767.424341. Tests: `tests/test_cli.py` (16).
- [x] **Language sweep — DONE** (2026-08-01). The whole port is in English:
      the eight source modules, the ten test files (renamed too:
      `test_red.py` -> `test_network.py`, `test_identificacion*.py` ->
      `test_identification.py` / `test_network_identification.py`,
      `test_transferencia.py` -> `test_transfer.py`,
      `test_homologacion_c.py` -> `test_homologation_c.py`,
      `test_baseline_univariante.py` -> `test_baseline_univariate.py`), the
      README, this file and `docs/PORTE.md`.
      Public identifiers were renamed with it, not just the prose:
      `Identificacion` -> `Identification` (and its fields `umbral` -> `bands`,
      `alternativas` -> `alternatives`, `exogena` -> `exogenous`,
      `p_exogeneidad` -> `p_exogeneity`, …), `RedIdentificada` ->
      `IdentifiedNetwork` (`enlaces` -> `candidates`, `covarianzas` ->
      `covariances`, `retroalimentacion` -> `feedback`, `banda` -> `band`,
      `nombres` -> `names`, `ciclo` -> `cycle`), `Candidato` -> `Candidate`
      (`pico` -> `peak`), `Fit.estado` -> `Fit.status`,
      `identify(..., banda="constante")` -> `band="constant"`,
      `report(..., nombres=)` -> `names=`, `check_scale(..., minimo=)` ->
      `minimum=`, and `serie=` -> `series=` in `to_level`, `Forecast.se` and
      `report_forecast`. The package is unpublished (0.0.1), so renaming was free now and
      would not have been later.
      All user-visible output is English too, including the `-p` report, which used
      to come out in Spanish from an English CLI — and which broke during the
      sweep, because nothing exercised it. `tests/test_cli.py` now covers it.
      Tests: 112 passing.
- [x] **Standard errors — DONE.** From the Hessian recomputed **at the optimum**
      by finite differences (`fdhess`, ported into drvarma's `_qnewt.py` — it was
      the one routine of `qnewtopt.c` the Python side lacked), then
      `cov = 2*F(x_hat)*H^-1/n`, exactly as drtran's `est()` does it.
      All 17 standard errors of the canonical case match the binary (worst
      relative difference 2.8e-04, which is finite-difference noise): omega1[0]
      0.001703 (t = 9.633), omega1[1] 0.001693 (t = -6.349). The CLI prints the
      `std.error / t-stat / p-val` columns with the C's significance codes; `-Q`
      skips them, since they cost (k^2+3k)/2 likelihood evaluations.
      **Not** the optimiser's BFGS matrix: it is path-dependent (fue C uses it
      and reports different s.e. for identical estimates across runs — see
      `fue-1.13.1/ERRORES_ESTANDAR.md`) and it is never even built when the
      search starts at the optimum, which is drtran's normal case.
      Added a guard the C does not have: if the Hessian at that point is **not
      positive definite** the result is refused with `ifault=2` rather than
      passed through the MODIFIED Cholesky, which would patch the pivots and
      publish plausible-looking numbers. It fires on m6 at the `.pre` seeds (2 of
      55 eigenvalues <= 0 there; all 55 positive at the real optimum).
      See `docs/PORTE.md` §9 for why Mauricio left `fdhess` commented out — the
      cost and truncation hypotheses were both measured and falsified.
      Tests: `tests/test_stderr.py` (10).
- [x] **Variance decomposition and the variation columns — DONE.** The forecast
      report now carries LEVEL / PERIOD / ANNUAL with their standard errors, and
      the decomposition of the level's forecast error variance. Both identical to
      the C: 1/2020 gives 82.01 (0.24), −1.00 (0.24), 1.07 (0.24); and ES_CPI's
      error is 68.2 % own noise / 31.8 % WTI at h=1, 46.3 % / 53.7 % at h=6.
      The decomposition runs on the **structural** representation, not the
      reduced form: with b=0 the reduced-form Sigma is correlated by
      construction, so `Q = Phi0*Sigma*Phi0'` and `psi*_struct = psi* Phi0^-1`.
      The total variance is invariant under that, so the s.e. do not move. If Q
      is not diagonal it is **declared** impossible rather than resolved with an
      arbitrary ordering. Tests: `test_forecast.py` (+3), `test_cli.py` (+1).
- [ ] **The ERR column.** The C prints the one-step residual on the observed
      rows; the port leaves the column out. `elf` returns residuals in its own
      internal scale and the factor that restores the series' units is not
      established — measured against the binary, they differ by ~1.21 on the
      first series and ~23.6 on the second, so it is per-series, not common. It
      never mattered because the only consumer was the portmanteau and the CCF
      is scale-invariant (which is why the diagnostics homologate exactly while
      the residuals do not). Find the factor, then print the column.
- [ ] The estimated nu(k) weights with their standard errors, which the C prints
      and the CLI does not.
- [ ] **fue's own standard errors — APLAZADO a propósito (2026-08-01).** fue C
      computes them from the BFGS matrix; its `fdhess` call sits commented out at
      `drvmlest.c:112` and uncommenting it is a one-line change that drtran now
      shows to work. **It is not being done here.** fue is a general-purpose
      program: it will be handed models that do not converge, and there the
      finite-difference Hessian can come out non-positive-definite where the
      BFGS matrix cannot (see `docs/PORTE.md` §9 — that trade-off is the most
      likely reason Mauricio left the call commented in the first place).
      Switching the default for every fue user needs its own session, with an
      empirical sweep over the battery and the theory behind the choice, not a
      one-line change borrowed from a program with much stronger preconditions.
      The full note lives in `fue-1.13.1/ERRORES_ESTANDAR.md`.
- [ ] The C-only options: aggregates (`-a`), fixed-window estimation
      (`-estwin`/`-R`), the rolling out-of-sample errors (`-C`) and the LaTeX
      report (`-L`).

## Inherited from the C — to watch in the port

- [ ] **The optimizer degrades with `refactor=1`.** In the C it hangs for over 2
      minutes without converging at Delta-log ~0.002, and converges in 23
      iterations with refactor=100. Cause: `cdgrad`'s finite-difference step
      (~6e-6 absolute) has a terrible signal-to-step ratio at raw scale. The port
      inherits `_qnewt` from drvarma, so it probably inherits the fragility.
      `drtran.check_scale()` already warns; whether to also condition internally is
      still to be decided.
- [ ] **`termcode 3` is NOT a failure here.** Starting from the `.pre`'s seeds
      (which on the diagonal rung already ARE the optimum) the line search cannot
      improve and stops. The correct classification (the C's M1 milestone): 1-2
      convergence, **3 stopped without improvement**, 4-5 a real failure. The right
      test is to perturb the pre-estimates and check that it converges by gradient
      to the same point.
      NB: multiart (drvarma) rejects termcode 3 in its order search, where the
      seeds are OLS. The two criteria coexist, but they are worth not confusing.
