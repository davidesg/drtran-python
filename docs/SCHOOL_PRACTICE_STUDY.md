# What the theses codify, and what of it belongs in drtran

A study of how the Treadway school actually builds transfer models, read from
three sources in `drtran/literature/`, and an assessment of what can be lifted
into code.

| source | author | year | why it is here |
|---|---|---|---|
| `T24986.pdf`, 409 pp | **M. Silvia Muñoz Polo** | 2000 | six transfer cases worked end to end — the modus operandi in action |
| `T28619.pdf`, 329 pp | **Sonia María Brajín Rodríguez** | 2004 | the formal definitions and the overfitting doctrine, stated as rules |
| `9720.pdf`, 21 pp | **Silvia Relloso Pereda** | — | working paper; her 1997 thesis is cited but not present |

All three directed by Arthur B. Treadway. Muñoz §2.6 states the procedure;
Muñoz §6.4 shows it applied six times; Brajín §2 gives the formulas and the
diagnosis rules.

**The value is not in the theory.** It is in what these analysts DO when the
numbers come back — which signals they read, in what order they react, and what
they refuse to conclude. Most of that is absent from drtran today.

---

## 1. The procedure, as §2.6 states it

> «La construcción de cada modelo de transferencia es el resultado de un proceso
> iterativo que emplea los modelos univariantes del input y del output. **El
> modelo U del input permanece inalterado desde el inicio hasta el fin del
> proceso.** La estructura estocástica del modelo U del output se emplea como
> especificación inicial del modelo univariante del ruido, **aunque este modelo
> puede ser reformulado después de completada la especificación de la
> relación**.»

Three things there, and drtran already honours the first two:

* the input's univariate model is **frozen** — it comes from the `.pre` and is
  never re-identified. This is drtran's design, and it is confirmed as doctrine
  rather than convenience;
* the output's univariate model **seeds the noise**, and is revisable;
* the initial CCF is the output filtered by the INPUT's model against the
  input's residuals — which is `prewhitened_pair`, and the reference is Box et
  al. (1994) ch. 11.

### 1.1 The asymmetry that fixes the order of reformulation

This is the single most codifiable sentence in the whole methodology chapter:

> «La especificación inadecuada de la relación v(B) puede generar la apariencia
> (en acf/pacf residuales) de especificación inadecuada del ruido θ(B) a la vez
> que una ccf que requiere reformulación de la relación. **Sin embargo, la
> especificación inadecuada del ruido NO puede dar la impresión en ccf de
> especificación inadecuada de la relación.** Por estas razones, se reformula
> v(B) hasta que parezca adecuada ANTES de reformular θ(B).»

A bad **relation** contaminates the residual ACF/PACF *and* the CCF. A bad
**noise** contaminates only the ACF/PACF. The contamination runs one way, so the
repair order is forced: **relation first, noise second**. An analyst who fixes
the noise first is chasing a symptom, and can iterate for a long time without
converging.

`mtram` does not say this anywhere, and it is exactly the kind of rule an
assistant should carry.

---

## 2. What the six cases actually do

### 2.1 Identification is iterative, not a single reading of the CCF

The first specification of `v(B)` is a **generous pure MA**, which Muñoz
describes precisely:

> «v(B) = .35 + .21B + .40B² + .16B³ + .64B⁴ + .29B⁵ + .34B⁶ … **De hecho, esto
> equivale a una estimación de los primeros términos de la ccf.**»

Then the *pattern of the estimated weights* decides the denominator:

> «Se observa que el valor absoluto de los mismos **decrece conforme aumenta el
> retardo**, lo que parece indicar que la relación requiere un factor AR(1) con
> parámetro positivo.»

and finally the non-significant terms are dropped (here, the odd lags). So
`(b, r, s)` is arrived at by **estimating and reading**, not by reading the CCF
once. drtran's `identify` does the single reading; the iterative refinement is
left entirely to the analyst.

### 2.2 Signals read from the ESTIMATES — these are the gold

Four diagnostics that have nothing to do with portmanteaus, each appearing more
than once across the cases:

| signal | what it means | source |
|---|---|---|
| **ω₀ ≈ −1.00** | the dead time is wrong, and short by at least one period | 6.4.4, 6.4.5 |
| **weights decreasing in \|·\| with lag** | a denominator AR(1) with positive parameter is called for | 6.4.1, 6.4.2, 6.4.3 |
| **δ ≈ .99 (.01)** | "poco plausible": the gain and the mean lag blow up and come out **non-significant** | 6.4.5 |
| **high correlations among estimated parameters** | "la situación de estimación está mal definida"; an overfit experiment under this condition is declared **failed** | 6.4.3 |

The first is startling in its directness. Twice, an estimated leading MA
coefficient of exactly −1.00 is read as *"el tiempo muerto especificado es
erróneo y es al menos superior en un trimestre"*, and in 6.4.4 the parameter
stays at −1.00 through several reformulations until the dead time is corrected
to two periods and the restriction imposed.

drtran computes every one of these numbers and reads none of them.

### 2.3 Anomalies, in both directions

Muñoz's cases settle empirically two claims this session has been circling.

**An anomaly distorts the CCF, and removing it REVEALS structure:**

> «La anomalía en I/87 se revela distorsionante de la ccf residual. Al incorporar
> la variable rampa en I/87, **en la ccf se detecta MÁS estructura**, lo que
> motiva el empleo de dos términos MA adicionales.»

**And the input may explain an anomaly the univariate analysis flagged:**

> «el residuo en IV/92 del modelo univariante es de −3.1σ, pero el
> correspondiente al modelo de transferencia tan sólo es de −2.4σ. Este
> resultado sugiere que parte de tal efecto podría explicarse por la influencia
> que ΔlnM1 ejerce sobre lnQ.»

That is precisely what `calibrate`'s docstring asserts — "an anomaly in the
output's univariate residuals may be explained by the INPUT once the transfer is
in the model" — and here it is measured, in σ, on real data.

They also trace residual-CCF spikes to **pairs of dates**: lag 8 justified "por
distorsión negativa entre el ruido preblanqueado y los residuos del input en
II/94 y II/92, y III/86 y III/84". Not an outlier in one series — a distortion
between two specific observations, one in each.

### 2.4 The prewhitening artifice when the seasonalities differ

Case 6.4.1 hits a problem drtran would hit identically and handle worse. The
output's model has **mixed** seasonality (stochastic at frequency 1,
deterministic at frequency 2); the input's is **fully deterministic**. Filtering
the output by the input's stochastic model therefore leaves it non-stationary at
frequency 1, and:

> «genera una serie todavía no estacionaria en la frecuencia uno y **una ccf muy
> poco informativa**.»

The fix is a deliberate split between the model used to IDENTIFY and the model
used to ESTIMATE:

> «Los parámetros estimados de estacionalidad determinista obtenidos en M3.Q se
> emplean para calcular la desviación del output de sus componentes
> deterministas, que posteriormente se filtra por el modelo univariante del
> input. … **No obstante, el modelo M2.Q, con estacionalidad mixta, se emplea
> como modelo univariante del ruido en cada una de las estimaciones
> eficientes.**»

An alternative output model, with the frequency-1 seasonality made
deterministic, exists only to make the CCF readable. The real model is used for
the fit.

### 2.5 Overparametrisation that is kept on purpose

In 6.4.4, μ and an intervention parameter correlate at **−.93**, and the
decision is to keep both:

> «Esta sobreparametrización **es necesaria**. Si se suprime el parámetro de
> intervención, muchos de los demás parámetros cambian significativamente… Si se
> suprime μ, la media de los residuos difiere de cero.»

So a high correlation is not automatically a defect to remove. It is a question
to answer by testing what breaks when each is dropped.

### 2.6 What a finished case reports

Every one of the six closes the same way, and only two of these are in drtran's
output today:

1. the **gain**, with its economic reading — "si ΔlnM1 aumenta un 1 %, lnQ a
   largo plazo aumenta un 3 %"  ✅ *(we report it since this session)*
2. the **mean lag** — "aproximadamente un año", "siete trimestres", "un año y
   medio"  ❌
3. the **irf/srf shape** read out loud — "todos los valores de la irf son
   positivos, por eso la srf es monótona creciente"  ~
4. the **% reduction in residual variance versus the univariate model** — 44 %,
   53 %, 23 %, 37 %  ❌
5. **how many intervention parameters it needed** versus the univariate — often
   one FEWER  ❌
6. any **change in the order of integration** — lnE is I(2) univariately and
   I(1) once ΔlnM1's effects are removed, which they flag as paradoxical and
   possible only in finite samples  ❌
7. the **LR test of the economic hypothesis**, using the final parametrisation
   with the restriction relaxed  ~ *(we do LR against the diagonal rung, which
   is the same machinery pointed at a different hypothesis)*

---

## 3. Brajín's formulas and rules

**The mean lag** (2.8, 2.9), defined only when the response is monotone:

```
l = Σ k·ν_k / Σ ν_k  =  ν'(B)/ν(B) evaluated at B = 1
```

**Overparametrisation is detected** "buscando parámetros redundantes, empleando
las formas factorizadas en factores simples irreducibles… Estos se revelan con
**errores estándar altos en relación con el valor estimado** y/o **correlaciones
altas entre parámetros estimados**".

**Overfitting is a required stage, with a direction**: add one parameter (two if
the operator has imaginary roots) "en direcciones en que se sospecha que puede
estar presente más estructura" — and she gives an example of a *reasoned*
direction: with an AR(2) with imaginary roots, add an MA(1) "para facilitar una
condición inicial paramétrica a la forma sinusoidal amortiguada".

**A model is adequate** when all of: the residual plot looks centred and
homoscedastic with no major anomalies; the residual mean is small relative to
its standard deviation; the ACF/PACF show no missing ARMA structure; and
**Ljung-Box Q does not exceed its degrees of freedom** — a blunter rule of thumb
than a p-value, and worth knowing as the school's own working criterion.

---

## 4. What to build, ranked

### Tier 1 — small, direct, and every case uses them

1. **Mean lag** with its standard error, beside the gain. ✅ implemented in
   `irf.py` (it shares the gain's Jacobian, so the delta method is exact rather
   than numerical). Formula in §3, and the
   delta method already gives the gain's error the same way. **Report it only
   when the response is monotone**, which is Brajín's own condition — an
   average of a sign-changing response is not a lag.
2. **Residual variance reduction versus the univariate model.** ✅ implemented
   — 36.7 % on the canonical case. The diagonal
   rung already computes the univariate fit, so this is a division. It is the
   school's headline answer to "was the transfer worth it", stated in the units
   an analyst cares about rather than in log-likelihood.
3. **`ω₀ ≈ −1` ⇒ the dead time is wrong.** ✅ implemented — **and testing it
   changed what it is.** Against known truth (generated with b=2, fitted at
   b=0,1,2,3) ω₀ came out −0.169, +0.036, +0.794, +0.471. The rule never fires,
   and it should not: those cases run on a **transformed output**, with the
   long-run restriction imposed by subtracting the input from it, and Muñoz's
   own algebra (§2.6 p. 37) gives that parametrisation's numerator as
   **ω\*(B) = ω_s(B) − δ_r(B)**. Since δ's leading coefficient is 1, an
   understated dead time leaves ω\*₀ = 0 − 1 = −1. **The −1 is the denominator
   showing through the subtraction**, not a property of transfer models.
   It ships, because where that parametrisation is used the reflex is real and
   valuable — but documented as conditional. Presenting it as a general
   dead-time test would be reading a 25-year-old reflex out of its setting.
4. **Implausible denominator.** ✅ implemented. Flag δ close to 1: report that the gain and the
   mean lag are then unreliable, as 6.4.5 found (both "excesivamente altos" and
   non-significant).
5. **The parameter correlation matrix.** ✅ implemented. We compute `cov`; report the largest
   off-diagonal correlations and flag any above .9 — with the 6.4.4 caveat that
   high correlation is a question, not a verdict: test what breaks when each is
   dropped.

### Tier 2 — procedural, changes what the assistant says

6. **The reformulation order** (§1.1) into `diagnose`'s branch: when the CCF and
   the residual ACF are both bad, fix the RELATION first. The asymmetry
   argument, stated, so the analyst knows why.
7. **An `overfit` tool.** After adequacy, add MA terms to `v(B)` one at a time
   and report whether they are significant and whether the estimation situation
   degrades. Every case does this; none of them skip it, even when nothing looks
   wrong.
8. **A `refine` path for identification**: estimate a generous pure-MA `v(B)`,
   show the weights, and read the decay pattern back to the analyst as evidence
   for or against a denominator. This is what the cases actually do, and it is
   strictly more informative than reading the CCF once.

### Tier 3 — real design work

9. **The identification/estimation model split** (§2.4). Needs an alternative
   output model with deterministic seasonality, used only for prewhitening. The
   payoff is that seasonal mismatch between input and output stops producing
   "una ccf muy poco informativa" — which today we would simply misread.
10. **Residual CCF spikes traced to observation pairs** (§2.3). `calibrate` does
    leave-one-out on one series; this needs the pair.
11. **Report the change in the integration order** of the noise versus the
    univariate model. A conceptual finding, and one of the most interesting
    results in the whole thesis.

### What NOT to build

The economic hypotheses themselves (monetary neutrality, gain = 0 or 1) are
domain content, not method. What is general is the **shape**: a theory-driven
restriction, imposed to make the parametrisation tractable, and **relaxed and
tested by LR at the end using the final parametrisation**. drtran already has
the machinery for both halves — `.cns` constraints and, since this session, an
LR against a restricted alternative. What is missing is only the documentation
that this is the intended workflow.

---

## 5. What changed on contact with the data

Tier 1 is implemented, in `drtran/school.py` (the computations, mute) and read
out in `mcp_server` (the narrative). One of the five did not survive contact
intact — see item 3 above: the `ω₀ ≈ −1` reflex turned out to be an artefact of
a specific parametrisation rather than a general diagnostic, which is only
visible if you test it against a case whose answer you already know. It is the
same lesson as everything else in this session: a rule that never fires and a
rule that fires wrongly are indistinguishable from the outside.

## 6. The one-line summary

drtran computes almost everything these theses read, and reads almost none of
it. The gap is not numerical: it is that a table of parameters with standard
errors is the *input* to the school's method, not its output.
