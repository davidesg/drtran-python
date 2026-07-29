# drtran (Python) — TODO

Puerto de `drtran` (C) reutilizando `fue` y `drvarma`. Ver README para el diseño,
el principio no negociable y el criterio de validación.

## Paso 0 — entrada — HECHO

- [x] **El `.pre` llega íntegro.** `fue.load()` lee `.pre`/`.inp`; verificado campo
      a campo (λ/d/D, refactor, mu0 + estimate_mu, ar/ma regulares y anuales con
      free-flags, ar_f/ma_f, deterministas con omega/omega_free, la serie).
      Decisión: **no se porta `fue_pre_reader.c`** (601 líneas). Duplicar el
      parser sería crear una segunda fuente de verdad que se desincroniza.
      Tests: `tests/test_pre_roundtrip.py` (8).
- [x] **Línea base univariante.** fue Python reproduce el caso canónico:
      φ 0.402839 / 0.299193, logL −7.3917 / −760.0326, y la SUMA da
      **−767.424341** (dif 7.6e-09), que es la diana de la conjunta.
      Tests: `tests/test_baseline_univariante.py` (3).
- [ ] **Round-trip de escritura.** `model.write_pre()` exige el modelo ajustado,
      así que el ciclo `estimar → .pre → releer` no se ha probado todavía. Es lo
      que cierra la continuidad hacia el siguiente escalón. No bloquea el cast.

## Paso 1 — el cast — HECHO (caso diagonal)

- [x] **Cast diagonal** (`cast.py`). Se reutiliza el cast univariante de fue por
      serie (`build_est_spec` + `cast_us_py`), que ya devuelve `w` con Box-Cox,
      diferencias y deterministas restados — es `build_stationary_series` del C.
      drtran sólo ENSAMBLA: Φ y Θ diagonales por bloques, μ por serie, w alineadas
      por el final, Q normalizada.
- [x] **Verosimilitud CONCENTRADA.** `est()` no estima la escala: descompone
      Σ = sigma2·Q con Q[1][1]=1 y concentra sigma2. Evaluar `elf_varma` con un Σ
      absoluto da otra cosa (pasar la identidad daba −7802 en vez de −767).
      Se usa `_elf_f1f2` de drvarma + la fórmula de `drvmlest.c:est [4]`.
- [x] **Semilla de las razones de varianza.** No se dejan en cero: las escalas
      difieren ×1098 en el caso canónico y arrancar en 1 deja el punto inicial en
      −1371. Se calculan con el mismo `elf`, m=1, sobre las semillas del `.pre`.
- [ ] Transferencias: ω/δ por enlace, `compute_irf`, y el cast EMPOTRADO
      (`embed_varma`), que mete la transferencia dentro del VARMA sin restar nada
      y deja la inicialización pre-muestral a la verosimilitud exacta.
- [ ] `expand_params`: parámetros fijos, COMPARTIDOS, productos y combinaciones
      lineales (la tabla de slots del `.cns`, por nombres: `omega1[1]`,
      `theta_2[B^1]`, `q[5,2]`).

## Paso 2 — la puerta — SUPERADA

- [x] **Conjunta diagonal ≡ fue por separado.** logL = **−767.424341**,
      diferencia 3.9e-07. Y se alcanza YA en las semillas del `.pre`, lo que
      confirma empíricamente por qué el C reporta `termcode 3` en este escalón:
      las semillas son el óptimo. Tests: `tests/test_cast_diagonal.py` (4).

## Paso 3 — lo demás

- [ ] Transferencia ω(B)/δ(B)·B^b: identificación por preblanqueo + CCF.
- [ ] Diagnósticos de `diagnose.c`: portmanteau de la transferencia (k ≥ 0,
      incluye el contemporáneo) y **portmanteau de exogeneidad** (k < 0, detecta
      retroalimentación Y → X).
- [ ] Previsión y CLI.

## Heredado del C — vigilar en el puerto

- [ ] **El optimizador se degrada con `refactor=1`.** En el C cuelga >2 min sin
      converger con Δlog ~0.002, y converge en 23 iteraciones con refactor=100.
      Causa: el paso de diferencias finitas de `cdgrad` (~6e-6 absoluto) tiene
      relación señal/paso pésima a escala cruda. El puerto hereda `_qnewt` de
      drvarma, así que probablemente hereda la fragilidad. `drtran.check_scale()`
      ya avisa; falta decidir si además condicionar internamente.
- [ ] **`termcode 3` NO es fallo aquí.** Arrancando en las semillas del `.pre`
      (que en el escalón diagonal ya SON el óptimo) la búsqueda lineal no puede
      mejorar y para. Clasificación correcta (hito M1 del C): 1-2 convergencia,
      **3 parada sin mejora**, 4-5 fallo real. El test adecuado es perturbar las
      preestimaciones y comprobar que converge por gradiente al mismo punto.
      NB: multiart (drvarma) rechaza termcode 3 en su búsqueda de orden, donde las
      semillas son OLS. Los dos criterios conviven, pero conviene no confundirlos.
