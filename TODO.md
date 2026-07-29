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

## Paso 1 — el cast (`tran_shootx.c`, 660 líneas)

- [ ] Portar el cast paramétrico: vector de parámetros → estructura VARMA.
      Patrón a seguir: `fue/cast_us.py` (`build_est_spec(model)` precomputa lo
      fijo; `cast_us_py(x, spec)` mapea x → (p, q, phi, theta, mu, w, ifault)).
- [ ] Recast del par (Y, X) como VARMA bivariante diagonal, según BRIDGE_DESIGN:
      serie 1 = `w_Y − transfer_t` (el ruido N_t) con el ARMA de Y; serie 2 =
      `w_X` con el ARMA de X; covarianza diagonal; el acoplamiento entero vive en
      `transfer_t = Σⱼ νⱼ · w_X(t−j)`. Con ω = 0 el modelo se parte en dos
      univariantes independientes — y ahí está la prueba de que el puente es
      correcto.

## Paso 2 — la puerta

- [ ] **Conjunta diagonal ≡ fue por separado**, logL = −767.424341. Sin esto no
      se sigue: cualquier discrepancia es un bug del cast, nunca de `elf`.

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
