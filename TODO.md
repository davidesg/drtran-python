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
- [x] **Transferencias por RESTA.** `Link(out, inp, b, r, s)`, `compute_irf`
      (convención BJR: ω(B)=ω₀−ω₁B−…, el líder suma y los demás restan) y la
      resta a la salida: la serie 1 del VARMA pasa a ser el ruido
      N_t = w_Y − Σⱼ transferenciaⱼ. Verificado: con ω=0 la verosimilitud es
      EXACTAMENTE la del diagonal (diferencia 0.0).
- [x] **Estimación conjunta** (`estimate.py`): objetivo escalado de Mauricio
      (1995 ec. 3.5) normalizado a 1 en x₀ — en multivariante (f1/f1₀)^m·(f2/f2₀),
      porque ll = C − 0.5n(m·log f1 + log f2) — minimizado con el `raxopt` de
      drvarma. Un punto rechazado devuelve 1.0 y el optimizador se aleja.
      Medido en el caso canónico Y←X (b=0,r=0,s=0): **ω₀ = 0.016002**,
      logL −736.774 frente a −767.424 del diagonal, **LR = 61.3 (1 gl,
      p=4.9e-15)**. Converge por gradiente en 21 iteraciones.
- [x] **Cast EMPOTRADO (`embed.py`) — DESBLOQUEADO y homologado.**
      Álgebra de polinomios: fila i = diagonal φᵢ·Dᵢ, fuera de diagonal
      −φᵢ·ωₖ·B^bₖ·(Dᵢ/δₖ), MA Dᵢ·θᵢ, con Dᵢ = Πₖ δₖ de los enlaces entrantes;
      medias en orden topológico; series SIN restar; y `normalize_phi0`
      (Φ₀⁻¹ por la izquierda) porque una transferencia contemporánea mete ω₀ en
      el retardo cero. Verificado: sin enlaces y con ω=0 coincide EXACTAMENTE
      con el diagonal.
      Estuvo bloqueado por un fallo de port en drvarma: `_chol_lower` usaba
      `np.linalg.cholesky` (estricta) donde el C usa la Cholesky MODIFICADA
      (`nlatools.c:choldcp`), que acepta matrices semidefinidas. Como el
      empotrado produce Φ_p singular por construcción, `elf` lo rechazaba con
      ifault=3. **Arreglado en drvarma** (commit `fix(as311)`), y de paso cerró
      los tres tests de paridad con el C que arrastraba su suite.
      Homologa con el binario (`-V`) a ~1e-7: (0,0,0) −736.774158,
      (0,1,0) −721.801539, (0,0,1) −718.287406, (1,1,1) −756.602851.
      `fit(..., embed=True)` es ahora el DEFECTO, como en el C.
      **Ojo:** el empotrado NO da mayor verosimilitud que el de resta — las dos no
      miden lo mismo (el de resta modela el RUIDO y trunca; el empotrado modela la
      serie OBSERVADA). El C muestra el mismo patrón.
- [x] **Homologación con el binario C.** El cast por resta reproduce el
      `drtran` compilado a ~1e-7 en cuatro combinaciones de b/r/s:
      (0,0,0) −736.774158, (0,1,0) −721.720197, (0,0,1) −718.183933,
      (1,1,1) −756.528944; y el diagonal −767.424341. Un test relanza el binario
      en vivo para no arrastrar referencias obsoletas.
      Tests: `tests/test_homologacion_c.py` (7).
- [x] **Identificación de (b, r, s) por preblanqueo + CCF** (`identify.py`).
      Puerto de `prewhiten_and_identify`. Preblanquea la entrada con SU ARMA,
      aplica EL MISMO filtro a la salida, calcula la CCF y lee ν(k)=r(k)·s_β/s_a.
      Homologa con el binario: banda 0.13640, r(0)=0.492, r(1)=0.310,
      r(2)=0.025, r(−1)=−0.077, r(−6)=−0.128, y la misma propuesta **b=0 r=0 s=1**.
      Exogeneidad por portmanteau sobre k<0: **Q(24)=18.2969, p=0.7884**, idéntico
      al C (el divisor de `ChiTestC` es n−i+1, no n−i).
      Se replican las dos decisiones del C que evitan disparates: la estructura es
      el **bloque CONTIGUO** desde b (hay un pico significativo en el lag 24 que NO
      entra en la propuesta — con bandas al 5 % se espera 1 de cada 20 fuera), y la
      exogeneidad se juzga por portmanteau, no contando cuántos cruzan.
      Tests: `tests/test_identificacion.py` (9), incluida una transferencia
      sintética con retardo conocido (b=3).
- [ ] Red de transferencias (`-n`), DAG y `expand_params` (fijos, compartidos,
      productos, combinaciones lineales). Objetivos del C para cuando llegue:
      m6 diagonal −1709.511575, red libre −1697.613401.
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
