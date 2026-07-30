# drtran (Python) — TODO

Puerto de `drtran` (C) reutilizando `fue` y `drvarma`. Ver README para el diseño,
el principio no negociable y el criterio de validación, y
[`docs/PORTE.md`](docs/PORTE.md) para el **registro del proceso**: las decisiones
que no son traducción, las cifras de homologación y los defectos que el porte le
encontró al original.

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
- [x] **La MEDIA es la media, no un intercepto** (`embed.py`). El modelo se escribe
      en DESVIACIONES, (w_Y − μ_Y) = ν(B)(w_X − μ_X) + N_t ⇒ E[w_Y] = μ_Y: la
      salida **no hereda** nada de la entrada, y multiplicando por δ(B) sale la
      fila 1 de Φ(B)(w − μ) = Θ(B)a con μ = (μ_Y, μ_X), sin término adicional.
      Antes se hacía `MU[i] += (ω(1)/δ(1))·MU[inp]` en orden topológico, que es la
      parametrización con INTERCEPTO. Misma familia mientras μ_Y sea libre (mismo
      óptimo a 1e-12); divergen con μ_Y FIJADA. Manda la coherencia con fue: el μ
      del `.pre` es la media que fue estimó, y un cero significa que la serie no
      tiene deriva. **El mismo defecto estaba en el C** y se corrigió allí
      (`tran_shootx.c:build_embedded_varma`, commit `fix(cast)`), donde figuraba
      como sospecha de signo en su TODO.
      No se ve en el caso canónico: su entrada (WTI) tiene μ = 0 y el término vale
      cero con cualquier convención. Hace falta la entrada con **media libre**.
      Verificado con `ES_CPI_m10` ← `DE_CPI_mar3sar`: fue C = fue Python = drtran C
      = drtran Python = 3.8139613 en el diagonal, y con transferencia C ≡ Python en
      (0,0,0) 24.408974, (0,0,1) 35.487981, (0,1,1) 35.555382.
      Tests: `tests/test_transferencia.py`, dos nuevos.
- [x] **RED de transferencias, DAG y `expand_params` — HECHO y homologado.**
      Tres piezas nuevas:
      * `network.py` — el `.dag` (`SALIDA <- ENTRADA b r s`, series por NOMBRE) y
        el rechazo de ciclos, que **dice cuál es el ciclo**. Con un ciclo no hay
        orden topológico: dejaría de ser un DAG recursivo para ser un sistema de
        ecuaciones simultáneas, que no es lo que el cast representa.
      * `slots.py` — la tabla de slots con los NOMBRES del C (`omega1[1]`,
        `theta_2[B^1]`, `q[5,2]`, `log(var3/var1)`), el `.cns` y `expand_params`
        con las cinco naturalezas: libre, fijo, **compartido**, **producto**
        (`x = -y * z`) y **combinación lineal** (`x = t1 + t2 - t3`, cada término
        slot o slot·slot). El gradiente no necesita regla de la cadena: `expand`
        va DENTRO del objetivo y `cdgrad` lo deriva por diferencias finitas, la
        misma decisión del C.
      * covarianza **no diagonal** en los dos casts (`build_sigma`): las `q[i,j]`
        entran siempre al mapa pero **nacen fijas en cero**, y liberarlas es una
        decisión del analista (`q[5,2] = free`), no un interruptor global — el
        legacy m6-1 libera TRES de sus 15. Fuera de la región PSD se rechaza el
        punto (objetivo 1.0) en vez de evaluar lo imposible.

      El orden de los slots NO es el del C (él agrupa por clase, aquí por serie,
      porque el bloque univariante lo produce fue), pero los NOMBRES sí — y el
      `.cns` va por nombres. `build_slots` verifica el total contra
      `cast_spec.npar`: si la enumeración se separara de la de fue, los nombres
      dejarían de corresponder a las posiciones **en silencio**.

      Homologado con el binario sobre una red de 5 series con cadena
      EC → EU → EP, una entrada con dos salidas, un denominador r=1 y dos
      covarianzas libres: **−1434.696068** (dif 1.9e-10); con un producto y una
      combinación lineal, **−1439.505804** (dif 9.4e-08), y `expand` reconstruye
      los slots derivados desde sólo los libres (dif 3e-07, que es el redondeo a
      6 decimales del informe del C). El **optimizador** también llega: red de
      3 series, 24 libres, **−912.244333 en 180 iteraciones** contra las 181 del
      C. Tests: `tests/test_red.py` (15).
- [x] **El m6 canónico — VALIDADO** (tras arreglar fue, 2026-07-30). Estuvo
      bloqueado porque **fue Python degradaba el determinista `compimp` a un
      `pulse`**: el impulso compensado es +1 en la fecha y **−1 en la siguiente**,
      y el lector se comía el −1. Sólo `M6_EI.pre` lo usa, y sólo por eso:
      evaluando en el óptimo de fue C, cinco de las seis series clavaban a 5e-8 y
      EI daba −292.495 en vez de −290.613. Arreglado en **fue 0.1.9**
      (BUG-0006/0007); ahora el puerto reproduce las dos dianas del C:
      **diagonal −1709.511575** (dif 5.0e-07) y **red libre −1697.613401**
      (dif 5.9e-07), más las variantes con productos y con la estructura completa.
      Tests: `tests/test_red.py`. Ver `docs/PORTE.md` §5.4.

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
