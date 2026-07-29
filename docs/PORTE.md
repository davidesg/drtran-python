# El porte de drtran a Python — registro del proceso

Este documento cuenta **cómo** se hizo el porte y **con qué evidencia** se dio por
bueno cada escalón. `../TODO.md` dice qué está hecho y qué falta; el `README.md`
dice qué es el programa. Esto es el diario de a bordo: las decisiones, las cifras
que las sostienen y las trampas en las que no hay que volver a caer.

Escrito el 2026-07-29, con los pasos 0, 1 y 2 cerrados.

---

## 1. El método: una escalera de puertas, no un porte línea a línea

El porte **no** traduce el C función a función. Traduce el *cast* —el mapa de
parámetros a estructura VARMA— y reutiliza todo lo demás, que ya existe en Python.
Cada escalón tiene una **puerta**: una igualdad numérica que debe cumplirse antes
de seguir. Si no cierra, no se avanza; y la culpa es siempre del cast.

> **Principio no negociable.** El `elf` de drvarma se usa **tal cual**: no se
> modifica, no se parchea, no se caso-especializa. Cualquier discrepancia con fue
> es un bug del cast de drtran, **nunca** de `elf`.

Ese principio es lo que convierte el porte en un instrumento de medida: cuando algo
no cuadra, el sitio donde buscar está acotado de antemano.

La segunda regla, heredada del trabajo con drvarma: **antes de declarar un defecto,
construir un caso con respuesta conocida.** Las tres discrepancias que aparecen en
§5 se cerraron así, no leyendo código.

## 2. Qué se reutiliza y qué se porta

De las **12.615** líneas de C, la mayor parte no se porta: se reutiliza.

| C | destino | líneas |
|---|---|---|
| `elfvarma.c` + `drvmlest.c` + `qnewtopt.c` + `nlatools.c` | ya están en **drvarma** Python | 3.350 |
| `fue_pre_reader.c` | `fue.load()` — **no se porta** (ver §3.1) | 601 |
| `gnuplot_i.c` + `fuf_graphic.c` | matplotlib | 1.860 |
| `tran_shootx.c` | `cast.py` + `embed.py` | 668 |
| `diagnose.c` | parcialmente en `identify.py`; el resto pendiente | 1.687 |
| `drtran.c` | CLI y orquestación; pendiente, se encoge mucho | 4.201 |

El puerto ocupa hoy **1.243 líneas** de Python en `src/drtran/` (`cast.py` 335,
`identify.py` 321, `embed.py` 263, `estimate.py` 171, `pre.py` 109) y **593** de
tests, con **52 tests**.

## 3. Los escalones, con su evidencia

### Paso 0 — la entrada llega íntegra

**Puerta:** que el `.pre` que escribe fue llegue al cast **campo a campo**, y que
fue Python reproduzca los univariantes de referencia.

Verificado: λ/d/D, `refactor`, μ **con su flag de fijado**, órdenes y coeficientes
AR/MA regulares y anuales con sus *free-flags*, AR(2)/MA(2) de frecuencia fija, los
deterministas con sus ω y sus flags, y la serie. Caso canónico: φ 0.402839 /
0.299193, logL −7.3917 / −760.0326, **suma −767.424341** (dif. 7.6e-09 con fue).

Tests: `test_pre_roundtrip.py` (8), `test_baseline_univariante.py` (3).

#### 3.1. Decisión: no se porta `fue_pre_reader.c`

601 líneas de parser. Duplicarlo sería crear una **segunda fuente de verdad** que
se desincroniza en cuanto fue cambie una línea de su formato. `fue.load()` ya lee
`.pre` e `.inp`. El `.pre` es el contrato entre los dos programas, y un contrato
con dos lectores distintos no es un contrato.

Coste conocido: `model.write_pre()` exige el modelo ajustado, así que el ciclo
`estimar → .pre → releer` sigue sin probarse. Es lo que cierra la continuidad hacia
el siguiente escalón de la escalera metodológica; no bloquea el cast.

### Paso 1 — el cast diagonal

`cast.py` reutiliza el cast **univariante** de fue por serie (`build_est_spec` +
`cast_us_py`), que ya devuelve la `w` con Box–Cox aplicado, diferencias tomadas y
deterministas restados — es el `build_stationary_series` del C. drtran sólo
**ENSAMBLA**: Φ y Θ diagonales por bloques, μ por serie, las `w` alineadas por el
final y Q normalizada.

Dos cosas que no son obvias y costaron la primera tarde:

- **La verosimilitud es CONCENTRADA.** `est()` no estima la escala: descompone
  Σ = σ²·Q con Q[1][1] = 1 y concentra σ². Evaluar `elf_varma` con un Σ absoluto da
  otra cosa — pasar la identidad daba **−7802** en vez de −767. Se usa `_elf_f1f2`
  de drvarma más la fórmula de `drvmlest.c:est [4]`.
- **Las razones de varianza no se dejan en cero.** Las escalas difieren ×1098 en el
  caso canónico; arrancar en 1 deja el punto inicial en **−1371**. Se calculan con
  el mismo `elf`, m = 1, sobre las semillas del `.pre`.

### Paso 2 — LA PUERTA

**Conjunta diagonal ≡ fue por separado.** Con estructura diagonal la verosimilitud
exacta se factoriza, así que la conjunta debe reproducir la **suma** de las
univariantes.

| | logL |
|---|---|
| ES_CPI_m10 (fue) | −7.3917 |
| WTI_ar1 (fue) | −760.0326 |
| **suma = diana** | **−767.424341** |
| **conjunta diagonal (puerto)** | **−767.424341** (dif. 3.9e-07) |

Y se alcanza **ya en las semillas del `.pre`**, lo que confirma empíricamente por
qué el C reporta `termcode 3` en este escalón: las semillas *son* el óptimo.

Tests: `test_cast_diagonal.py` (4).

### Paso 3 — las transferencias

Dos casts, como en el C, y **no miden lo mismo**:

- **Por resta.** `Link(out, inp, b, r, s)`, `compute_irf` y la resta a la salida:
  la serie 1 del VARMA pasa a ser el ruido `N_t = w_Y − Σⱼ transferenciaⱼ`. Modela
  el RUIDO y **trunca** al principio de la muestra.
- **Empotrado** (`embed.py`), el **defecto**, como en el C. Álgebra de polinomios:
  fila i = diagonal φᵢ·Dᵢ, fuera de diagonal −φᵢ·ωₖ·B^bₖ·(Dᵢ/δₖ), MA Dᵢ·θᵢ, con
  Dᵢ = Πₖ δₖ de los enlaces entrantes; series **sin restar**. Modela la serie
  OBSERVADA, sin truncar.

Que el empotrado **no** dé mayor verosimilitud que el de resta no es un defecto:
son objetivos distintos sobre datos distintos. El C muestra el mismo patrón.

Prueba de consistencia en las dos: **con ω = 0 la verosimilitud es exactamente la
del diagonal, diferencia 0.0.** Es la misma prueba de la puerta, un escalón más
arriba.

La estimación conjunta (`estimate.py`) usa el objetivo escalado de Mauricio (1995,
ec. 3.5) normalizado a 1 en x₀ — en multivariante (f1/f1₀)^m·(f2/f2₀), porque
ll = C − 0.5n(m·log f1 + log f2) — minimizado con el `raxopt` de drvarma. Un punto
rechazado devuelve 1.0 y el optimizador se aleja.

Medido en el caso canónico Y ← X, (b,r,s) = (0,0,0): **ω₀ = 0.016002**, logL
−736.774 frente a −767.424 del diagonal, **LR = 61.3** (1 gl, p = 4.9e-15), por
gradiente en 21 iteraciones.

### Paso 4 — identificación de (b, r, s)

`identify.py` porta `prewhiten_and_identify`: preblanquea la entrada con SU ARMA,
aplica **el mismo filtro** a la salida, calcula la CCF y lee ν(k) = r(k)·s_β/s_a.

Homologa con el binario: banda 0.13640, r(0) = 0.492, r(1) = 0.310, r(2) = 0.025,
r(−1) = −0.077, r(−6) = −0.128, y la misma propuesta **b = 0, r = 0, s = 1**.
Exogeneidad por portmanteau sobre k < 0: **Q(24) = 18.2969, p = 0.7884**, idéntico
al C — el divisor de `ChiTestC` es n−i+1, no n−i.

Se replican **a propósito** las dos decisiones del C que evitan disparates:

1. la estructura es el **bloque CONTIGUO** desde b (hay un pico significativo en el
   lag 24 que NO entra en la propuesta: con bandas al 5 % se espera 1 de cada 20
   fuera);
2. la exogeneidad se juzga por **portmanteau**, no contando cuántos cruzan.

Revisado contra Haugh–Box (1977) y Tsay (1985). Tests: `test_identificacion.py`
(12), incluida una transferencia sintética con retardo conocido (b = 3).

## 4. Decisiones de porte que no son traducción

| decisión | por qué |
|---|---|
| No portar el lector de `.pre` | una sola fuente de verdad del contrato (§3.1) |
| Verosimilitud **concentrada** | es lo que hace `drvmlest.c:est`; con Σ absoluto sale −7802 |
| Semilla de las razones de varianza | escalas ×1098; arrancar en 1 deja el inicio en −1371 |
| Convención **BJR** de signos: ω(B) = ω₀ − ω₁B − … | el líder suma, los demás **restan**; es la del cast del C |
| `normalize_phi0` (Φ₀⁻¹ por la izquierda) | una transferencia contemporánea mete ω₀ en el retardo cero y `elf` exige Φ(0) = I |
| **μ es la media, no un intercepto** | coherencia con fue; ver §5.2 |
| Empotrado por defecto | como en el C; no trunca la muestra |

## 5. Lo que el porte le encontró al C

El porte resultó ser un banco de pruebas: tres discrepancias, y en dos de ellas el
que estaba mal era el original.

### 5.1. La respuesta al impulso invertía el signo (bug del C)

El informe de `drtran.c:1371` sumaba los términos no líderes del numerador donde el
cast resta. Publicaba ganancia 0.005610 (= ω₀+ω₁) donde la real es ω₀−ω₁ =
**0.027195**: un factor de casi 5, en todo s > 0. **La documentación tenía el mismo
error**, así que código y documento se confirmaban mutuamente. Corregidos los dos
en el repo C, más una sección nueva de la batería que fija el signo.

### 5.2. La media del cast empotrado (bug del C)

El C hacía, en orden topológico, `mu_i += (Σₖ ωₖ/δ(1))·mu_inp`. La sospecha anotada
en su TODO era un **signo**; el defecto era la **parametrización entera**.

μ es LA MEDIA de la serie, no un intercepto. Box–Jenkins escribe el modelo en
**desviaciones**,

```
(w_Y − μ_Y) = ν(B)·(w_X − μ_X) + N_t     ⇒     E[w_Y] = μ_Y
```

así que la media de la salida **no hereda nada** de la entrada. Multiplicando por
δ(B),

```
φ_Y·δ·(w_Y − μ_Y) − φ_Y·ω·B^b·(w_X − μ_X) = δ·θ_Y·a_Y
```

que es exactamente la fila 1 de Φ(B)(w − μ) = Θ(B)a con μ = (μ_Y, μ_X). **No hay
término que añadir.**

La alternativa (`w_Y = c + ν(B)·w_X + N`) es la parametrización con **intercepto**.
Son la misma familia reparametrizada **mientras μ_Y sea libre** — verificado, mismo
óptimo a 1e-12. Divergen cuando μ_Y está **fijada**: en desviaciones μ_Y = 0
significa E[w_Y] = 0; con intercepto, E[w_Y] = ν(1)·μ_X ≠ 0. Manda la coherencia
con fue: si fue fijó la media en cero es porque la serie no tiene deriva.

**Por qué sobrevivió tanto:** el caso canónico tiene entrada WTI con μ = 0, y ahí
el término vale cero con cualquier convención. Hace falta **una entrada con media
libre** para verlo. Corregido en los dos lados (`embed.py:cast_embedded` y
`tran_shootx.c:build_embedded_varma`) y documentado en la nota técnica del C
(observación *The means are means, not intercepts*).

### 5.3. La Cholesky modificada (bug del porte de drvarma)

El cast empotrado estuvo bloqueado: `elf` lo rechazaba con `ifault = 3`. La causa
no estaba en drtran: `_chol_lower` de drvarma usaba `np.linalg.cholesky`
(**estricta**) donde el C usa la Cholesky **MODIFICADA** (`nlatools.c:choldcp`),
que acepta matrices semidefinidas. Como el empotrado produce Φ_p singular **por
construcción**, la estricta lo tumbaba.

Arreglado en drvarma (`fix(as311): porta fielmente la Cholesky MODIFICADA del C`),
y de paso cerró los tres tests de paridad con el C que su suite arrastraba.

## 6. Homologación con el binario

`test_homologacion_c.py` (12) **relanza el binario en vivo** en vez de comparar
contra referencias guardadas, para no arrastrar cifras obsoletas cuando el C
cambia.

Caso canónico `ES_CPI_m10` ← `WTI_ar1` (entrada con μ = 0):

| (b,r,s) | cast por resta | cast empotrado |
|---|---|---|
| (0,0,0) | −736.774158 | −736.774158 |
| (0,1,0) | −721.720197 | −721.801539 |
| (0,0,1) | −718.183933 | −718.287406 |
| (1,1,1) | −756.528944 | −756.602851 |
| diagonal | −767.424341 | −767.424341 |

Caso que discrimina la convención de medias, `ES_CPI_m10` ← `DE_CPI_mar3sar`
(**las dos con media libre**), cast empotrado:

| | drtran C | drtran Python |
|---|---|---|
| (0,0,0) | 24.408974 | 24.408974 |
| (0,0,1) | 35.487981 | 35.487981 |
| (0,1,1) | 35.555382 | 35.555382 |

Y la cadena entera en el diagonal de ese mismo caso: fue C (−7.3917271 +
11.2056885) = **3.8139613** = fue Python = drtran C (3.813961) = drtran Python
(3.8139611).

## 7. Cómo reproducirlo

```sh
# el puerto
cd ~/Dropbox/SRC/drtran-python && python -m pytest -q          # 52 passed, ~3 min

# el original
cd ~/Dropbox/SRC/drtran && make && ./test_battery.sh           # 296 PASS, 0 FAIL

# una comparación puntual
./bin/drtran tests/cases/ES_CPI_m10.pre tests/cases/DE_CPI_mar3sar.pre \
             -b 0 -r 0 -s 1 -V -o /tmp/t.out | grep '^Log-likelihood'
```

### Trampas al comparar con el binario

- **No extraigas la verosimilitud con `grep -oE '[0-9]+\.[0-9]+' | tail -1`.** Eso
  captura la **p de exogeneidad** del bloque de diagnósticos, no el logL. Un falso
  positivo de divergencia C ↔ Python vino exactamente de ahí: parecía 0.1630 contra
  35.487981 cuando los dos daban 35.487981. Filtra por `^Log-likelihood`.
- `-V` es el cast **empotrado** y `-S` el de **resta**; comparar uno con otro mide
  el truncamiento, no un error.
- Las series con μ = 0 **no discriminan** la convención de medias (§5.2).

## 8. Lo que falta

- **Round-trip de escritura**: `estimar → .pre → releer` (§3.1).
- **Red de transferencias** (`-n`), DAG y `expand_params`: parámetros fijos,
  **compartidos**, productos y combinaciones lineales, con la tabla de slots del
  `.cns` por nombres (`omega1[1]`, `theta_2[B^1]`, `q[5,2]`). Dianas del C:
  m6 diagonal **−1709.511575**, red libre **−1697.613401**.
- **Diagnósticos** de `diagnose.c`: portmanteau de la transferencia (k ≥ 0, incluye
  el contemporáneo) y de exogeneidad (k < 0, detecta retroalimentación Y → X).
- **Previsión y CLI.**

### Heredado del C — vigilar

- **El optimizador se degrada con `refactor = 1`.** En el C cuelga > 2 min sin
  converger con Δlog ~0.002, y converge en 23 iteraciones con `refactor = 100`. El
  paso de diferencias finitas de `cdgrad` (~6e-6 absoluto) tiene relación
  señal/paso pésima a escala cruda. El puerto hereda `_qnewt` de drvarma, así que
  probablemente hereda la fragilidad. `check_scale()` ya avisa; falta decidir si
  además condicionar internamente.
- **`termcode 3` NO es fallo aquí.** Arrancando en las semillas del `.pre` (que en
  el diagonal ya SON el óptimo) la búsqueda lineal no puede mejorar y para.
  Clasificación correcta: 1–2 convergencia, **3 parada sin mejora**, 4–5 fallo real.
  El test adecuado es perturbar las preestimaciones y comprobar que converge por
  gradiente al mismo punto. NB: multiart (drvarma) **sí** rechaza termcode 3 en su
  búsqueda de orden, donde las semillas son OLS. Los dos criterios conviven; no
  confundirlos.
