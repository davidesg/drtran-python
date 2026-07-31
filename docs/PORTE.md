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

El puerto ocupa hoy **2.229 líneas** de Python en `src/drtran/` (`slots.py` 440,
`cast.py` 373, `identify.py` 321, `netid.py` 315, `embed.py` 262,
`estimate.py` 215, `network.py` 140, `pre.py` 109) y **1.361** de tests, con
**77 tests**.

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

### Paso 5 — la red, el DAG y `expand_params`

Hasta aquí el modelo era una salida y sus entradas. La **red** es lo general: una
serie puede recibir transferencias y ser a la vez entrada de otra, que es lo que
son de verdad los sistemas de la escuela (m6-1: EC → EU → EI → EP, más EC → EP).
Y con la red llega lo que hacía a mano el `shootx` del legacy: parámetros
compartidos entre la transferencia y el ARMA de la entrada, numeradores
factorizados, factores fijos.

Son tres piezas, deliberadamente separadas:

| pieza | fichero | qué dice |
|---|---|---|
| **el grafo** | `.dag` → `network.py` | quién mueve a quién, con qué (b, r, s) |
| **los parámetros** | `.cns` → `slots.py` | qué es libre, fijo, compartido o una expresión |
| **la contemporaneidad** | `.cns`, `q[i,j]` | qué se mueve junto en el mismo instante |

Separar el grafo de los parámetros no es orden por el orden: el DAG dice
**dinámica con retardo** y Σ dice **simultaneidad**. Mezclarlas es justo el error
que el aviso de casi-colinealidad del C persigue — una transferencia
contemporánea (b=0) y la covarianza de esas dos innovaciones explican lo mismo en
el retardo cero.

**El `.dag`** son líneas `SALIDA <- ENTRADA b r s`, con las series por su
**nombre**, no por su posición: un `.dag` no debe depender del orden de la línea
de órdenes. Un ciclo se rechaza, y el mensaje **dice cuál es** — sin orden
topológico el sistema deja de ser un DAG recursivo y pasa a ser un modelo de
ecuaciones simultáneas, que no es lo que este cast representa.

**La tabla de slots** es el DSL. Cada posición del vector completo tiene un nombre
estable y una de cinco naturalezas: `free`, `fixed`, `alias` (COMPARTIDO),
`product` (`x = -y * z`) y `lincomb` (`x = t1 + t2 - t3`, con cada término un slot
o un producto de dos). El optimizador ve sólo los libres:

```
xfree ──expand──▶ xfull ──cast──▶ Φ, Θ, μ, w, Σ ──elf──▶ ℓ
```

`expand` va **dentro** del objetivo, así que el gradiente sale por diferencias
finitas sin regla de la cadena: añadir expresiones al DSL no toca el optimizador.
Es la decisión del C, y es la razón de que el producto y la combinación lineal
cupieran sin tocar `_qnewt`.

Dos cosas del diseño que conviene no perder:

- **El orden de los slots no es el del C, los nombres sí.** El C agrupa por clase
  (todos los ARMA, luego todos los deterministas); aquí se agrupa por serie,
  porque el bloque univariante lo produce `fue._build_initial_x` y el orden lo
  manda fue. Da igual, porque **el `.cns` va por nombres** — los mismos `.cns` del
  repo C se leen aquí. Lo que sí se comprueba es que el total cuadre con
  `cast_spec.npar`: si las dos enumeraciones se separaran, los nombres dejarían de
  corresponder a las posiciones y el `.cns` restringiría el parámetro equivocado,
  **en silencio**.
- **Las covarianzas nacen fijas en cero.** Entran siempre al mapa, pero la
  covarianza diagonal es el caso por defecto y liberar una es una decisión del
  analista (`q[5,2] = free`), no algo que se active en bloque: el legacy m6-1
  libera **tres** de sus quince. Fuera de la región donde Q es definida positiva
  el punto se rechaza (el objetivo devuelve 1.0 y la búsqueda se aleja), que es la
  estrategia de Mauricio (1995 §3); esa frontera no ha mordido nunca en los casos
  reales.

Homologación, sobre cinco series de m6 con cadena EC → EU → EP, una entrada con
dos salidas, un denominador r=1 y dos covarianzas libres:

| | drtran C | puerto | dif |
|---|---|---|---|
| red libre (40 libres de 48 slots) | −1434.696068 | −1434.696068 | 1.9e-10 |
| + producto + combinación lineal (38 libres) | −1439.505804 | −1439.505804 | 9.4e-08 |

y `expand` reconstruye los slots derivados partiendo **sólo de los libres** del
óptimo del C (dif 3e-07, que es el redondeo a 6 decimales de su informe). Eso
prueba el cast; que la **búsqueda** llegue es otra cosa, y se prueba aparte: red
de 3 series con 24 libres, el puerto converge a **−912.244333 en 180 iteraciones**
contra las 181 del C.

### Paso 6 — identificar la red

Cerrado el escalón diagonal, la escalera dice: **leer las CCF de sus residuos**
para descubrir las relaciones dinámicas del sistema (Muñoz Polo 2001, §2.6).
`netid.py` lo porta:

| k | lectura | propuesta |
|---|---|---|
| k > 0 | a_i antecede a a_j | enlace i → j, con b y s del bloque contiguo |
| k < 0 | a_j antecede a a_i | enlace j → i |
| k = 0 | contemporáneo | liberar q[i,j] |
| ambos | retroalimentación | no cabe en un DAG: se toma el dominante, avisando |

Los residuos **no se recalculan**: los devuelve el mismo `elf` que puntúa la
verosimilitud (`atf=True`), así que son los exactos, con su inicialización
pre-muestral. Reconstruirlos con un filtro a mano habría sido crear una segunda
fuente de verdad para algo que ya existe — el mismo criterio que con `fue.load()`
(§3.1) y con `cast_us_py`.

Homologa con el binario **línea por línea** en m6: las tres covarianzas
(EI·EU +0.358, EI·EA −0.314, EC·EA −0.408), los ocho enlaces con sus picos y sus
(b, s), y el mismo orden.

**Una trampa que costó un rato:** `-i` a secas **no** identifica desde el
diagonal. Monta su propio modelo —en m6, 61 slots, 46 libres, logL −1716.36, con
Σ diagonal— y lee las CCF de *esos* residuos. Comparando contra él, las cifras
del puerto no cuadraban y parecía un fallo; pidiéndole `-0 -i` con las mismas
restricciones, coinciden exactamente. Comparar dos ajustes distintos es la forma
más fácil de inventarse un bug.

**La propuesta puede salir cíclica, y en m6 sale.** Leer las CCF par a par no
impone aciclicidad: el borrador trae EP → EC → EA → EP, y hacen falta **dos**
podas para dejarlo estimable. El modo guiado escribe el `.dag` igualmente, con el
ciclo anotado en cabecera, y `read_dag` lo rechaza mientras siga ahí. La librería
avisa y no poda: cuál de los enlaces cae es juicio, no aritmética, y podar en
silencio invita a estimar el borrador — que es exactamente lo que la doctrina de
la escuela dice que no se haga.

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

### 5.4. `compimp` degradado a `pulse` (bug de fue Python) — CORREGIDO

Buscando reproducir los objetivos de m6 apareció una discrepancia de 1.9 en el
escalón diagonal, que no era de drtran. La bisección la puso donde estaba:

1. la conjunta diagonal de m6 no coincidía con el C ni evaluando en su óptimo;
2. con Σ estrictamente diagonal tampoco ⇒ no era la covarianza ni la red;
3. serie a serie, evaluando **en el óptimo de fue C**, cinco de las seis clavaban
   a 5e-8 y sólo **EI** difería: −292.495 frente a −290.613.

EI es la única de las seis con un determinista **`compimp`**, el impulso
*compensado*: +1 en la fecha y **−1 en la siguiente** (`fue_pre_reader.c:194`,
`fue.c:317`). El lector de fue Python lo mapea a `pulse` a secas
(`fue/inp.py:276`), y se come el −1.

Confirmado con respuesta conocida: reconstruyendo a mano el regresor compensado
(+1, −1) sobre el mismo `.pre` y los mismos coeficientes, fue Python da
**−290.613205**, exactamente fue C.

**Corregido en fue 0.1.9** (BUG-0006), junto con otros dos huecos que la revisión
de los nueve deterministas destapó: `easter` y `trend` no existían en el puerto, y
—esto en el propio fue C, BUG-0007— su escritor del `.pre` los perdía, de modo que
**fue C no podía releer su propio `.pre`**. De paso se unificó el vocabulario:
`impulse` es el nombre canónico, porque fue C **no rechaza** una palabra que no
conoce, la toma por variable no estándar y estima otra cosa en silencio.

Con eso, los objetivos canónicos de m6 se reproducen: **diagonal −1709.511575**
(dif 5.0e-07) y **red libre −1697.613401** (dif 5.9e-07). La validación sobre las
cinco series limpias se conserva: ejercita la misma maquinaria sin depender de la
versión de fue que haya instalada.

## 6. Homologación con el binario

`test_homologacion_c.py` (12) y `test_red.py` (15) **relanzan el binario en vivo**
en vez de comparar contra referencias guardadas, para no arrastrar cifras
obsoletas cuando el C cambia.

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

**La RED**, sobre cinco series de m6 (EP, EU, EC, EA, P) con el DAG
EP ← EU, EP ← EC, EU ← EC (cadena EC → EU → EP, una entrada con dos salidas y un
denominador r=1) y dos covarianzas libres:

| | drtran C | puerto | dif |
|---|---|---|---|
| red libre (40 libres / 48 slots) | −1434.696068 | −1434.696068 | 1.9e-10 |
| + producto + comb. lineal (38 libres) | −1439.505804 | −1439.505804 | 9.4e-08 |
| red de 3 series, **optimizada** (24 libres) | −912.244333 (181 it) | −912.244333 (180 it) | 9.0e-08 |

Las dos primeras filas evalúan **en el óptimo del C** y prueban el *cast*; la
tercera arranca en el escalón diagonal y prueba la *búsqueda*. Son cosas
distintas y conviene no confundirlas: un cast correcto con un optimizador que no
llega, y un optimizador que llega sobre un cast torcido, fallan de maneras muy
diferentes.

## 7. Cómo reproducirlo

```sh
# el puerto
cd ~/Dropbox/SRC/drtran-python && python -m pytest -q          # 77 passed, ~7 min

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
- **El m6 canónico**, bloqueado por el `compimp` de fue Python (§5.4). Dianas del
  C para cuando se desbloquee: diagonal **−1709.511575**, red libre
  **−1697.613401**. La maquinaria de la red ya está validada sobre las cinco
  series limpias de m6 (§3, paso 5).
- **Identificación de la red** (`-i` / `-g` del C): leer las CCF de los residuos
  del diagonal para PROPONER el DAG y las covarianzas, y escribir el `.dag` y el
  `.cns` de arranque. `identify.py` ya hace la parte bivariante.
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
