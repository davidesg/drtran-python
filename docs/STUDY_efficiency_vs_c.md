# Estudio: eficiencia de drtran-python frente al drtran en C

**Estado: ESTUDIO. No hay cambios de código.** El árbol se dejó exactamente como
estaba antes de empezar (`git checkout -- src/drtran/embed.py`); `fue` nunca
llegó a tocarse. Lo que sigue es lo medido, lo que se probó y se revirtió, y las
líneas de investigación con su coste y su beneficio.

Medido el 2026-08-15. Los dos lados evalúan **la misma verosimilitud en C**:
`drtran-python` llama al núcleo de `drvarma` (`elf_c`), y el binario en C llama
al suyo. Todo lo que se mida por encima de eso es envoltorio, no aritmética.

## 1. El resultado que ordena todo lo demás: hay dos regímenes

No hay «un factor» de sobrecoste. Depende de la configuración, y la variable que
manda **no es el operador racional**.

### Régimen A — ajuste de biblioteca, sin deterministas pesados

`scripts/repro_perf_regression_transfer.py`, transferencia racional
ν(B) = ω₀/(1−δ₁B), b=0 r=1 s=0, llamando a `drtran.fit` directamente:

| caso | reloj | iter | s/iter |
|---|---|---|---|
| ES_CPI_m10 (P=0, sin AR estacional) ← WTI | 3,6 s | 24 | 0,15 |
| EA_HICP_sar (P=1, AR estacional en B¹²) ← Brent | 4,6 s | 24 | 0,19 |
| referencia C | — | ~23 | ~0,04 |

**Sobrecoste ×4**, y el AR estacional **no lo empeora**: la razón entre los dos
es 1×.

⚠ **Esto contradice la cabecera del propio repro**, que describe un
«~15× per-iteration» atribuido a que el estado de la verosimilitud exacta crece
con el retardo AR máximo. Ejecutado hoy, ese efecto **no aparece**. Cualquiera de
las dos cosas es cierta y ninguna se ha comprobado: o lo arregló alguno de los
cambios posteriores —el candidato natural es la corrección de `check_scale` del
8-ago, que dejó de mirar `refactor` para medir el tamaño de la serie
estacionaria—, o la cabecera se escribió sobre una medición que no vuelve.
**Primera línea de investigación, y prioritaria**, porque hoy el fichero afirma
en el repositorio algo que su propia ejecución desmiente.

### Régimen B — la línea de órdenes completa, con muchos deterministas

`examples/passthrough`, IPC_ES ← WTI. **Las verosimilitudes coinciden hasta el
último dígito** en las tres especificaciones, así que se está midiendo el mismo
trabajo:

| | C puro | Python | factor |
|---|---|---|---|
| b0 r0 s1 | 0,08 s | 5,76 s | ×72 |
| b0 r1 s1 | 0,15 s | 8,67 s | ×58 |
| b0 r2 s1 | 0,16 s | 8,64 s | ×54 |

El operador racional **no empeora la proporción** — la mejora, porque r>0 sólo
añade iteraciones y el sobrecoste es por evaluación. Tampoco es el arranque del
intérprete (`import drtran`: 0,23 s) ni los errores estándar (con `-Q` siguen
siendo 7,8 s).

### Lo que separa A de B

`IPC_ES.pre` lleva **11 intervenciones sobre 288 observaciones**; `WTI.pre`,
ninguna. El coste dominante es el camino determinista del cast, un bucle doble
`11 × 288` en Python interpretado. **El regresor determinista es la variable que
manda**, no el orden racional ni el AR estacional. Un modelo con estacionalidad
determinista completa —que es la especificación preferible para el trabajo
multivariante— es justo el caso caro.

## 2. Dónde está el tiempo

Perfil del caso r=1 con `-Q`:

| | tiempo | % |
|---|---|---|
| `fue.cast_us.cast_us_py` (Python puro) | 4,36 s | **56 %** |
| `drvarma_elf` (núcleo C) | 0,13 s | **2 %** |
| todo lo demás (Python repartido) | 3,23 s | **42 %** |

Por llamada: el cast cuesta ~1,9–3 ms y la verosimilitud que alimenta, 81 µs.
**El cast cuesta más de veinte veces que la verosimilitud**, y se ejecuta dos
veces por evaluación, una por serie.

El renglón siguiente es `ar_is_stationary` (0,64 s): una llamada por serie y
evaluación que pasa por `numpy.roots` — matriz compañera y `eigvals`— por delante
de la propia verosimilitud.

**El 42 % restante no está desglosado.** Es la deuda de medición de este estudio:
son muchos renglones pequeños, y ninguna de las opciones de abajo lo toca.

## 3. Lo que se probó y se revirtió

Un memo sobre `cast_us_py` y otro sobre `ar_is_stationary`, los dos con clave en
los **bytes crudos** del subvector de parámetros. Funcionan porque en el gradiente
por diferencias finitas cada evaluación perturba **un** parámetro, que pertenece a
**una** serie: la otra llega bit a bit igual que en la llamada anterior.

Resultado: llamadas al cast de 2.964 a **1.268** (−57 %), reloj **−30 %**
(×58 → ×41 en r=1), e **informes byte a byte idénticos**.

**Revertido**, por dos razones y no por una:

1. La ganancia es real pero insuficiente, y **no queda holgura por ese camino**:
   las 1.268 llamadas restantes están cerca del mínimo alcanzable, porque hay
   1.481 evaluaciones y casi todas cambian al menos una serie.
2. Un memo es estado oculto en la ruta caliente. Aciertos y fallos dependen de
   la trayectoria del optimizador, así que un fallo futuro no es reproducible sin
   reproducir la trayectoria entera. A cambio de un 30 % no compensa.

Si alguna vez se retoma, lo que había que preservar: clave por bytes exactos (un
memo que casara aproximadamente devolvería el cast de otro punto y **aplanaría la
diferencia que el gradiente está midiendo**); el `est_spec` guardado dentro del
valor para que su `id()` no lo recicle otro objeto; arrays devueltos por copia; y
los `ifault` sin cachear.

## 4. Las opciones, con coste y beneficio

### (a) Vectorizar `cast_us_py` en `fue` — **recomendada**

El bucle que domina es ensamblado y convolución sobre `n_interv × nobs`, no una
recursión difícil.

*Beneficio:* estimado ×5–10 en el cast; el caso r=1 caería a ~3,2–3,5 s. Aprovecha
a `fue` y a todos sus consumidores, no sólo a `drtran`.

*Coste:* cambio de Python puro, **sin ABI y sin recompilar ruedas** (viaja en las
mismas). Contrastable **bit a bit** contra la implementación actual sobre toda la
batería de specs. Sin estado, sin reentrancia, **una sola implementación del
espejo**. Requiere versión de `fue` para que llegue a los usuarios.

### (b) Exponer `cast_us` en el CFFI de `fue` — **estudio, no ahora**

*Beneficio:* el cast en C cuesta del orden de 15 µs contra 3 ms, ×200 en esa
pieza. Pero **no cierra la brecha**: aunque costara cero quedan los 3,23 s del
42 %, así que el caso r=1 pasaría de ×41 a **×18**, no a ×2.

*Coste*, y es permanente:

1. `cast_us` es `static` y lee el modelo de los globales `Tm`, `Ts`, `DataMat`:
   hace falta un ciclo de vida `open/eval/close` que hoy no existe.
2. Los globales **no son reentrantes** y `mtram` es un servidor MCP en vivo.
   Además `drtran` necesita **dos casts vivos a la vez** (salida y entrada), lo
   que obliga a desglobalizar — y eso rompe a propósito la correspondencia con
   `fue.c:3645` que `fue_api.c:10` documenta y que es lo que hace auditable la
   extracción.
3. `Tvarma` no cruza tal cual: `real ***phi` exige struct plano y conversor.
4. Tren de publicación de tres paquetes con ABI nueva: fue → ruedas de todas las
   plataformas menos macOS Intel → suelo de drtran → cotas de atsw, en el orden
   de `RELEASING.md`.
5. Vuelve a poner al alcance `fue/BUG-0013` —la escritura fuera de rango con
   p=q=0—, hoy sólo esquivada desviando al motor Python.
6. **El puerto pierde su oráculo en la ruta caliente.** Que `cast_us_py`
   reproduzca la verosimilitud del binario a 1e-9 es lo que valida el porte.
7. Dos implementaciones que sincronizar para siempre, en un proyecto donde ya hay
   tres copias del C de drvarma sin sincronizar y `qnewtopt.c` divergió.

### (c) Reducir el número de evaluaciones

Fuera de alcance por decisión establecida: `raxopt` y `elf` son trabajo
publicado. Cualquier cosa aquí es estudio documentado, no cambio.

### (d) No hacer nada

Defendible en el régimen A (×4). Insostenible en el B si el servidor MCP tiene
que responder en vivo sobre modelos con estacionalidad determinista completa.

## 5. Lo que este estudio NO puede prometer

**La paridad con el binario no es alcanzable por ninguna de estas vías.**
Requeriría todo el camino por evaluación en C, que es exactamente lo que es el
binario. El objetivo razonable es «suficientemente rápido para el servidor MCP»,
no ×1. Conviene fijar ese objetivo en segundos antes de optimizar nada más.

## 6. Líneas de investigación, en orden

1. **Resolver la contradicción del régimen A.** El repro afirma ~15× y mide 1×.
   Bisecar contra los commits posteriores al 8-ago; si lo arregló `check_scale`,
   la cabecera se corrige y el caso pasa a ser una prueba de no regresión.
2. **Desglosar el 42 % que no está desglosado.** Es la mitad del reloj después de
   (a) y hoy no se sabe qué es. Ninguna decisión debería tomarse sin ese número.
3. **Vectorizar `cast_us_py`** con contraste bit a bit contra la versión actual.
4. **Fijar el objetivo en segundos** para el servidor MCP, y medir contra él.
5. **Desglobalizar `cast_us`** — estudio propio en `fue`, con la reentrancia como
   requisito, no como consecuencia.
6. **Instrumentar el conteo de evaluaciones** de las dos implementaciones sobre el
   mismo caso. Aquí C y Python hicieron 40 y 38 iteraciones, pero **el C paró por
   gradiente y el puerto por paso** (`termcode=2`). Coinciden en el resultado; no
   coinciden en el criterio. Ver §7.

## 7. Defectos encontrados de camino — abiertos, y no es una decisión menor

Ninguno de estos se ha arreglado. Dejarlos abiertos es una decisión que conviene
tomar mirándolos, no por omisión:

- **BUG-10** (`docs/BUGS.md`): la varianza del nivel en `forecast()` integra con
  `d + D·s` e **ignora `ifadf`** — el mismo defecto que BUG-9 cerró en la puerta
  diagonal, en un segundo sitio. La media del nivel está bien; sólo las **bandas**
  salen mal, y por eso es invisible en el punto de previsión. Medido: factor 19
  en varianza a l=12. Con repro. **Abierto.**
- **`estimate.py:203` cuenta el `termcode 2` como convergencia**, que es lo que
  `fue` abandonó en 0.1.10 y lo que `art` tuvo que separar. En el caso medido no
  cambia el resultado, pero el criterio de parada de las dos implementaciones no
  es el mismo.
- **Regresión de identificación en `art`** (destapada por
  `tests/test_end_to_end_passthrough.py`, 5 fallos): `art` decide ahora **λ=0**
  para IPC_ES donde el 7-ago decidía **λ=1**. Los otros cuatro fallos cuelgan de
  ése: con otra λ la serie es otra. Verificado que **no** lo causa nada de este
  estudio — falla igual contra el árbol limpio. Causa sin establecer; candidatos:
  el commit `f8ee98e` de art, el cambio a instalaciones editables, o un
  estadístico que decide al filo. Pide ficha propia en `art` y bisección.
