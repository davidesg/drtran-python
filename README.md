# drtran (Python)

**Modelos de transferencia de Box–Jenkins por máxima verosimilitud exacta.**

Puerto a Python de [`drtran`](../drtran) (C). Es el **puente** entre dos programas
que ya funcionan:

- **[fue]** — identifica y estima modelos **univariantes** (ARIMA + Box–Cox +
  deterministas). Produce un `.pre` por serie.
- **[drvarma]** — evalúa la **verosimilitud exacta VARMA** de Mauricio (`elf`) y
  la maximiza con BFGS factorizado.

drtran lee los `.pre` ya especificados en fue —una salida y una o varias
entradas— y los estima **conjuntamente**, todos los parámetros a la vez:

```
Y_t  =  Σⱼ  ωⱼ(B)/δⱼ(B) · B^bⱼ · Xⱼ,t  +  N_t
```

## El `.pre` es el contrato, no un formato de entrada

fue deja en el `.pre` el mejor modelo univariante estimado de cada serie; drtran
lo toma como semilla. `.pre` e `.inp` comparten formato — la diferencia es que las
semillas del `.pre` son las estimaciones de la última iteración. Por eso la cadena
es **iterativa y con continuidad**: la salida de un escalón alimenta el siguiente.

```
fue → .pre → drtran (diagonal) → CCF de residuos → .dag/.cns → drtran (red) → …
```

Los artefactos son ficheros de texto inspeccionables **a propósito**: el punto de
intervención del analista entre escalones es parte del método (la red identificada
es una guía, no la final), no una limitación a abstraer.

## Principio de diseño, no negociable

> El `elf` de drvarma se usa **tal cual**. No se modifica, no se parchea, no se
> caso-especializa. Es la implementación de referencia de la verosimilitud exacta.
> Cualquier discrepancia con fue es un bug del cast de drtran, **nunca** de `elf`.

## Criterio de validación (puerta de entrada a todo lo demás)

**Estimación conjunta diagonal ≡ fue por separado.** Con estructura diagonal la
verosimilitud exacta se factoriza, así que la conjunta debe reproducir la suma de
las univariantes. Caso canónico `ES_CPI_m10` ← `WTI_ar1`:

| | φ | logL |
|---|---|---|
| ES_CPI | 0.402839 | −7.3917 |
| WTI | 0.299193 | −760.0326 |
| **suma = objetivo conjunto** | | **−767.424341** |

## Estado

**Paso 0 — completado.** La entrada está validada:

- `fue.load()` lee `.pre` y `.inp`, y se ha verificado **campo a campo** que
  preserva todo lo que drtran necesita: λ/d/D, `refactor`, μ **con su flag de
  fijado**, órdenes y coeficientes AR/MA regulares y anuales, AR(2)/MA(2) de
  frecuencia fija, los deterministas con sus ω y sus flags, y la serie. **No se
  porta `fue_pre_reader.c`.**
- fue Python reproduce los univariantes de referencia, y su suma da la diana de la
  conjunta con diferencia 7.6e-09.

**Paso 1 — siguiente.** Portar el cast (`tran_shootx.c`, 660 líneas), siguiendo la
estructura de `fue/cast_us.py` (`build_est_spec` / `cast(x) → estructura`).

**Paso 2 — la puerta.** Conjunta diagonal = −767.424341.

**Paso 3.** Transferencia ω/δ, diagnósticos (portmanteau de transferencia y de
exogeneidad), previsión.

## Alcance del puerto

La mayor parte del C **no se porta**, se reutiliza: `elfvarma` + `drvmlest` +
`qnewtopt` + `nlatools` ya están en drvarma Python; `gnuplot_i` → matplotlib; el
lector de `.pre` → `fue.load()`. Son ~4.500 de las 12.600 líneas del C.

Queda: `tran_shootx.c` (el cast, 660), `diagnose.c` (1687) y `drtran.c`
(CLI/orquestación, 4189 — se encoge mucho en Python).
