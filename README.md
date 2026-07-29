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

## Un ejemplo

```python
import drtran
from drtran.cast import build_cast_spec, Link
from drtran.estimate import fit, unpack

Y = drtran.load_pre("ES_CPI_m10.pre")       # la salida
X = drtran.load_pre("WTI_ar1.pre")          # la entrada

cs = build_cast_spec([Y, X], links=[Link(out=0, inp=1, b=0, r=0, s=1)])
f  = fit(cs)                                # empotrado por defecto, como en el C
print(f.loglik, unpack(f)["links"])
```

`identify(cs, link)` propone (b, r, s) por preblanqueo y CCF antes de estimar.

## Estado

**Pasos 0, 1 y 2 — cerrados.** La entrada está validada campo a campo, el cast
diagonal supera la puerta (−767.424341, dif. 3.9e-07 con la suma de fue), están los
dos casts de transferencia —por resta y **empotrado**, este el defecto— con
estimación conjunta, y la identificación de (b, r, s) por preblanqueo + CCF. Todo
homologado contra el binario C a ~1e-7, con un test que lo **relanza en vivo**.
**52 tests**, verdes.

**Falta:** red de transferencias (`-n`), DAG y `expand_params`; el resto de los
diagnósticos de `diagnose.c`; previsión y CLI. Detalle en [`TODO.md`](TODO.md).

> **[`docs/PORTE.md`](docs/PORTE.md) — el registro del proceso.** Cómo se hizo,
> qué decisiones no son traducción y por qué, las cifras de homologación, los tres
> defectos que el porte le encontró al original (entre ellos que **μ es la media,
> no un intercepto**) y las trampas al comparar contra el binario.

## Alcance del puerto

La mayor parte del C **no se porta**, se reutiliza: `elfvarma` + `drvmlest` +
`qnewtopt` + `nlatools` ya están en drvarma Python; `gnuplot_i` → matplotlib; el
lector de `.pre` → `fue.load()`. Son ~5.800 de las 12.615 líneas del C.

Portado: `tran_shootx.c` (el cast, 668) → `cast.py` + `embed.py`, y la parte de
identificación de `diagnose.c` → `identify.py`. Queda el resto de `diagnose.c`
(1687) y `drtran.c` (CLI/orquestación, 4201 — se encoge mucho en Python).
