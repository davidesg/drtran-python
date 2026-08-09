# Changelog — drtran (motor) y mtram (su servidor MCP)

Los informes completos están en `docs/BUGS.md`. Etiquetas de publicación: `v*`.

## 0.2.1 — 2026-08-10

Corrige el EMPAQUETADO de 0.2.0, no el código: el motor y mtram son los mismos.
Aquella llevaba `recursive-include docs *.md`, así que el sdist repartía los diez
documentos de `docs/` incluidos dos que son BITÁCORAS DE TRABAJO y no
documentación: el plan de documentación, que trata de decisiones pendientes, y
el banco pass-through × MEG, que es un experimento sobre investigación aún no
publicada y apunta a ficheros que no están en ningún repositorio. Nada
comprometido, pero tampoco cosas que repartir.

- El `MANIFEST.in` lista los documentos uno a uno Y excluye explícitamente esos
  dos. Las dos cosas hacen falta: `include` sólo añade --nunca quita-- y el
  `SOURCES.txt` del `egg-info` anterior arrastra lo que se incluyó alguna vez.
- Cualificadas dos referencias que cruzan de repositorio y quedaban colgando
  para quien lea la copia distribuida: `drvarma/docs/DESIGN_MCP.md` en
  ARCHITECTURE_MCP y `drtran/docs/OPTIMIZER_STOPPING_STUDY.md` --el repo del
  C-- en BUGS.
- El README dice cuáles viajan y cuáles viven sólo en el repositorio.

Viajan: SCHOOL_PRACTICE_STUDY, DECISION_NODES, LADDER_AS_OPTIMISATION,
ARCHITECTURE_MCP, PORTE, BUGS y LEVEL_TRANSFER_PLAN.

## 0.2.0 — 2026-08-09

La versión de **BUG-8**: el cast relacionaba series diferenciadas cada una por
SU propio operador, y el modelo dice que la transferencia relaciona los NIVELES
con la diferenciación en el ruido. Cuando los dos operadores coincidían daba lo
mismo —∇ conmuta con ν(B)— y cuando no, lo que se ajustaba no era ν sino **ν·Δ**,
con la ganancia equivocada por **Δ(1)**.

- **El despacho.** `Δ = op_salida / op_entrada` se calcula antes de estimar
  comparando los dos POLINOMIOS, no el par (d, D): así el ∇∇₄ de la escuela
  —`d=2, ifadf=[0,1,1]`— se reconoce como el mismo operador que `d=1, D=1`.
  Si Δ = 1 corre el cast empotrado exactamente como antes; si no, corre el de
  RESTA, que es el único con sitio para un segundo vector, alimentado con la
  entrada re-diferenciada por el operador de la SALIDA.
  **Los casos emparejados no se mueven ni un bit**: todo el legacy, m6 y la red.
- **Medido.** Un banco sintético de tres brazos establece la ley
  `ν̂(1) = ν(1)·Δ(1)`: con exceso en frecuencia cero Δ(1)=0 y la ganancia se
  ANIQUILA; con exceso sólo estacional Δ(1)=s y se MULTIPLICA —doce veces en un
  mensual—. Las frecuencias estacionales **no** están exentas.
- **Contra el oráculo.** Los tres casos mixtos del pass-through pasan del 41-53 %
  de discrepancia con TASTE al **0.24-1.01 %**, la banda en la que estaban los
  emparejados. Cinco casos nuevos del banco SF_MEG, con el brazo Z validado.
- **La muestra previa se retropronostica** en vez de ponerse a cero. El cast de
  resta calculaba "la verosimilitud exacta de la serie equivocada"; ahora usa
  primero las observaciones reales que el recorte deja fuera y retropronostica
  el resto. FR pasa de 1.01 % a 0.13 % del oráculo, y su ω₁ de 2.3e-4 a 6.0e-6.
- **La previsión de la ruta de resta es la de TASTE**, no la del empotrado
  corregida: cada input previsto por su modelo, el ruido por su ARMA, unidos
  sobre los NIVELES. RMSE a un paso 6.6-20.0 % menor. Con ω=0 devuelve la
  previsión univariante EXACTA, y sobre un modelo emparejado coincide con la
  ruta homologada.
- **La ventana común** se infiere del `.pre`, que ya trae `nobs`, `start` y
  `freq`. Dos series pueden acabar la misma fecha —alineamiento correcto— y una
  tener más historia: el cast recorta, y el `.pre` de la larga deja de ser
  óptimo de la muestra que se ajusta. Ahora se avisa antes, no después.
- **mtram lo dice todo.** `estimate` anuncia cuándo se pidió el empotrado y
  corrió el de resta —y que las dos verosimilitudes no son comparables—,
  `check_operators` es herramienta nueva, y `forecast` dice por qué ruta salió.
- **La documentación viaja en el sdist** (`MANIFEST.in`), así que es alcanzable
  sin depender de que el repositorio lo sea.

Portado al C y rehomologado: batería del C 304 PASS 0 FAIL con las
verosimilitudes canónicas de m6 intactas, homologación C↔Python 12, batería
Python 404.

## 0.1.1 — 2026-08-08

La versión de la CONSISTENCIA con art. El analista llega a mtram desde art, y
se encontraba con otro formato, otros instrumentos y medio panel de diagnosis.

- **El modelo se presenta en las DOS ECUACIONES de art**, con el error típico
  DEBAJO de cada coeficiente, y lo dibuja el renderizador de art —no una copia—
  sobre `drtran.fitted_model`. La ecuación (2), la del RUIDO, no existía en
  mtram: se estima aquí, se mueve aquí, y no aparecía por ninguna parte.
  El renderizador se comprueba a sí mismo, porque ω(B) = ω₀ − ω₁B y lo que se
  IMPRIME en el retardo k≥1 es −ω_k: la suma de lo impreso debe ser la ganancia.
- **¿Es INFLUYENTE la transferencia?** Comparando contra el escalón diagonal
  —que ya está estimado y lleva el mismo modelo sin transferencia— qué
  parámetros del ruido y qué deterministas mueve, en errores típicos. Es el
  traspaso que la escuela anota al cerrar un caso.
- **Las tres cifras de Brajín**, no una: reducción de varianza, desviación
  típica residual *pasando de* la univariante a la de transferencia, y el R²
  (A.28) igual. El R² va sobre la serie ESTACIONARIA, que es lo que lo hace
  utilizable y comparable entre los dos ajustes.
- **La diagnosis en el formato de art**: ruido blanco, normalidad (JB),
  asimetría y curtosis, y residuos extremos. Sin duplicar `calibrate`, que
  contesta mejor la pregunta de los anómalos. Y el veredicto no se traga una
  normalidad que falla.
- **Los gráficos vienen DENTRO de la respuesta**, como en art, e `identify_link`
  trae su CCF consigo: el nodo N1 se decide mirándola.
- **BUG-4 — corregido.** `identify_link` se declaraba `-> str` devolviendo una
  lista, y FastMCP rechazaba la respuesta: el tool funcionaba sólo cuando el
  gráfico fallaba. Con él, no. La batería de 349 pasaba igual porque todos los
  tests llamaban a la función, no al tool registrado. Dos guardas nuevas, las
  dos verificadas reintroduciendo el defecto.
- **BUG-1 — cerrado en el Python y en el C.** La estacionariedad AR se lee de
  las RAÍCES vía `chekma`. Verificarlo corrigió la ficha: en el camino por
  defecto esa guarda nunca se ejecutaba, así que el daño era del PUERTO.
  Rehomologado: 48 ejecuciones idénticas.
- **BUG-3 — corregido.** `write_pre` escribía ficheros que no eran `.pre`.
  Ahora `write_inp` escribe `.inp`: lo que sale de aquí es un punto de partida,
  no un óptimo, y sólo el programa que estimó puede afirmar lo segundo.
- Los niveles 1, 2 y 3 del estudio de las tesis, completos.
- `docs/LADDER_AS_OPTIMISATION.md`: la escalera como algoritmo de optimización,
  qué garantiza cada fichero, y dónde se rompe la analogía.

Batería: 351 passed.

## 0.1.0 — 2026-07

Primera publicación: el puerto de drtran a Python y mtram, su servidor MCP.
