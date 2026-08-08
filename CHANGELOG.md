# Changelog — drtran (motor) y mtram (su servidor MCP)

Los informes completos están en `docs/BUGS.md`. Etiquetas de publicación: `v*`.

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
