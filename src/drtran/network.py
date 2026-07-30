"""La RED de transferencias: el `.dag` y su validación.

Puerto de `read_network` (`drtran.c`). Un enlace bivariante Y ← X es el caso de
una serie; la red es lo general: **una serie puede recibir transferencias y ser a
la vez entrada de otra**, que es lo que son de verdad los sistemas de la escuela
(m6-1 de Mauricio: EC → EU → EI → EP, más EC → EP).

El fichero, una línea por enlace y `#` para comentarios::

    # salida  <-  entrada    b  r  s
    EP <- EI   1 0 1
    EP <- EC   1 0 2
    EI <- EU   1 0 3
    EU <- EC   2 0 1

Las series van por su NOMBRE (el del `.pre`, tal como lo devuelve `load_pre`), no
por posición: un `.dag` no debe depender del orden de la línea de órdenes. Las
covarianzas contemporáneas NO viven aquí — son parámetros, y se liberan en el
`.cns` (`q[5,2] = free`). Separar las dos cosas es deliberado: el DAG dice quién
mueve a quién con retardo, y Σ dice qué se mueve junto en el mismo instante.

Por qué se rechaza un ciclo
---------------------------
El cast empotrado multiplica la fila de cada salida por los denominadores de sus
enlaces entrantes y ordena las series topológicamente; con un ciclo no hay orden
topológico y el sistema deja de ser un DAG recursivo: sería un modelo de
ecuaciones simultáneas, que no es lo que este cast representa. El C lo asume; aquí
se comprueba y se dice cuál es el ciclo.
"""

from __future__ import annotations

from .cast import Link


def _indice_serie(nombre, names):
    """Índice 0-based de una serie por nombre, o por posición 1-based si es un número."""
    nombre = nombre.strip()
    if nombre in names:
        return names.index(nombre)
    if nombre.isdigit():
        i = int(nombre) - 1
        if 0 <= i < len(names):
            return i
    return -1


def find_cycle(links, m):
    """Un ciclo del grafo de enlaces como lista de índices, o `None` si es un DAG.

    Se devuelve el ciclo, no un booleano: un mensaje que dice «hay un ciclo» sin
    decir cuál obliga al usuario a buscarlo a mano.
    """
    sucesores = {i: [] for i in range(m)}
    for l in links:
        sucesores[l.inp].append(l.out)          # la entrada precede a la salida

    estado = {}                                  # 0 = en curso, 1 = cerrado
    camino = []

    def visita(u):
        estado[u] = 0
        camino.append(u)
        for v in sucesores[u]:
            if estado.get(v) == 0:               # arista hacia atrás: ciclo
                return camino[camino.index(v):] + [v]
            if v not in estado:
                c = visita(v)
                if c:
                    return c
        camino.pop()
        estado[u] = 1
        return None

    for i in range(m):
        if i not in estado:
            c = visita(i)
            if c:
                return c
    return None


def check_acyclic(links, m, names=None):
    """Levanta `ValueError` si los enlaces forman un ciclo."""
    c = find_cycle(links, m)
    if c is None:
        return
    etiqueta = (lambda i: names[i]) if names else str
    raise ValueError("la red tiene un ciclo, y un DAG no lo admite: "
                     + " → ".join(etiqueta(i) for i in c))


def read_dag(path, names):
    """Lee el fichero de red y devuelve la lista de `Link`. Puerto de `read_network`.

    `names` son los nombres de las series **en el orden en que se pasaron al
    cast** (`cast_spec.names`). Se valida que el grafo sea acíclico.
    """
    names = list(names)
    links = []
    with open(path) as f:
        for nlin, linea in enumerate(f, 1):
            linea = linea.split("#", 1)[0].strip()
            if not linea:
                continue
            campos = linea.split()
            if len(campos) != 6:
                raise ValueError(
                    f"{path}:{nlin}: se esperaba 'SALIDA <- ENTRADA b r s': {linea!r}")
            lhs, flecha, rhs, b, r, s = campos
            if flecha != "<-":
                raise ValueError(f"{path}:{nlin}: se esperaba '<-', encontré {flecha!r}")
            io, ii = _indice_serie(lhs, names), _indice_serie(rhs, names)
            if io < 0 or ii < 0:
                desconocida = lhs if io < 0 else rhs
                raise ValueError(
                    f"{path}:{nlin}: serie desconocida {desconocida!r}; "
                    f"las cargadas son {names}")
            if io == ii:
                raise ValueError(f"{path}:{nlin}: una serie no puede alimentarse "
                                 f"a sí misma ({lhs})")
            try:
                b, r, s = int(b), int(r), int(s)
            except ValueError:
                raise ValueError(f"{path}:{nlin}: b, r, s deben ser enteros: "
                                 f"{linea!r}") from None
            if b < 0 or r < 0 or s < 0:
                raise ValueError(f"{path}:{nlin}: b, r, s no pueden ser negativos")
            links.append(Link(out=io, inp=ii, b=b, r=r, s=s))

    check_acyclic(links, len(names), names)
    return links


def write_dag(path, links, names):
    """Escribe un `.dag` legible. La contraparte de `read_dag`, para el `-g` del C."""
    with open(path, "w") as f:
        f.write("# salida  <-  entrada    b  r  s\n")
        for l in links:
            f.write(f"{names[l.out]} <- {names[l.inp]}   {l.b} {l.r} {l.s}\n")
