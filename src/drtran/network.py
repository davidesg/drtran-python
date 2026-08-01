"""The transfer NETWORK: the `.dag` file and its validation.

Port of `read_network` (`drtran.c`). A bivariate link Y <- X is the one-series
case; the network is the general one: **a series may receive transfers and be an
input to another at the same time**, which is what the school's systems really
are (Mauricio's m6-1: EC -> EU -> EI -> EP, plus EC -> EP).

The file, one line per link and `#` for comments::

    # output  <-  input    b  r  s
    EP <- EI   1 0 1
    EP <- EC   1 0 2
    EI <- EU   1 0 3
    EU <- EC   2 0 1

Series go by NAME (the one in the `.pre`, as `load_pre` returns it), not by
position: a `.dag` must not depend on the order of the command line.
Contemporaneous covariances do NOT live here — they are parameters, and they are
freed in the `.cns` (`q[5,2] = free`). Keeping the two apart is deliberate: the
DAG says who moves whom with a delay, and Sigma says what moves together within
the same instant.

Why a cycle is rejected
-----------------------
The embedded cast multiplies each output's row by the denominators of its
incoming links and orders the series topologically; with a cycle there is no
topological order and the system stops being a recursive DAG: it would be a
simultaneous-equations model, which is not what this cast represents. The C
assumes it; here it is checked, and the cycle itself is reported.
"""

from __future__ import annotations

from .cast import Link


def _series_index(name, names):
    """0-based index of a series by name, or by 1-based position if it is a number."""
    name = name.strip()
    if name in names:
        return names.index(name)
    if name.isdigit():
        i = int(name) - 1
        if 0 <= i < len(names):
            return i
    return -1


def find_cycle(links, m):
    """A cycle of the link graph as a list of indices, or `None` if it is a DAG.

    The cycle is returned, not a boolean: a message that says "there is a cycle"
    without saying which one leaves the user to find it by hand.
    """
    successors = {i: [] for i in range(m)}
    for l in links:
        successors[l.inp].append(l.out)          # the input precedes the output

    state = {}                                   # 0 = in progress, 1 = closed
    path = []

    def visit(u):
        state[u] = 0
        path.append(u)
        for v in successors[u]:
            if state.get(v) == 0:                # back edge: a cycle
                return path[path.index(v):] + [v]
            if v not in state:
                c = visit(v)
                if c:
                    return c
        path.pop()
        state[u] = 1
        return None

    for i in range(m):
        if i not in state:
            c = visit(i)
            if c:
                return c
    return None


def check_acyclic(links, m, names=None):
    """Raise `ValueError` if the links form a cycle."""
    c = find_cycle(links, m)
    if c is None:
        return
    label = (lambda i: names[i]) if names else str
    raise ValueError("the network has a CYCLE, and a DAG does not admit one: "
                     + " -> ".join(label(i) for i in c))


def read_dag(path, names):
    """Read the network file and return the list of `Link`. Port of `read_network`.

    `names` are the series names **in the order they were passed to the cast**
    (`cast_spec.names`). The graph is checked to be acyclic.
    """
    names = list(names)
    links = []
    with open(path) as f:
        for nline, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{nline}: expected 'OUTPUT <- INPUT b r s': {line!r}")
            lhs, arrow, rhs, b, r, s = fields
            if arrow != "<-":
                raise ValueError(f"{path}:{nline}: expected '<-', found {arrow!r}")
            io, ii = _series_index(lhs, names), _series_index(rhs, names)
            if io < 0 or ii < 0:
                unknown = lhs if io < 0 else rhs
                raise ValueError(
                    f"{path}:{nline}: unknown series {unknown!r}; "
                    f"the ones loaded are {names}")
            if io == ii:
                raise ValueError(f"{path}:{nline}: a series cannot feed itself "
                                 f"({lhs})")
            try:
                b, r, s = int(b), int(r), int(s)
            except ValueError:
                raise ValueError(f"{path}:{nline}: b, r, s must be integers: "
                                 f"{line!r}") from None
            if b < 0 or r < 0 or s < 0:
                raise ValueError(f"{path}:{nline}: b, r, s cannot be negative")
            links.append(Link(out=io, inp=ii, b=b, r=r, s=s))

    check_acyclic(links, len(names), names)
    return links


def write_dag(path, links, names):
    """Write a readable `.dag`. The counterpart of `read_dag`, for the C's `-g`."""
    with open(path, "w") as f:
        f.write("# output  <-  input    b  r  s\n")
        for l in links:
            f.write(f"{names[l.out]} <- {names[l.inp]}   {l.b} {l.r} {l.s}\n")
