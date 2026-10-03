"""BUG-52 — the .cns named covariances by command-line POSITION and transfers
by the .dag's LINE ORDER, neither written anywhere: the same .dag, .cns and
.pre in another order estimated another model, silently. Names are now
accepted where positions were (positional files still read), and the guided
mode writes names."""
import os

import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec  # noqa: E402
from drtran.network import read_dag  # noqa: E402
from drtran.slots import build_slots, read_cns, resolve_names  # noqa: E402

DATA = "/home/david/Dropbox/SRC/drtran/tests/data/m6"
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "M6_EP.pre")),
                                reason="the m6 .pre files from the C repo are missing")

DAG = "EP <- EU   1 0 1\nEP <- EC   1 0 2\nEU <- EC   2 1 1\n"
DAG_REORDERED = "EU <- EC   2 1 1\nEP <- EC   1 0 2\nEP <- EU   1 0 1\n"
CNS_NAMES = """q[EA,EU] = free
q[EC,EA] = free
omega[EP<-EU][1] = omega[EP<-EU][0] * theta[EU][B^1]
omega[EP<-EC][0] = omega[EP<-EC][1] + omega[EP<-EC][2]
"""
CNS_POSITIONS = """q[4,2] = free
q[4,3] = free
omega1[1] = omega1[0] * theta_2[B^1]
omega2[0] = omega2[1] + omega2[2]
"""


def _table(order, dag, tmp_path):
    specs = [drtran.load_pre(os.path.join(DATA, f"M6_{n}.pre")) for n in order]
    cs0 = build_cast_spec(specs)
    p = tmp_path / f"{'_'.join(order)}_{abs(hash(dag))}.dag"
    p.write_text(dag)
    cs = build_cast_spec(specs, links=read_dag(str(p), cs0.names))
    return cs, build_slots(cs)


def _apply(table, text, tmp_path, tag):
    p = tmp_path / f"{tag}.cns"
    p.write_text(text)
    return read_cns(str(p), table)


def _by_meaning(cs, table):
    """The constrained slots, renamed to what they MEAN (series names and link
    pairs), so two orderings can be compared."""
    names = cs.names
    pairs = [(names[l.out], names[l.inp]) for l in cs.links]
    out = {}
    for sl in table.slots:
        if sl.kind == 0 and not sl.name.startswith("q["):
            continue
        n = sl.name
        if n.startswith("q["):
            i, j = (int(v) for v in n[2:-1].split(","))
            n = "q" + str(sorted([names[i - 1], names[j - 1]]))
        for k, (o, i) in enumerate(pairs, 1):
            n = n.replace(f"omega{k}[", f"omega({o}<-{i})[").replace(f"delta{k}[", f"delta({o}<-{i})[")
        for k, nm in enumerate(names, 1):
            n = n.replace(f"theta_{k}[", f"theta({nm})[").replace(f"phi_{k}[", f"phi({nm})[")
        out[n] = sl.kind
    return out


def test_names_resolve_to_the_positions(tmp_path):
    cs, t = _table(["EP", "EU", "EC", "EA", "P"], DAG, tmp_path)
    assert resolve_names("q[EA,EU] = free", t) == "q[4,2] = free"
    assert resolve_names("q[EU,EA] = free", t) == "q[4,2] = free"
    assert resolve_names("omega[EP<-EC][1] = theta_EU[B^1]", t) == "omega2[1] = theta_2[B^1]"
    assert resolve_names("theta[EU][B^1] + mu[EA]", t) == "theta_2[B^1] + mu[4]"
    assert resolve_names("q[4,2] = theta_2[B^1]", t) == "q[4,2] = theta_2[B^1]"
    with pytest.raises(KeyError, match="no series 'XX'"):
        resolve_names("q[EA,XX] = free", t)
    with pytest.raises(KeyError, match="no link EU <- EP"):
        resolve_names("omega[EU<-EP][0] = 0", t)


def test_the_named_cns_is_the_positional_one(tmp_path):
    cs, a = _table(["EP", "EU", "EC", "EA", "P"], DAG, tmp_path)
    _cs, b = _table(["EP", "EU", "EC", "EA", "P"], DAG, tmp_path)
    assert _apply(a, CNS_NAMES, tmp_path, "n") == _apply(b, CNS_POSITIONS, tmp_path, "p") == 4
    assert [(s.name, s.kind) for s in a.slots] == [(s.name, s.kind) for s in b.slots]


def test_reordering_files_and_dag_lines_keeps_the_model(tmp_path):
    """The report's two experiments: other file order, other .dag line order.
    By name the constraints mean the same thing in all three."""
    ref = None
    for order, dag in ((["EP", "EU", "EC", "EA", "P"], DAG),
                       (["EP", "EU", "EC", "P", "EA"], DAG),
                       (["EP", "EU", "EC", "EA", "P"], DAG_REORDERED)):
        cs, t = _table(order, dag, tmp_path)
        _apply(t, CNS_NAMES, tmp_path, "x")
        meaning = _by_meaning(cs, t)
        if ref is None:
            ref = meaning
        assert meaning == ref, order


def test_the_positional_cns_is_what_moved(tmp_path):
    """The same positional file under the permuted order frees another
    covariance — the defect, kept visible."""
    cs1, t1 = _table(["EP", "EU", "EC", "EA", "P"], DAG, tmp_path)
    cs2, t2 = _table(["EP", "EU", "EC", "P", "EA"], DAG, tmp_path)
    _apply(t1, CNS_POSITIONS, tmp_path, "a")
    _apply(t2, CNS_POSITIONS, tmp_path, "b")
    assert _by_meaning(cs1, t1) != _by_meaning(cs2, t2)
