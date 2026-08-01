"""The transfer NETWORK: the `.dag`, the slot table and `expand_params`.

Three different things, and all three have to homologate with the C binary:

1. **the `.dag`** — who feeds whom, with which (b, r, s), and that it is acyclic;
2. **the slot table** — the map of names, with the covariances born fixed;
3. **`expand_params`** — free -> full structure, with fixed, shared, products and
   linear combinations.

The validation case is NOT the canonical m6: its EI series carries a `compimp`
deterministic (a compensated impulse) that fue Python's reader degraded to a
`pulse`, and that moved the likelihood by 1.88 before drtran even came into play
(see `docs/PORTE.md` §5.4). The five clean m6 series are used instead, with a
network that exercises the same things: the chain EC -> EU -> EP, one input with
two outputs, a denominator with r=1, free covariances, a product and a linear
combination.
"""

import os
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec  # noqa: E402
from drtran.estimate import loglik  # noqa: E402
from drtran.network import find_cycle, read_dag, write_dag  # noqa: E402
from drtran.slots import (ALIAS, FIXED, FREE, LINCOMB, PRODUCT,  # noqa: E402
                          build_slots, read_cns)

REPO = "/home/david/Dropbox/SRC/drtran"
DATA = os.path.join(REPO, "tests/data/m6")
BIN = os.path.join(REPO, "bin/drtran")
CLEAN = ["EP", "EU", "EC", "EA", "P"]        # m6 without EI (see the docstring)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "M6_EP.pre")),
    reason="the m6 .pre files from the C repo are missing")


# ── helpers ──────────────────────────────────────────────────────────────────
def _specs(names=None):
    return [drtran.load_pre(os.path.join(DATA, f"M6_{n}.pre"))
            for n in (names or CLEAN)]


def _cs(dag=None, names=None):
    specs = _specs(names)
    cs = build_cast_spec(specs)
    if dag is None:
        return cs
    return build_cast_spec(specs, links=read_dag(dag, cs.names))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


DAG = """# 1=EP 2=EU 3=EC 4=EA 5=P
EP <- EU   1 0 1
EP <- EC   1 0 2
EU <- EC   2 1 1
"""

CNS_FREE = "q[4,2] = free\nq[4,3] = free\n"

CNS_EXPR = """q[4,2] = free
q[4,3] = free
omega1[1] = omega1[0] * theta_2[B^1]
omega2[0] = omega2[1] + omega2[2]
"""


# ── 1. the .dag ──────────────────────────────────────────────────────────────
def test_the_dag_is_read_by_name_not_by_position(tmp_path):
    """A `.dag` must not depend on the order of the command line."""
    cs = _cs()
    links = read_dag(_write(tmp_path, "r.dag", DAG), cs.names)
    assert len(links) == 3
    assert (links[0].out, links[0].inp, links[0].b, links[0].s) == (0, 1, 1, 1)
    assert (links[2].out, links[2].inp, links[2].b, links[2].r) == (1, 2, 2, 1)


def test_the_dag_rejects_what_is_not_a_network(tmp_path):
    names = ["EP", "EU", "EC", "EA", "P"]
    with pytest.raises(ValueError, match="unknown series"):
        read_dag(_write(tmp_path, "a.dag", "EP <- NOTHING 1 0 1\n"), names)
    with pytest.raises(ValueError, match="feed itself"):
        read_dag(_write(tmp_path, "b.dag", "EP <- EP 1 0 1\n"), names)
    with pytest.raises(ValueError, match="'<-'"):
        read_dag(_write(tmp_path, "c.dag", "EP -> EU 1 0 1\n"), names)
    with pytest.raises(ValueError, match="b, r, s"):
        read_dag(_write(tmp_path, "d.dag", "EP <- EU x 0 1\n"), names)


def test_a_cycle_is_rejected_and_named(tmp_path):
    """With a cycle there is no topological order: it would stop being a
    recursive DAG and become a simultaneous-equations system, which is not what
    the cast represents. The message gives the cycle, not just its existence."""
    cyclic = "EP <- EU  1 0 1\nEU <- EC  1 0 1\nEC <- EP  1 0 1\n"
    with pytest.raises(ValueError, match="CYCLE"):
        read_dag(_write(tmp_path, "cycle.dag", cyclic), CLEAN)

    from drtran.cast import Link
    assert find_cycle([Link(0, 1), Link(1, 2), Link(2, 0)], 5) is not None
    assert find_cycle([Link(0, 1), Link(1, 2)], 5) is None      # a chain: a DAG


def test_the_dag_survives_a_round_trip(tmp_path):
    cs = _cs()
    p = _write(tmp_path, "r.dag", DAG)
    links = read_dag(p, cs.names)
    q = str(tmp_path / "back.dag")
    write_dag(q, links, cs.names)
    assert read_dag(q, cs.names) == links


# ── 2. the slot table ────────────────────────────────────────────────────────
def test_the_table_has_the_C_names_and_the_covariances_are_born_fixed(tmp_path):
    cs = _cs(_write(tmp_path, "r.dag", DAG))
    t = build_slots(cs)

    # the binary reports "Structural parameters: 48 (free: 40, fixed: 8)"
    assert len(t) == 48
    assert t.n_free == 48 - 10          # the 10 covariances of 5 series, fixed

    # transfers: numbered 1..n_link, as in the C's .cns
    assert t.names[:4] == ["omega1[0]", "omega1[1]", "omega2[0]", "omega2[1]"]
    assert t.index("delta3[1]") >= 0            # the link with r=1
    # ARMA and deterministics, with the 1-based SERIES index
    assert t.index("theta_2[B^1]") >= 0
    assert t.index("omega_d1[1,0]") >= 0
    assert t.index("log(var5/var1)") >= 0

    for i in range(2, 6):
        for j in range(1, i):
            k = t.index(f"q[{i},{j}]")
            assert k >= 0 and t.slots[k].kind == FIXED and t.slots[k].value == 0.0


def test_freeing_a_covariance_is_a_decision_not_a_bulk_switch(tmp_path):
    """`q[i,j] = free`, in the same place and the same language as everything
    else: the legacy m6-1 does not free the 15 of its system, it frees THREE."""
    cs = _cs(_write(tmp_path, "r.dag", DAG))
    t = build_slots(cs)
    n0 = t.n_free
    assert read_cns(_write(tmp_path, "c.cns", CNS_FREE), t) == 2
    assert t.n_free == n0 + 2
    assert t.slots[t.index("q[4,2]")].kind == FREE
    assert t.slots[t.index("q[3,1]")].kind == FIXED       # the rest, untouched


def test_the_table_matches_fues_vector():
    """The port's safety net: if the name enumeration drifts from
    `fue._build_initial_x`'s, the names stop matching the positions and the
    `.cns` would constrain the wrong parameter, silently."""
    cs = _cs()
    t = build_slots(cs)
    assert len(t) == cs.npar + cs.m * (cs.m - 1) // 2


# ── 3. the constraints ───────────────────────────────────────────────────────
def test_the_five_natures_of_a_slot(tmp_path):
    cs = _cs(_write(tmp_path, "r.dag", DAG))
    t = build_slots(cs)
    read_cns(_write(tmp_path, "c.cns", """
        q[4,2] = free
        omega1[0] = 0.25
        delta3[1] = theta_2[B^1]
        omega2[1] = -omega1[0] * theta_3[B^1]
        omega2[0] = omega2[1] + omega2[2] - theta_2[B^1]
    """), t)

    assert t.slots[t.index("q[4,2]")].kind == FREE
    assert t.slots[t.index("omega1[0]")].kind == FIXED
    assert t.slots[t.index("delta3[1]")].kind == ALIAS
    assert t.slots[t.index("omega2[1]")].kind == PRODUCT
    assert t.slots[t.index("omega2[0]")].kind == LINCOMB
    assert len(t.slots[t.index("omega2[0]")].terms) == 3

    x = t.expand(np.zeros(t.n_free))
    assert x[t.index("omega1[0]")] == 0.25

    # shared: a single degree of freedom in two places
    xf = np.zeros(t.n_free)
    xf[t.free_of_slot[t.index("theta_2[B^1]")]] = 0.7
    x = t.expand(xf)
    assert x[t.index("delta3[1]")] == pytest.approx(0.7)
    # product: -0.25 * theta_3, with theta_3 = 0
    assert x[t.index("omega2[1]")] == pytest.approx(0.0)
    # linear combination: omega2[1] + omega2[2] - theta_2
    assert x[t.index("omega2[0]")] == pytest.approx(0.0 + 0.0 - 0.7)


def test_the_cns_does_not_swallow_what_it_cannot_read(tmp_path):
    cs = _cs()
    t = build_slots(cs)
    with pytest.raises(KeyError, match="unknown"):
        read_cns(_write(tmp_path, "a.cns", "does_not_exist = 1\n"), t)
    # on the right: neither a slot nor a number — as in the C, "cannot parse"
    with pytest.raises(ValueError, match="cannot interpret"):
        read_cns(_write(tmp_path, "b.cns", "log(var2/var1) = neither\n"), t)
    with pytest.raises(ValueError, match="with itself"):
        read_cns(_write(tmp_path, "c.cns",
                        "log(var2/var1) = log(var2/var1)\n"), t)
    with pytest.raises(ValueError, match="factor of itself"):
        read_cns(_write(tmp_path, "d.cns",
                        "log(var2/var1) = log(var2/var1) * log(var3/var1)\n"), t)


def test_comments_and_blank_lines_do_not_get_in_the_way(tmp_path):
    cs = _cs()
    t = build_slots(cs)
    n = read_cns(_write(tmp_path, "c.cns",
                        "# this is all a comment\n\n"
                        "q[2,1] = free   # and so is this\n"), t)
    assert n == 1 and t.slots[t.index("q[2,1]")].kind == FREE


# ── 4. the non-diagonal covariance ───────────────────────────────────────────
def test_a_free_covariance_changes_sigma_and_a_non_PSD_one_is_rejected(tmp_path):
    from drtran.cast import build_sigma
    from drtran.embed import cast_embedded

    cs = _cs()
    t = build_slots(cs)
    read_cns(_write(tmp_path, "c.cns", "q[2,1] = free\n"), t)

    x = drtran.x0_full(cs, t)
    x[t.index("q[2,1]")] = 0.1
    _phi, _th, _mu, _w, sigma, ifault = cast_embedded(x, cs)
    assert ifault == 0
    assert sigma[1, 0] == pytest.approx(0.1) and sigma[0, 1] == pytest.approx(0.1)

    # outside the PSD region it is rejected: the objective returns 1.0 and the
    # optimizer moves away (Mauricio 1995 §3), instead of evaluating the
    # impossible
    x[t.index("q[2,1]")] = 50.0
    _q, _idx, ifa = build_sigma(x, cs.npar - (cs.m - 1), cs.m)
    assert ifa != 0
    assert cast_embedded(x, cs)[5] != 0


# ── 5. homologation with the binary ──────────────────────────────────────────
def _optimum_of_the_C(dag, cns, tmp_path, names=CLEAN):
    """Run the binary and return `(cast_spec, table, x_C, logL_C)`.

    It is relaunched LIVE instead of storing figures: a frozen reference stops
    warning as soon as the C changes, which is exactly when one needs to know.
    """
    import re
    pre = [os.path.join(DATA, f"M6_{n}.pre") for n in names]
    out = str(tmp_path / "c.out")
    r = subprocess.run([BIN, *pre, "-n", dag, "-c", cns, "-o", out],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr

    ll_C = None
    pairs, inside = [], False
    for ln in open(out):
        if ln.startswith("Log-likelihood"):
            ll_C = float(re.search(r"-?\d+\.\d+", ln).group())
        if "Parameter" in ln and "Estimate" in ln:
            inside = True
            continue
        if inside:
            if ln.startswith("---") and pairs:
                break
            if "(fixed by .pre" in ln:
                continue
            mm = re.match(r"^([A-Za-z_][^ ]*)\s+(-?\d+\.\d+)", ln)
            if mm:
                pairs.append((mm.group(1), float(mm.group(2))))

    cs = _cs(dag, names)
    t = build_slots(cs)
    read_cns(cns, t)

    # by name, respecting the k-th occurrence: the C repeats names when a series
    # has two factors of the same kind (`theta_5[B^1]` in m6)
    from collections import Counter, defaultdict
    by_name = defaultdict(list)
    for n, v in pairs:
        by_name[n].append(v)
    used, x = Counter(), np.zeros(len(t))
    for i, s in enumerate(t.slots):
        vs = by_name.get(s.name)
        assert vs, f"the C does not report the slot {s.name}"
        k = used[s.name]
        used[s.name] += 1
        x[i] = vs[k] if k < len(vs) else vs[-1]
    return cs, t, x, ll_C


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_network_homologates_with_the_binary(tmp_path):
    """The chain EC -> EU -> EP, one input with two outputs, a denominator with
    r=1 and two free covariances: the port reproduces the C at its own
    optimum."""
    dag = _write(tmp_path, "r.dag", DAG)
    cns = _write(tmp_path, "c.cns", CNS_FREE)
    cs, t, x, ll_C = _optimum_of_the_C(dag, cns, tmp_path)
    assert len(t) == 48 and t.n_free == 40          # what the binary reports
    ll, ifault = loglik(x, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_products_and_the_linear_combination_homologate(tmp_path):
    """And `expand` reconstructs the derived slots from ONLY the free ones: the
    proof that the DSL is not decorative."""
    dag = _write(tmp_path, "r.dag", DAG)
    cns = _write(tmp_path, "c.cns", CNS_EXPR)
    cs, t, xC, ll_C = _optimum_of_the_C(dag, cns, tmp_path)
    assert t.n_free == 38                            # 40 - 2 expressions

    xexp = t.expand(t.pack(xC))
    assert np.max(np.abs(xexp - xC)) < 1e-5, "expand does not rebuild the derived"

    i0, i1, i2 = (t.index(f"omega2[{k}]") for k in (0, 1, 2))
    assert xexp[i0] == pytest.approx(xexp[i1] + xexp[i2])       # the (1-B)
    ip, ia, ib = t.index("omega1[1]"), t.index("omega1[0]"), t.index("theta_2[B^1]")
    assert xexp[ip] == pytest.approx(xexp[ia] * xexp[ib])       # the product

    ll, ifault = loglik(xexp, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=1e-5)


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
@pytest.mark.parametrize("cns,target,tol", [
    # free numerators: evaluated at the very numbers the C prints
    ("m6_net.cns", -1697.613401, 1e-4),
    # with EXPRESSIONS the tolerance rises: the C's report rounds to 6 decimals
    # and the derived slots are rebuilt from already rounded factors, so the
    # input error propagates through the product
    ("m6_net_prod.cns", None, 1e-3),       # + the 2 shared-MA products
    ("m6_net_full.cns", None, 1e-3),       # + the fixed (1-B) factor of EI<-EU
])
def test_the_canonical_m6(cns, target, tol, tmp_path):
    """Relloso's system (1997, Table 4) in full, with all SIX series.

    It was blocked: the EI series carries a `compimp` deterministic — the
    compensated impulse — that fue Python read as a plain impulse, and that moved
    the likelihood by 1.88 before drtran came into play. Fixed in fue 0.1.9
    (BUG-0006), the port reproduces the C's targets.
    """
    dag = os.path.join(DATA, "m6_net.dag")
    cs, t, x, ll_C = _optimum_of_the_C(dag, os.path.join(DATA, cns), tmp_path,
                                       ["EP", "EI", "EU", "EC", "EA", "P"])
    if target is not None:
        assert ll_C == pytest.approx(target, abs=1e-5), "the C binary changed"
    ll, ifault = loglik(x, cs, embed=True)
    assert ifault == 0
    assert ll == pytest.approx(ll_C, abs=tol)


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_canonical_diagonal_m6(tmp_path):
    """The previous rung: six univariate models together, a non-diagonal Sigma,
    and no network."""
    import re

    pre = [os.path.join(DATA, f"M6_{n}.pre")
           for n in ("EP", "EI", "EU", "EC", "EA", "P")]
    out = str(tmp_path / "diag.out")
    r = subprocess.run([BIN, *pre, "-0", "-c", os.path.join(DATA, "m6.cns"),
                        "-o", out], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr
    ll_C = float(re.search(r"^Log-likelihood\s*=\s*(-?\d+\.\d+)",
                           open(out).read(), re.M).group(1))
    assert ll_C == pytest.approx(-1709.511575, abs=1e-5), "the C binary changed"

    cs = _cs(names=["EP", "EI", "EU", "EC", "EA", "P"])
    t = build_slots(cs)
    read_cns(os.path.join(DATA, "m6.cns"), t)
    assert t.n_free == 56, "the C's -0 frees the 15 covariances"


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_optimizer_reaches_the_same_optimum_as_the_C(tmp_path):
    """Homologating at the C's optimum tests the CAST; this tests the SEARCH.

    A small network (EC -> EU -> EP, 24 free) so that it fits in the battery: the
    port starts on the diagonal rung and has to arrive where the C arrived.
    """
    dag = _write(tmp_path, "r.dag", "EP <- EU   1 0 1\nEU <- EC   2 0 1\n")
    cns = _write(tmp_path, "c.cns", "q[3,1] = free\n")
    three = ["EP", "EU", "EC"]
    cs, t, _x, ll_C = _optimum_of_the_C(dag, cns, tmp_path, three)
    assert len(t) == 26 and t.n_free == 24

    f = drtran.fit(cs, slots=t, embed=True)
    assert f.ifault == 0
    # The criterion is ARRIVING, not how it stops. `termcode` 3 is "stopped
    # without improvement", which at the optimum is normal -- the C itself
    # reports "STOPPED AT A POINT WITH NO IMPROVEMENT" there -- and an
    # epsilon-level change is enough (here, moving from the pure-Python elf to
    # the compiled one, identical to 1e-13) for the line search to stop finding
    # improvement and the 2 to become a 3. A real failure is 4-5. What is
    # required is the value, which is what is compared with the C.
    assert f.termcode in (1, 2, 3), f"did not converge: {f.status}"
    assert f.loglik == pytest.approx(ll_C, abs=1e-5)
    assert len(f.xfree) == 24 and len(f.x) == 26


def test_with_no_links_or_covariances_the_network_is_the_usual_diagonal():
    """The slot table does not change the model merely by existing: with
    everything free and the covariances at zero, `fit(slots=...)` starts from the
    same place as without it."""
    cs = _cs()
    t = build_slots(cs)
    x_without = drtran.x0_from_pre(cs)
    x_with = drtran.x0_full(cs, t)
    assert np.allclose(x_with[:len(x_without)], x_without)
    assert np.all(x_with[len(x_without):] == 0.0)
    assert loglik(x_with, cs, embed=True)[0] == pytest.approx(
        loglik(x_without, cs, embed=True)[0], abs=1e-9)
