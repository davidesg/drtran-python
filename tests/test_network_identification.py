"""Identifying the NETWORK: reading the CCFs of the diagonal model's residuals.

This is step 3 of the ladder. What is checked here is that the port reads the
same CCFs as the C and proposes the same thing — not that the proposal is any
good, which is the analyst's judgement.

**Careful when comparing with the binary:** a bare `-i` does NOT identify from
the diagonal model. It builds its own model (in m6: 61 slots, 46 free, logL
-1716.36, with a diagonal Sigma and a default transfer) and reads the CCFs of
THOSE residuals. To compare, it must be asked for `-0 -i` with the same
constraints. Confusing the two fits makes the figures disagree and look like a
bug in the port.
"""

import os
import re
import subprocess

import numpy as np
import pytest

drtran = pytest.importorskip("drtran")
from drtran.cast import build_cast_spec  # noqa: E402
from drtran.netid import (_bs_from_side, identify_network,  # noqa: E402
                          report_network, residuals, write_guided)
from drtran.slots import build_slots, read_cns  # noqa: E402

REPO = "/home/david/Dropbox/SRC/drtran"
DATA = os.path.join(REPO, "tests/data/m6")
BIN = os.path.join(REPO, "bin/drtran")
M6 = ["EP", "EI", "EU", "EC", "EA", "P"]

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "M6_EP.pre")),
    reason="the m6 .pre files from the C repo are missing")


def _cs(names=None):
    return build_cast_spec([drtran.load_pre(os.path.join(DATA, f"M6_{n}.pre"))
                            for n in (names or M6)])


# ── the contiguous block ─────────────────────────────────────────────────────
def test_the_structure_is_the_contiguous_block_not_the_stray_peaks():
    """An isolated peak far from the block is noise: with 5% bands one lag in
    twenty is expected outside by chance. It is the same decision as in the
    bivariate identification, and the one that keeps the answers sane."""
    c = np.zeros(13)
    c[2] = 0.40; c[3] = 0.35          # a block at k = 2..3
    c[9] = 0.45                        # an isolated peak, HIGHER
    b, s, _peak = _bs_from_side(c, 12, 0.25)
    assert (b, s) == (2, 1), "the block is contiguous from b, not up to the peak"


def test_with_nothing_significant_there_is_no_link():
    assert _bs_from_side(np.full(13, 0.05), 12, 0.25) is None


# ── the residuals ────────────────────────────────────────────────────────────
def test_the_residuals_come_from_elf_not_from_a_hand_filter():
    """`elf` with `atf=True`: these are the EXACT residuals, with their
    pre-sample initialisation, not a truncated approximation."""
    cs = _cs()
    a, ifault = residuals(drtran.x0_from_pre(cs), cs)
    assert ifault == 0
    assert a.shape == (64, 6)
    assert np.all(np.isfinite(a))


# ── homologation with the binary ─────────────────────────────────────────────
def _identification_from_the_C(out):
    """(covariances, links) from the identification block of an `.out`."""
    txt = open(out).read()
    cov = [(m[1], m[2], float(m[3])) for m in
           re.finditer(r"^\s+(\w+)\s+-\s+(\w+)\s+r\(0\) = ([+-][\d.]+)", txt, re.M)]
    lnk = [(m[1], m[2], float(m[3]), int(m[4]), int(m[5])) for m in
           re.finditer(r"^\s+(\w+)\s+->\s+(\w+)\s+peak ([+-][\d.]+)\s+"
                       r"proposal\s+b=(\d+) r=0 s=(\d+)", txt, re.M)]
    return cov, lnk


@pytest.mark.skipif(not os.path.exists(BIN), reason="the C binary is missing")
def test_the_identified_network_homologates_with_the_binary(tmp_path):
    """The reference case: the whole of m6, identifying from the SAME diagonal.

    Compared against the binary live. The comparison is demanding on purpose: the
    same pairs, in the same order, with the same peaks and the same (b, s)
    proposals.
    """
    pre = [os.path.join(DATA, f"M6_{n}.pre") for n in M6]
    out = str(tmp_path / "i.out")
    r = subprocess.run([BIN, *pre, "-0", "-i", "-c", os.path.join(DATA, "m6.cns"),
                        "-o", out], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr
    ll_C = float(re.search(r"^Log-likelihood\s*=\s*(-?\d+\.\d+)",
                           open(out).read(), re.M).group(1))
    assert ll_C == pytest.approx(-1709.511575, abs=1e-5), "the C binary changed"
    cov_C, lnk_C = _identification_from_the_C(out)
    assert cov_C and lnk_C, "could not read the identification from the .out"

    # the port, AT THE SAME POINT: the parameters the C reports are mapped over
    from collections import Counter, defaultdict
    pairs, inside = [], False
    for ln in open(out):
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
    by_name = defaultdict(list)
    for n, v in pairs:
        by_name[n].append(v)
    cs = _cs()
    t = build_slots(cs)
    read_cns(os.path.join(DATA, "m6.cns"), t)
    used, x = Counter(), np.zeros(len(t))
    for i, s in enumerate(t.slots):
        vs = by_name.get(s.name)
        assert vs, f"the C does not report {s.name}"
        k = used[s.name]; used[s.name] += 1
        x[i] = vs[k] if k < len(vs) else vs[-1]

    net = identify_network(cs, x=x)
    nb = net.names

    assert net.band == pytest.approx(2.0 / 8.0)       # n = 64
    assert net.nlags == 8                              # 2 x the quarterly freq

    assert [(nb[i], nb[j], round(r0, 3)) for i, j, r0 in net.covariances] == \
           [(a, b, round(v, 3)) for a, b, v in cov_C]

    assert [(nb[c.inp], nb[c.out], round(c.peak, 3), c.b, c.s)
            for c in net.candidates] == \
           [(a, b, round(v, 3), bb, ss) for a, b, v, bb, ss in lnk_C]


def test_guided_mode_writes_the_DRAFT_cycle_included(tmp_path):
    """The `.dag` and the `.cns` are written unpruned, and the cycle is NOTED.

    In m6 the proposal comes out cyclic (EP -> EC -> EA -> EP): reading the CCFs
    pair by pair does not impose acyclicity. That is not a failure — it is the
    part of the job that falls to the analyst. Hiding it, or pruning
    unilaterally, would be worse: the file is the draft the decision is made on.

    The covariances go by NUMERIC index, not by name: the `.cns` does not read
    names in the `q`, and the C's own `-i` prints them with names under a heading
    that says "paste into a -c file" — which cannot be pasted.
    """
    from drtran.network import read_dag

    cs = _cs()
    net = identify_network(cs, x=drtran.x0_from_pre(cs))
    assert net.cycle is not None, "in m6 the raw proposal is cyclic"

    dag, cns = write_guided(net, str(tmp_path / "guide"))
    header = open(dag).read()
    assert "CYCLE" in header and "Prune" in header

    with pytest.raises(ValueError, match="CYCLE"):
        read_dag(dag, cs.names)                # and the reader does not let it pass

    # Prune: drop the smallest-peak link of each cycle until it is a DAG. More
    # than one pass is needed — m6 has several cycles — which is exactly why the
    # library does not do this: deciding which one falls is judgement, not
    # arithmetic. Here it is pruned by the peak only to exercise the full cycle
    # of the method.
    pruned = 0
    while net.cycle is not None:
        cycle = set(net.cycle)
        drop = min((c for c in net.candidates
                    if c.out in cycle and c.inp in cycle),
                   key=lambda c: abs(c.peak))
        net.candidates.remove(drop)
        pruned += 1
        assert pruned < 10, "pruning in circles"
    assert pruned >= 2, "in m6 more than one link has to be pruned"
    dag2, cns2 = write_guided(net, str(tmp_path / "pruned"))

    links = read_dag(dag2, cs.names)
    assert len(links) == len(net.candidates)
    cs2 = build_cast_spec([s.spec for s in cs.series], links=links)
    t = build_slots(cs2)
    n0 = t.n_free
    assert read_cns(cns2, t) == len(net.covariances)
    assert t.n_free == n0 + len(net.covariances)
    assert "q[" in open(cns2).read() and "q[EI" not in open(cns2).read()


def test_the_report_says_it_is_a_guide():
    """The identified network is a GUIDE, not the final one. That the report says
    so is not courtesy: it is the school's doctrine, and removing it invites
    estimating the draft."""
    cs = _cs()
    txt = report_network(identify_network(cs, x=drtran.x0_from_pre(cs)))
    assert "GUIDE" in txt and "prune" in txt
    assert "exogeneity" in txt and "acyclicity" in txt
