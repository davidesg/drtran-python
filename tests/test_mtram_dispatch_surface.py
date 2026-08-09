"""mtram must SAY what drtran now does — the dispatch, and the operators.

`estimate(embed=True)` returning a fit that was not embedded, without a word, is
the same class of silence BUG-8 was. These pin that it speaks, and that it stays
quiet when there is nothing to say.
"""

import asyncio
import os

import pytest

pytest.importorskip("mcp")
drtran = pytest.importorskip("drtran")
from drtran import mcp_server as m  # noqa: E402

PT8 = "/home/david/Dropbox/SRC/atws/Taste/oracle/data/passthrough8"

pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.skipif(not os.path.exists(PT8),
                       reason="the oracle's passthrough data is missing"),
]


def _caso(nombre, salida, s):
    m.load_pre(nombre, f"{PT8}/{salida},{PT8}/PT8_WTI.pre")
    m.set_network(nombre, '[{"out":0,"inp":1,"b":0,"r":0,"s":%d}]' % s)
    return nombre


def test_every_tool_is_actually_registered():
    """Including the new one. A tool that exists and is not registered is
    invisible, and that has happened here before (BUG-4)."""
    nombres = {t.name for t in asyncio.run(m.mcp.list_tools())}
    for t in ("load_pre", "check_operators", "estimate", "forecast",
              "set_network", "diagnose"):
        assert t in nombres, f"{t} is not registered"


def test_check_operators_names_the_annihilated_gain_and_the_dispatch():
    out = m.check_operators(_caso("d_fr", "PT8_FR.pre", 1))
    assert "Delta(1) = 0" in out
    assert "FRECUENCIA CERO" in out and "ANIQUILADA" in out
    assert "cast de RESTA" in out
    assert "NO dividas la ganancia" in out          # the correction that looks obvious
    assert "216 observaciones" in out               # the common window, reported


def test_check_operators_is_quiet_when_the_operators_agree():
    out = m.check_operators(_caso("d_es", "PT8_ES.pre", 1))
    assert "Delta(1) = 1" in out and "IGUALES" in out
    assert "ANIQUILADA" not in out and "MULTIPLICADA" not in out


def test_estimate_says_when_it_did_not_honour_embed():
    r = m.estimate(_caso("e_fr", "PT8_FR.pre", 1), embed=True)
    assert "EL CAST" in r and "RESTA" in r
    assert "NO son" in r and "comparables" in r     # the likelihoods, warned about


def test_estimate_says_nothing_when_it_did():
    r = m.estimate(_caso("e_es", "PT8_ES.pre", 1), embed=True)
    assert "EL CAST" not in r
