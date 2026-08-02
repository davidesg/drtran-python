"""mtram — the transfer assistant.

Most of what is tested here is not arithmetic: the arithmetic has its own tests.
It is the **architecture's rules**, the ones that make three assistants worth
having instead of one. If they erode, the analyst stops knowing which bench they
are at, and that is the failure this design exists to prevent.
"""

import json
import os

import pytest

pytest.importorskip("mcp")
drtran = pytest.importorskip("drtran")
import drtran.mcp_server as mtram  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
DATA = "/home/david/Dropbox/SRC/drtran/tests/data"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(
    not os.path.exists(ES),
    reason="the canonical .pre files from the C repo are missing")


@pytest.fixture
def caso():
    name = "t"
    mtram.load_pre(name, f"{ES},{WTI}")
    mtram.set_network(name, json.dumps([{"out": 0, "inp": 1, "b": 0, "r": 0,
                                         "s": 1}]))
    return name


# ── the entry contract: .pre, not raw series ─────────────────────────────────
def test_it_starts_from_pre_files_and_the_first_is_the_output():
    out = mtram.load_pre("x", f"{ES},{WTI}")
    assert "SALIDA : ES_CPI" in out
    assert "entrada: WTI" in out
    # it reports what each .pre carries, so the analyst can see the rung below
    assert "lambda=0 d=1" in out and "deterministas=11" in out


def test_one_series_is_not_a_transfer_model():
    with pytest.raises(ValueError, match="al menos dos"):
        mtram.load_pre("x", ES)


def test_a_missing_file_says_which():
    with pytest.raises(ValueError, match="no_existe"):
        mtram.load_pre("x", f"{ES},/tmp/no_existe.pre")


# ── THE handoff rule ─────────────────────────────────────────────────────────
def test_a_cyclic_network_is_refused_and_routed_to_sima():
    """The rule that makes three assistants a design and not an accident: a DAG
    with a cycle has no topological order, cannot be cast as a triangular VARMA,
    and therefore describes a SIMULTANEOUS system — which is `sima`'s subject,
    not this one's.

    The assistant must not prune the cycle itself to make the model fit.
    """
    mtram.load_pre("c", f"{ES},{WTI}")
    cyclic = json.dumps([{"out": 0, "inp": 1, "b": 1},
                         {"out": 1, "inp": 0, "b": 1}])
    with pytest.raises(ValueError, match="CYCLE|ciclo"):
        mtram.set_network("c", cyclic)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "m6")),
                    reason="the m6 data is missing")
def test_the_cycle_message_names_sima_and_the_cycle():
    """On m6 the raw proposal IS cyclic — the case the rule was written for. The
    message has to say which cycle and where to go, or the analyst is stuck."""
    m6 = os.path.join(DATA, "m6")
    names = ["EP", "EI", "EU", "EC", "EA", "P"]
    mtram.load_pre("m6", ",".join(os.path.join(m6, f"M6_{n}.pre") for n in names))
    txt = mtram.identify_network("m6")
    assert "CICLO" in txt
    assert "sima" in txt
    assert "->" in txt, "it must name the cycle, not just its existence"
    assert "poda" in txt.lower() or "PODE" in txt


# ── the refusals, which are answers ──────────────────────────────────────────
def test_the_variance_decomposition_refuses_rather_than_ordering(caso):
    """With correlated innovations the decomposition needs an ordering, which is
    the VAR's problem. Refusing IS the answer; the tool must say so in words the
    assistant can pass on."""
    from drtran.forecast import forecast as _f

    f = mtram._require_fit(caso) if caso in mtram._FITS else None
    if f is None:
        mtram.estimate(caso)
    # force the non-diagonal case by pretending there was no normalisation
    fc = _f(mtram._FITS[caso], L=3, embed=True)
    fc.phi0 = None
    from drtran.forecast import variance_decomposition as _v
    shares, why = _v(fc, series=0)
    assert shares is None
    assert "NOT UNIQUE" in why and "ORDERING" in why


def test_tools_refuse_before_the_model_exists():
    mtram.load_pre("u", f"{ES},{WTI}")
    with pytest.raises(ValueError, match="no está estimado"):
        mtram.diagnose("u")
    with pytest.raises(ValueError, match="no hay ningún caso"):
        mtram.diagnose("no_cargado")


# ── the numbers, spot-checked against the C ──────────────────────────────────
def test_the_canonical_case_end_to_end(caso):
    est = mtram.estimate(caso)
    assert "-718.287406" in est
    assert "0.016400" in est and "0.001703" in est
    assert "```" in est, "the equation must come in a verbatim block"

    assert "0.1966" in mtram.diagnose(caso)
    assert "0.027146" in mtram.impulse_response(caso)      # the gain

    fc = mtram.forecast(caso, 3)
    assert "82.0149" in fc and "81.6280" in fc and "82.4035" in fc
    assert "NO es simétrica" in fc, "the band's shape must be stated"


def test_the_forecast_band_is_the_asymmetric_one(caso):
    """A symmetric band from 1.96 s.e. would give [81.54, 82.49]. The right one
    is [81.63, 82.40] — formed in the transformed scale and mapped back."""
    mtram.estimate(caso)
    fc = mtram.forecast(caso, 1)
    assert "81.5421" not in fc and "82.4877" not in fc
