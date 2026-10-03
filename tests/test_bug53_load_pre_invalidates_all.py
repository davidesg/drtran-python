"""BUG-53 — `load_pre` invalidated half of the case's state: the diagonal logL
(the base of estimate's LR test) survived a reload with other series, and a
stale base gives a number, not the warning a missing one gives.

Now load_pre clears all six dicts, and the diagonal certificate carries the
files it is of (path and content): used against anything else, it is absent.
"""
import json
import os
import shutil

import pytest

pytest.importorskip("mcp")
import drtran.mcp_server as mtram  # noqa: E402

CASES = "/home/david/Dropbox/SRC/drtran/tests/cases"
ES = os.path.join(CASES, "ES_CPI_m10.pre")
WTI = os.path.join(CASES, "WTI_ar1.pre")

pytestmark = pytest.mark.skipif(not os.path.exists(ES),
                                reason="the canonical .pre files are missing")


def test_reload_with_check_false_drops_the_old_base():
    """Load and certify; reload the same NAME without the gate: the LR must
    say there is no base, not use the first case's."""
    mtram.load_pre("b53", f"{ES},{WTI}")
    assert "b53" in mtram._DIAG
    mtram.load_pre("b53", f"{ES},{WTI}", check=False)
    for d in (mtram._LINKS, mtram._FITS, mtram._TABLES, mtram._DIAG,
              mtram._DIAG_FIT, mtram._DIAG_OF):
        assert "b53" not in d
    mtram.set_network("b53", json.dumps([{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]))
    est = mtram.estimate("b53")
    assert "No hay verosimilitud diagonal guardada" in est


def test_an_edited_pre_makes_the_certificate_stale(tmp_path):
    """The same files, edited after the gate: the certificate is of other
    contents and is not used as the base."""
    es, wti = str(tmp_path / "ES.pre"), str(tmp_path / "WTI.pre")
    shutil.copy(ES, es)
    shutil.copy(WTI, wti)
    mtram.load_pre("b53e", f"{es},{wti}")
    assert mtram._diag_vigente("b53e")[0] is not None
    with open(wti, "a") as fh:
        fh.write("\n")
    assert mtram._diag_vigente("b53e") == (None, None)
    mtram.set_network("b53e", json.dumps([{"out": 0, "inp": 1, "b": 0, "r": 0, "s": 1}]))
    assert "es de OTROS ficheros" in mtram.estimate("b53e")
