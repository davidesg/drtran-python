"""Where the tests find what is not in this repository.

Some tests read the C programs' repositories (drtran, atws, atsw-gui): they
are looked for NEXT TO this one (the folder that holds the suite's
checkouts), or under ATSW_SRC. Private data is never located by a written
path: its path comes in an environment variable, and the test skips without
it. Nothing here names a personal folder (tests/test_sin_rutas_personales.py)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("ATSW_SRC") or os.path.dirname(ROOT)


def src(rel):
    """A path under the folder of the suite's checkouts: src("drtran/tests/cases")."""
    p = os.path.join(SRC, *[x for x in rel.split("/") if x])
    return p + os.sep if rel.endswith("/") else p


def private(var):
    """The path a private dataset is given in `var`, or "" (the test skips)."""
    return os.environ.get(var, "")
