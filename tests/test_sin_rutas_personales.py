"""Nothing that ships names a personal folder.

The sdist carries src/, tests/ and the documents MANIFEST.in lists. Until
0.2.7 thirty of its test files named folders of the author's machine. The
C repositories are found next to this one (tests/_rutas.py, or ATSW_SRC);
private data comes in an environment variable, and its test skips without
it (docs/PUBLISHING.md of art-python: no personal path in the sdist)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPS = ("src", "tests", "scripts", "docs", "README.md", "CHANGELOG.md", "TODO.md")
#: written split, so that this file does not find itself
PATTERNS = ("/" + "home/", "/" + "Users/", "C:" + "\\\\Users")
TEXT = (".py", ".md", ".txt", ".json", ".toml", ".cfg", ".inp", ".pre", ".cns")


def _files():
    for top in SHIPS:
        p = os.path.join(ROOT, top)
        if os.path.isfile(p):
            yield p
        for d, _, fs in os.walk(p):
            for f in fs:
                if f.endswith(TEXT):
                    yield os.path.join(d, f)


def test_nothing_that_ships_names_a_personal_folder():
    if not os.path.isdir(os.path.join(ROOT, "src", "drtran")):
        pytest.skip("not a source tree (the suite runs against an installed drtran)")
    hits = []
    for f in _files():
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(pat in line for pat in PATTERNS):
                hits.append(f"{os.path.relpath(f, ROOT)}:{n}: {line.strip()[:100]}")
    assert not hits, "personal paths in what ships:\n" + "\n".join(hits)
