"""BUG-65: three tests of tests/test_cli.py read the canonical .pre files of
/home/david/Dropbox/SRC/drtran/tests/cases without the `needs_cases` marker the
other eight carry, so wherever those files are missing they FAIL instead of
skipping:

    test_estwin_is_the_same_option_as_R
    test_prewhiten_only_reports_and_does_not_estimate
    test_the_report_carries_the_variation_columns_and_the_decomposition

Found 2026-10-30 by the clean-install test of atsw 2.1.0 in ten distributions
(podman, drtran 0.2.6 from PyPI): the three fail in all ten with "drtran:
cannot open /home/david/Dropbox/SRC/drtran/tests/cases/ES_CPI_m10.pre".

The repro reads the test file and checks that those three carry the marker
(the other two that name ES without it only exercise paths where the file is
never opened, and pass without it). Expected: all three marked.

    python3 scripts/repro_test_cli_sin_needs_cases.py   # exits 1 while open
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "tests", "test_cli.py")
tree = ast.parse(open(SRC, encoding="utf-8").read())
FAILING = {"test_estwin_is_the_same_option_as_R",
           "test_prewhiten_only_reports_and_does_not_estimate",
           "test_the_report_carries_the_variation_columns_and_the_decomposition"}
bad = []
for f in tree.body:
    if isinstance(f, ast.FunctionDef) and f.name in FAILING and not any(
            isinstance(d, ast.Name) and d.id == "needs_cases" for d in f.decorator_list):
        bad.append(f.name)
for name in bad:
    print("reads the canonical cases without needs_cases:", name)
sys.exit(1 if bad else 0)
